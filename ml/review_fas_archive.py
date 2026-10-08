"""Offline ESRQS identity audit; pypdf is an evidence-review dependency only.

No network, credentials, model fits, features or source-admission side effects.
"""
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.fas_archive import (
    embargo_declaration,
    identity_check,
    parse_index,
    reconcile_upland,
    report_period_end,
)
from cottonlens_ml.sources.fas_country_review import (
    reconcile_countries,
    subtype_precision,
)
from cottonlens_ml.sprint import freeze_record, read_record

HOST = 'apps.fas.usda.gov'
API = '/esrqs/api/reports/'


def checked_response(folder, method, query_key):
    receipt_path = folder / 'retrieval.json'
    receipt = read_record(receipt_path)
    url = urlsplit(receipt['source_url'])
    query = parse_qs(url.query, strict_parsing=True)
    if (url.scheme != 'https' or url.netloc != HOST or url.path != API + method
            or url.fragment or set(query) != {query_key} or len(query[query_key]) != 1
            or receipt['kind'] != 'export_sales'):
        raise ValueError('Wrong public ESRQS response identity')
    raw = folder / 'source.bin'
    checksum = digest(raw)
    if checksum != receipt['sha256']:
        raise ValueError('Evidence bytes corrupted')
    return raw, query[query_key][0], {
        'source_url': receipt['source_url'], 'sha256': checksum,
        'retrieved_at': receipt['retrieved_at'], 'receipt_sha256': digest(receipt_path),
    }


def checked_release_cover(folder):
    receipt_path = folder / 'retrieval.json'
    receipt = read_record(receipt_path)
    url = urlsplit(receipt['source_url'])
    if (url.scheme != 'https' or url.netloc != 'content.govdelivery.com'
            or not url.path.startswith('/attachments/USDAFAS/') or url.query or url.fragment
            or receipt.get('publisher_account') != 'USDAFAS'):
        raise ValueError('USDAFAS public attachment identity required')
    raw = folder / 'govdelivery-wr06042020.pdf'
    if digest(raw) != receipt['sha256']:
        raise ValueError('Release-cover evidence bytes corrupted')
    return raw, {'source_url': receipt['source_url'], 'sha256': receipt['sha256'],
                 'retrieved_at': receipt['retrieved_at'], 'receipt_sha256': digest(receipt_path)}


def review(index_folder, report_folders, year, output, *, api_snapshot=None, commodity_catalog=None,
           release_cover=None, country_reference=None, subtype_review=False):
    import pypdf

    if subtype_review and country_reference is None:
        raise ValueError('Subtype review requires the pinned country reference')
    raw, requested_year, index_evidence = checked_response(
        index_folder, 'GetArchivedWeeklyReportsList', 'selectedYear')
    if requested_year != str(year):
        raise ValueError('Requested archive year differs from frozen index')
    records = parse_index(raw.read_bytes(), year)
    if (api_snapshot is None) != (commodity_catalog is None):
        raise ValueError('Both pinned API snapshot and commodity catalog required')
    numeric_inputs = None
    reference_text, reference_evidence = None, None
    if country_reference is not None:
        if api_snapshot is None or commodity_catalog is None:
            raise ValueError('Country review requires pinned API and commodity catalog')
        receipt = json.loads((country_reference / 'retrieval.json').read_text(encoding='utf-8'))
        source = country_reference / 'source.html'
        if (receipt['source_url'] != 'https://www.census.gov/foreign-trade/schedules/c/countrycodes.html'
                or receipt.get('http_status') != 200 or digest(source) != receipt['sha256']):
            raise ValueError('Country reference origin/bytes mismatch')
        reference_text = source.read_text(encoding='utf-8')
        reference_evidence = {'source_url': receipt['source_url'], 'sha256': receipt['sha256'],
                              'retrieved_at': receipt['retrieved_at'],
                              'receipt_sha256': digest(country_reference / 'retrieval.json')}
    if api_snapshot is not None:
        payload = json.loads(api_snapshot.read_bytes())
        catalog = json.loads(commodity_catalog.read_bytes())
        if not isinstance(payload, list) or not isinstance(catalog, list):
            raise TypeError('API snapshot and catalog must be arrays')
        numeric_inputs = {'api_snapshot_sha256': digest(api_snapshot),
                          'commodity_catalog_sha256': digest(commodity_catalog)}
    responses, ids = [], set()
    for folder in report_folders:
        pdf, archive_id, evidence = checked_response(folder, 'GetPdfFile', 'Id')
        if archive_id in ids:
            raise ValueError('Duplicate requested archive ID')
        ids.add(archive_id)
        if not pdf.read_bytes().startswith(b'%PDF-'):
            raise ValueError('Expected a PDF payload')
        document = pypdf.PdfReader(pdf)
        if not document.pages:
            raise ValueError('Empty PDF document')
        text = document.pages[0].extract_text()
        result = identity_check(records, archive_id, text)
        if numeric_inputs is not None:
            if not result['index_date_equals_pdf_period_end']:
                raise ValueError('Selected archive period differs; no numerical reconciliation')
            full_text = '\n'.join(page.extract_text() for page in document.pages)
            result['cotton_content_comparison'] = reconcile_upland(payload, catalog, full_text)
            result['numeric_reconciliation_allowed'] = True
            result['numeric_reconciliation_scope'] = 'content_diagnostic_only_not_publication_or_vintage'
            if reference_text is not None:
                result['country_stock_comparison'] = reconcile_countries(payload, catalog, full_text, reference_text)
                if subtype_review:
                    result['subtype_precision_diagnostic'] = subtype_precision(payload, catalog, full_text, reference_text)
        responses.append({**result, **evidence, 'pdf_pages': len(document.pages),
                          'pdf_creation_date_literal': str((document.metadata or {}).get('/CreationDate'))})
    if not responses:
        raise ValueError('At least one report response required')
    cover = None
    if release_cover is not None:
        pdf, evidence = checked_release_cover(release_cover)
        document = pypdf.PdfReader(pdf)
        full_text = '\n'.join(page.extract_text() for page in document.pages)
        period = report_period_end(full_text).isoformat()
        if period not in {r['pdf_report_period_end'] for r in responses}:
            raise ValueError('Release cover refers to a different report period')
        cover = {**evidence, **embargo_declaration(document.pages[0].extract_text()),
                 'report_period_end': period, 'pdf_pages': len(document.pages),
                 'numeric_comparison_performed': False,
                 'same_version_as_selected_archive_verified': False}
        if reference_text is not None:
            comparison = reconcile_countries(payload, catalog, full_text, reference_text)
            archive = next(r['country_stock_comparison'] for r in responses if r['pdf_report_period_end'] == period)
            cover['country_stock_comparison'] = comparison
            cover['numeric_comparison_performed'] = True
            cover['selected_country_content_matches_archive'] = comparison['countries'] == archive['countries']
    country_match = (all(r['country_stock_comparison']['all_stock_values_match'] for r in responses)
                     and (cover is None or cover['country_stock_comparison']['all_stock_values_match'])) \
        if reference_text is not None else None
    body = {
        'scope': 'observed public archive responses, not a historical publication ledger',
        'year': year, 'index': index_evidence, 'index_rows': len(records),
        'placeholder_created_time_rows': sum(r['created_time_literal'] == '0001-01-01T00:00:00'
                                             for r in records),
        'responses': sorted(responses, key=lambda r: r['archive_id']),
        'distinct_pdf_payloads': len({r['sha256'] for r in responses}),
        'pdf_parser': {'name': 'pypdf', 'version': pypdf.__version__, 'page': 1},
        'review_code_sha256': digest(Path(__file__)),
        'identity_code_sha256': digest(Path(__file__).parent / 'src/cottonlens_ml/sources/fas_archive.py'),
        'model_eligible': False, 'publication_timestamp_verified': False,
        'numeric_inputs': numeric_inputs,
        'country_reference': reference_evidence,
        'subtype_precision_diagnostic_requested': subtype_review,
        'country_stock_values_match': country_match,
        'country_review_code_sha256': digest(Path(__file__).parent / 'src/cottonlens_ml/sources/fas_country_review.py')
            if country_reference is not None else None,
        'release_cover': cover,
        'numeric_reconciliation_allowed': numeric_inputs is not None, 'cause_verified': False,
        'decision': ('country stock mismatch; no source admission; publication/version metadata still required'
                     if country_match is False else
                     'content comparison only; publication/version metadata still required' if numeric_inputs
                     else 'identity review only; no publication/version admission'),
        'pdf_creation_date_is_publication_proof': False,
    }
    freeze_record(output, body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--api-snapshot', type=Path)
    parser.add_argument('--commodity-catalog', type=Path)
    parser.add_argument('--release-cover', type=Path)
    parser.add_argument('--country-reference', type=Path, help='Pinned current Census reference folder; no historical admission')
    parser.add_argument('--subtype-precision', action='store_true', help='Diagnostic rounding assumption only; direct stock gate remains fixed')
    args = parser.parse_args()
    result = review(args.index, args.report, args.year, args.output,
                    api_snapshot=args.api_snapshot, commodity_catalog=args.commodity_catalog,
                    release_cover=args.release_cover, country_reference=args.country_reference,
                    subtype_review=args.subtype_precision)
    print(f"Reviewed {len(result['responses'])} responses; "
          f"unique PDF payloads={result['distinct_pdf_payloads']}; model_eligible=false")
    if result['country_stock_values_match'] is False:
        print('Country stock mismatch: review preserved; source admission blocked.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
