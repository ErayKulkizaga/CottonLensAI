"""Offline PDF/CSV verification. Numeric agreement never grants historical admission."""
import argparse
import hashlib
import io
import re
from decimal import InvalidOperation
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.public import utc_timestamp, validate_url
from cottonlens_ml.sources.wasde import (
    ReleasePage,
    compare_pdf_layout_rows,
    compare_pdf_pages,
    cotton_rows,
)
from cottonlens_ml.sources.wasde_regional import LEVELS, add_revisions, regional_values
from cottonlens_ml.sources.wasde_regional import VERSION as CANDIDATE_VERSION

VERSION = 'wasde-regional-numeric-verification-v1'


def archive_source(root, sha):
    if not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha):
        raise ValueError('Safe SHA-256 source identity required')
    folder = Path(root) / 'wasde' / sha
    receipt = read_record(folder / 'retrieval.json')
    if receipt['kind'] != 'wasde' or receipt['sha256'] != sha or digest(folder / 'source.bin') != sha:
        raise ValueError('WASDE retrieval/source checksum mismatch')
    utc_timestamp(receipt['retrieved_at'])  # An ingestion bound, never a historical publication.
    return folder / 'source.bin', receipt


def archive_index(root):
    by_url, hashes = {}, {}
    for path in root.glob('wasde/*/retrieval.json'):
        receipt = read_record(path)
        by_url.setdefault(receipt['source_url'], set()).add(receipt['sha256'])
    for path in root.glob('wasde_url_aliases/*/alias.json'):
        alias = read_record(path)
        validate_url('wasde', alias['source_url'])
        validate_url('wasde', alias['canonical_source_url'])
        if hashlib.sha256(alias['source_url'].encode()).hexdigest() != path.parent.name:
            raise ValueError('WASDE URL alias path mismatch')
        _, receipt = archive_source(root, alias['sha256'])
        if receipt['source_url'] != alias['canonical_source_url']:
            raise ValueError('WASDE URL alias source mismatch')
        by_url.setdefault(alias['source_url'], set()).add(alias['sha256'])
        hashes[path.relative_to(root).as_posix()] = digest(path)
    return by_url, hashes


def verify_pdf(xml_rows, path):
    """Fresh extraction of the original PDF, not a previous investigator's pass flag."""
    import pypdf

    raw = path.read_bytes()
    reader = pypdf.PdfReader(io.BytesIO(raw))
    # Page positions change across vintages. Finding one Cotton page in a fixed
    # window does not establish full crop-year/forecast coverage.
    indexes = list(range(len(reader.pages)))
    pages = [reader.pages[i].extract_text(extraction_mode='layout') for i in indexes]
    relevant = [i for i, page in zip(indexes, pages, strict=True) if 'World Cotton Supply and Use' in ' '.join(page.split())]
    try:
        comparison = compare_pdf_layout_rows(xml_rows, pages)
        method = 'pypdf fixed-column layout'
    except (ValueError, InvalidOperation):
        import pdfplumber

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            # A damaged text layer can hide an entire title in the first
            # extractor. Reinspect ALL pages with the independent extractor.
            selected = list(range(len(pdf.pages)))
            for x, y in [(3, 3), (1, 5)]:
                pages = [pdf.pages[i].extract_text(x_tolerance=x, y_tolerance=y) or '' for i in selected]
                try:
                    comparison = compare_pdf_layout_rows(xml_rows, pages)
                except (ValueError, InvalidOperation):
                    try:
                        comparison = compare_pdf_pages(xml_rows, pages)
                    except (ValueError, InvalidOperation):
                        if (x, y) == (1, 5):
                            raise
                        continue
                # Numeric mismatches stop; a different extraction setting is
                # allowed only for an unsupported layout, never to erase errors.
                method = f'pdfplumber independent text extraction x={x} y={y}'
                relevant = [i for i, page in enumerate(pages) if 'World Cotton Supply and Use' in ' '.join(page.split())] or selected
                break
    if not comparison['all_values_match']:
        raise ValueError(f"PDF/XML mismatch: {len(comparison['mismatches'])} cells")
    return {'method': method, 'compared_cells': comparison['compared_cells'],
        'pdf_page_indexes': relevant, 'pdf_sha256': hashlib.sha256(raw).hexdigest(),
        'pdf_creation_metadata': str(reader.metadata.get('/CreationDate')) if reader.metadata else None,
        'pdf_modification_metadata': str(reader.metadata.get('/ModDate')) if reader.metadata else None,
        'metadata_is_publication_evidence': False}


def verify_candidate_values(frame, records):
    if (frame.report_date.isna().any() or frame.report_date.duplicated().any()
            or not frame.report_date.is_monotonic_increasing):
        raise ValueError('Unique chronological candidate dates required')
    expected = pd.DataFrame([{'report_date': pd.Timestamp(r['report_date']), **r['values']} for r in records])
    expected = expected.sort_values('report_date').reset_index(drop=True)
    for _, copies in expected.groupby('report_date'):
        if len(copies[['crop_year', *LEVELS]].drop_duplicates()) != 1:
            raise ValueError('Conflicting source versions cannot be collapsed')
    expected = expected.drop_duplicates('report_date').reset_index(drop=True)
    same_month = expected.report_date.dt.to_period('M').eq(expected.report_date.shift().dt.to_period('M'))
    identical = expected[['crop_year', *LEVELS]].eq(expected[['crop_year', *LEVELS]].shift()).all(axis=1)
    expected = add_revisions(expected.loc[~(same_month & identical)].reset_index(drop=True))
    columns = ['crop_year', *LEVELS, *[name + '_revision' for name in LEVELS]]
    if not set(columns).issubset(frame) or not np.array_equal(expected.report_date, frame.report_date):
        raise ValueError('Candidate/source date or field coverage mismatch; no intersection')
    if not np.allclose(frame[columns], expected[columns], rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError('Candidate values or revisions differ from verified source rows')
    return expected


def analyze(candidate, archive_root, output):
    candidate, archive_root, output = map(Path, (candidate, archive_root, output))
    if (output / 'complete.json').exists():
        raise ValueError('Completed audit is immutable; use a new namespace')
    complete = read_record(candidate / 'complete.json')
    if (complete.get('completed') is not True or complete.get('fits') != 0
            or complete.get('version') != CANDIDATE_VERSION):
        raise ValueError('Completed pinned regional candidate required')
    inputs = {name: digest(candidate / name) for name in ['complete.json', 'candidate-manifest.json', 'regional.csv']}
    if any(inputs[name] != expected for name, expected in complete['files'].items()):
        raise ValueError('Candidate payload checksum mismatch')
    manifest = read_record(candidate / 'candidate-manifest.json')
    if (manifest['version'] != CANDIDATE_VERSION or manifest['files']['regional.csv'] != inputs['regional.csv']
            or any(manifest[key] is not False for key in ['model_eligible', 'publication_timestamp_verified', 'first_version_verified'])):
        raise ValueError('Unverified-time candidate required; no status promotion')
    import pypdf

    identity = {'inputs': inputs, 'version': VERSION, 'pypdf': pypdf.__version__,
        'source_code': {Path(fn.__code__.co_filename).name: digest(Path(fn.__code__.co_filename))
                        for fn in [analyze, compare_pdf_layout_rows, regional_values, utc_timestamp, read_record, digest, manifest_id]}}
    # Bind fallback dependency too; a changed extractor cannot reuse an old pass.
    try:
        import pdfplumber
        identity['pdfplumber'] = pdfplumber.__version__
    except ImportError:
        identity['pdfplumber'] = None
    namespace = manifest_id(identity)
    checkpoints = output / 'checkpoints' / namespace
    checkpoints.mkdir(parents=True, exist_ok=True)
    by_url, hashes = archive_index(archive_root)
    alias_hashes = hashes.copy()
    results = []
    for index, source in enumerate(manifest['source_versions']):
        xml, xml_receipt = archive_source(archive_root, source['xml_sha256'])
        page, _ = archive_source(archive_root, source['page_sha256'])
        validate_url('wasde', source['release_url'])
        match = re.fullmatch(r'https://esmis\.nal\.usda\.gov/publication/world-agricultural-supply-and-demand-estimates/(\d{4}-\d{2}-\d{2})(?:-\d+)?', source['release_url'])
        if not match or match[1] != source['report_date']:
            raise ValueError('Dated release/report identity mismatch')
        if by_url.get(source['release_url']) != {source['page_sha256']}:
            raise ValueError('Release page belongs to another source URL/version')
        html = ReleasePage()
        html.feed(page.read_text(encoding='utf-8'))
        xml_links = {urljoin(source['release_url'], url) for url in html.links if urlsplit(url).path.lower().endswith('.xml')}
        if len(xml_links) != 1 or by_url.get(next(iter(xml_links))) != {source['xml_sha256']}:
            raise ValueError('Release XML link/version mismatch')
        links = {urljoin(source['release_url'], url) for url in html.links if urlsplit(url).path.lower().endswith('.pdf')}
        if len(links) != 1:
            raise ValueError('One exact official archived PDF link required')
        url = next(iter(links))
        options = set(by_url.get(url, []))
        if len(options) != 1:
            raise ValueError('Missing or multiple archived PDF versions; no implicit latest choice')
        pdf, pdf_receipt = archive_source(archive_root, next(iter(options)))
        source_id = {'release_url': source['release_url'], 'report_date': source['report_date'],
            'xml_sha256': source['xml_sha256'], 'page_sha256': source['page_sha256'], 'pdf_sha256': next(iter(options)),
            'retrieval_sha256': {kind: digest(file.parent / 'retrieval.json') for kind, file in [('xml', xml), ('page', page), ('pdf', pdf)]},
            'url_alias_sha256': alias_hashes}
        for file in [xml, page, pdf, xml.parent / 'retrieval.json', page.parent / 'retrieval.json', pdf.parent / 'retrieval.json']:
            hashes[file.relative_to(archive_root).as_posix()] = digest(file)
        path = checkpoints / (manifest_id(source_id) + '.json')
        if path.exists():
            item = read_record(path)
            if item['identity'] != source_id or item['namespace'] != namespace or item['numeric_verified'] is not True:
                raise ValueError('Checkpoint identity/status mismatch')
        else:
            rows = cotton_rows(xml.read_bytes())
            values, qualifiers = regional_values(rows, source['report_date'])
            comparison = verify_pdf(rows, pdf)
            if comparison['pdf_sha256'] != source_id['pdf_sha256']:
                raise ValueError('PDF changed while extracting')
            item = {'identity': source_id, 'namespace': namespace, 'report_date': source['report_date'],
                'values': values, 'qualifiers': qualifiers, 'numeric_verified': True,
                'comparison': comparison, 'xml_retrieved_at': xml_receipt['retrieved_at'],
                'pdf_retrieved_at': pdf_receipt['retrieved_at'],
                'publication_timestamp_verified': False, 'first_version_verified': False, 'model_eligible': False}
            freeze_record(path, item)
        results.append(item)
        if index % 10 == 0 or index + 1 == len(manifest['source_versions']):
            print(f"WASDE numeric check {index + 1}/{len(manifest['source_versions'])}; historical admission remains blocked", flush=True)
    frame = pd.read_csv(candidate / 'regional.csv', parse_dates=['report_date'], float_precision='round_trip')
    verified = verify_candidate_values(frame, results)
    for name, expected in inputs.items():
        if digest(candidate / name) != expected:
            raise ValueError('Candidate changed during review')
    for name, expected in hashes.items():
        if digest(archive_root / name) != expected:
            raise ValueError('Source changed during review')
    for fn in [analyze, compare_pdf_layout_rows, regional_values, utc_timestamp, read_record, digest, manifest_id]:
        if digest(Path(fn.__code__.co_filename)) != identity['source_code'][Path(fn.__code__.co_filename).name]:
            raise ValueError('Verification code changed during review')
    coverage, previous = [], None
    for row in verified.itertuples(index=False):
        related = [r for r in results if pd.Timestamp(r['report_date']) == row.report_date]
        observed = min(utc_timestamp(r['xml_retrieved_at']) for r in related)
        revision_valid = (previous is not None and previous.crop_year == row.crop_year
            and 1 <= (row.report_date - previous.report_date).days <= 62)
        coverage.append({'report_date': row.report_date.strftime('%Y-%m-%d'), 'numeric_verified': True,
            'earliest_recorded_xml_retrieval': observed.isoformat(), 'available_at': None,
            'publication_timestamp_verified': False, 'first_version_verified': False,
            'historical_model_eligible': False,
            'revision_base_report_date': previous.report_date.strftime('%Y-%m-%d') if revision_valid else None})
        previous = row
    report = {'version': VERSION, 'fits': 0, 'identity': identity, 'namespace': namespace,
        'inputs': inputs, 'source_hashes': hashes, 'rows': len(verified), 'source_versions_verified': len(results),
        'unique_xml_versions_verified': len({r['identity']['xml_sha256'] for r in results}),
        'unique_pdf_versions_verified': len({r['identity']['pdf_sha256'] for r in results}),
        'pdf_xml_cells_compared': sum(r['comparison']['compared_cells'] for r in results),
        'pdf_xml_mismatch_cells': 0,
        'numeric_verified_rows': len(verified), 'historical_model_eligible_rows': 0,
        'derived_feature_cells_checked': int(verified[[*LEVELS, *[n + '_revision' for n in LEVELS]]].notna().sum().sum()),
        'derived_feature_missing_cells_preserved': int(verified[[*LEVELS, *[n + '_revision' for n in LEVELS]]].isna().sum().sum()),
        'release_independent_innovations_at_most': len(verified), 'daily_forward_fill_creates_no_new_reports': True,
        'model_eligible': False, 'decision': 'NUMERIC_VERIFIED_HISTORICAL_AVAILABILITY_UNPROVEN',
        'policy': 'Report dates, PDF creation/modification and current retrieval clocks cannot backdate these specific bytes. Future-season forecasts known at publication are not future realized crop data. No source or old candidate changed.',
        'records': results}
    freeze_record(output / 'report.json', report)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    freeze_record(output / 'complete.json', {'version': VERSION, 'completed': True, 'fits': 0,
        'namespace': namespace, 'files': {name: digest(output / name) for name in ['report.json', 'coverage.csv']},
        'model_eligible': False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['candidate', 'archive-root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.candidate, args.archive_root, args.output)
    print({key: result[key] for key in ['rows', 'source_versions_verified', 'historical_model_eligible_rows', 'fits']})


if __name__ == '__main__':
    main()
