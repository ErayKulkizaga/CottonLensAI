"""Offline XML/PDF numeric review of a frozen WASDE release manifest.

Requires pypdf and pdfplumber for evidence review, not for model inference.
No network requests, model training, or publication-time certification.
"""
import argparse
import hashlib
from decimal import InvalidOperation
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.wasde import (
    ReleasePage,
    compare_pdf_layout_rows,
    compare_pdf_pages,
    inspect_archive,
)
from cottonlens_ml.sprint import freeze_record, read_record


def verify_pdf(xml_rows, pdf_path):
    from pypdf import PdfReader

    try:
        pages = [page.extract_text(extraction_mode='layout')
                 for page in PdfReader(pdf_path).pages]
        return compare_pdf_layout_rows(xml_rows, pages)
    except (ValueError, InvalidOperation):
        import pdfplumber

        with pdfplumber.open(pdf_path) as document:
            pages = [page.extract_text(x_tolerance=1, y_tolerance=5)
                     for page in document.pages]
        return compare_pdf_pages(xml_rows, pages)


def review(manifest_path, archive_root, output_path):
    manifest_path, archive_root = Path(manifest_path), Path(archive_root)
    output_path = Path(output_path)
    manifest = read_record(manifest_path)
    review_sha = digest(Path(__file__))
    wasde_sha = digest(Path(inspect_archive.__code__.co_filename))
    checkpoint_id = hashlib.sha256(
        (digest(manifest_path) + review_sha + wasde_sha).encode()).hexdigest()
    checkpoint_root = output_path.parent / 'history-review-checkpoints' / checkpoint_id
    cached = {}
    for receipt_path in archive_root.glob('wasde/*/retrieval.json'):
        receipt = read_record(receipt_path)
        source = receipt_path.parent / 'source.bin'
        if receipt['kind'] != 'wasde' or digest(source) != receipt['sha256']:
            raise ValueError('Cached WASDE checksum/kind mismatch')
        cached.setdefault(receipt['source_url'], []).append(receipt_path.parent)
    for alias_path in archive_root.glob('wasde_url_aliases/*/alias.json'):
        alias = read_record(alias_path)
        if hashlib.sha256(alias['source_url'].encode()).hexdigest() != alias_path.parent.name:
            raise ValueError('WASDE URL alias path mismatch')
        folder = archive_root / 'wasde' / alias['sha256']
        receipt = read_record(folder / 'retrieval.json')
        if (receipt['source_url'] != alias['canonical_source_url']
                or receipt['sha256'] != alias['sha256']
                or digest(folder / 'source.bin') != alias['sha256']):
            raise ValueError('WASDE URL alias content mismatch')
        cached.setdefault(alias['source_url'], []).append(folder)

    def one(url):
        matches = cached.get(url, [])
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous immutable WASDE source')
        return matches[0]

    results = []
    for url in manifest['release_urls']:
        page = ReleasePage()
        page.feed((one(url) / 'source.bin').read_text(encoding='utf-8-sig'))
        files = {}
        for extension in ('.xml', '.pdf'):
            links = {urljoin(url, link) for link in page.links if link.endswith(extension)}
            if len(links) != 1 or urlsplit(next(iter(links))).hostname != 'esmis.nal.usda.gov':
                raise ValueError('Missing or ambiguous official XML/PDF link')
            files[extension] = one(next(iter(links)))
        xml = inspect_archive(files['.xml'])
        pdf_path = files['.pdf'] / 'source.bin'
        identity = {'url': url, 'xml_sha256': xml['source_sha256'],
                    'pdf_sha256': digest(pdf_path)}
        checkpoint = checkpoint_root / (url.rsplit('/', 1)[-1] + '.json')
        if checkpoint.exists():
            item = read_record(checkpoint)
            if any(item.get(key) != value for key, value in identity.items()):
                raise ValueError('Completed PDF review disagrees with archived source')
        else:
            item = {**identity, 'numeric_status': 'blocked'}
            try:
                comparison = verify_pdf(xml['rows'], pdf_path)
                item.update({'method': comparison['format'],
                             'compared_cells': comparison['compared_cells'],
                             'numeric_status': 'passed' if comparison['all_values_match'] else 'mismatch',
                             'mismatch_count': len(comparison['mismatches'])})
            except (ValueError, InvalidOperation) as exc:
                item['reason'] = str(exc)
            freeze_record(checkpoint, item)
        results.append(item)
        print(f"WASDE PDF {len(results)}/{len(manifest['release_urls'])} "
              f"{url.rsplit('/', 1)[-1]} {item['numeric_status']}", flush=True)
    report = {'manifest_sha256': digest(manifest_path), 'release_count': len(results),
              'review_code_sha256': review_sha, 'wasde_code_sha256': wasde_sha,
              'checkpoint_id': checkpoint_id,
              'numeric_passed': sum(item['numeric_status'] == 'passed' for item in results),
              'numeric_mismatch': sum(item['numeric_status'] == 'mismatch' for item in results),
              'numeric_blocked': sum(item['numeric_status'] == 'blocked' for item in results),
              'missing_months': manifest['missing_months'],
              'ambiguous_months': manifest['ambiguous_months'],
              'model_eligible': False, 'publication_timing_verified': False,
              'scope': 'Selected-region XML/PDF numeric agreement, not first-publication proof',
              'releases': results}
    freeze_record(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = review(args.manifest, args.archive_root, args.output)
    print(f"Numeric checks: {report['numeric_passed']}/{report['release_count']}; "
          'publication timing unverified; model eligibility false')


if __name__ == '__main__':
    main()
