"""Reconcile blocked WASDE XML rows against independently extracted official PDFs.

Local evidence review only. Requires pypdf; never certifies publication timing or
creates a model-ready publication package.
"""
import argparse
from pathlib import Path

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.wasde import compare_pdf_layout_rows, inspect_archive
from cottonlens_ml.sprint import freeze_record, read_record


def review(manifest_path, prior_path, output_path):
    from pypdf import PdfReader

    manifest = read_record(Path(manifest_path))
    prior = read_record(Path(prior_path))
    entries = {item['url']: item for item in manifest['items']}
    if len(entries) != len(manifest['items']) or prior['release_count'] != len(prior['releases']):
        raise ValueError('Duplicate or inconsistent release evidence')
    results = []
    for entry in prior['releases']:
        item = dict(entry)
        if item['numeric_status'] == 'blocked' and item['url'] in entries:
            paths = entries[item['url']]
            pdf = Path(paths['pdf_folder'])
            receipt = read_record(pdf / 'retrieval.json')
            if receipt['kind'] != 'wasde' or digest(pdf / 'source.bin') != receipt['sha256']:
                raise ValueError('PDF evidence checksum/kind mismatch')
            xml = inspect_archive(Path(paths['xml_folder']))
            pages = [page.extract_text(extraction_mode='layout')
                     for page in PdfReader(pdf / 'source.bin').pages]
            comparison = compare_pdf_layout_rows(xml['rows'], pages)
            item.update({'method': 'XML/PDF fixed-column layout',
                         'pdf_sha256': receipt['sha256'], 'xml_sha256': xml['source_sha256'],
                         'compared_cells': comparison['compared_cells']})
            if comparison['all_values_match']:
                item.update({'numeric_status': 'passed', 'reason': None})
            else:
                item['reason'] = f"{len(comparison['mismatches'])} PDF/XML numeric mismatches"
        results.append(item)
    report = {'release_count': len(results),
              'numeric_passed': sum(x['numeric_status'] == 'passed' for x in results),
              'numeric_blocked': sum(x['numeric_status'] != 'passed' for x in results),
              'model_eligible': False, 'publication_timing_verified': False,
              'historical_coverage_complete': False,
              'method': 'Prior immutable review plus independent fixed-column PDF layout reconciliation',
              'parent_evidence_sha256': digest(Path(prior_path)),
              'pdf_manifest_sha256': digest(Path(manifest_path)), 'releases': results}
    freeze_record(Path(output_path), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = review(args.manifest, args.prior, args.output)
    print(f"Numeric checks: {report['numeric_passed']}/{report['release_count']}; "
          'publication timing unverified; model eligibility false')


if __name__ == '__main__':
    main()
