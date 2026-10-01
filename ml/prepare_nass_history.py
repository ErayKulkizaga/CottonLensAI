"""Check annual NASS Cotton condition snapshots against official report bytes.

The output is a content audit, never a publication-time or model-use approval.
"""
import argparse
import json
from pathlib import Path

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.nass_archive import (
    audit_condition_year,
    verified_cached_release,
)
from cottonlens_ml.sprint import freeze_record, read_record


def snapshot_for_year(raw_root, inventory, year):
    expected = [row['source_sha256'] for row in inventory['archives']
                if row['provider'] == 'nass' and row['year'] == year and row['checksum_ok']]
    if len(expected) != 1:
        raise ValueError(f'Expected one checked NASS source identity for {year}')
    matches = [path for path in (Path(raw_root) / 'nass').glob('*/*/source.json')
               if digest(path) == expected[0]]
    if not matches:
        raise FileNotFoundError(f'NASS {year} source with acquisition checksum is unavailable')
    for path in matches:
        receipt = json.loads((path.parent / 'retrieval.json').read_text(encoding='utf-8'))
        if (receipt.get('provider') == 'nass'
                and receipt.get('source_sha256') == expected[0]
                and receipt.get('parameters', {}).get('year') == year):
            return path, expected[0]
    raise ValueError(f'NASS {year} source receipts disagree with acquisition inventory')


def prepare_year(year, *, raw_root, inventory, output):
    source, checksum = snapshot_for_year(raw_root, inventory, year)
    report = output / f'year-audit-{year}.json'
    if report.exists():
        saved = read_record(report)
        if saved['year'] != year or saved['source_sha256'] != checksum:
            raise ValueError(f'NASS {year} completed audit uses another source')
        for row in saved['results']:
            if (row['status'] in ('numeric_match', 'reference_missing_zero')
                    and verified_cached_release(row['release_page'], output) is None):
                raise ValueError(f'NASS {year} completed audit lost a source report')
        print(f'NASS {year}: cached {saved["weeks"]} weeks {saved["counts"]}', flush=True)
        return saved

    def progress(done, total, item):
        print(f'NASS {year}: {done}/{total} {item["status"]}', flush=True)

    result = audit_condition_year(source, output, progress=progress)
    freeze_record(report, result)
    print(f'NASS {year}: saved {result["weeks"]} weeks {result["counts"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-year', type=int, default=2010)
    parser.add_argument('--end-year', type=int, default=2023)
    args = parser.parse_args()
    if not 2010 <= args.start_year <= args.end_year <= 2023:
        parser.error('Years must lie between 2010 and 2023')
    inventory = json.loads(args.inventory.read_text(encoding='utf-8'))
    for year in range(args.start_year, args.end_year + 1):
        prepare_year(year, raw_root=args.raw_root, inventory=inventory, output=args.output)


if __name__ == '__main__':
    main()
