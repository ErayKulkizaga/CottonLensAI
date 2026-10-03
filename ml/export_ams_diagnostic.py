"""Build an offline, explicitly ineligible AMS spot-quotation research table."""
import argparse
import csv
import io
import json
from pathlib import Path

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.ams_archive import verified_cached_excerpt
from cottonlens_ml.sprint import freeze_record, read_record

FIELDS = ('report_date', 'spot_41_4_34_cents_per_lb',
          'spot_31_3_35_cents_per_lb', 'reported_spot_bales',
          'release_page_url', 'report_sha256')


def collapse_identical_report_aliases(rows):
    """Keep one row per date only when duplicate pages point to identical bytes."""
    retained, aliases = [], 0
    for row in sorted(rows, key=lambda item: (item['report_date'], item['release_page_url'])):
        if retained and row['report_date'] == retained[-1]['report_date']:
            if any(row[key] != retained[-1][key] for key in FIELDS
                   if key != 'release_page_url'):
                raise ValueError('Conflicting same-date AMS report versions')
            aliases += 1
        else:
            retained.append(row)
    return retained, aliases


def _year(root, year):
    path = root / f'year-content-audit-{year}.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    summary = read_record(path) if 'record_id' in raw else raw
    if summary['year'] != year or len(summary['months']) != 12:
        raise ValueError('Incomplete AMS year summary')
    rows = []
    for month_summary in summary['months']:
        month = month_summary['month']
        listing_path = root / f'ams-month-listing-{month}.json'
        audit_path = root / f'ams-month-audit-{month}.json'
        if digest(audit_path) != month_summary['audit_sha256']:
            raise ValueError('AMS month audit checksum mismatch')
        listing = json.loads(listing_path.read_text(encoding='utf-8'))
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if (month[:4] != str(year) or listing['month'] != month or audit['month'] != month
                or digest(listing_path) != audit['inventory_sha256']
                or len(listing['entries']) != audit['release_count']
                or len(audit['results']) != audit['release_count']):
            raise ValueError('AMS month inventory/audit mismatch')
        entries = {(item['date'], item['release_page_url']) for item in listing['entries']}
        if len(entries) != len(listing['entries']):
            raise ValueError('Repeated AMS listing entry')
        for item in audit['results']:
            day, url = item['date'], item['release_page_url']
            if (day, url) not in entries:
                raise ValueError('AMS audit release absent from listing')
            cached = verified_cached_excerpt(day, url, root)
            if cached is None:
                raise ValueError('AMS archived release missing')
            folder, receipt = cached
            if (str(folder) != item['evidence_path']
                    or receipt['report_sha256'] != item['report_sha256']
                    or receipt['reported_spot_bales'] != item['reported_spot_bales']
                    or receipt['seven_market_average_41_4_34_cents_per_lb']
                    != item['spot_41_4_34_cents_per_lb']):
                raise ValueError('AMS audited price or receipt changed')
            rows.append({'report_date': day,
                         'spot_41_4_34_cents_per_lb': receipt['seven_market_average_41_4_34_cents_per_lb'],
                         'spot_31_3_35_cents_per_lb': receipt['seven_market_average_31_3_35_cents_per_lb'],
                         'reported_spot_bales': receipt['reported_spot_bales'],
                         'release_page_url': url, 'report_sha256': receipt['report_sha256']})
    return rows, digest(path)


def export(roots, output):
    by_year = {}
    for root in map(Path, roots):
        for path in root.glob('year-content-audit-*.json'):
            year = int(path.stem.rsplit('-', 1)[-1])
            if year in by_year:
                raise ValueError('Duplicate AMS year')
            by_year[year] = root
    if not by_year:
        raise ValueError('No completed AMS years')
    rows, source_hashes = [], {}
    for year, root in sorted(by_year.items()):
        year_rows, source_hashes[str(year)] = _year(root, year)
        rows.extend(year_rows)
    source_report_links = len(rows)
    rows, aliases = collapse_identical_report_aliases(rows)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = stream.getvalue().encode('utf-8')
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError('Frozen AMS diagnostic table changed')
    else:
        with output.open('xb') as handle:
            handle.write(data)
    manifest = {'years': sorted(by_year), 'row_count': len(rows),
                'source_report_links': source_report_links,
                'identical_report_url_aliases': aliases,
                'first_report_date': rows[0]['report_date'],
                'last_report_date': rows[-1]['report_date'],
                'source_year_sha256': source_hashes,
                'export_code_sha256': digest(Path(__file__)),
                'table_sha256': digest(output),
                'publication_clock_verified': False, 'model_eligible': False,
                'policy': 'Diagnostic only; no as-published timestamp or first-version certification'}
    freeze_record(output.with_suffix('.manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.root, args.output)
    print(f"AMS diagnostic: {result['row_count']} rows; model eligible: false")


if __name__ == '__main__':
    main()
