"""Export checked NASS Upland condition reports for research inspection only."""
import argparse
import csv
import io
from pathlib import Path

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.nass_archive import (
    cotton_condition_summary,
    verified_cached_release,
)
from cottonlens_ml.sprint import freeze_record, read_record

CATEGORIES = ('VERY POOR', 'POOR', 'FAIR', 'GOOD', 'EXCELLENT')
FIELDS = ('week_ending', 'release_page_date', 'very_poor_pct', 'poor_pct',
          'fair_pct', 'good_pct', 'excellent_pct', 'quickstats_check',
          'release_page_url', 'report_sha256')
ACCEPTED = {'numeric_match', 'reference_missing_zero'}


def checked_rows(root, years):
    rows, audits = [], {}
    for year in years:
        path = root / f'year-audit-{year}.json'
        audit = read_record(path)
        if (audit['year'] != year or audit['weeks'] != len(audit['results'])
                or any(item['status'] not in ACCEPTED for item in audit['results'])):
            raise ValueError(f'NASS {year} content audit is incomplete')
        audits[str(year)] = digest(path)
        for item in audit['results']:
            url = item['release_page']
            cached = verified_cached_release(url, root)
            if cached is None:
                raise ValueError('NASS archived report is missing')
            folder, receipt = cached
            if receipt['report_sha256'] != item['report_sha256']:
                raise ValueError('NASS audit/report checksum mismatch')
            summary = cotton_condition_summary((folder / 'report.txt').read_text(encoding='utf-8'))
            if summary['week_ending'] != item['week_ending']:
                raise ValueError('NASS report observation week changed')
            values = summary['national_condition']
            rows.append({'week_ending': item['week_ending'],
                         'release_page_date': url.rsplit('/', 1)[-1],
                         **{name.lower().replace(' ', '_') + '_pct': values[name]
                            for name in CATEGORIES},
                         'quickstats_check': item['status'],
                         'release_page_url': url,
                         'report_sha256': receipt['report_sha256']})
    rows.sort(key=lambda item: item['week_ending'])
    if len({item['week_ending'] for item in rows}) != len(rows):
        raise ValueError('Duplicate NASS observation week')
    return rows, audits


def export(root, output, *, years=range(2010, 2024)):
    root, output = Path(root), Path(output)
    years = list(years)
    if not years:
        raise ValueError('At least one NASS year required')
    rows, audits = checked_rows(root, years)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError('Frozen NASS diagnostic table changed')
    else:
        with output.open('xb') as handle:
            handle.write(data)
    manifest = {'years': years, 'row_count': len(rows),
                'first_week': rows[0]['week_ending'], 'last_week': rows[-1]['week_ending'],
                'content_check_counts': {status: sum(row['quickstats_check'] == status for row in rows)
                                         for status in sorted(ACCEPTED)},
                'source_year_audit_sha256': audits,
                'export_code_sha256': digest(Path(__file__)),
                'table_sha256': digest(output),
                'publication_clock_verified': False, 'model_eligible': False,
                'policy': 'Current API versus archived report content only; not an as-published vintage'}
    freeze_record(output.with_suffix('.manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.root, args.output)
    print(f'NASS diagnostic: {result["row_count"]} weeks; model eligible: false')


if __name__ == '__main__':
    main()
