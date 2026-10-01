"""Archive and verify complete years of AMS spot quotations, without training.

Resumes from checksum-verified release files. Publication clocks remain unverified.
"""
import argparse
import json
from pathlib import Path

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.ams_archive import (
    archive_spot_month_listing,
    audit_spot_inventory,
    verified_cached_excerpt,
)
from cottonlens_ml.sprint import freeze_record

INDEPENDENT_DATE_PAGES = {
    # The excerpt's dateline is literal #########, but the separately dated
    # official front page repeats both quotations and the spot-bale total.
    '2021-10-20': ('https://esmis.nal.usda.gov/publication/'
                   'daily-spot-quotations-pg-1-front-page/2021-10-20'),
}


def prepare_year(year, root):
    root = Path(root)
    months = []
    for number in range(1, 13):
        month = f'{year}-{number:02d}'
        listing_path = root / f'ams-month-listing-{month}.json'
        if not listing_path.exists():
            listing = archive_spot_month_listing(month, root)
            root.mkdir(parents=True, exist_ok=True)
            with listing_path.open('x', encoding='utf-8') as handle:
                json.dump(listing, handle, indent=2)
        listing = json.loads(listing_path.read_text(encoding='utf-8'))
        if listing['month'] != month or listing['release_count'] != len(listing['entries']):
            raise ValueError('Stored AMS month listing mismatch')
        audit_path = root / f'ams-month-audit-{month}.json'
        if audit_path.exists():
            previous = json.loads(audit_path.read_text(encoding='utf-8'))
            for entry in listing['entries']:
                if verified_cached_excerpt(entry['date'], entry['release_page_url'], root) is None:
                    raise ValueError('Completed AMS month lost an archived report')
        current = audit_spot_inventory(listing_path,
                                       independent_date_pages=INDEPENDENT_DATE_PAGES)
        if audit_path.exists():
            if previous != current:
                raise ValueError('Completed AMS month changed on verification')
        else:
            with audit_path.open('x', encoding='utf-8') as handle:
                json.dump(current, handle, indent=2)
        months.append({'month': month, 'listing_sha256': digest(listing_path),
                       'audit_sha256': digest(audit_path),
                       'report_count': current['release_count'],
                       'zero_reported_spot_bales': sum(
                           row['reported_spot_bales'] == 0 for row in current['results'])})
        print(f"AMS {month}: {current['release_count']} reports verified", flush=True)
    summary = {'year': year, 'months': months,
               'report_count': sum(item['report_count'] for item in months),
               'zero_reported_spot_bales': sum(item['zero_reported_spot_bales'] for item in months),
               'publication_clock_verified': False, 'model_eligible': False,
               'policy': 'Content archive only; not an as-published model package'}
    freeze_record(root / f'year-content-audit-{year}.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-year', type=int, required=True)
    parser.add_argument('--end-year', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 2010 <= args.start_year <= args.end_year <= 2024:
        parser.error('Review years must lie between 2010 and 2024')
    for year in range(args.start_year, args.end_year + 1):
        summary = prepare_year(year, args.output)
        print(f"AMS {year}: {summary['report_count']} checked; model eligible: false", flush=True)


if __name__ == '__main__':
    main()
