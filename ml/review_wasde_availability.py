"""Join frozen WASDE numeric reviews into a fail-closed availability inventory.

This is an offline audit, not evidence of an actual publication instant.
"""
import argparse
import re
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sprint import freeze_record, read_record


def inventory(earlier_path, later_path):
    earlier_path, later_path = Path(earlier_path), Path(later_path)
    earlier, later = read_record(earlier_path), read_record(later_path)
    releases = earlier['releases'] + later['releases']
    if len(releases) != earlier['release_count'] + later['release_count']:
        raise ValueError('WASDE review count mismatch')
    if any(item['numeric_status'] != 'passed' for item in releases):
        raise ValueError('WASDE numeric review incomplete')
    urls = [item['url'] for item in releases]
    if len(urls) != len(set(urls)):
        raise ValueError('Duplicate WASDE release URL')
    months = Counter()
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or parsed.netloc != 'esmis.nal.usda.gov'
                or parsed.query or parsed.fragment or not parsed.path.startswith(
                    '/publication/world-agricultural-supply-and-demand-estimates/')):
            raise ValueError('Nonofficial WASDE release URL')
        slug = parsed.path.rsplit('/', 1)[-1]
        if not re.fullmatch(r'20\d\d-\d\d-\d\d(?:-\d+)?', slug):
            raise ValueError('Invalid WASDE release date')
        day = date.fromisoformat(slug[:10])
        if day.year not in range(2016, 2024):
            raise ValueError('WASDE release outside reviewed years')
        months[day.strftime('%Y-%m')] += 1
    expected = {f'{year}-{month:02d}' for year in range(2016, 2024)
                for month in range(1, 13)}
    missing = sorted(expected - months.keys())
    repeated = {month: count for month, count in sorted(months.items()) if count > 1}
    ambiguous = earlier['ambiguous_months']
    if (missing != earlier['missing_months'] or set(repeated) != set(ambiguous)
            or any(repeated[month] != len(links) or set(links) != {
                url for url in urls if url.rsplit('/', 1)[-1].startswith(month)
            } for month, links in ambiguous.items())):
        raise ValueError('WASDE manifest and numeric reviews disagree on release coverage')
    return {
        'review_code_sha256': digest(Path(__file__)),
        'inputs_sha256': {str(earlier_path.name): digest(earlier_path),
                          str(later_path.name): digest(later_path)},
        'period': '2016-01 through 2023-12',
        'calendar_months': len(expected),
        'published_months': len(months),
        'release_page_links': len(urls),
        'numeric_passed_links': len(releases),
        'missing_months': missing,
        'multiple_page_months': repeated,
        'publication_clock_verified_months': 0,
        'first_version_verified_months': 0,
        'model_eligible': False,
        'decision': 'research archive only; no historical feature admission',
        'reason': ('Release dates and numeric equality do not prove actual first availability '
                   'or exclude later reposts; a schedule is not a per-release timestamp.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--earlier', type=Path, required=True)
    parser.add_argument('--later', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.earlier, args.later)
    freeze_record(args.output, result)
    print(f"WASDE {result['published_months']}/{result['calendar_months']} months, "
          f"{result['numeric_passed_links']} numeric links; model eligible: false")


if __name__ == '__main__':
    main()
