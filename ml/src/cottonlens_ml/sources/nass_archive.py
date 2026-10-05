"""Archive an explicitly named NASS Crop Progress release and its TXT report.

The release page supplies both the report link and the date; URL dates are never
accepted as evidence. A date without a verified clock does not admit model data.
"""
import argparse
import hashlib
import json
import re
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from cottonlens_ml.code_identity import digest


class ArchiveHTTPError(RuntimeError):
    def __init__(self, status_code, stage):
        self.status_code = status_code
        self.stage = stage
        super().__init__(f'Official {stage} returned HTTP {status_code}')


class HistoricalDelayNotice(RuntimeError):
    """The named official publication is a delay notice, not the weekly report."""


class ReleaseHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_time = False
        self.dates = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'time' and attrs.get('datetime'):
            self.dates.append(attrs['datetime'])
        if tag == 'a' and attrs.get('href', '').lower().endswith('.txt'):
            self.links.append(attrs['href'])


def release_links(page_html, page_url):
    parser = ReleaseHTML()
    parser.feed(page_html)
    dates = set(parser.dates)
    links = [urljoin(page_url, path) for path in parser.links
             if re.search(r'/[A-Za-z0-9_-]+\.txt$', path, re.IGNORECASE)]
    links = [u for u in links if urlsplit(u).hostname == 'esmis.nal.usda.gov'
             and '/release-files/' in urlsplit(u).path]
    if len(dates) != 1 or not links or len(links) != len(set(links)):
        raise ValueError('Expected one dated release and distinct official TXT links')
    date = next(iter(dates))
    if not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', date):
        raise ValueError('Unknown release page date format')
    return date, sorted(links)


def release_details(page_html, page_url):
    date, links = release_links(page_html, page_url)
    if len(links) != 1:
        raise ValueError('Expected one dated release and one official TXT file')
    return date, links[0]


def cotton_condition_summary(report_text):
    """Extract only the five published national condition percentages.

    Other cotton tables and comparison years have different column semantics.
    """
    lines = report_text.splitlines()
    modern = [(i, line) for i, line in enumerate(lines)
              if line.startswith('Cotton Condition - Selected States: Week Ending ')]
    legacy = [(i, line) for i, line in enumerate(lines)
              if re.match(r'^\s*Cotton:\s+Crop Condition by Percent,\s*$', line)]
    if len(modern) + len(legacy) != 1:
        raise ValueError('Expected one Cotton Condition section')
    if modern:
        start, heading = modern[0]
        week_text = heading.rsplit('Week Ending ', 1)[1].strip()
        block = lines[start:start + 75]
        if not any(all(name in line for name in ('Very poor', 'Poor', 'Fair', 'Good', 'Excellent'))
                   for line in block[:10]):
            raise ValueError('Cotton condition category order changed')
        totals = [line for line in block if re.match(r'^15 States\s*\.+:', line)]
    else:
        start, _ = legacy[0]
        block = lines[start:start + 75]
        headings = re.findall(r'Week Ending\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',
                              '\n'.join(block[:5]))
        if (len(headings) != 1 or not re.search(
                r'State\s*:\s*VP\s*:\s*P\s*:\s*F\s*:\s*G\s*:\s*EX',
                '\n'.join(block[:10]))):
            raise ValueError('Legacy Cotton condition heading/category order changed')
        week_text = headings[0]
        totals = [line for line in block if re.match(r'^15 Sts\s*:', line)]
    week = datetime.strptime(week_text, '%B %d, %Y').replace(tzinfo=UTC).date().isoformat()
    if len(totals) != 1:
        raise ValueError('Expected one 15 States national condition row')
    values = totals[0].split(':', 1)[1].split()
    if len(values) != 5 or any(not re.fullmatch(r'\d+|-', v) for v in values):
        raise ValueError('Unexpected Cotton condition percentages')
    result = dict(zip(('VERY POOR', 'POOR', 'FAIR', 'GOOD', 'EXCELLENT'),
                      (0 if v == '-' else int(v) for v in values), strict=True))
    if sum(result.values()) != 100:
        raise ValueError('National condition percentages do not sum to 100')
    return {'week_ending': week, 'national_condition': result}


def compare_condition_to_quickstats(summary, rows):
    """Match report categories to five unique current national Upland records."""
    differences, missing_zero = {}, []
    for category, published in summary['national_condition'].items():
        label = f'COTTON, UPLAND - CONDITION, MEASURED IN PCT {category}'
        matching = [row for row in rows if row.get('week_ending') == summary['week_ending']
                    and row.get('short_desc') == label and row.get('agg_level_desc') == 'NATIONAL'
                    and row.get('class_desc') == 'UPLAND']
        if not matching and published == 0:
            differences[category] = None
            missing_zero.append(category)
            continue
        if len(matching) != 1 or not re.fullmatch(r'\d+', str(matching[0]['Value'])):
            raise ValueError('Missing, duplicate or suppressed national Upland condition value')
        actual = int(matching[0]['Value'])
        differences[category] = actual - published
    return {'week_ending': summary['week_ending'], 'differences': differences,
            'missing_zero_categories': missing_zero,
            'all_values_match': not missing_zero and all(value == 0 for value in differences.values()),
            'model_eligible': False, 'publication_clock_verified': False}


def bounded_get(url, session, *, stage='archive'):
    with session.get(url, timeout=(10, 35), stream=True, allow_redirects=False) as response:
        if response.status_code != 200:
            raise ArchiveHTTPError(response.status_code, stage)
        parts, size = [], 0
        for part in response.iter_content(65536):
            size += len(part)
            if size > 2 * 1024 * 1024:
                raise ValueError('Unexpectedly large archive page/report')
            parts.append(part)
    raw = b''.join(parts)
    if not raw:
        raise ValueError('Empty official archive response')
    return raw


def archive_release(page_url, root, *, session=None):
    parsed = urlsplit(page_url)
    if (parsed.scheme != 'https' or parsed.hostname != 'esmis.nal.usda.gov'
            or not re.fullmatch(r'/publication/crop-progress/\d{4}-\d\d-\d\d', parsed.path)
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('Explicit official Crop Progress release page required')
    session = session or requests.Session()
    page = bounded_get(page_url, session, stage='release page')
    date, report_urls = release_links(page.decode('utf-8'), page_url)
    if date[:10] != parsed.path.rsplit('/', 1)[1]:
        raise ValueError('Release page date disagrees with its official URL')
    reports = [bounded_get(url, session, stage='TXT report') for url in report_urls]
    if any(raw != reports[0] for raw in reports[1:]):
        raise ValueError('Multiple Crop Progress TXT links have different bytes')
    report_url, report = report_urls[0], reports[0]
    delay_markers = {
        '2012-10-29': b'NASS Delays October 29, 2012 Reports',
        '2018-09-10': b'USDA September 10 Crop Progress Report Delayed',
    }
    notice_day = date[:10]
    if notice_day in delay_markers and delay_markers[notice_day] in report[:600]:
        notice_hash = hashlib.sha256(report).hexdigest()
        notice = Path(root) / 'nass_crop_progress_notices' / notice_day / notice_hash
        notice.mkdir(parents=True, exist_ok=True)
        for name, raw in [('release-page.html', page), ('notice.txt', report)]:
            path = notice / name
            if path.exists() and digest(path) != hashlib.sha256(raw).hexdigest():
                raise ValueError('Archived Crop Progress delay notice corrupted')
            if not path.exists():
                with path.open('xb') as handle:
                    handle.write(raw)
        notice_receipt = {'release_page_url': page_url, 'notice_url': report_url,
                          'page_sha256': digest(notice / 'release-page.html'),
                          'notice_sha256': notice_hash, 'model_eligible': False}
        saved_notice = notice / 'retrieval.json'
        if saved_notice.exists():
            if json.loads(saved_notice.read_text(encoding='utf-8')) != notice_receipt:
                raise ValueError('Archived Crop Progress delay notice receipt mismatch')
        else:
            with saved_notice.open('x', encoding='utf-8') as handle:
                json.dump(notice_receipt, handle, indent=2)
        raise HistoricalDelayNotice(f'{notice_day} official report was delayed')
    if b'Crop Progress' not in report[:500] or b'Released ' not in report[:1000]:
        raise ValueError('Unexpected Crop Progress TXT content')
    checksum = hashlib.sha256(report).hexdigest()
    folder = Path(root) / 'nass_crop_progress' / date[:10] / checksum
    folder.mkdir(parents=True, exist_ok=True)
    for name, raw in [('release-page.html', page), ('report.txt', report)]:
        path = folder / name
        if path.exists():
            if digest(path) != hashlib.sha256(raw).hexdigest():
                raise ValueError('Archived release file corrupted')
        else:
            with path.open('xb') as handle:
                handle.write(raw)
    receipt = folder / 'retrieval.json'
    content = {'release_page_url': page_url, 'report_url': report_url,
               'candidate_report_urls': report_urls,
               'release_page_sha256': digest(folder / 'release-page.html'),
               'report_sha256': checksum, 'release_page_date_field': date,
               'retrieved_at': datetime.now(UTC).isoformat(),
               'publication_clock_verified': False, 'model_eligible': False}
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if any(saved[k] != content[k] for k in content if k != 'retrieved_at'):
            raise ValueError('Archived release receipt mismatch')
    else:
        with receipt.open('x', encoding='utf-8') as handle:
            json.dump(content, handle, indent=2)
    return folder, content


def verified_cached_release(page_url, root):
    """Resume a completed official report only after rechecking its raw bytes."""
    parsed = urlsplit(page_url)
    if (parsed.scheme != 'https' or parsed.hostname != 'esmis.nal.usda.gov'
            or not re.fullmatch(r'/publication/crop-progress/\d{4}-\d\d-\d\d', parsed.path)
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('Explicit official Crop Progress release page required')
    day = parsed.path.rsplit('/', 1)[-1]
    matches = []
    for receipt_path in (Path(root) / 'nass_crop_progress' / day).glob('*/retrieval.json'):
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt.get('release_page_url') == page_url:
            matches.append((receipt_path.parent, receipt))
    if len(matches) > 1:
        raise ValueError('Multiple archived Crop Progress versions require review')
    if not matches:
        return None
    folder, receipt = matches[0]
    page, report = folder / 'release-page.html', folder / 'report.txt'
    if (digest(page) != receipt['release_page_sha256']
            or digest(report) != receipt['report_sha256']):
        raise ValueError('Cached Crop Progress checksum mismatch')
    page_date, report_urls = release_links(page.read_text(encoding='utf-8'), page_url)
    if (page_date != receipt['release_page_date_field']
            or report_urls != receipt.get('candidate_report_urls', [receipt['report_url']])
            or report_urls[0] != receipt['report_url']):
        raise ValueError('Cached Crop Progress page/link mismatch')
    return folder, receipt


def audit_condition_year(snapshot_file, archive_root, *, session=None, progress=None):
    """Compare every current national Upland condition week to an archived report.

    Candidate page dates are discovery attempts, not assumed publication times.
    Missing pages and mismatches remain explicit, and all rows stay ineligible.
    """
    snapshot_file = Path(snapshot_file)
    receipt = json.loads((snapshot_file.parent / 'retrieval.json').read_text(encoding='utf-8'))
    if (snapshot_file.name != 'source.json' or receipt.get('provider') != 'nass'
            or receipt.get('source_sha256') != digest(snapshot_file)):
        raise ValueError('Verified NASS source snapshot required')
    params = receipt.get('parameters', {})
    if any(params.get(k) != v for k, v in {
        'commodity_desc': 'COTTON', 'source_desc': 'SURVEY', 'freq_desc': 'WEEKLY',
        'domain_desc': 'TOTAL', 'agg_level_desc': 'NATIONAL',
    }.items()) or not isinstance(params.get('year'), int):
        raise ValueError('National weekly Cotton snapshot required')
    rows = json.loads(snapshot_file.read_text(encoding='utf-8'))['data']
    label = 'COTTON, UPLAND - CONDITION, MEASURED IN PCT GOOD'
    weeks = sorted({row['week_ending'] for row in rows if row.get('short_desc') == label})
    if not weeks:
        raise ValueError('No national Upland condition observations')
    session = session or requests.Session()
    results = []
    for week in weeks:
        observed = date.fromisoformat(week)
        if observed.year != params['year'] or observed.weekday() != 6:
            raise ValueError('Unexpected condition observation date')
        result = {'week_ending': week, 'status': 'release_not_found',
                  'attempted_release_dates': [], 'delay_notice_dates': [],
                  'model_eligible': False}
        for offset in range(1, 5):
            candidate = (observed + timedelta(days=offset)).isoformat()
            result['attempted_release_dates'].append(candidate)
            url = f'https://esmis.nal.usda.gov/publication/crop-progress/{candidate}'
            try:
                cached = verified_cached_release(url, archive_root)
                folder, metadata = cached if cached is not None else archive_release(
                    url, archive_root, session=session)
            except ArchiveHTTPError as exc:
                if exc.status_code == 404 and exc.stage == 'release page':
                    continue
                raise
            except HistoricalDelayNotice:
                result['delay_notice_dates'].append(candidate)
                continue
            summary = cotton_condition_summary((folder / 'report.txt').read_text(encoding='utf-8'))
            if summary['week_ending'] != week:
                result.update(status='report_week_mismatch', release_page=url,
                              report_week_ending=summary['week_ending'])
                break
            checked = compare_condition_to_quickstats(summary, rows)
            status = ('reference_missing_zero' if checked['missing_zero_categories']
                      and all(value in (None, 0) for value in checked['differences'].values())
                      else 'numeric_match' if checked['all_values_match'] else 'numeric_mismatch')
            result.update(status=status,
                          release_page=url, report_sha256=metadata['report_sha256'],
                          differences=checked['differences'],
                          missing_zero_categories=checked['missing_zero_categories'])
            break
        results.append(result)
        if progress is not None:
            progress(len(results), len(weeks), result)
    counts = {status: sum(row['status'] == status for row in results)
              for status in sorted({row['status'] for row in results})}
    return {'year': params['year'], 'source_sha256': receipt['source_sha256'],
            'weeks': len(weeks), 'counts': counts, 'results': results,
            'publication_clock_verified': False, 'model_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--release-url')
    group.add_argument('--audit-snapshot', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.release_url:
        folder, metadata = archive_release(args.release_url, args.output)
        print(json.dumps({'archive': str(folder), 'report_sha256': metadata['report_sha256'],
                          'model_eligible': False}))
    else:
        def report_progress(done, total, item):
            print(f'{done}/{total} {item["week_ending"]} {item["status"]}', flush=True)

        result = audit_condition_year(args.audit_snapshot, args.output.parent / 'archive',
                                      progress=report_progress)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps({'audit': str(args.output), 'weeks': result['weeks'],
                          'counts': result['counts'], 'model_eligible': False}))


if __name__ == '__main__':
    main()
