"""Archive official correction notices; never infer release clocks or eligibility."""
import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import requests

URL = 'https://www.nass.usda.gov/Corrections/'


class CorrectionTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.cells = None
        self.parts = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.cells = []
        elif tag in ('td', 'th') and self.cells is not None:
            self.parts = []
        elif tag == 'br' and self.parts is not None:
            self.parts.append(' ')

    def handle_data(self, data):
        if self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.parts is not None:
            self.cells.append(' '.join(' '.join(self.parts).split()))
            self.parts = None
        elif tag == 'tr' and self.cells is not None:
            self.rows.append(self.cells)
            self.cells = None


def crop_notices(html):
    parser = CorrectionTable()
    parser.feed(html)
    if not any('Release Date' in row and 'Notification Date' in row for row in parser.rows):
        raise ValueError('Official correction table headers missing')
    result = []
    for row in parser.rows:
        if row and row[0].strip().lower() == 'crop progress':
            if len(row) != 4 or not all(row):
                raise ValueError('Incomplete Crop Progress correction row')
            result.append(dict(zip(('report', 'release_date_text', 'description',
                                    'notification_date_text'), row, strict=True)))
    if not result:
        raise ValueError('No Crop Progress notices found; inspect source layout')
    return result


def archive(output):
    response = requests.get(URL, timeout=(10, 30), allow_redirects=False)
    response.raise_for_status()
    if response.status_code != 200 or len(response.content) > 5_000_000:
        raise ValueError('Unexpected official corrections response')
    notices = crop_notices(response.text)
    checksum = hashlib.sha256(response.content).hexdigest()
    folder = Path(output) / checksum
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'corrections.html'
    if source.exists() and source.read_bytes() != response.content:
        raise ValueError('Existing source checksum conflict')
    if not source.exists():
        source.write_bytes(response.content)
    receipt = folder / 'crop-progress-notices.json'
    if not receipt.exists():
        receipt.write_text(json.dumps({'source_url': URL, 'sha256': checksum,
            'retrieved_at': datetime.now(UTC).isoformat(), 'notices': notices,
            'model_eligible': False,
            'limits': 'Published notice inventory, not a complete vintage history. '
                      'Absence of a notice does not prove no revision. Dates/clocks remain source text.'},
            indent=2), encoding='utf-8')
    return folder, notices


def notice_dates(notice):
    """Use only the stated release date and explicit rescheduling date."""
    def parse(value):
        value = value.replace('.', '')
        for fmt in ('%b %d, %Y', '%B %d, %Y'):
            try:
                return datetime.strptime(value, fmt).date().isoformat()  # noqa: DTZ007 - date only, no clock inferred
            except ValueError:
                pass
        raise ValueError(f'Unrecognized notice date: {value}')
    dates = {parse(notice['release_date_text'])}
    match = re.search(r'rescheduled for ([A-Za-z]+\.? \d{1,2}, \d{4})',
                      notice['description'], re.IGNORECASE)
    if match:
        dates.add(parse(match[1]))
    return sorted(dates)


def reconcile_archives(notice_folder, archive_root):
    """Match verified report bytes to known exceptions without certifying vintage."""
    notice_folder, archive_root = Path(notice_folder), Path(archive_root)
    receipt = json.loads((notice_folder / 'crop-progress-notices.json').read_text(encoding='utf-8'))
    raw = (notice_folder / 'corrections.html').read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise ValueError('Correction source checksum mismatch')
    notices = crop_notices(raw.decode('utf-8'))
    if notices != receipt['notices']:
        raise ValueError('Correction receipt differs from source table')
    releases = {}
    for path in sorted(archive_root.glob('nass_crop_progress/*/*/retrieval.json')):
        metadata = json.loads(path.read_text(encoding='utf-8'))
        report = (path.parent / 'report.txt').read_bytes()
        page = (path.parent / 'release-page.html').read_bytes()
        if (hashlib.sha256(report).hexdigest() != metadata['report_sha256']
                or hashlib.sha256(page).hexdigest() != metadata['release_page_sha256']):
            raise ValueError('Archived report/page checksum mismatch')
        match = re.search(r'Released ([A-Za-z]+ \d{1,2}, \d{4}),', report.decode('utf-8')[:1500])
        if not match:
            raise ValueError('Report header lacks a release date')
        released = datetime.strptime(match[1], '%B %d, %Y').date().isoformat()  # noqa: DTZ007 - date only
        if released != metadata['release_page_date_field'][:10]:
            raise ValueError('Report header and archive page dates disagree')
        releases.setdefault(released, []).append({'report_sha256': metadata['report_sha256'],
            'release_page_url': metadata['release_page_url'], 'header_date_matches_page': True})
    matches = []
    for notice in notices:
        dates = notice_dates(notice)
        found = [{'date': day, **report} for day in dates for report in releases.get(day, [])]
        matches.append({'notice': notice, 'explicit_release_dates': dates,
            'matched_reports': found, 'status': 'exception_report_found' if found else 'report_not_archived',
            'model_eligible': False})
    covered = {day for row in matches for day in row['explicit_release_dates']}
    return {'correction_source_sha256': receipt['sha256'], 'verified_archive_dates': len(releases),
            'notices': matches, 'dates_without_matched_notice': sorted(set(releases) - covered),
            'model_eligible': False,
            'limits': 'Date matching is not first-publication clock or original-vintage certification. '
                      'No matched notice does not prove absence of revisions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notice-folder', type=Path)
    parser.add_argument('--archive-root', type=Path)
    args = parser.parse_args()
    if args.notice_folder or args.archive_root:
        if not (args.notice_folder and args.archive_root):
            parser.error('--notice-folder and --archive-root must be supplied together')
        result = reconcile_archives(args.notice_folder, args.archive_root)
        args.output.mkdir(parents=True, exist_ok=True)
        content = json.dumps(result, sort_keys=True, indent=2).encode()
        target = args.output / ('reconciliation-' + hashlib.sha256(content).hexdigest() + '.json')
        if target.exists() and target.read_bytes() != content:
            raise ValueError('Reconciliation output changed')
        if not target.exists():
            target.write_bytes(content)
        print(json.dumps({'report': str(target), 'verified_archive_dates': result['verified_archive_dates'],
            'matched_notices': sum(bool(r['matched_reports']) for r in result['notices']),
            'model_eligible': False}))
        return
    folder, notices = archive(args.output)
    print(json.dumps({'archive': str(folder), 'crop_progress_notices': len(notices),
                      'model_eligible': False}))


if __name__ == '__main__':
    main()
