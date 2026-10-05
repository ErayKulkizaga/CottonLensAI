"""Reconcile NASS Crop Progress reports with dated official calendar entries.

Calendar time is scheduled time plus current publication status, not an independently
observed first-publication clock. This diagnostic never makes model data eligible.
"""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import requests

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.nass_archive import bounded_get

CALENDAR = 'https://data.nass.usda.gov/Publications/Calendar/reports_by_date.php'


class CalendarHTML(HTMLParser):
    def __init__(self, year, month):
        super().__init__()
        self.year, self.month = year, month
        self.in_row = False
        self.in_cell = False
        self.cells = []
        self.parts = []
        self.link = None
        self.current_date = None
        self.entries = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'tr' and 'calendar' in attrs.get('class', '').split():
            self.in_row, self.cells, self.link = True, [], None
        elif tag == 'td' and self.in_row:
            self.in_cell, self.parts = True, []
        elif tag == 'a' and self.in_cell and 'calendar-landing.php' in attrs.get('href', ''):
            self.link = attrs['href']

    def handle_data(self, data):
        if self.in_cell:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'td' and self.in_cell:
            self.cells.append(' '.join(' '.join(self.parts).split()))
            self.in_cell = False
        elif tag == 'tr' and self.in_row:
            if len(self.cells) >= 4:
                if self.cells[0]:
                    parsed = datetime.strptime(self.cells[0], '%a, %m/%d/%y').replace(tzinfo=UTC).date()
                    if parsed.year != self.year or parsed.month != self.month:
                        raise ValueError('NASS calendar date outside requested month')
                    self.current_date = parsed.isoformat()
                if self.cells[2] == 'Crop Progress':
                    if self.current_date is None or self.link is None:
                        raise ValueError('Undated or unlinked Crop Progress calendar row')
                    url = urljoin(CALENDAR, self.link)
                    parsed_url = urlsplit(url)
                    query = parse_qs(parsed_url.query)
                    day = self.current_date
                    expected = {'source': ['n'], 'year': [day[2:4]],
                                'month': [day[5:7]], 'day': [day[8:10]], 'report_id': ['17011']}
                    if (parsed_url.scheme != 'https' or parsed_url.hostname != 'data.nass.usda.gov'
                            or not parsed_url.path.endswith('/calendar-landing.php')
                            or query != expected):
                        raise ValueError('NASS Crop Progress calendar link disagrees with row date')
                    self.entries.append({'release_date': day,
                                         'scheduled_time_et': self.cells[1],
                                         'status_at_retrieval': self.cells[3],
                                         'landing_url': url})
            self.in_row = False


def archive_calendar_month(year, month, root, *, session=None):
    if not 1900 <= year <= datetime.now(UTC).year or not 1 <= month <= 12:
        raise ValueError('Valid historical NASS calendar month required')
    session = session or requests.Session()
    url = f'{CALENDAR}?month={month:02d}&view=l&year={year}'
    raw = bounded_get(url, session, stage='NASS publication calendar')
    parser = CalendarHTML(year, month)
    parser.feed(raw.decode('utf-8'))
    if not parser.entries or len({item['release_date'] for item in parser.entries}) != len(parser.entries):
        raise ValueError('Empty or duplicated Crop Progress calendar month')
    checksum = hashlib.sha256(raw).hexdigest()
    folder = Path(root) / 'nass_crop_progress_calendar' / f'{year}-{month:02d}' / checksum
    folder.mkdir(parents=True, exist_ok=True)
    page = folder / 'calendar.html'
    if page.exists():
        if digest(page) != checksum:
            raise ValueError('Archived NASS calendar corrupted')
    else:
        with page.open('xb') as handle:
            handle.write(raw)
    receipt = folder / 'retrieval.json'
    content = {'calendar_url': url, 'calendar_sha256': checksum,
               'retrieved_at': datetime.now(UTC).isoformat(), 'entries': parser.entries,
               'actual_publication_clock_verified': False, 'model_eligible': False}
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if any(saved[k] != content[k] for k in content if k != 'retrieved_at'):
            raise ValueError('Archived NASS calendar receipt mismatch')
    else:
        with receipt.open('x', encoding='utf-8') as handle:
            json.dump(content, handle, indent=2)
    return folder, content


def audit_condition_calendar(condition_audit_file, calendar_root, *, session=None):
    condition_audit_file = Path(condition_audit_file)
    audit = json.loads(condition_audit_file.read_text(encoding='utf-8'))
    if audit.get('model_eligible') is not False or not audit.get('results'):
        raise ValueError('Ineligible NASS condition audit required')
    dates = [row['release_page'].rsplit('/', 1)[1] for row in audit['results']
             if row['status'] == 'numeric_match']
    if len(dates) != len(audit['results']):
        raise ValueError('All condition weeks must have matched report evidence')
    months = sorted({(int(day[:4]), int(day[5:7])) for day in dates})
    calendars = [archive_calendar_month(year, month, calendar_root, session=session)
                 for year, month in months]
    by_date = {entry['release_date']: entry for _, receipt in calendars for entry in receipt['entries']}
    results = []
    for day in dates:
        entry = by_date.get(day)
        if entry is None:
            results.append({'release_date': day, 'status': 'calendar_entry_missing'})
        else:
            scheduled = entry['scheduled_time_et'] == '4:00 pm ET'
            published = entry['status_at_retrieval'] == 'Published'
            results.append({'release_date': day,
                            'status': 'calendar_match' if scheduled and published else 'calendar_unverified',
                            'scheduled_time_et': entry['scheduled_time_et'],
                            'publication_status_at_retrieval': entry['status_at_retrieval'],
                            'landing_url': entry['landing_url']})
    return {'condition_audit_sha256': digest(condition_audit_file),
            'calendar_sha256': [receipt['calendar_sha256'] for _, receipt in calendars],
            'matched': sum(row['status'] == 'calendar_match' for row in results),
            'total': len(results), 'results': results,
            'actual_publication_clock_verified': False, 'model_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition-audit', required=True, type=Path)
    parser.add_argument('--calendar-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit_condition_calendar(args.condition_audit, args.calendar_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'audit': str(args.output), 'matched': result['matched'],
                      'total': result['total'], 'model_eligible': False}))


if __name__ == '__main__':
    main()
