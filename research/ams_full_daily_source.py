"""Strict value-only parsing of full daily USDA ICE settlement witnesses."""

import re
from datetime import datetime
from decimal import Decimal
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

MONTHS = {'Mar': 3, 'May': 5, 'Jul': 7, 'Oct': 10, 'Dec': 12}


class DailyPDFLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.dates = set(); self.links = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'time' and a.get('datetime'):
            self.dates.add(a['datetime'])
        if tag == 'a' and a.get('href', '').upper().endswith('/CNDDSQ.PDF'):
            self.links.add(a['href'])


def pdf_release(html, page_url, expected):
    parser = DailyPDFLinks(); parser.feed(html)
    if len(parser.dates) != 1 or len(parser.links) != 1:
        raise ValueError('One dated full-daily PDF release required')
    stamp = next(iter(parser.dates))
    if not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', stamp) or stamp[:10] != expected:
        raise ValueError('Release date differs; latest release cannot fill a historical day')
    url = urljoin(page_url, next(iter(parser.links))); parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname != 'esmis.nal.usda.gov'
            or '/release-files/' not in parsed.path or parsed.query or parsed.fragment
            or parsed.username or parsed.password):
        raise ValueError('Exact official full-daily source required')
    return stamp, url  # Page field is not a verified historical publication clock.


def settlements(first_page, expected):
    if ('Settlement prices provided by ICE Futures US' not in first_page
            or not re.search(r'ICE\s+FUTURES\s+US\s*-\s*CONTRACT\s+NO\.\s*2', first_page)
            or not re.search(r'Month\s+Today\s+Previous\s+Change', first_page)):
        raise ValueError('Explicit ICE settlement field and column roles required')
    headers = set(re.findall(r'(?:ICE Futures|Cotton and Tobacco Market News)\s+(\d{1,2}-[A-Za-z]{3}-\d{2})', first_page))
    if len(headers) != 1:
        raise ValueError('One printed daily header date required')
    stamp = datetime.strptime(next(iter(headers)), '%d-%b-%y').date().isoformat()  # noqa: DTZ007 - printed day, not a timestamp
    if stamp != expected:
        raise ValueError('Printed PDF date differs from requested day')
    rows = []
    for line in first_page.splitlines():
        m = re.match(r'^(Mar|May|Jul|Oct|Dec)-(\d{2})\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s+(-?\d+\.\d{2})(?:\s|$)', line)
        if not m:
            continue
        month, year, today, previous, change = m.groups()
        contract = f'20{year}-{MONTHS[month]:02d}-01'
        if contract[:7] < expected[:7] or min(Decimal(today), Decimal(previous)) <= 0:
            raise ValueError('Expired contract or nonpositive settlement')
        if Decimal(today)-Decimal(previous) != Decimal(change):
            raise ValueError('Today/previous/change arithmetic differs')
        rows.append({'contract': contract, 'today': today, 'previous': previous, 'change': change})
    contracts = [r['contract'] for r in rows]
    if len(rows) != 10 or contracts != sorted(set(contracts)):
        raise ValueError('Exactly ten unique chronological ICE settlement rows required')
    return {'observation_date': stamp, 'rows': rows, 'quotes': {r['contract']: r['today'] for r in rows},
            'source_field': 'ICE settlement Today', 'historical_availability_verified': False,
            'first_vintage_verified': False, 'training_allowed': False,
            'previous_column_date': None, 'previous_column_admitted_as_dated_quote': False}
