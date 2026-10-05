"""Archive one official USDA AMS daily spot-quotation excerpt as ineligible evidence."""
import argparse
import hashlib
import json
import re
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.nass_archive import bounded_get, release_details

BASE = 'https://esmis.nal.usda.gov/publication/daily-spot-quotations-excerpts/'
INDEX = BASE.rstrip('/')


class MonthListing(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_row = False
        self.row_dates = []
        self.row_links = []
        self.entries = []
        self.next_page = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'tr':
            self.in_row = True
            self.row_dates, self.row_links = [], []
        elif tag == 'time' and self.in_row and attrs.get('datetime'):
            self.row_dates.append(attrs['datetime'][:10])
        elif tag == 'a':
            href = attrs.get('href', '')
            if self.in_row and re.fullmatch(
                r'/publication/daily-spot-quotations-excerpts/\d{4}-\d\d-\d\d(?:-\d+)?', href
            ):
                self.row_links.append(href)
            if ('usa-pagination__next-page' in attrs.get('class', '').split()
                    and attrs.get('aria-label') == 'Next page'):
                page = parse_qs(urlsplit(href).query).get('page', [])
                if len(page) != 1 or not page[0].isdigit():
                    raise ValueError('Invalid AMS pagination link')
                self.next_page = int(page[0])

    def handle_endtag(self, tag):
        if tag == 'tr' and self.in_row:
            if self.row_links:
                dates = set(self.row_dates)
                if len(dates) != 1 or len(self.row_links) != 1:
                    raise ValueError('Ambiguous AMS archive listing row')
                day = next(iter(dates))
                if not self.row_links[0].rsplit('/', 1)[1].startswith(day):
                    raise ValueError('AMS listing date/link mismatch')
                self.entries.append({'date': day,
                                     'release_page_url': 'https://esmis.nal.usda.gov' + self.row_links[0]})
            self.in_row = False


def archive_spot_month_listing(month, root, *, session=None):
    """Inventory official links and preserve every filtered HTML page by hash."""
    if not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])', month):
        raise ValueError('Expected YYYY-MM')
    session = session or requests.Session()
    entries, pages, seen_urls = [], [], set()
    number = 0
    while True:
        if number >= 30:
            raise ValueError('AMS month exceeds 30 listing pages')
        url = INDEX + '?' + urlencode({'date': month, 'page': number})
        raw = bounded_get(url, session, stage='AMS month listing')
        listing = MonthListing()
        listing.feed(raw.decode('utf-8'))
        if any(not item['date'].startswith(month) for item in listing.entries):
            raise ValueError('AMS month filter returned an out-of-month release')
        if any(item['release_page_url'] in seen_urls for item in listing.entries):
            raise ValueError('AMS month listing repeated a release')
        checksum = hashlib.sha256(raw).hexdigest()
        path = Path(root) / 'ams_month_listings' / month / str(number) / checksum / 'listing.html'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if digest(path) != checksum:
                raise ValueError('Archived AMS month listing corrupted')
        else:
            with path.open('xb') as handle:
                handle.write(raw)
        pages.append({'url': url, 'sha256': checksum,
                      'archive': str(path.relative_to(Path(root)))})
        entries.extend(listing.entries)
        seen_urls.update(item['release_page_url'] for item in listing.entries)
        if listing.next_page is None:
            break
        if listing.next_page != number + 1:
            raise ValueError('Non-sequential AMS pagination')
        number = listing.next_page
    if not entries:
        raise ValueError('AMS month listing contained no releases')
    return {'month': month, 'listing_pages': pages, 'entries': entries,
            'release_count': len(entries), 'distinct_dates': len({item['date'] for item in entries}),
            'publication_clock_verified': False, 'model_eligible': False}


def parse_independent_spot_date(report_text, expected_date, averages):
    """Check the dated MP_CN002 front page against the excerpt's average row."""
    dates = re.findall(r'^MP_CN002\b[^\n]*?\b(\d{1,2}-[A-Za-z]{3}-\d{2})\s*$',
                       report_text, re.MULTILINE)
    values = re.findall(r'^Average\s+[-\d]+\s+\S+\s+(\d+\.\d{2})\s+'
                        r'(\d+\.\d{2})\s+([\d,]+)\b', report_text, re.MULTILINE)
    if len(dates) != 1 or len(values) != 1:
        raise ValueError('Independent AMS date/average evidence missing')
    day = datetime.strptime(dates[0], '%d-%b-%y').replace(tzinfo=UTC).date().isoformat()
    if day != expected_date or tuple(values[0]) != (
            averages[0], averages[1], f'{int(averages[2]):,}'):
        raise ValueError('Independent AMS date or average disagrees with excerpt')


def decode_ams_text(raw):
    """Preserve legacy CP1252 punctuation without altering ASCII quote fields."""
    try:
        return raw.decode('utf-8-sig'), 'utf-8-sig'
    except UnicodeDecodeError:
        return raw.decode('cp1252'), 'cp1252'


def parse_spot_excerpt(report_text, expected_date, *, independent_text=None):
    """Read the named seven-market averages, never futures or a transaction VWAP."""
    if not re.search(r'^\s*MARKET\s+41-4/34\s+31-3/35\s+BALES\b', report_text, re.MULTILINE):
        raise ValueError('Unknown spot quotation quality/header')
    headers = re.findall(r'^MEMPHIS,\s*TN\s+(\d{1,2}/\d{1,2}/\d{4}'
                         r'|\d{1,2}-[A-Za-z]{3}-\d{2})\s+USDA\b',
                         report_text, re.MULTILINE)
    averages = re.findall(r'^AVERAGE\s+(\d+\.\d{2})\s+(\d+\.\d{2})\s+(\d+)\b',
                          report_text, re.MULTILINE)
    if len(averages) != 1:
        raise ValueError('Expected one report date and one seven-market AVERAGE row')
    if len(headers) == 1:
        fmt = '%m/%d/%Y' if '/' in headers[0] else '%d-%b-%y'
        report_date = datetime.strptime(headers[0], fmt).replace(tzinfo=UTC).date()
        if report_date != date.fromisoformat(expected_date):
            raise ValueError('Spot excerpt date disagrees with release page')
    elif (independent_text is not None and len(headers) == 0
          and len(re.findall(r'^MEMPHIS,\s*TN\s+#+\s+USDA\b',
                             report_text, re.MULTILINE)) == 1):
        parse_independent_spot_date(independent_text, expected_date, averages[0])
        report_date = date.fromisoformat(expected_date)
    else:
        raise ValueError('Expected one report date and one seven-market AVERAGE row')
    base, alternate, bales = averages[0]
    return {'report_date': report_date.isoformat(),
            'seven_market_average_41_4_34_cents_per_lb': float(base),
            'seven_market_average_31_3_35_cents_per_lb': float(alternate),
            'reported_spot_bales': int(bales),
            'semantics': 'published seven-market quality quotations; not futures or transaction VWAP',
            'published_at': None, 'publication_clock_verified': False,
            'model_eligible': False}


def archive_spot_excerpt(report_date, root, *, session=None, release_page_url=None,
                         independent_page_url=None):
    day = date.fromisoformat(report_date).isoformat()
    session = session or requests.Session()
    page_url = release_page_url or BASE + day
    parsed_url = urlsplit(page_url)
    if (parsed_url.scheme != 'https' or parsed_url.hostname != 'esmis.nal.usda.gov'
            or not re.fullmatch(r'/publication/daily-spot-quotations-excerpts/'
                                + re.escape(day) + r'(?:-\d+)?', parsed_url.path)
            or parsed_url.query or parsed_url.fragment or parsed_url.username or parsed_url.password):
        raise ValueError('Official AMS release URL matching report date required')
    page = bounded_get(page_url, session, stage='AMS release page')
    page_date, report_url = release_details(page.decode('utf-8'), page_url)
    if page_date[:10] != day or urlsplit(report_url).hostname != 'esmis.nal.usda.gov':
        raise ValueError('Official AMS page/report date or host mismatch')
    report = bounded_get(report_url, session, stage='AMS TXT report')
    report_text, report_encoding = decode_ams_text(report)
    independent = None
    try:
        parsed = parse_spot_excerpt(report_text, day)
    except ValueError:
        if independent_page_url is None:
            raise
        independent_url = urlsplit(independent_page_url)
        if (independent_url.scheme != 'https' or independent_url.hostname != 'esmis.nal.usda.gov'
                or independent_url.path != '/publication/daily-spot-quotations-pg-1-front-page/' + day
                or independent_url.query or independent_url.fragment):
            raise ValueError('Exact official independent AMS date page required')
        independent_page = bounded_get(independent_page_url, session, stage='AMS independent date page')
        independent_date, independent_report_url = release_details(
            independent_page.decode('utf-8'), independent_page_url)
        if (independent_date[:10] != day
                or urlsplit(independent_report_url).hostname != 'esmis.nal.usda.gov'):
            raise ValueError('Independent AMS page date or host mismatch')
        independent_report = bounded_get(independent_report_url, session,
                                         stage='AMS independent dated report')
        independent_text, _ = decode_ams_text(independent_report)
        parsed = parse_spot_excerpt(report_text, day,
                                    independent_text=independent_text)
        independent = {'page_url': independent_page_url,
                       'report_url': independent_report_url,
                       'page_sha256': hashlib.sha256(independent_page).hexdigest(),
                       'report_sha256': hashlib.sha256(independent_report).hexdigest(),
                       'page': independent_page, 'report': independent_report}
    page_sha, report_sha = hashlib.sha256(page).hexdigest(), hashlib.sha256(report).hexdigest()
    extra_sha = '' if independent is None else independent['page_sha256'] + independent['report_sha256']
    evidence_id = hashlib.sha256((page_sha + report_sha + extra_sha).encode()).hexdigest()
    folder = Path(root) / 'ams_daily_spot_excerpts' / day / evidence_id
    folder.mkdir(parents=True, exist_ok=True)
    for name, raw in (('release-page.html', page), ('report.txt', report)):
        path = folder / name
        if path.exists():
            if digest(path) != hashlib.sha256(raw).hexdigest():
                raise ValueError('Archived AMS evidence corrupted')
        else:
            with path.open('xb') as handle:
                handle.write(raw)
    if independent is not None:
        for name, raw in (('independent-page.html', independent['page']),
                          ('independent-report.txt', independent['report'])):
            path = folder / name
            if path.exists():
                if digest(path) != hashlib.sha256(raw).hexdigest():
                    raise ValueError('Archived independent AMS evidence corrupted')
            else:
                with path.open('xb') as handle:
                    handle.write(raw)
    receipt = folder / 'retrieval.json'
    content = {'release_page_url': page_url, 'report_url': report_url,
               'release_page_sha256': page_sha, 'report_sha256': report_sha,
               'release_page_date_field': page_date, 'retrieved_at': datetime.now(UTC).isoformat(),
               **parsed}
    if report_encoding != 'utf-8-sig':
        content['report_text_encoding'] = report_encoding
    if independent is not None:
        content.update({f'independent_{key}': independent[key] for key in (
            'page_url', 'report_url', 'page_sha256', 'report_sha256')})
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if any(saved[k] != content[k] for k in content if k != 'retrieved_at'):
            raise ValueError('Archived AMS receipt mismatch')
    else:
        with receipt.open('x', encoding='utf-8') as handle:
            json.dump(content, handle, indent=2)
    return folder, content


def verified_cached_excerpt(day, page_url, root):
    """Reuse a complete local receipt; never treat a partial file as finished."""
    folder = Path(root) / 'ams_daily_spot_excerpts' / day
    matches = []
    if folder.exists():
        for receipt in folder.glob('*/retrieval.json'):
            saved = json.loads(receipt.read_text(encoding='utf-8'))
            if saved.get('release_page_url') == page_url:
                matches.append((receipt.parent, saved))
    if len(matches) > 1:
        raise ValueError('Multiple archived AMS versions require explicit review')
    if not matches:
        return None
    path, saved = matches[0]
    page, report = path / 'release-page.html', path / 'report.txt'
    if digest(page) != saved['release_page_sha256'] or digest(report) != saved['report_sha256']:
        raise ValueError('Cached AMS evidence checksum mismatch')
    release_date, report_url = release_details(page.read_text(encoding='utf-8'), page_url)
    if release_date != saved['release_page_date_field'] or report_url != saved['report_url']:
        raise ValueError('Cached AMS page/link mismatch')
    independent_text = None
    if 'independent_page_url' in saved:
        independent_page = path / 'independent-page.html'
        independent_report = path / 'independent-report.txt'
        if (digest(independent_page) != saved['independent_page_sha256']
                or digest(independent_report) != saved['independent_report_sha256']):
            raise ValueError('Cached independent AMS evidence checksum mismatch')
        independent_date, independent_url = release_details(
            independent_page.read_text(encoding='utf-8'), saved['independent_page_url'])
        if independent_date[:10] != day or independent_url != saved['independent_report_url']:
            raise ValueError('Cached independent AMS page/link mismatch')
        independent_text, _ = decode_ams_text(independent_report.read_bytes())
    report_text, report_encoding = decode_ams_text(report.read_bytes())
    if saved.get('report_text_encoding', 'utf-8-sig') != report_encoding:
        raise ValueError('Cached AMS report text encoding mismatch')
    parsed = parse_spot_excerpt(report_text, day,
                                independent_text=independent_text)
    if any(saved[key] != value for key, value in parsed.items()):
        raise ValueError('Cached AMS parsed values mismatch')
    return path, saved


def rebuild_month_inventory(year_inventory_file, month):
    """Reconstruct a month from checksummed saved listings, without network access."""
    year_inventory_file = Path(year_inventory_file)
    root = year_inventory_file.parent
    year = json.loads(year_inventory_file.read_text(encoding='utf-8'))
    if not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])', month) or year.get('year') != int(month[:4]):
        raise ValueError('Month and year inventory disagree')
    months = [item for item in year['months'] if item['month'] == month]
    if len(months) != 1:
        raise ValueError('Expected exactly one archived AMS month')
    saved = months[0]
    entries, seen = [], set()
    for number, item in enumerate(saved['listing_pages']):
        url = INDEX + '?' + urlencode({'date': month, 'page': number})
        path = (root / 'ams_month_listings' / month / str(number)
                / item['sha256'] / 'listing.html')
        if item['url'] != url or digest(path) != item['sha256']:
            raise ValueError('Archived AMS month listing checksum or order mismatch')
        listing = MonthListing()
        listing.feed(path.read_text(encoding='utf-8'))
        expected_next = number + 1 if number + 1 < len(saved['listing_pages']) else None
        if listing.next_page != expected_next:
            raise ValueError('Archived AMS month pagination mismatch')
        if any(not entry['date'].startswith(month) or entry['release_page_url'] in seen
               for entry in listing.entries):
            raise ValueError('Archived AMS month contains out-of-month or repeated release')
        entries.extend(listing.entries)
        seen.update(entry['release_page_url'] for entry in listing.entries)
    if (not entries or len(entries) != saved['release_count']
            or len({entry['date'] for entry in entries}) != saved['distinct_dates']):
        raise ValueError('Archived AMS month count mismatch')
    return {'month': month, 'listing_pages': saved['listing_pages'], 'entries': entries,
            'release_count': len(entries), 'distinct_dates': len({entry['date'] for entry in entries}),
            'publication_clock_verified': False, 'model_eligible': False}


def audit_spot_inventory(inventory_file, *, session=None, independent_date_pages=None):
    """Review every release in a checksummed monthly listing, resuming saved files."""
    inventory_file = Path(inventory_file)
    root = inventory_file.parent
    inventory = json.loads(inventory_file.read_text(encoding='utf-8'))
    month = inventory['month']
    if not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])', month):
        raise ValueError('Invalid AMS inventory month')
    listing_entries = []
    for number, item in enumerate(inventory['listing_pages']):
        expected_url = INDEX + '?' + urlencode({'date': month, 'page': number})
        path = (root / 'ams_month_listings' / month / str(number)
                / item['sha256'] / 'listing.html')
        if item['url'] != expected_url or digest(path) != item['sha256']:
            raise ValueError('AMS listing evidence checksum or page order mismatch')
        listing = MonthListing()
        listing.feed(path.read_text(encoding='utf-8'))
        if listing.next_page != (number + 1 if number < len(inventory['listing_pages']) - 1 else None):
            raise ValueError('AMS listing pagination mismatch')
        listing_entries.extend(listing.entries)
    if (listing_entries != inventory['entries']
            or len(listing_entries) != inventory['release_count']
            or len({item['date'] for item in listing_entries}) != inventory['distinct_dates']):
        raise ValueError('AMS inventory entries disagree with archived listing')
    results = []
    for item in listing_entries:
        day, url = item['date'], item['release_page_url']
        evidence = verified_cached_excerpt(day, url, root)
        if evidence is None:
            evidence = archive_spot_excerpt(
                day, root, session=session, release_page_url=url,
                independent_page_url=(independent_date_pages or {}).get(day))
        path, receipt = evidence
        results.append({'date': day, 'release_page_url': url,
                        'report_sha256': receipt['report_sha256'],
                        'evidence_path': str(path),
                        'spot_41_4_34_cents_per_lb': receipt['seven_market_average_41_4_34_cents_per_lb'],
                        'reported_spot_bales': receipt['reported_spot_bales']})
    return {'month': month, 'inventory_sha256': digest(inventory_file),
            'release_count': len(results), 'distinct_dates': len({item['date'] for item in results}),
            'results': results, 'publication_clock_verified': False, 'model_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--date')
    group.add_argument('--month')
    group.add_argument('--audit-inventory', type=Path)
    parser.add_argument('--release-page-url')
    parser.add_argument('--year-inventory', type=Path,
                        help='With --month, rebuild from archived year listings without network access')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.year_inventory and not args.month:
        parser.error('--year-inventory requires --month')
    if args.audit_inventory:
        if args.release_page_url:
            parser.error('--release-page-url requires --date')
        audit = audit_spot_inventory(args.audit_inventory)
        report = args.output / f'ams-month-audit-{audit["month"]}.json'
        args.output.mkdir(parents=True, exist_ok=True)
        with report.open('x', encoding='utf-8') as handle:
            json.dump(audit, handle, indent=2)
        print(json.dumps({'audit': str(report), 'release_count': audit['release_count'],
                          'distinct_dates': audit['distinct_dates'], 'model_eligible': False}))
    elif args.month:
        if args.release_page_url:
            parser.error('--release-page-url requires --date')
        inventory = (rebuild_month_inventory(args.year_inventory, args.month)
                     if args.year_inventory else archive_spot_month_listing(args.month, args.output))
        report = args.output / f'ams-month-listing-{args.month}.json'
        with report.open('x', encoding='utf-8') as handle:
            json.dump(inventory, handle, indent=2)
        print(json.dumps({'inventory': str(report), 'release_count': inventory['release_count'],
                          'distinct_dates': inventory['distinct_dates'], 'model_eligible': False}))
    else:
        folder, evidence = archive_spot_excerpt(args.date, args.output,
                                                release_page_url=args.release_page_url)
        print(json.dumps({'archive': str(folder), 'date': evidence['report_date'],
                          'spot_41_4_34_cents_per_lb': evidence['seven_market_average_41_4_34_cents_per_lb'],
                          'model_eligible': False}))


if __name__ == '__main__':
    main()
