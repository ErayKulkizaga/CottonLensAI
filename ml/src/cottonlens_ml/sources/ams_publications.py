"""Match versioned MMN publication records to frozen ESMIS spot-report bytes.

The archive's displayed clock has no timezone. Preserve it literally, keep all
listed versions, and never turn this content check into model eligibility.
"""
import argparse
import csv
import hashlib
import re
import time
from contextlib import contextmanager
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

import requests

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.sources.public import archive, validate_url
from cottonlens_ml.sprint import freeze_record, read_record

INDEX = 'https://mymarketnews.ams.usda.gov/filerepo/reports'
HEADERS = {
    'view-field-slug-id-table-column': 'slug_id',
    'view-name-1-table-column': 'slug_name',
    'view-field-slug-title-table-column': 'title',
    'view-field-published-date-table-column': 'published_local_text',
    'view-field-report-date-table-column': 'report_date',
    'view-field-report-status-table-column': 'status',
    'view-field-document-table-column': 'document',
}


class PublicationTransport:
    """Adapt the locked browser-compatible client to the public archive interface.

    No credentials, proxies, certificate bypasses or persistent browser cookies.
    HTTP denial is an error; only interrupted network transfers are retryable.
    """
    def __init__(self, client, network_error, *, renew=None):
        self.client = client
        self.network_error = network_error
        self.renew = renew

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    @contextmanager
    def get(self, url, **kwargs):
        response = None
        try:
            response = self.client.get(url, **kwargs)
            yield response
        except self.network_error as exc:
            if response is not None and response.status_code >= 400:
                raise requests.HTTPError(f'AMS HTTP {response.status_code}', response=response) from exc
            if self.renew is not None:
                if response is not None:
                    response.close()
                    response = None
                self.client.close()
                self.client = self.renew()
            raise requests.ConnectionError('AMS public archive transfer interrupted') from exc
        finally:
            if response is not None:
                response.close()


def validate_listing_url(url):
    parts = urlsplit(url)
    if (parts.scheme != 'https' or parts.netloc != 'mymarketnews.ams.usda.gov'
            or parts.path != '/filerepo/reports' or parts.fragment
            or parse_qs(parts.query).get('field_slug_id_value') != ['3004']):
        raise ValueError('Exact public MMN slug 3004 listing required')


class PublicationListing(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.table_count = 0
        self.headers = set()
        self.cell = None
        self.cells = {}
        self.rows = []
        self.next_links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'table' and attrs.get('id') == 'filerepo-reports-table':
            self.in_table = True
            self.table_count += 1
        if self.in_table:
            if tag == 'th':
                self.headers.add(attrs.get('id'))
            elif tag == 'tr':
                self.cells = {}
            elif tag == 'td':
                self.cell = HEADERS.get(attrs.get('headers'))
                if self.cell is None or self.cell in self.cells:
                    raise ValueError('Changed or repeated MMN table column')
                self.cells[self.cell] = {'text': [], 'links': []}
            elif tag == 'a' and self.cell:
                self.cells[self.cell]['links'].append(attrs.get('href', ''))
        if tag == 'a' and 'next' in attrs.get('rel', '').split():
            self.next_links.append(attrs.get('href', ''))

    def handle_data(self, text):
        if self.in_table and self.cell:
            self.cells[self.cell]['text'].append(text)

    def handle_endtag(self, tag):
        if tag == 'td':
            self.cell = None
        elif tag == 'tr' and self.in_table and self.cells:
            self.rows.append(self.cells)
            self.cells = {}
        elif tag == 'table':
            self.in_table = False


def parse_listing(raw, url):
    validate_listing_url(url)
    parser = PublicationListing()
    parser.feed(raw.decode('utf-8-sig'))
    if parser.table_count != 1 or parser.headers != set(HEADERS):
        raise ValueError('Expected one complete MMN publication table')
    rows = []
    requested = parse_qs(urlsplit(url).query).get('field_report_date_end_value')
    for cells in parser.rows:
        if set(cells) != set(HEADERS.values()):
            raise ValueError('Incomplete publication row')
        values = {key: ' '.join(''.join(cell['text']).split()) for key, cell in cells.items()}
        if values['slug_id'] != '3004' or values['slug_name'] != 'MP_CN001':
            raise ValueError('Archive filter returned a different report')
        day = date.fromisoformat(values['report_date']).isoformat()
        if requested and requested != [day]:
            raise ValueError('MMN report-date filter ignored')
        # Keep the publisher's unzoned display literal; do not invent UTC.
        clock = datetime.strptime(values['published_local_text'], '%m-%d-%Y %I:%M:%S %p')  # noqa: DTZ007
        if clock.date() < date.fromisoformat(day) or not values['status']:
            raise ValueError('Publication precedes report date or lacks status')
        links = cells['document']['links']
        if len(links) != 1:
            raise ValueError('Expected exactly one versioned report link')
        document = urljoin(url, links[0])
        parts = urlsplit(document)
        if (parts.scheme != 'https' or parts.netloc != 'mymarketnews.ams.usda.gov'
                or parts.query or parts.fragment or not re.fullmatch(
                    r'/filerepo/sites/default/files/3004/' + re.escape(day)
                    + r'/\d+/(?:ams_3004_[A-Za-z0-9_]+\.txt|MP_CN001'
                    + day.replace('-', '') + r'\.TXT)', parts.path)):
            raise ValueError('Exact dated versioned MMN report URL required')
        rows.append({key: values[key] for key in ('report_date', 'status', 'published_local_text')} |
                    {'published_local_naive': clock.isoformat(), 'document_url': document,
                     'published_at': None, 'timezone_verified': False, 'model_eligible': False})
    if len(parser.next_links) > 1:
        raise ValueError('Ambiguous publication pagination')
    next_url = urljoin(url, parser.next_links[0]) if parser.next_links else None
    if next_url:
        validate_listing_url(next_url)
        current = parse_qs(urlsplit(url).query)
        following = parse_qs(urlsplit(next_url).query)
        if (following.get('page') != [str(int(current.get('page', ['0'])[0]) + 1)]
                or {k: v for k, v in following.items() if k != 'page'}
                != {k: v for k, v in current.items() if k != 'page'}):
            raise ValueError('Nonsequential or changed-filter pagination')
    return rows, next_url


def cached_archive(url, root, *, session=None):
    validate_url('ams', url)
    root = Path(root)
    key = hashlib.sha256(url.encode()).hexdigest()
    pointer = root / 'requests' / (key + '.json')
    if pointer.exists():
        receipt = read_record(pointer)
        if receipt['url'] != url or not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256']):
            raise ValueError('Request cache identity changed')
        path = root / 'raw' / 'ams' / receipt['sha256'] / 'source.bin'
        if digest(path) != receipt['sha256']:
            raise ValueError('Cached AMS evidence corrupted')
        return path
    for attempt in range(1, 4):
        try:
            folder = archive('ams', url, root / 'raw', session=session)
            break
        except (requests.Timeout, requests.ConnectionError):
            print(f'AMS network interruption: attempt {attempt}/3; completed downloads preserved', flush=True)
            if attempt == 3:
                raise
            time.sleep(attempt)
    path = folder / 'source.bin'
    freeze_record(pointer, {'url': url, 'sha256': digest(path)})
    return path


def review_history(table, manifest, output, *, session=None, max_pages=80):
    table, output = Path(table), Path(output)
    reference = read_record(Path(manifest))
    if digest(table) != reference['table_sha256']:
        raise ValueError('Frozen AMS diagnostic table changed')
    with table.open(encoding='utf-8', newline='') as handle:
        reference_rows = list(csv.DictReader(handle))
    expected = {row['report_date']: row['report_sha256'] for row in reference_rows}
    if len(expected) != len(reference_rows) or len(expected) != reference['row_count']:
        raise ValueError('Diagnostic row count or dates changed')
    report_path = output / 'publication-content-review.json'
    if report_path.exists():
        report = read_record(report_path)
        if report['table_sha256'] != reference['table_sha256']:
            raise ValueError('Review belongs to a different diagnostic table')
        for name, checksum in report['files'].items():
            path = (output / name).resolve()
            if (not safe_member(name) or not path.is_relative_to(output.resolve())
                    or digest(path) != checksum):
                raise ValueError('Completed publication evidence corrupted')
        print('AMS publication review: cached completed evidence; no downloads', flush=True)
        return report
    parameters = {'field_slug_id_value': '3004', 'name': '', 'field_slug_title_value': '',
                  'field_published_date_value': '', 'field_report_date_end_value': '',
                  'field_api_market_types_target_id': 'All', 'order': '', 'sort': ''}
    url = INDEX + '?' + urlencode(parameters)
    files, versions, seen, pages = {}, [], set(), 0
    while url:
        if pages >= max_pages or url in seen:
            raise ValueError('Publication pagination exceeded bound or repeated')
        seen.add(url)
        path = cached_archive(url, output, session=session)
        files[path.relative_to(output).as_posix()] = digest(path)
        rows, url = parse_listing(path.read_bytes(), url)
        versions.extend(row for row in rows if row['report_date'] in expected)
        pages += 1
        print(f'AMS metadata page {pages}: {len(versions)} relevant version records retained', flush=True)
    if len({row['document_url'] for row in versions}) != len(versions):
        raise ValueError('Repeated MMN version link; inspect pagination before accepting coverage')
    matched_dates = set()
    for number, row in enumerate(versions, 1):
        path = cached_archive(row['document_url'], output, session=session)
        checksum = digest(path)
        files[path.relative_to(output).as_posix()] = checksum
        row['document_sha256'] = checksum
        row['matches_esmis_bytes'] = checksum == expected[row['report_date']]
        if row['matches_esmis_bytes']:
            matched_dates.add(row['report_date'])
        if number % 50 == 0 or number == len(versions):
            print(f'AMS version files verified {number}/{len(versions)}; matching dates {len(matched_dates)}/{len(expected)}', flush=True)
    report = {'table_sha256': reference['table_sha256'], 'reference_rows': len(expected),
              'listing_pages': pages, 'version_records': versions, 'matched_dates': len(matched_dates),
              'missing_metadata_dates': sorted(set(expected) - {r['report_date'] for r in versions}),
              'unmatched_dates': sorted(set(expected) - matched_dates), 'files': files,
              'publication_timezone_verified': False, 'first_version_reviewed': False,
              'model_eligible': False,
              'next_gate': 'Review displayed timezone, all versions/statuses and source-use terms; no historical time is fabricated'}
    freeze_record(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from curl_cffi import requests as browser_requests

    def new_client():
        return browser_requests.Session(impersonate='chrome136', verify=True, trust_env=False)

    print('AMS transport: locked curl-cffi 0.13.0 / chrome136; TLS verification enabled', flush=True)
    try:
        with PublicationTransport(new_client(), browser_requests.exceptions.RequestException, renew=new_client) as session:
            result = review_history(args.table, args.manifest, args.output, session=session)
    except requests.RequestException as exc:
        print(f'AMS review paused ({type(exc).__name__}); no completed review was recorded. '
              'Verified downloads remain in --output. Rerun the same Data Workbench after access recovers.', flush=True)
        raise SystemExit(2) from None
    print(f"AMS publication review complete: {result['matched_dates']}/{result['reference_rows']} dates byte-matched; model_eligible=false")


if __name__ == '__main__':
    main()
