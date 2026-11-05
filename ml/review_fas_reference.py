"""Offline audit of four country-code witnesses, never source/vintage admission."""
import argparse
import base64
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cottonlens_ml.code_identity import digest
from cottonlens_ml.sprint import freeze_record

ORIGINAL = 'https://apps.fas.usda.gov/esrquery/esrq.aspx'
COUNTRIES = {'CHINA, PEOPLES REPUBLIC OF': 5700, 'VIETNAM': 5520,
             'TURKEY': 4890, 'PAKISTAN': 5350}


class Options(HTMLParser):
    def __init__(self):
        super().__init__()
        self.select = None
        self.option = None
        self.items = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'select':
            self.select = attrs.get('name')
        if tag == 'option':
            self.option = {'select': self.select, 'value': attrs.get('value'), 'label': ''}

    def handle_data(self, text):
        if self.option is not None:
            self.option['label'] += text

    def handle_endtag(self, tag):
        if tag == 'option' and self.option is not None:
            self.option['label'] = ' '.join(self.option['label'].split())
            self.items.append(self.option)
            self.option = None
        if tag == 'select':
            self.select = None


def selected_pairs(html):
    parser = Options()
    parser.feed(html)
    country_options = [o for o in parser.items if o['select'] == 'ctl00$MainContent$lbCountry']
    pairs = []
    for label, code in COUNTRIES.items():
        matches = [o for o in country_options if o['label'] == label]
        if len(matches) != 1:
            raise ValueError('One exact historical country option required')
        option = matches[0]
        match = re.fullmatch(r'\d+:(\d{4})', option['value'] or '')
        if not match or int(match[1]) != code:
            raise ValueError('Historical country code mismatch; no inferred replacement')
        if sum(bool(re.fullmatch(r'\d+:' + str(code), o['value'] or '')) for o in country_options) != 1:
            raise ValueError('Ambiguous historical country code')
        pairs.append({'label': label, 'country_code': code, 'literal_option_value': option['value']})
    cotton = [o for o in parser.items if o['select'] == 'ctl00$MainContent$lbCommodity'
              and o['value'] == '1404']
    if len(cotton) != 1 or cotton[0]['label'] != 'All Upland Cotton':
        raise ValueError('Explicit historical All Upland option required')
    return pairs


def review(index_path, receipt_paths, output):
    index = json.loads(index_path.read_text(encoding='utf-8'))
    expected_header = ['urlkey', 'timestamp', 'original', 'mimetype', 'statuscode', 'digest', 'length']
    if not isinstance(index, list) or not index or index[0] != expected_header:
        raise ValueError('Pinned archive index header required')
    if any(not isinstance(r, list) or len(r) != len(expected_header) for r in index[1:]):
        raise ValueError('Invalid archive index row')
    rows = [dict(zip(expected_header, r, strict=True)) for r in index[1:]]
    snapshots = []
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        row = receipt['archive_index_row']
        stamp = row.get('timestamp', '')
        if not re.fullmatch(r'\d{14}', stamp):
            raise ValueError('Explicit archive capture timestamp required')
        datetime.strptime(stamp, '%Y%m%d%H%M%S').replace(tzinfo=UTC)
        url = f'https://web.archive.org/web/{stamp}id_/{ORIGINAL}'
        if (rows.count(row) != 1 or row['original'] != ORIGINAL or row['statuscode'] != '200'
                or row['mimetype'] != 'text/html' or receipt.get('status') != 200
                or receipt.get('url') != url or receipt.get('final_url') != url
                or receipt.get('credentials_sent') is not False):
            raise ValueError('Archive receipt/index/original identity mismatch')
        source = (receipt_path.parent / receipt['file']).resolve()
        if not source.is_relative_to(receipt_path.parent.resolve()) or digest(source) != receipt['sha256']:
            raise ValueError('Unsafe or changed archived source')
        # CDX uses SHA1/Base32 for legacy WARC payload identity, not security.
        archive_digest = base64.b32encode(hashlib.sha1(source.read_bytes(), usedforsecurity=False).digest()).decode().rstrip('=')
        if archive_digest != row['digest']:
            raise ValueError('Archive payload digest mismatch')
        pairs = selected_pairs(source.read_text(encoding='utf-8-sig'))
        snapshots.append({'archive_capture_timestamp_literal': stamp,
                          'original_url': ORIGINAL, 'archive_url': url, 'sha256': digest(source),
                          'receipt_sha256': digest(receipt_path), 'pairs': pairs,
                          'cdx_payload_digest_matches': True,
                          'commodity_code': 1404, 'commodity_label': 'All Upland Cotton',
                          'archive_timestamp_is_usda_publication_proof': False})
    if len(snapshots) != 2 or len({s['archive_capture_timestamp_literal'] for s in snapshots}) != 2:
        raise ValueError('Two distinct historical snapshots required')
    snapshots.sort(key=lambda s: s['archive_capture_timestamp_literal'])
    if not (snapshots[0]['archive_capture_timestamp_literal'] < '20200528000000'
            and snapshots[1]['archive_capture_timestamp_literal'] > '20200528235959'):
        raise ValueError('Selected snapshots must bracket the observation date')
    if snapshots[0]['pairs'] != snapshots[1]['pairs']:
        raise ValueError('Historical country options differ; no interpolation')
    body = {'scope': 'four code/name witnesses in two archived public query forms',
            'observation_date': '2020-05-28', 'index_sha256': digest(index_path),
            'review_code_sha256': digest(Path(__file__)), 'snapshots': snapshots,
            'four_code_name_pairs_observed': True,
            'continuous_validity_between_snapshots_verified': False,
            'full_historical_catalog_verified': False, 'unit_verified': False,
            'exact_accumulated_exports_cell_verified': False,
            'publication_timestamp_verified': False, 'first_version_verified': False,
            'model_eligible': False, 'market_fits': 0,
            'decision': 'Historical reference witnesses only; no numerical cell or vintage admission'}
    freeze_record(output, body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.index, args.receipt, args.output)
    print('Four historical code witnesses reviewed; original cell not verified; model_eligible=false')


if __name__ == '__main__':
    main()
