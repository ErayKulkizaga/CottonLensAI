"""Pull NASS QuickStats upland-cotton series for the development years only.

The API key is read from the NASS_API_KEY environment variable and is never
written anywhere. MAX_YEAR is hard-capped at 2021: the 2022-2025 test years stay
sealed until the preregistration is pushed. Raw JSON responses are stored with
SHA-256 receipts; the key is stripped from every recorded URL.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = 'https://quickstats.nass.usda.gov/api/api_GET/'
MIN_YEAR, MAX_YEAR = 2008, 2021
SEALED_FROM = 2022

QUERIES = {
    # county estimates (final, annual)
    'county_yield': dict(agg_level_desc='COUNTY', statisticcat_desc='YIELD', unit_desc='LB / ACRE'),
    'county_production': dict(agg_level_desc='COUNTY', statisticcat_desc='PRODUCTION', unit_desc='480 LB BALES'),
    'county_harvested': dict(agg_level_desc='COUNTY', statisticcat_desc='AREA HARVESTED', unit_desc='ACRES'),
    'county_planted': dict(agg_level_desc='COUNTY', statisticcat_desc='AREA PLANTED', unit_desc='ACRES'),
    # state and national: annual values AND monthly forecasts (reference_period_desc keeps them apart)
    'state_yield': dict(agg_level_desc='STATE', statisticcat_desc='YIELD', unit_desc='LB / ACRE'),
    'state_production': dict(agg_level_desc='STATE', statisticcat_desc='PRODUCTION', unit_desc='480 LB BALES'),
    'state_harvested': dict(agg_level_desc='STATE', statisticcat_desc='AREA HARVESTED', unit_desc='ACRES'),
    'state_planted': dict(agg_level_desc='STATE', statisticcat_desc='AREA PLANTED', unit_desc='ACRES'),
    'us_yield': dict(agg_level_desc='NATIONAL', statisticcat_desc='YIELD', unit_desc='LB / ACRE'),
    'us_production': dict(agg_level_desc='NATIONAL', statisticcat_desc='PRODUCTION', unit_desc='480 LB BALES'),
    'us_harvested': dict(agg_level_desc='NATIONAL', statisticcat_desc='AREA HARVESTED', unit_desc='ACRES'),
    'us_planted': dict(agg_level_desc='NATIONAL', statisticcat_desc='AREA PLANTED', unit_desc='ACRES'),
}
BASE = dict(source_desc='SURVEY', commodity_desc='COTTON', class_desc='UPLAND', domain_desc='TOTAL', format='JSON')


def nass_key():
    """NASS key from NASS_API_KEY, else from ~/.nass_api_key (never printed or written anywhere)."""
    k = os.environ.get('NASS_API_KEY') or (Path.home() / '.nass_api_key').read_text(encoding='utf-8').strip()
    if not k:
        raise SystemExit('NASS key not found')
    return k


def fetch(params, key, year):
    # year__GE/__LE were observed to be ignored; query one year at a time instead.
    q = dict(BASE, **params, year=str(year), key=key)
    url = API + '?' + urllib.parse.urlencode(q)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research-client'}), timeout=120) as r:
                return r.read()
        except Exception as exc:                      # noqa: BLE001
            if attempt == 3:
                raise
            print('retry', attempt, type(exc).__name__, flush=True)
            time.sleep(5 * (attempt + 1))


def main(out_dir, first=MIN_YEAR, last=MAX_YEAR):
    first, last = int(first), int(last)
    key = nass_key()
    if last >= SEALED_FROM and '--unseal' not in sys.argv:  # T39: pushed prereg docs/prereg/SEALED_TEST_T39_PREREG_20261115.md
        raise SystemExit('Sealed test years need --unseal after a pushed preregistration')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'api': API, 'base_params': BASE, 'years': [first, last], 'queries': {}}
    for name, params in QUERIES.items():
        rows, hashes = [], {}
        for year in range(first, last + 1):
            try:
                raw = fetch(params, key, year)
            except urllib.error.HTTPError as exc:
                if exc.code == 400:                   # QuickStats answers 400 when a year has no rows
                    hashes[year] = None
                    continue
                raise
            part = json.loads(raw).get('data', [])
            if any(int(r['year']) != year for r in part):
                raise SystemExit(f'{name} {year}: response contains other years; nothing written')
            rows += part
            hashes[year] = hashlib.sha256(raw).hexdigest()
        body = json.dumps({'data': rows}, separators=(',', ':')).encode()
        (out / f'{name}.json').write_bytes(body)
        receipt['queries'][name] = {'params': params, 'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest(),
                                    'per_year_response_sha256': hashes,
                                    'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        print(name, len(rows), flush=True)
    text = json.dumps(receipt, indent=2)
    if key in text:
        raise SystemExit('key leaked into receipt')
    (out / 'receipt.json').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
