"""Free USDA API acquisition. Latest API values remain quarantined for vintage review.

Credentials come from environment/Colab Secrets, never command-line arguments.
NASS load_time and ESR release dates are not treated as original publication proof.
"""
import argparse
import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.sprint import freeze_record

ENDPOINTS = {'ams': 'https://marsapi.ams.usda.gov/services/v1.2/',
             'fas': 'https://api.fas.usda.gov/api/esr/',
             'nass': 'https://quickstats.nass.usda.gov/api/'}
KEYS = {'ams': 'USDA_AMS_API_KEY', 'fas': 'USDA_FAS_API_KEY', 'nass': 'USDA_NASS_API_KEY'}
NASS_ATTRIBUTION = 'This product uses the NASS API but is not endorsed or certified by NASS.'


class USDAClient:
    def __init__(self, provider, *, session=None, credential=None, pause=time.sleep):
        if provider not in ENDPOINTS:
            raise ValueError('Unsupported USDA provider')
        self.provider = provider
        self._key = credential or os.environ.get(KEYS[provider])
        if not self._key:
            raise ValueError(f'Add the free {KEYS[provider]} credential to Colab Secrets; never paste it in logs')
        self._session = session or requests.Session()
        self._pause = pause

    def get(self, resource, parameters, root):
        if not re.fullmatch(r'[A-Za-z0-9_/]+', resource) or '..' in resource:
            raise ValueError('Fixed USDA resource path required')
        if any(k.lower() in ('key', 'api_key', 'token', 'password') for k in parameters):
            raise ValueError('Credentials cannot be query configuration')
        url = ENDPOINTS[self.provider] + resource
        query, headers, auth = dict(parameters), {'Accept': 'application/json'}, None
        if self.provider == 'nass':
            query.update(key=self._key, format='JSON')
        elif self.provider == 'fas':
            headers['X-Api-Key'] = self._key
        else:
            auth = (self._key, '')
        # One request at a time with bounded retries; never print prepared URLs.
        started = datetime.now(UTC).isoformat()
        for attempt in range(3):
            self._pause(1)
            try:
                with self._session.get(url, params=query, headers=headers, auth=auth,
                        stream=True, allow_redirects=False, timeout=(10, 45)) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        delay = response.headers.get('Retry-After', str(2 ** attempt))
                        if attempt == 2 or not delay.isdigit() or int(delay) > 30:
                            raise RuntimeError(f'{self.provider} temporarily unavailable; retry later')
                        self._pause(int(delay))
                        continue
                    if response.status_code != 200:
                        raise RuntimeError(f'{self.provider} HTTP {response.status_code}; response body omitted')
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 25 * 1024 * 1024:
                            raise ValueError('USDA response too large; narrow the date/geography query')
                        chunks.append(chunk)
                    raw = b''.join(chunks)
            except requests.RequestException:
                if attempt == 2:
                    raise RuntimeError(f'{self.provider} request failed; credential-bearing details suppressed') from None
                continue
            if self._key.encode() in raw:
                raise ValueError('Response echoed a credential; refusing to archive')
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeError):
                raise ValueError('USDA response is not valid JSON') from None
            if isinstance(payload, dict) and any(k in payload for k in ('error', 'errors', 'ERROR')):
                raise ValueError('USDA API returned an error; response omitted')
            checksum = hashlib.sha256(raw).hexdigest()
            request_id = content_id({'provider': self.provider, 'resource': resource, 'parameters': parameters})
            folder = Path(root) / self.provider / request_id / checksum
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / 'source.json'
            if target.exists():
                if digest(target) != checksum:
                    raise ValueError('Archived USDA response corrupted')
            else:
                pending = folder / (uuid.uuid4().hex + '.pending')
                pending.write_bytes(raw)
                pending.rename(target)
            if not (folder / 'retrieval.json').exists():
                freeze_record(folder / 'retrieval.json', {'provider': self.provider, 'source_url': url,
                    'parameters': parameters, 'source_sha256': checksum, 'retrieved_at': datetime.now(UTC).isoformat(),
                    'model_eligible': False, 'vintage_policy': 'latest_api_snapshot_requires_review',
                    'attribution': NASS_ATTRIBUTION if self.provider == 'nass' else 'USDA', 'cost_tl': 0})
            observed = {'provider': self.provider, 'source_url': url, 'parameters': parameters,
                'request_started_at': started, 'observed_available_at': datetime.now(UTC).isoformat(),
                'source_sha256': checksum, 'source_file': 'source.json', 'published_at': None,
                'model_eligible': False, 'availability_basis': 'actual_download_completion_not_first_publication'}
            freeze_record(folder / 'observations' / (content_id(observed) + '.json'), observed)
            return folder, payload
        raise RuntimeError('USDA request attempts exhausted')

    def cotton_catalog(self, root):
        resource = 'reports' if self.provider == 'ams' else 'commodities'
        if self.provider == 'nass':
            raise ValueError('Use NASS cotton_by_year, not an ESR/AMS catalog')
        folder, payload = self.get(resource, {}, root)
        rows = payload.get('results', []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise TypeError('USDA catalog schema changed')
        cotton = [row for row in rows if 'cotton' in json.dumps(row).lower()]
        return folder, cotton

    def cotton_by_year(self, year, root, *, state=None):
        if self.provider != 'nass' or not 1900 <= int(year) <= datetime.now(UTC).year:
            raise ValueError('NASS requires a valid year')
        query = {'commodity_desc': 'COTTON', 'source_desc': 'SURVEY', 'freq_desc': 'WEEKLY',
                 'year': int(year), 'domain_desc': 'TOTAL', 'agg_level_desc': 'STATE' if state else 'NATIONAL'}
        if state:
            if not re.fullmatch('[A-Z]{2}', state):
                raise ValueError('Two-letter state required')
            query['state_alpha'] = state
        _, count = self.get('get_counts/', query, root)
        if int(count['count']) > 50000:
            raise ValueError('NASS 50,000-record ceiling; split this query')
        return self.get('api_GET/', query, root)

    def exports(self, commodity_code, market_year, root):
        if self.provider != 'fas' or not str(commodity_code).isdigit() or not 1900 <= int(market_year) <= datetime.now(UTC).year + 1:
            raise ValueError('FAS requires a catalog commodity code and market year')
        return self.get(f'exports/commodityCode/{commodity_code}/allCountries/marketYear/{market_year}', {}, root)

    def report(self, slug_id, start, end, root):
        if self.provider != 'ams' or not str(slug_id).isdigit():
            raise ValueError('AMS requires a catalog slug ID')
        first, last = pd.Timestamp(start), pd.Timestamp(end)
        if last < first or (last - first).days > 31:
            raise ValueError('AMS requests must span at most 31 days')
        return self.get(f'reports/{slug_id}', {'q': f'report_begin_date={first:%m/%d/%Y}:{last:%m/%d/%Y}',
                                              'allSections': 'true'}, root)


def normalize_nass(payload):
    """Preserve units/geography/suppression and load timestamp without inventing availability."""
    rows = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise TypeError('Expected NASS data array')
    records = []
    for row in rows:
        required = ('commodity_desc', 'short_desc', 'unit_desc', 'year', 'week_ending', 'Value', 'load_time')
        if any(name not in row for name in required) or row['commodity_desc'] != 'COTTON':
            raise ValueError('Unexpected NASS cotton schema')
        text = str(row['Value']).strip()
        numeric = pd.to_numeric(text.replace(',', ''), errors='coerce')
        records.append({'series': row['short_desc'], 'unit': row['unit_desc'],
            'state': row.get('state_alpha', ''), 'geography': row.get('agg_level_desc', ''),
            'observation_end': row['week_ending'], 'year': int(row['year']),
            'value': numeric, 'suppression_code': None if pd.notna(numeric) else text,
            'database_load_time': row['load_time'], 'published_at': None, 'timestamp_verified': False})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=tuple(ENDPOINTS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--year', type=int)
    parser.add_argument('--state')
    parser.add_argument('--commodity-code')
    parser.add_argument('--slug-id')
    parser.add_argument('--start')
    parser.add_argument('--end')
    args = parser.parse_args()
    client = USDAClient(args.provider)
    if args.provider == 'nass':
        if args.year is None:
            parser.error('NASS requires --year')
        folder, payload = client.cotton_by_year(args.year, args.output, state=args.state)
        frame = normalize_nass(payload)
        normalized = folder / 'normalized.parquet'
        if not normalized.exists():
            frame.to_parquet(normalized, index=False)
        print(json.dumps({'archive': str(folder), 'rows': len(frame), 'model_eligible': False}))
    elif args.provider == 'fas' and args.commodity_code and args.year:
        folder, _ = client.exports(args.commodity_code, args.year, args.output)
        print(json.dumps({'archive': str(folder), 'model_eligible': False}))
    elif args.provider == 'ams' and args.slug_id and args.start and args.end:
        folder, _ = client.report(args.slug_id, args.start, args.end, args.output)
        print(json.dumps({'archive': str(folder), 'model_eligible': False}))
    else:
        folder, rows = client.cotton_catalog(args.output)
        print(json.dumps({'archive': str(folder), 'cotton_catalog': rows}, indent=2))


if __name__ == '__main__':
    main()
