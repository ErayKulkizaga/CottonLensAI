"""Archive ECB daily reference rates without assuming historical availability.

ECB's includeHistory API may expose database-version validity rather than each
observation's first public release. The archive is diagnostic and never eligible
for model features by itself.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from cottonlens_ml.code_identity import digest

API = 'https://data-api.ecb.europa.eu/service/data/EXR/'
CURRENCIES = ('BRL', 'CNY', 'INR', 'USD')
SERIES = 'D.' + '+'.join(CURRENCIES) + '.EUR.SP00.A'
REQUIRED = {'KEY', 'FREQ', 'CURRENCY', 'CURRENCY_DENOM', 'EXR_TYPE', 'EXR_SUFFIX',
            'TIME_PERIOD', 'OBS_VALUE', 'OBS_STATUS'}


def parse_ecb_csv(raw, year):
    """Validate raw series identities and values; preserve any revised duplicates."""
    try:
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        if reader.fieldnames is None or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError('ECB CSV missing required columns')
        counts, statuses, missing_statuses, dates = Counter(), Counter(), Counter(), []
        seen = set()
        duplicates = 0
        for row in reader:
            currency = row['CURRENCY']
            if (currency not in CURRENCIES or row['KEY'] != f'EXR.D.{currency}.EUR.SP00.A'
                    or row['FREQ'] != 'D' or row['CURRENCY_DENOM'] != 'EUR'
                    or row['EXR_TYPE'] != 'SP00' or row['EXR_SUFFIX'] != 'A'):
                raise ValueError('Unexpected ECB series identity')
            day = date.fromisoformat(row['TIME_PERIOD'])
            if day.year != year:
                raise ValueError('ECB observation outside requested year')
            if row['OBS_VALUE'] == '':
                if not row['OBS_STATUS']:
                    raise ValueError('Missing ECB rate needs an observation status')
                missing_statuses[row['OBS_STATUS']] += 1
            else:
                try:
                    value = float(row['OBS_VALUE'])
                except ValueError:
                    raise ValueError('Invalid ECB reference rate') from None
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('ECB reference rate must be positive and finite')
            key = (currency, day.isoformat())
            duplicates += key in seen
            seen.add(key)
            counts[currency] += 1
            statuses[row['OBS_STATUS']] += 1
            dates.append(day.isoformat())
        if set(counts) != set(CURRENCIES):
            raise ValueError('ECB response missing a requested currency')
    except (UnicodeError, csv.Error, TypeError, OverflowError) as exc:
        raise ValueError('Invalid ECB CSV response') from exc
    return {'year': year, 'rows': sum(counts.values()), 'series_rows': dict(sorted(counts.items())),
            'observation_statuses': dict(sorted(statuses.items())),
            'missing_value_statuses': dict(sorted(missing_statuses.items())),
            'missing_value_rows': sum(missing_statuses.values()),
            'repeated_currency_dates': duplicates, 'first_observation': min(dates),
            'last_observation': max(dates), 'publication_clock_verified': False,
            'vintage_verified': False, 'model_eligible': False}


def archive_ecb_year(year, root, *, session=None):
    if not 1999 <= year <= datetime.now(UTC).year:
        raise ValueError('Historical ECB year required')
    params = {'startPeriod': f'{year}-01-01', 'endPeriod': f'{year}-12-31',
              'includeHistory': 'true', 'format': 'csvdata'}
    session = session or requests.Session()
    with session.get(API + SERIES, params=params, timeout=(10, 45), stream=True,
                     allow_redirects=False) as response:
        if response.status_code != 200:
            raise RuntimeError(f'ECB download HTTP {response.status_code}')
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > 8 * 1024 * 1024:
                raise ValueError('ECB annual response exceeds 8 MiB')
            chunks.append(chunk)
        raw = b''.join(chunks)
    summary = parse_ecb_csv(raw, year)
    checksum = hashlib.sha256(raw).hexdigest()
    folder = Path(root) / 'ecb_fx' / str(year) / checksum
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / 'source.csv'
    if source.exists():
        if digest(source) != checksum:
            raise ValueError('Archived ECB CSV corrupted')
    else:
        with source.open('xb') as handle:
            handle.write(raw)
    receipt = folder / 'retrieval.json'
    content = {'source_url': API + SERIES, 'query': params, 'sha256': checksum,
               'retrieved_at': datetime.now(UTC).isoformat(), **summary,
               'usage': 'Public ECB statistics; quote source and retain original bytes',
               'unit': 'currency units per 1 EUR; not direct USD crosses'}
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if any(saved.get(key) != value for key, value in content.items() if key != 'retrieved_at'):
            raise ValueError('Archived ECB receipt mismatch')
    else:
        with receipt.open('x', encoding='utf-8') as handle:
            json.dump(content, handle, indent=2)
    return folder, content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder, receipt = archive_ecb_year(args.year, args.output)
    print(json.dumps({'archive': str(folder), 'rows': receipt['rows'],
                      'repeated_currency_dates': receipt['repeated_currency_dates'],
                      'model_eligible': False}))


if __name__ == '__main__':
    main()
