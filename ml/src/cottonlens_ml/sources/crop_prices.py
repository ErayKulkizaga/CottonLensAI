"""Pinned Yahoo corn/soybean proxy bars; no vintage or roll certification."""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record

TICKERS = {'corn': 'ZC=F', 'soybean': 'ZS=F'}
START, END = '2010-01-01', '2024-01-01'
QUERY = {'period1': 1262304000, 'period2': 1704067200, 'interval': '1d'}
API = 'https://query1.finance.yahoo.com/v8/finance/chart/'


def parse_chart(raw, series):
    if series not in TICKERS:
        raise ValueError('Only the preregistered corn/soybean proxy series are allowed')
    try:
        chart = json.loads(raw)['chart']
        if chart.get('error') or len(chart['result']) != 1:
            raise ValueError('Yahoo error or ambiguous chart result')
        result = chart['result'][0]
        meta = result['meta']
        if meta['symbol'] != TICKERS[series] or meta['instrumentType'] != 'FUTURE':
            raise ValueError('Unexpected futures identity')
        # Provider-declared timezone date; not a certified exchange publication clock.
        dates = pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert(
            meta['exchangeTimezoneName']).normalize().tz_localize(None)
        quotes = result['indicators']['quote']
        if len(quotes) != 1:
            raise ValueError('One unadjusted quote array required')
        frame = pd.DataFrame({name: quotes[0][name] for name in ('open', 'high', 'low', 'close', 'volume')})
        if len(frame) != len(dates) or len(frame) == 0:
            raise ValueError('Aligned nonempty timestamps and quotes required')
        frame.insert(0, 'date', dates)
        frame.insert(1, 'series', series)
        if (dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing
                or dates.min() < pd.Timestamp(START) or dates.max() >= pd.Timestamp(END)):
            raise ValueError('Unique chronological requested pre-audit dates required')
        values = frame[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
        if np.isinf(values).any() or frame.volume.dropna().lt(0).any():
            raise ValueError('Invalid numeric quotes')
    except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed Yahoo chart response') from exc
    valid = frame.close.gt(0) & np.isfinite(frame.close)
    complete = frame[['open', 'high', 'low', 'close']].notna().all(axis=1)
    coherent = (frame.low.le(frame[['open', 'close']].min(axis=1))
                & frame.high.ge(frame[['open', 'close']].max(axis=1)) & frame.high.ge(frame.low))
    returns = np.log(frame.close.where(valid)).diff()
    audit = {'series': series, 'ticker': TICKERS[series], 'rows': len(frame),
        'first_date': str(dates.min().date()), 'last_date': str(dates.max().date()),
        'exchange_timezone': meta['exchangeTimezoneName'], 'currency_label': meta.get('currency'),
        'valid_positive_closes': int(valid.sum()), 'missing_or_nonpositive_closes': int((~valid).sum()),
        'zero_volume_rows': int(frame.volume.eq(0).sum()),
        'inconsistent_ohlc_rows': int((complete & ~coherent).sum()),
        'absolute_log_jumps_over_10pct': int(returns.abs().gt(.1).sum()),
        'roll_verified': False, 'publication_timestamp_verified': False, 'first_version_verified': False}
    return frame, audit


def archive(series, root, *, session=None):
    if series not in TICKERS:
        raise ValueError('Unknown crop series')
    client = session or requests.Session()
    url = API + quote(TICKERS[series], safe='')
    started = datetime.now(UTC).isoformat()
    with client.get(url, params=QUERY, headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=(10, 35), stream=True, allow_redirects=False) as response:
        if response.status_code != 200:
            raise RuntimeError(f'Yahoo crop HTTP {response.status_code}; no silent provider switch or retry loop')
        pieces, size = [], 0
        for piece in response.iter_content(65536):
            size += len(piece)
            if size > 8 * 1024 * 1024:
                raise ValueError('Daily crop chart exceeds 8 MiB')
            pieces.append(piece)
        raw = b''.join(pieces)
    _, audit = parse_chart(raw, series)
    sha = hashlib.sha256(raw).hexdigest()
    folder = Path(root) / series / sha
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'chart.json'
    if target.exists():
        if digest(target) != sha:
            raise ValueError('Corrupt archived crop bytes preserved')
    else:
        with target.open('xb') as handle:
            handle.write(raw)
    receipt = folder / 'retrieval.json'
    if not receipt.exists():
        freeze_record(receipt, {'source_url': url, 'query': QUERY, 'sha256': sha,
            'request_started_at': started, 'retrieved_at': datetime.now(UTC).isoformat(),
            'summary': audit, 'source_tier': 'A_exploration_only', 'model_eligible': False,
            'release_allowed': False, 'first_version_verified': False,
            'publication_timestamp_verified': False, 'cost_tl': 0, 'redistribution_reviewed': False,
            'usage_scope': 'private research; not licensed CME/CBOT or certified contract chain',
            'terms_url': 'https://github.com/ranaroussi/yfinance',
            'transforms': 'raw quote arrays; adjusted-close ignored, no repair or back adjustment'})
    return folder


def compile_prices(archives, output):
    if set(archives) != set(TICKERS):
        raise ValueError('Pinned corn AND soybean archives required')
    frames, sources = [], {}
    for series, folder in archives.items():
        folder = Path(folder)
        raw, receipt = folder / 'chart.json', folder / 'retrieval.json'
        record = read_record(receipt)
        if (digest(raw) != record['sha256'] or record['source_url'] != API + quote(TICKERS[series], safe='')
                or record['query'] != QUERY or record['model_eligible'] or record['release_allowed']
                or record['first_version_verified'] or record['publication_timestamp_verified']):
            raise ValueError('Pinned crop identity/availability changed')
        frame, audit = parse_chart(raw.read_bytes(), series)
        if record['summary'] != audit:
            raise ValueError('Crop source audit changed')
        frames.append(frame)
        sources[series] = {'source_sha256': digest(raw), 'receipt_sha256': digest(receipt),
            'source_url': record['source_url'], 'summary': audit}
    frame = pd.concat(frames, ignore_index=True).sort_values(['date', 'series'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, date_format='%Y-%m-%d').encode()
    if output.exists():
        if output.read_bytes() != raw:
            raise ValueError('Conflicting compiled crop table preserved')
    else:
        output.write_bytes(raw)
    manifest = {'table_sha256': digest(output), 'rows': len(frame), 'sources': sources,
        'source_tier': 'A_exploration_only', 'model_eligible': False, 'release_allowed': False,
        'publication_timestamp_verified': False, 'first_version_verified': False,
        'roll_verified': False, 'cost_tl': 0, 'redistribution_reviewed': False,
        'scope': 'Yahoo ZC=F/ZS=F daily futures proxies, not physical crop prices or clean contract returns',
        'policy': 'strictly after provider-declared local bar day, then Cotton observation lag1/2/6; no audit dates'}
    freeze_record(output.with_suffix('.manifest.json'), manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Exactly two requests; completed pinned packets are compiled without new requests.
    archives = {series: archive(series, args.output / 'raw') for series in TICKERS}
    print(json.dumps(compile_prices(archives, args.output / 'crop-prices-2010-2023.csv')))


if __name__ == '__main__':
    main()
