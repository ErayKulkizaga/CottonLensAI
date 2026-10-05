"""Read-only reconciliation against existing AMS FUTURES TODAY tables; no training."""
import argparse
import hashlib
import json
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.execution_audit import ROUNDING_TOLERANCE
from cottonlens_ml.research.ledger import freeze_record

VERSION = 'ams-cotton-price-reconciliation-v1'


def parse_quotes(data, reference_date):
    text = data.decode('utf-8', errors='replace')
    dates = re.findall(r'\b(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-[A-Za-z]{3}-\d{2})\b', text[:600])
    if not dates or pd.Timestamp(dates[0]) != pd.Timestamp(reference_date):
        raise ValueError('AMS document reference date differs from listing')
    if 'FUTURES' not in text or 'COTTON QUOTATIONS' not in text:
        raise ValueError('Missing AMS futures table boundaries')
    table = text.split('FUTURES', 1)[1].split('COTTON QUOTATIONS', 1)[0]
    if 'TODAY' not in table[:80]:
        raise ValueError('Expected FUTURES TODAY column')
    quotes = re.findall(r'\b(Mar|May|Jul|Oct|Dec)-(\d\d)\s+([0-9]+\.[0-9]+)\s*$', table, re.MULTILINE)
    names = [f'{month}-{year}' for month, year, _ in quotes]
    values = [float(value) for _, _, value in quotes]
    if not quotes or len(set(names)) != len(names) or not all(value > 0 for value in values):
        raise ValueError('Unique positive quoted contracts required')
    return dict(zip(names, values, strict=True))


def reconcile(market, review, archive):
    cotton = market.loc[market.series.eq('cotton')].copy()
    cotton['date'] = pd.to_datetime(cotton.date)
    if cotton.date.isna().any() or cotton.date.duplicated().any():
        raise ValueError('Unique Cotton dates required')
    cotton = cotton.set_index('date').sort_index()
    records, hashes, seen, rejected = [], {}, set(), []
    for entry in review['version_records']:
        date = pd.Timestamp(entry['report_date'])
        sha = entry['document_sha256']
        if date in seen or entry['status'] != 'Final' or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('Unique final AMS records and safe SHA-256 paths required')
        seen.add(date)
        path = Path(archive) / 'raw' / 'ams' / sha / 'source.bin'
        data = path.read_bytes()
        # Bind the exact bytes parsed, not a different read of the same path.
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('AMS document checksum mismatch')
        hashes[date.strftime('%Y-%m-%d')] = sha
        if '#########' in data[:180].decode('utf-8', errors='replace'):
            rejected.append({'date': entry['report_date'], 'document_sha256': sha,
                'reason': 'Printed reference date is #########; listing date alone not admitted'})
            continue
        quotes = parse_quotes(data, date)
        if date not in cotton.index:
            raise ValueError('AMS date absent from frozen market; no silent intersection')
        bar = cotton.loc[date]
        close = float(bar.close)
        if not np.isfinite(close) or close <= 0:
            raise ValueError('Positive finite Close required')
        first = next(iter(quotes))
        records.append({'date': date.strftime('%Y-%m-%d'), 'close': close,
            'first_quoted_contract': first, 'first_quote': quotes[first],
            'first_quote_matches_close': abs(close - quotes[first]) <= ROUNDING_TOLERANCE,
            'matching_contracts': [name for name, value in quotes.items() if abs(close - value) <= ROUNDING_TOLERANCE],
            'close_outside_range': bool(close < bar.low - ROUNDING_TOLERANCE or close > bar.high + ROUNDING_TOLERANCE),
            'quotes': quotes, 'document_sha256': sha, 'document_url': entry['document_url']})
    if not records:
        raise ValueError('Existing AMS reports required')
    records.sort(key=lambda row: row['date'])
    transitions = []
    index = pd.Index(cotton.index)
    for prior, current in pairwise(records):
        if prior['first_quoted_contract'] == current['first_quoted_contract']:
            continue
        old_date, new_date = map(pd.Timestamp, (prior['date'], current['date']))
        consecutive = index.get_loc(new_date) == index.get_loc(old_date) + 1
        previous_quote = prior['quotes'].get(current['first_quoted_contract'])
        verified = consecutive and prior['first_quote_matches_close'] and current['first_quote_matches_close'] and previous_quote is not None
        transitions.append({'previous_date': prior['date'], 'date': current['date'],
            'from': prior['first_quoted_contract'], 'to': current['first_quoted_contract'],
            'consecutive_cotton_observations': bool(consecutive), 'price_decomposition_supported': bool(verified),
            'observed_change_cents': current['close'] - prior['close'],
            'same_contract_change_cents': current['close'] - previous_quote if verified else None,
            'previous_day_contract_spread_cents': previous_quote - prior['close'] if verified else None})
    summary = {'version': VERSION, 'fits': 0, 'reports': len(records),
        'documents_checksum_verified': len(hashes), 'rejected_reference_dates': rejected,
        'first_quote_matches': sum(r['first_quote_matches_close'] for r in records),
        'any_quote_matches': sum(bool(r['matching_contracts']) for r in records),
        'close_outside_range': sum(r['close_outside_range'] for r in records),
        'outside_range_first_quote_matches': sum(r['close_outside_range'] and r['first_quote_matches_close'] for r in records),
        'mismatches': [r for r in records if not r['first_quote_matches_close']],
        'quoted_contract_transitions': transitions, 'document_hashes': hashes,
        'forecast_labels_repaired': False, 'model_eligible': False, 'historical_availability_verified': False,
        'limit': 'FUTURES TODAY quote correspondence is not a vendor settlement specification, certified contract map, executable price or historical availability proof. Quote-column transitions and adjacent-contract spreads are retrospective diagnostics only.'}
    return summary, records


def analyze(market_file, review_file, archive, output):
    market_file, review_file, output = map(Path, (market_file, review_file, output))
    if output.exists():
        raise ValueError('Never overwrite source data or prior evidence')
    inputs = {'market.parquet': digest(market_file), 'publication-content-review.json': digest(review_file)}
    summary, records = reconcile(pd.read_parquet(market_file), json.loads(review_file.read_text(encoding='utf-8')), archive)
    for name, path in [('market.parquet', market_file), ('publication-content-review.json', review_file)]:
        if digest(path) != inputs[name]:
            raise ValueError('Input changed during reconciliation')
    summary['inputs'] = inputs
    output.mkdir(parents=True, exist_ok=False)
    freeze_record(output / 'report.json', summary)
    freeze_record(output / 'quotes.json', {'version': VERSION, 'records': records})
    freeze_record(output / 'complete.json', {'version': VERSION, 'completed': True, 'fits': 0,
        'inputs': inputs, 'files': {name: digest(output / name) for name in ['report.json', 'quotes.json']}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['market', 'review', 'archive', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.market, args.review, args.archive, args.output)
    print({key: result[key] for key in ['reports', 'first_quote_matches', 'close_outside_range', 'fits']})


if __name__ == '__main__':
    main()
