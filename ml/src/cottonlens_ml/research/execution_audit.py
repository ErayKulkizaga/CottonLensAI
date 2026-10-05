"""No-fit OHLC/source-semantics audit. Flags never repair prices or drop origins."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.trading_diagnostic import MULTIPLIER, match_market

VERSION = 'cotton-ohlc-semantics-audit-v1'
ROUNDING_TOLERANCE = 1e-5  # Far smaller than one 0.01 cent/lb tick.


def audit(rows, market):
    rows = rows.reset_index(drop=True)
    predictions = rows.rename(columns={'control_predicted_return': 'predicted_return'}).copy()
    predictions['horizon'] = 1
    matched = match_market(predictions, market)
    cotton = market.loc[market.series.eq('cotton')].set_index('date')
    target = cotton.loc[pd.to_datetime(rows.target_date)].reset_index(drop=True)
    prices = target[['open', 'high', 'low', 'close', 'volume']].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices[:, :4] <= 0).any()
            or (target.high < target.low).any() or (target.volume < 0).any()):
        raise ValueError('Finite positive OHLC, nonnegative volume and ordered range required')
    distance = np.maximum(target.low - target.close, target.close - target.high).clip(lower=0)
    flags = rows[['date', 'target_date']].copy()
    flags['close_outside_range'] = distance > ROUNDING_TOLERANCE
    flags['close_outside_more_than_one_tick'] = distance > .01 + ROUNDING_TOLERANCE
    flags['open_outside_range'] = (target.open < target.low - ROUNDING_TOLERANCE) | (target.open > target.high + ROUNDING_TOLERANCE)
    flags['zero_volume'] = target.volume.eq(0)
    flags['zero_range'] = target.high.eq(target.low)
    flags['any_flag'] = flags[['close_outside_range', 'open_outside_range', 'zero_volume', 'zero_range']].any(axis=1)
    flags['close_distance_cents'] = distance.to_numpy()
    for field in ['open', 'high', 'low', 'close', 'volume']:
        flags['target_' + field] = target[field].to_numpy()
    flags['execution_verified'] = False
    masks = [c for c in flags if c.endswith('_range') or c in ('zero_volume', 'any_flag', 'close_outside_more_than_one_tick')]
    summary = {'version': VERSION, 'fits': 0, 'origins': len(rows),
        'rounding_tolerance_cents': ROUNDING_TOLERANCE,
        'flag_counts': {name: int(flags[name].sum()) for name in masks},
        'max_close_distance_cents': float(distance.max()),
        'median_volume_close_outside': float(target.loc[flags.close_outside_range, 'volume'].median()) if flags.close_outside_range.any() else None,
        'median_volume_close_inside': float(target.loc[~flags.close_outside_range, 'volume'].median()) if (~flags.close_outside_range).any() else None,
        'strategies': {}, 'execution_verified': False, 'price_semantics_verified': False,
        'prices_repaired': False, 'origins_dropped': 0,
        'decision': 'SOURCE_SEMANTICS_REQUIRED_FOR_EXECUTION_CLAIMS',
        'limit': 'Outside-range Close may be settlement/mark or mixed definitions, not necessarily bad forecast labels. No contract or bar-clock proof is inferred. Projection and filtered subsets are sensitivity diagnostics, not alternative prices or strategies.'}
    projected = target.close.clip(lower=target.low, upper=target.high).to_numpy()
    movement = matched.open_to_close_cents.to_numpy() * MULTIPLIER
    years = pd.to_datetime(rows.date).dt.year.to_numpy()
    for name in [c.removesuffix('_position') for c in rows if c.endswith('_position')]:
        position = rows[name + '_position'].to_numpy(dtype=float)
        if not np.isfinite(position).all() or not np.isin(position, [-1, 0, 1]).all():
            raise ValueError('Fixed-unit positions required; no rule optimization')
        pnl = position * movement
        if not np.allclose(pnl, rows[name + '_gross_usd_equivalent'], atol=1e-9, rtol=0):
            raise ValueError('Saved proxy PnL differs from raw OHLC')
        buckets = {field: {'active': int(np.count_nonzero(position[flags[field]])),
            'flagged_gross_usd_equivalent': float(pnl[flags[field]].sum()),
            'unflagged_gross_usd_equivalent': float(pnl[~flags[field]].sum())} for field in masks}
        sensitivity = position * (projected - target.open.to_numpy()) * MULTIPLIER
        summary['strategies'][name] = {'original_gross_usd_equivalent': float(pnl.sum()),
            'range_projection_diagnostic_gross_usd_equivalent': float(sensitivity.sum()),
            'buckets': buckets, 'years': {str(year): {
                'original': float(pnl[years == year].sum()),
                'range_projection_diagnostic': float(sensitivity[years == year].sum())} for year in sorted(set(years))}}
    return summary, flags


def analyze(market_file, diagnostic, output):
    market_file, diagnostic, output = map(Path, (market_file, diagnostic, output))
    if output.exists():
        raise ValueError('Never overwrite data, prior evidence or an audit')
    complete = read_record(diagnostic / 'complete.json')
    if (complete.get('completed') is not True or complete.get('fits') != 0
            or complete.get('version') != 'clock-t1-open-close-proxy-v1'):
        raise ValueError('Completed original no-fit T+1 diagnostic required')
    files = {'market.parquet': market_file, **{name: diagnostic / name for name in ['complete.json', 'report.json', 'decisions.csv']}}
    inputs = {name: digest(path) for name, path in files.items()}
    for name in ['report.json', 'decisions.csv']:
        if inputs[name] != complete['files'][name]:
            raise ValueError('Original diagnostic payload checksum mismatch')
    report = read_record(diagnostic / 'report.json')
    if report['horizon'] != 1 or inputs['market.parquet'] != report['inputs']['market.parquet']:
        raise ValueError('Original T+1/raw-market identity required')
    rows = pd.read_csv(diagnostic / 'decisions.csv', float_precision='round_trip')
    result, flags = audit(rows, pd.read_parquet(market_file))
    for name, sha in inputs.items():
        if digest(files[name]) != sha:
            raise ValueError('Input changed while auditing')
    result['inputs'] = inputs
    output.mkdir(parents=True, exist_ok=False)
    flags.to_csv(output / 'flags.csv', index=False)
    freeze_record(output / 'report.json', result)
    freeze_record(output / 'complete.json', {'version': VERSION, 'completed': True, 'fits': 0,
        'inputs': inputs, 'files': {name: digest(output / name) for name in ['flags.csv', 'report.json']},
        'execution_verified': False, 'prices_repaired': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.market, args.diagnostic, args.output)
    print({'origins': result['origins'], 'flags': result['flag_counts'], 'fits': 0})


if __name__ == '__main__':
    main()
