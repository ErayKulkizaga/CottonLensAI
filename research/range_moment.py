"""Zero-fit test of recorded current range information and next-day EWMA errors.

Association diagnostics use within-year centering and are NOT causal forecast
parameters, a backtest of a trading strategy, or independent predictive skill.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signal_frame(history, paired):
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique sorted history required')
    if history.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('2024+ cannot enter selection diagnosis')
    rows = paired.loc[paired.horizon.eq(1)].copy()
    if len(rows) != 2006 or rows.date.duplicated().any() or set(rows.year) != set(range(2016, 2024)):
        raise ValueError('Exact eight-year/2006-origin risk cohort required')
    source = history.set_index('date').loc[pd.to_datetime(rows.date)]
    actual = source.variance_target_1.to_numpy(float)
    np.testing.assert_allclose(actual, rows.actual_variance, rtol=1e-12, atol=1e-14)
    index = history.set_index('date').index
    dates = pd.to_datetime(rows.date)
    target = history.date.shift(-1).iloc[index.get_indexer(dates)].dt.strftime('%Y-%m-%d').tolist()
    if target != rows.target_date.tolist():
        raise ValueError('Exact next-Cotton target identity required')
    hi, lo, opening, close = (source[c].to_numpy(float) for c in
                              ('cotton_high', 'cotton_low', 'cotton_open', 'cotton_close'))
    valid = (np.isfinite([hi, lo, opening, close]).all(axis=0) & (lo > 0)
             & (hi >= np.maximum(opening, close)) & (lo <= np.minimum(opening, close)))
    variance = np.where(valid, np.log(np.where(valid, hi/lo, 1.))**2/(4*np.log(2.)), np.nan)
    baseline = rows.ewma_variance.to_numpy(float)
    if not np.isfinite(actual).all() or not np.isfinite(baseline).all() or (baseline <= 0).any() or (actual < 0).any():
        raise ValueError('Valid original risk reference required')
    rows['valid_range'] = valid
    rows['range_variance'] = variance
    rows['range_relative_log'] = np.where(valid, np.log(np.maximum(variance, 1e-12)/baseline), np.nan)
    rows['ewma_relative_error'] = actual/baseline-1
    return rows


def covariance(part):
    valid = part[:, 2].astype(bool)
    if not valid.any():
        raise ValueError('No eligible range observations')
    z, e = part[valid, :2].T
    return float(np.mean(z*e)-z.mean()*e.mean())


def intervals(parts, block, repetitions=10000):
    rng = np.random.default_rng(42)
    count = sum(p[:, 2].sum() for p in parts)
    point = sum(covariance(p)*p[:, 2].sum() for p in parts)/count
    draws = []
    for start in range(0, repetitions, 128):
        batch = min(128, repetitions-start)
        total, weights = np.zeros(batch), np.zeros(batch)
        for part in parts:
            length = min(block, len(part))
            starts = rng.integers(0, len(part)-length+1, (batch, math.ceil(len(part)/length)))
            idx = (starts[:, :, None]+np.arange(length)).reshape(batch, -1)[:, :len(part)]
            sample = part[idx]; w = sample[:, :, 2]; n = w.sum(axis=1)
            if (n == 0).any():
                raise ValueError('Empty bootstrap range cohort')
            z, e = sample[:, :, 0], sample[:, :, 1]
            cov = (z*e*w).sum(axis=1)/n-(z*w).sum(axis=1)*(e*w).sum(axis=1)/n**2
            total += cov*n; weights += n
        draws.extend(total/weights)
    noise = np.array(draws)-np.mean(draws)
    return {'covariance': float(point), 'ci_95': (point-np.quantile(noise, [.975, .025])).tolist(),
            'block': block, 'replicates': repetitions, 'eligible': int(count)}


def prepare(history_path, predictions_path, root):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    contract = {'profile': 'risk-range-error-moment-t1-v1', 'new_fits': 0,
                'history': str(Path(history_path).resolve()), 'history_sha256': sha(history_path),
                'predictions': str(Path(predictions_path).resolve()), 'predictions_sha256': sha(predictions_path),
                'analysis_sha256': sha(__file__), 'horizon': 1, 'origins': 2006, 'years': list(range(2016, 2024)),
                'feature': 'log(max(log(high/low)^2/(4log(2)),1e-12)/origin_EWMA)',
                'statistic': 'eligible-count weighted within-year covariance with actual/origin_EWMA-1',
                'centering_scope': 'ex-post diagnostic; not learned causal prediction coefficients',
                'bootstrap': 'recenter each year in each block draw; full ordinal slots;20/60;10000;seed42',
                'missing': 'OHLC invalid explicitly masked, original2006ordinal slots retained',
                'positive_screen': 'both CI lower>0;>=6/8 positive yearly covariances;top1% EWMA loss trim covariance>0',
                'model_training_allowed': False, 'release_allowed': False, 'independent_holdout': False,
                'clock_vintage_verified': False, 'qualification': 'CT daily-range and squared-return proxies; association only'}
    (root/'contract.json').write_text(json.dumps(contract, indent=2)+'\n', encoding='utf-8')
    return contract


def run(root):
    root = Path(root); c = json.loads((root/'contract.json').read_bytes())
    if sha(__file__) != c['analysis_sha256'] or sha(c['history']) != c['history_sha256'] or sha(c['predictions']) != c['predictions_sha256']:
        raise ValueError('Frozen source/input changed')
    frame = signal_frame(pd.read_parquet(c['history']), pd.read_csv(c['predictions']))
    groups = list(frame.groupby('year'))
    parts = [g[['range_relative_log', 'ewma_relative_error', 'valid_range']].fillna(0).to_numpy(float) for _, g in groups]
    years = [{'year': int(year), 'eligible': int(g.valid_range.sum()), 'ordinal_slots': len(g),
              'covariance': covariance(part)} for (year, g), part in zip(groups, parts, strict=True)]
    bootstrap = {str(b): intervals(parts, b) for b in (20, 60)}
    trim = frame.copy()
    trim.loc[trim.nlargest(math.ceil(len(trim)*.01), 'baseline_loss').index, 'valid_range'] = False
    trimmed_parts = [g[['range_relative_log', 'ewma_relative_error', 'valid_range']].fillna(0).to_numpy(float) for _, g in trim.groupby('year')]
    total = sum(p[:, 2].sum() for p in trimmed_parts)
    trimmed = sum(covariance(p)*p[:, 2].sum() for p in trimmed_parts)/total
    passed = all(v['ci_95'][0] > 0 for v in bootstrap.values()) and sum(y['covariance'] > 0 for y in years) >= 6 and trimmed > 0
    result = {'profile': c['profile'], 'new_fits': 0, 'screen_passed': passed,
              'yearly': years, 'bootstrap': bootstrap, 'trim_top1pct_covariance': float(trimmed),
              'invalid_range_origins': frame.loc[~frame.valid_range, 'date'].tolist(),
              'contract_sha256': sha(root/'contract.json'), 'model_skill_established': False,
              'independent_holdout': False, 'release_allowed': False, 'clock_vintage_verified': False}
    if (root/'result.json').exists():
        raise ValueError('Immutable result exists')
    frame.to_csv(root/'moments.csv', index=False, lineterminator='\n')
    result['moments_sha256'] = sha(root/'moments.csv')
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'run')); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--history', type=Path); p.add_argument('--predictions', type=Path)
    a = p.parse_args()
    if a.stage == 'prepare':
        if a.history is None or a.predictions is None:
            p.error('history and predictions required')
        print(prepare(a.history, a.predictions, a.root))
    else:
        print(run(a.root))


if __name__ == '__main__':
    main()
