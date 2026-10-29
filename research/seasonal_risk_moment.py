"""Fixed cotton-season/EWMA error association; no forecast model or fit."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.range_moment import sha

PROFILE = 'cotton-season-ewma-error-moment-t1-v1'
USDA_2010 = 'https://www.nass.usda.gov/Publications/Todays_Reports/reports/fcdate10.pdf'


def centered_covariance(part):
    season, error = part.T
    if len(part) < 3 or season.min() == season.max():
        raise ValueError('Both seasons required within year')
    return float(np.mean((season-season.mean())*(error-error.mean())))


def intervals(parts, block, repetitions=10000):
    count = sum(map(len, parts))
    point = sum(len(p)*centered_covariance(p) for p in parts)/count
    rng = np.random.default_rng(42)
    draws = []
    for start in range(0, repetitions, 128):
        batch = min(128, repetitions-start)
        total = np.zeros(batch)
        for part in parts:
            length = min(block, len(part))
            begins = rng.integers(0, len(part)-length+1, (batch, math.ceil(len(part)/length)))
            indexes = (begins[:, :, None]+np.arange(length)).reshape(batch, -1)[:, :len(part)]
            sampled = part[indexes]
            season, error = sampled.transpose(2, 0, 1)
            centered = (season-season.mean(axis=1)[:, None])*(error-error.mean(axis=1)[:, None])
            total += centered.mean(axis=1)*len(part)
        draws.extend(total/count)
    noise = np.asarray(draws)-np.mean(draws)
    return {'covariance': float(point), 'ci_95': (point-np.quantile(noise, [.975, .025])).tolist(),
            'block': block, 'replicates': repetitions}


def prepare(root, paired):
    root.mkdir(parents=True, exist_ok=False)
    body = {'profile': PROFILE, 'new_fits': 0, 'paired_path': str(paired.resolve()),
            'paired_sha256': sha(paired), 'analysis_sha256': sha(__file__),
            'frozen_season': 'origin months April-November inclusive, fixed before result',
            'error': 'next CT squared log return/origin EWMA0.94 - 1',
            'statistic': 'origin-weighted within-year covariance(season,error)',
            'positive_hypothesis': 'crop-active months have higher next-day EWMA-relative risk error',
            'screen': 'both 20/60 moving-block CI lower>0, >=6/8 positive years, positive top1% EWMA-loss trim',
            'bootstrap': '10000 draws/seed42, resample within each ordinal year and recenter per draw',
            'scope': 'original CT T+1/2016-2023/2006 paired origins; not named-contract risk',
            'source': USDA_2010, 'source_role': 'pre-2016 crop timing only; volatility sign is a hypothesis',
            'model_skill_established': False, 'training_allowed': False,
            'independent_holdout': False, 'source_clock_verified': False,
            'multiplicity_adjusted': False, 'history_reused': True}
    (root/'contract.json').write_text(json.dumps(body, indent=2)+'\n', encoding='utf-8')
    return {'profile': PROFILE, 'new_fits': 0}


def run(root):
    c = json.loads((root/'contract.json').read_text(encoding='utf-8'))
    paired = Path(c['paired_path'])
    if sha(paired) != c['paired_sha256'] or sha(__file__) != c['analysis_sha256']:
        raise ValueError('Frozen input or analysis changed')
    f = pd.read_csv(paired, parse_dates=['date', 'target_date'])
    if (len(f) != 2006 or sorted(f.year.unique()) != list(range(2016, 2024))
            or not f.horizon.eq(1).all() or f.date.duplicated().any()
            or not f.target_date.gt(f.date).all()):
        raise ValueError('Original T+1 ordinal cohort mismatch')
    actual, ewma = f.actual_variance.to_numpy(float), f.ewma_variance.to_numpy(float)
    if (not np.isfinite(actual).all() or not np.isfinite(ewma).all()
            or (actual < 0).any() or (ewma <= 0).any()):
        raise ValueError('Invalid actual or EWMA')
    f['season'] = f.date.dt.month.between(4, 11).astype(int)
    f['relative_error'] = actual/ewma-1
    groups = list(f.groupby('year', sort=True))
    parts = [g[['season', 'relative_error']].to_numpy(float) for _, g in groups]
    yearly = [{'year': int(year), 'count': len(g), 'covariance': centered_covariance(part)}
              for (year, g), part in zip(groups, parts, strict=True)]
    bootstrap = {str(block): intervals(parts, block) for block in (20, 60)}
    trim = f.drop(f.nlargest(math.ceil(len(f)*.01), 'baseline_loss').index)
    trim_parts = [g[['season', 'relative_error']].to_numpy(float)
                  for _, g in trim.groupby('year', sort=True)]
    trimmed = sum(len(p)*centered_covariance(p) for p in trim_parts)/len(trim)
    passed = (all(x['ci_95'][0] > 0 for x in bootstrap.values())
              and sum(x['covariance'] > 0 for x in yearly) >= 6 and trimmed > 0)
    result = {'profile': PROFILE, 'new_fits': 0, 'retained_origins': len(f),
              'yearly': yearly, 'bootstrap': bootstrap,
              'top1pct_trim_covariance': float(trimmed), 'screen_passed': bool(passed),
              'model_skill_established': False, 'training_allowed': False,
              'independent_holdout': False, 'source_clock_verified': False,
              'multiplicity_adjusted': False, 'contract_sha256': sha(root/'contract.json')}
    if (root/'result.json').exists():
        raise ValueError('Immutable result already exists')
    f.to_csv(root/'moments.csv', index=False, lineterminator='\n')
    result['moments_sha256'] = sha(root/'moments.csv')
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--paired', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.paired) if args.stage == 'prepare' else run(args.root)))
