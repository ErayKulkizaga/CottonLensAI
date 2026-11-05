"""One magnitude-controlled signed-shock diagnostic; no forecast fit or grid."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research import range_moment as m


def partial_covariance(part):
    data = part[part[:, 3].astype(bool), :3]
    if len(data) < 3:
        raise ValueError('Insufficient eligible partial-covariance rows')
    z, e, w = (data - data.mean(axis=0)).T
    variance = np.mean(w*w)
    if variance <= 1e-15:
        raise ValueError('Unsigned shock has no usable variation')
    return float(np.mean(z*e) - np.mean(z*w)*np.mean(w*e)/variance)


def intervals(parts, block, repetitions=10000):
    rng = np.random.default_rng(42)
    count = sum(p[:, 3].sum() for p in parts)
    point = sum(partial_covariance(p)*p[:, 3].sum() for p in parts)/count
    draws = []
    for start in range(0, repetitions, 128):
        batch = min(128, repetitions-start)
        total, weights = np.zeros(batch), np.zeros(batch)
        for part in parts:
            length = min(block, len(part))
            starts = rng.integers(0, len(part)-length+1, (batch, math.ceil(len(part)/length)))
            ix = (starts[:, :, None]+np.arange(length)).reshape(batch, -1)[:, :len(part)]
            sample = part[ix]; mask = sample[:, :, 3]; n = mask.sum(axis=1)
            if (n < 3).any():
                raise ValueError('Insufficient bootstrap data')
            columns = []
            for column in range(3):
                value = sample[:, :, column]
                columns.append((value-(value*mask).sum(axis=1)[:, None]/n[:, None])*mask)
            z, e, w = columns
            variance = (w*w).sum(axis=1)/n
            if (variance <= 1e-15).any():
                raise ValueError('Bootstrap unsigned shock has no usable variation')
            cov = (z*e).sum(axis=1)/n - ((z*w).sum(axis=1)/n)*((w*e).sum(axis=1)/n)/variance
            total += cov*n; weights += n
        draws.extend(total/weights)
    noise = np.asarray(draws)-np.mean(draws)
    return {'partial_covariance': float(point), 'ci_95': (point-np.quantile(noise, [.975, .025])).tolist(),
            'eligible': int(count), 'replicates': repetitions, 'block': block}


def prepare(root, history, predictions):
    root.mkdir(parents=True, exist_ok=False)
    contract = {'profile': 'signed-shock-partial-risk-moment-t1-v1', 'new_fits': 0,
                'history': str(history.resolve()), 'history_sha256': m.sha(history),
                'predictions': str(predictions.resolve()), 'predictions_sha256': m.sha(predictions),
                'analysis_sha256': m.sha(__file__), 'base_math_sha256': m.sha(m.__file__),
                'feature': 'signed current log-return square/origin_EWMA',
                'control': 'unsigned current log-return square/origin_EWMA; ex-post linear projection within each year',
                'positive_hypothesis': 'positive shocks associate with higher next-day relative risk beyond unsigned size',
                'scope': 'same CT proxy T+1 squared-return/EWMA0.94,2006ordinal origins,2016-2023',
                'screen': 'both20/60CI lower>0;>=6/8positive years;top1pctbaseline-loss trim partialcov>0',
                'bootstrap': '10000/seed42; full ordinal mask; reproject and recenter in each draw',
                'prior_evidence': 'HAR and symmetric GARCH omit return sign; asymmetric registry query inspected, no match not novelty proof',
                'literature': 'Chen/Mu2021 inverse leverage commodity mechanism; cotton explicitly excluded, not evidence for CottonLens',
                'literature_url': 'https://discovery.dundee.ac.uk/ws/files/49087226/1_s2.0_S2405851320300167_main.pdf',
                'model_skill_established': False, 'training_allowed': False,
                'source_clock_verified': False, 'independent_holdout': False,
                'history_reused': True, 'multiplicity_adjusted': False}
    (root/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    return {'profile': contract['profile'], 'new_fits': 0}


def run(root):
    c = json.loads((root/'contract.json').read_text())
    for path, expected in [(Path(c['history']), c['history_sha256']), (Path(c['predictions']), c['predictions_sha256']),
                           (Path(__file__), c['analysis_sha256']), (Path(m.__file__), c['base_math_sha256'])]:
        if m.sha(path) != expected:
            raise ValueError('Frozen source/input changed')
    h = pd.read_parquet(c['history']); f = m.signal_frame(h, pd.read_csv(c['predictions']))
    returns = h.set_index('date').loc[pd.to_datetime(f.date)].cotton_ret_1.to_numpy(float)
    f['unsigned_shock'] = returns**2/f.ewma_variance.to_numpy(float)
    f['signed_shock'] = np.sign(returns)*f.unsigned_shock
    f['feature_eligible'] = np.isfinite(returns)
    columns = ['signed_shock', 'ewma_relative_error', 'unsigned_shock', 'feature_eligible']
    groups = list(f.groupby('year'))
    parts = [g[columns].fillna(0).to_numpy(float) for _, g in groups]
    years = [{'year': int(year), 'eligible': int(g.feature_eligible.sum()), 'partial_covariance': partial_covariance(part)}
             for (year, g), part in zip(groups, parts, strict=True)]
    bootstrap = {str(b): intervals(parts, b) for b in (20, 60)}
    trim = f.copy(); trim.loc[trim.nlargest(math.ceil(len(trim)*.01), 'baseline_loss').index, 'feature_eligible'] = False
    trimmed = [g[columns].fillna(0).to_numpy(float) for _, g in trim.groupby('year')]
    n = sum(p[:, 3].sum() for p in trimmed)
    trim_cov = sum(partial_covariance(p)*p[:, 3].sum() for p in trimmed)/n
    passed = (all(v['ci_95'][0] > 0 for v in bootstrap.values()) and
              sum(y['partial_covariance'] > 0 for y in years) >= 6 and trim_cov > 0)
    result = {'profile': c['profile'], 'new_fits': 0, 'retained_origins': len(f),
              'yearly': years, 'bootstrap': bootstrap, 'trim_top1pct_partial_covariance': float(trim_cov),
              'screen_passed': bool(passed), 'training_allowed': False, 'model_skill_established': False,
              'independent_holdout': False, 'source_clock_verified': False, 'multiplicity_adjusted': False,
              'contract_sha256': m.sha(root/'contract.json')}
    if (root/'result.json').exists():
        raise ValueError('Immutable result exists')
    f.to_csv(root/'moments.csv', index=False, lineterminator='\n'); result['moments_sha256'] = m.sha(root/'moments.csv')
    (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--history', type=Path)
    parser.add_argument('--predictions', type=Path); args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.history, args.predictions) if args.stage == 'prepare' else run(args.root)))
