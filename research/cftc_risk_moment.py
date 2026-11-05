"""One frozen absolute-position-turnover association screen; no learner or grid."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.research import cftc_exploration as source

from research import range_moment as moment


def prepare(root, history, predictions, positions):
    root.mkdir(parents=True, exist_ok=False)
    source_path = Path(source.__file__)
    rows, manifest = source.read_positions(positions)
    contract = {'profile': 'cftc-risk-turnover-moment-t1-v1', 'new_fits': 0,
                'history': str(history.resolve()), 'history_sha256': moment.sha(history),
                'predictions': str(predictions.resolve()), 'predictions_sha256': moment.sha(predictions),
                'positions': str(positions.resolve()), 'positions_sha256': moment.sha(positions),
                'position_manifest_sha256': moment.sha(positions.with_suffix('.manifest.json')),
                'analysis_sha256': moment.sha(__file__), 'math_sha256': moment.sha(moment.__file__),
                'source_helper_sha256': moment.sha(source_path), 'source_rows': len(rows),
                'feature': 'abs(cftc_managed_share_change_L1); one feature, no lag or transformation search',
                'clock': 'existing report+7days, strictly next Cotton observation then L1, carry<=10; 29 masked reports',
                'source_manifest_rows': manifest['rows'],
                'scope': 'CT proxy T+1 squared-return/EWMA0.94, 2006 origins,2016-2023; not named-contract skill',
                'hypothesis': 'Magnitude of past managed-money net/OI adjustment contains risk information beyond recorded EWMA',
                'statistic': 'within-year centered covariance with actual/origin_EWMA-1; full ordinal missing masks',
                'positive_screen': 'both20/60CI lower>0;>=6/8positive years;trimtop1pctbaseline-loss covariance>0',
                'bootstrap': {'blocks': [20, 60], 'repetitions': 10000, 'seed': 42, 'recenter_each_draw': True},
                'prior_evidence': 'T+5 signed-price CFTC controls are related, not T+1 risk tests; risk and HAR+CFTC lookups inspected; no match is not novelty proof',
                'research_history_reused': True, 'multiplicity_adjusted': False,
                'model_skill_established': False, 'training_allowed': False,
                'first_vintage_verified': False, 'publication_utc_verified': False}
    (root / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    return {'profile': contract['profile'], 'new_fits': 0, 'source_rows': len(rows)}


def run(root):
    contract = json.loads((root / 'contract.json').read_text())
    identities = [(Path(contract[k]), contract[k + '_sha256']) for k in ('history', 'predictions', 'positions')]
    identities += [(Path(__file__), contract['analysis_sha256']),
                   (Path(moment.__file__), contract['math_sha256']),
                   (Path(source.__file__), contract['source_helper_sha256']),
                   (Path(contract['positions']).with_suffix('.manifest.json'), contract['position_manifest_sha256'])]
    if any(moment.sha(path) != expected for path, expected in identities):
        raise ValueError('frozen source/input changed')
    history = pd.read_parquet(contract['history'])
    rows, _ = source.read_positions(Path(contract['positions']))
    attached = source.add_positions(history, rows)
    frame = moment.signal_frame(history, pd.read_csv(contract['predictions']))
    attached = attached.set_index('date').loc[pd.to_datetime(frame.date)]
    turnover = attached.cftc_managed_share_change_L1.to_numpy(float)
    frame['risk_feature'] = np.abs(turnover)
    frame['feature_eligible'] = np.isfinite(turnover)
    frame['risk_feature_age'] = attached.cftc_age_L1.to_numpy(float)
    groups = list(frame.groupby('year'))
    parts = [g[['risk_feature', 'ewma_relative_error', 'feature_eligible']].fillna(0).to_numpy(float) for _, g in groups]
    years = [{'year': int(year), 'eligible': int(g.feature_eligible.sum()), 'ordinal_slots': len(g),
              'covariance': moment.covariance(part)} for (year, g), part in zip(groups, parts, strict=True)]
    bootstrap = {str(b): moment.intervals(parts, b) for b in (20, 60)}
    trim = frame.copy()
    trim.loc[trim.nlargest(math.ceil(len(trim) * .01), 'baseline_loss').index, 'feature_eligible'] = False
    trimmed = [g[['risk_feature', 'ewma_relative_error', 'feature_eligible']].fillna(0).to_numpy(float)
               for _, g in trim.groupby('year')]
    total = sum(p[:, 2].sum() for p in trimmed)
    trimmed_cov = sum(moment.covariance(p) * p[:, 2].sum() for p in trimmed) / total
    passed = (all(v['ci_95'][0] > 0 for v in bootstrap.values())
              and sum(y['covariance'] > 0 for y in years) >= 6 and trimmed_cov > 0)
    result = {'profile': contract['profile'], 'new_fits': 0, 'retained_origins': len(frame),
              'eligible_origins': int(frame.feature_eligible.sum()), 'yearly': years, 'bootstrap': bootstrap,
              'trim_top1pct_covariance': float(trimmed_cov), 'screen_passed': bool(passed),
              'model_skill_established': False, 'training_allowed': False, 'independent_holdout': False,
              'publication_utc_verified': False, 'first_vintage_verified': False,
              'multiplicity_adjusted': False, 'contract_sha256': moment.sha(root / 'contract.json')}
    if (root / 'result.json').exists():
        raise ValueError('immutable result exists')
    frame.to_csv(root / 'moments.csv', index=False, lineterminator='\n')
    result['moments_sha256'] = moment.sha(root / 'moments.csv')
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--positions', type=Path)
    args = parser.parse_args()
    result = prepare(args.root, args.history, args.predictions, args.positions) if args.stage == 'prepare' else run(args.root)
    print(json.dumps(result))
