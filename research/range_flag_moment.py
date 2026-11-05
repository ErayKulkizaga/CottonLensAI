"""Fixed flat high/low bar flag versus future CT proxy EWMA error; no fits."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

helper = Path(__file__).with_name('range_moment.py')
spec = importlib.util.spec_from_file_location('fixed_range_math', helper)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prepare(history, predictions, root):
    root.mkdir(parents=True, exist_ok=False)
    c = {'profile': 'risk-flat-bar-error-moment-t1-v1', 'new_fits': 0,
         'history': str(history.resolve()), 'history_sha256': m.sha(history),
         'predictions': str(predictions.resolve()), 'predictions_sha256': m.sha(predictions),
         'analysis_sha256': m.sha(__file__), 'math_sha256': m.sha(helper),
         'feature': 'int(current_origin_high == current_origin_low)',
         'validity': 'finite high>=low>0; settlement/open containment not required for quoted flat-bar flag',
         'cohort': 'same2006T1/2016-2023risk origins; fullordinalmask',
         'statistic': 'within-year covariance flag with actual/originEWMA-1; ex-post diagnostic centering',
         'bootstrap': '20/60,10000,seed42,recenter every draw',
         'positive_screen': 'bothCI lower>0;>=6/8positive years;top1pctEWMA-loss trim covariance>0',
         'prior_evidence': 'Continuous range moment screen failed;248flat bars in2006source origins. No flag/error subgroup outcome was inspected before this registration.',
         'scope': 'observable CT proxy measurement state, not named-contract or physical/intraday risk',
         'research_history_reused': True, 'multiplicity_adjusted': False,
         'model_skill_established': False, 'release_allowed': False, 'clock_vintage_verified': False}
    (root/'contract.json').write_text(json.dumps(c, indent=2)+'\n', encoding='utf-8')
    return c


def run(root):
    c = json.loads((root/'contract.json').read_bytes())
    for path, expected in [(Path(__file__), c['analysis_sha256']), (helper, c['math_sha256']),
                           (Path(c['history']), c['history_sha256']), (Path(c['predictions']), c['predictions_sha256'])]:
        if m.sha(path) != expected:
            raise ValueError('Frozen source/input mismatch')
    h = pd.read_parquet(c['history']); f = m.signal_frame(h, pd.read_csv(c['predictions']))
    source = h.set_index('date').loc[pd.to_datetime(f.date)]
    hi, lo = source.cotton_high.to_numpy(float), source.cotton_low.to_numpy(float)
    valid = np.isfinite([hi, lo]).all(axis=0) & (hi >= lo) & (lo > 0)
    f['flat_bar'] = hi == lo; f['flag_eligible'] = valid
    f['range_relative_log'] = f.flat_bar.astype(float)
    parts = [g[['range_relative_log', 'ewma_relative_error', 'flag_eligible']].to_numpy(float) for _, g in f.groupby('year')]
    years = [{'year': int(year), 'eligible': int(g.flag_eligible.sum()), 'flat_count': int((g.flat_bar & g.flag_eligible).sum()),
              'covariance': m.covariance(part)} for (year, g), part in zip(f.groupby('year'), parts, strict=True)]
    bootstrap = {str(b): m.intervals(parts, b) for b in (20, 60)}
    trim = f.copy(); trim.loc[trim.nlargest(math.ceil(len(trim)*.01), 'baseline_loss').index, 'flag_eligible'] = False
    trimmed_parts = [g[['range_relative_log', 'ewma_relative_error', 'flag_eligible']].to_numpy(float) for _, g in trim.groupby('year')]
    n = sum(p[:, 2].sum() for p in trimmed_parts)
    trimmed = sum(m.covariance(p)*p[:, 2].sum() for p in trimmed_parts)/n
    passed = all(v['ci_95'][0] > 0 for v in bootstrap.values()) and sum(y['covariance'] > 0 for y in years) >= 6 and trimmed > 0
    r = {'profile': c['profile'], 'new_fits': 0, 'screen_passed': passed, 'yearly': years,
         'bootstrap': bootstrap, 'trim_top1pct_covariance': float(trimmed), 'retained_origins': len(f),
         'contract_sha256': m.sha(root/'contract.json'), 'model_skill_established': False,
         'scope': c['scope'], 'release_allowed': False, 'independent_holdout': False,
         'multiplicity_adjusted': False, 'clock_vintage_verified': False}
    if (root/'result.json').exists():
        raise ValueError('Immutable result exists')
    f.to_csv(root/'moments.csv', index=False, lineterminator='\n')
    r['moments_sha256'] = m.sha(root/'moments.csv')
    (root/'result.json').write_text(json.dumps(r, indent=2)+'\n', encoding='utf-8')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('--root', type=Path, required=True); p.add_argument('--history', type=Path); p.add_argument('--predictions', type=Path)
    a = p.parse_args()
    if a.stage == 'prepare':
        if a.history is None or a.predictions is None:
            p.error('history and predictions required')
        print(prepare(a.history, a.predictions, a.root))
    else:
        print(run(a.root))


if __name__ == '__main__':
    main()
