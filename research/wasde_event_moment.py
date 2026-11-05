"""Retrospective dated-WASDE event/error association, never a causal feature."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd

helper = Path(__file__).with_name('range_moment.py')
spec = importlib.util.spec_from_file_location('fixed_event_math', helper)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def prepare(history, predictions, corpus, root):
    root.mkdir(parents=True, exist_ok=False)
    c = {'profile': 'risk-wasde-event-error-moment-t1-v1', 'new_fits': 0,
         'history': str(history.resolve()), 'history_sha256': m.sha(history),
         'predictions': str(predictions.resolve()), 'predictions_sha256': m.sha(predictions),
         'corpus': str(corpus.resolve()), 'corpus_sha256': m.sha(corpus),
         'analysis_sha256': m.sha(__file__), 'math_sha256': m.sha(helper),
         'statistic': 'within-year covariance dated-report target-day bit with actual/originEWMA-1',
         'known_no_report_months': ['2019-01'],
         'unknown_months_policy': 'mask all target-month observations lacking a unique dated corpus report;retain2006ordinalslots',
         'bootstrap': '20/60,10000,seed42,recenter every draw',
         'positive_screen': 'bothCI lower>0;>=6/8positive years;top1pctEWMA-loss trim covariance>0',
         'uses_ex_post_report_day': True, 'causal_feature_availability_verified': False,
         'next_if_positive': 'Build prior-document announced calendar before freezing a learner; no direct event-label leakage into model',
         'research_history_reused': True, 'multiplicity_adjusted': False,
         'model_training_allowed': False, 'model_skill_established': False, 'release_allowed': False,
         'clock_vintage_verified': False,
         'rationale': 'USDA ERS primary research reports Cotton response to WASDE; published future WAP dates suggest a separate announcement calendar channel, not report numerical/text features'}
    (root/'contract.json').write_text(json.dumps(c, indent=2)+'\n', encoding='utf-8')
    return c


def run(root):
    c = json.loads((root/'contract.json').read_bytes())
    for path, expected in [(Path(__file__), c['analysis_sha256']), (helper, c['math_sha256']),
                           (Path(c['history']), c['history_sha256']), (Path(c['predictions']), c['predictions_sha256']),
                           (Path(c['corpus']), c['corpus_sha256'])]:
        if m.sha(path) != expected:
            raise ValueError('Frozen input/source mismatch')
    corpus = json.loads(Path(c['corpus']).read_bytes())
    if (corpus['model_eligible'] or corpus['first_version_verified'] or corpus['publication_timestamp_verified']
            or corpus['source_tier'] != 'A_exploration_only'):
        raise ValueError('Preserve existing Tier-A qualification')
    report_days = [r['report_day'] for r in corpus['records']]
    if len(report_days) != len(set(report_days)) or any(not '2016-01-01' <= d < '2024-01-01' for d in report_days):
        raise ValueError('Unique scoped report dates required')
    per_month = pd.Series(report_days).str[:7].value_counts()
    known = set(per_month[per_month.eq(1)].index) | set(c['known_no_report_months'])
    f = m.signal_frame(pd.read_parquet(c['history']), pd.read_csv(c['predictions']))
    f['event'] = f.target_date.isin(report_days)
    f['event_eligible'] = f.target_date.str[:7].isin(known)
    f['range_relative_log'] = f.event.astype(float)
    fields = ['range_relative_log', 'ewma_relative_error', 'event_eligible']
    parts = [g[fields].to_numpy(float) for _, g in f.groupby('year')]
    years = [{'year': int(year), 'eligible': int(g.event_eligible.sum()), 'event_count': int((g.event & g.event_eligible).sum()),
              'covariance': m.covariance(part)} for (year, g), part in zip(f.groupby('year'), parts, strict=True)]
    bootstrap = {str(b): m.intervals(parts, b) for b in (20, 60)}
    trim = f.copy(); trim.loc[trim.nlargest(math.ceil(len(trim)*.01), 'baseline_loss').index, 'event_eligible'] = False
    trimmed_parts = [g[fields].to_numpy(float) for _, g in trim.groupby('year')]
    n = sum(p[:, 2].sum() for p in trimmed_parts)
    trimmed = sum(m.covariance(p)*p[:, 2].sum() for p in trimmed_parts)/n
    passed = all(v['ci_95'][0] > 0 for v in bootstrap.values()) and sum(y['covariance'] > 0 for y in years) >= 6 and trimmed > 0
    r = {'profile': c['profile'], 'new_fits': 0, 'screen_passed': passed, 'yearly': years,
         'bootstrap': bootstrap, 'trim_top1pct_covariance': float(trimmed), 'retained_origins': len(f),
         'unknown_months': sorted(set(f.target_date.str[:7])-known),
         'uses_ex_post_report_day': True, 'causal_feature_availability_verified': False,
         'model_training_allowed': False, 'model_skill_established': False, 'release_allowed': False,
         'independent_holdout': False, 'multiplicity_adjusted': False, 'clock_vintage_verified': False,
         'contract_sha256': m.sha(root/'contract.json')}
    if (root/'result.json').exists():
        raise ValueError('Immutable result exists')
    f.to_csv(root/'moments.csv', index=False, lineterminator='\n'); r['moments_sha256'] = m.sha(root/'moments.csv')
    (root/'result.json').write_text(json.dumps(r, indent=2)+'\n', encoding='utf-8')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=('prepare', 'run'))
    p.add_argument('--root', type=Path, required=True)
    for field in ('history', 'predictions', 'corpus'):
        p.add_argument('--'+field, type=Path)
    a = p.parse_args()
    if a.stage == 'prepare':
        if any(v is None for v in (a.history, a.predictions, a.corpus)):
            p.error('history/predictions/corpus required')
        print(prepare(a.history, a.predictions, a.corpus, a.root))
    else:
        print(run(a.root))


if __name__ == '__main__':
    main()
