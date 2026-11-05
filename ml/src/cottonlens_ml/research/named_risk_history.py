"""Matched fixed Log-HAR input-history control, conditional clock/vintage."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import named_risk_control as base
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.full_year import volatility_recipe
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.volatility import FLOOR, qlike

PROFILE = 'named-risk-history-control-t1-v1'
ARMS = ('ct', 'named')
RECIPE = {**volatility_recipe('har', 1), 'feature_control': PROFILE, 'label_arm': 'named'}
LEARNER_FILES = ('ml/src/cottonlens_ml/research/models.py',
                 'ml/src/cottonlens_ml/research/volatility.py', 'ml/src/cottonlens_ml/runtime_guard.py')


def own_history(dates, quotes, available):
    """Last k OBSERVED exact adjacent-session pairs; do not invent missing returns."""
    events = defaultdict(list)
    positions = {d: i for i, d in enumerate(dates)}
    if len(positions) != len(dates) or dates != sorted(dates):
        raise ValueError('Unique sorted Cotton calendar required')
    for previous, current in pairwise(dates):
        a, b = quotes.get(previous), quotes.get(current)
        if a is None or b is None:
            continue
        for name in sorted(set(a['quotes']) & set(b['quotes'])):
            prices = (a['quotes'][name], b['quotes'][name])
            if any(not math.isfinite(p) or p <= 0 for p in prices):
                raise ValueError('Positive finite own-contract prices required')
            events[name].append((current, max(available(a), available(b)), math.log(prices[1]/prices[0])**2))
    values = []
    for day in dates:
        quote = quotes.get(day)
        row = {f'own_har_{k}': np.nan for k in (1, 5, 22)}
        row.update(own_feature_ready=False, own_observed_pairs=0, own_22_session_span=None,
                   own_latest_pair_age=None, own_latest_pair_date=None)
        if quote is not None and available(quote) <= base.decision(day):
            past = [(end, square) for end, clock, square in events[quote['second_contract']]
                    if end <= day and clock <= base.decision(day)]
            row['own_observed_pairs'] = len(past)
            if len(past) >= 22:
                row.update(own_feature_ready=True, own_latest_pair_date=past[-1][0],
                           own_latest_pair_age=positions[day]-positions[past[-1][0]],
                           own_22_session_span=positions[past[-1][0]]-positions[past[-22][0]]+1)
                for k in (1, 5, 22):
                    row[f'own_har_{k}'] = math.log(max(sum(v for _, v in past[-k:])/k, FLOOR))
        values.append(row)
    return pd.DataFrame(values)


def load_inputs(inputs):
    h, parent = base.load_inputs(inputs)
    label_ready = read_record(inputs/'label-ready.json')
    proof = read_record(inputs/'label-proof.json')
    if (digest(inputs/'label-history.parquet') != label_ready['history_sha256']
            or label_ready['identity']['source_id'] != proof['source_id']
            or digest(inputs/'label-report.json') != proof['result_sha256']):
        raise ValueError('Completed label-study binding differs')
    for name, expected in label_ready['identity']['inputs_sha256'].items():
        if digest(inputs/name) != expected:
            raise ValueError('Parent label-study input differs')
    pd.testing.assert_frame_equal(h, pd.read_parquet(inputs/'label-history.parquet'), check_exact=True)
    panel = read_record(inputs/'quotes.json')
    spec = importlib.util.spec_from_file_location('fixed_label_clock', inputs/'label-helper.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    own = own_history(h.date.dt.strftime('%Y-%m-%d').tolist(),
                      {q['report_date']: q for q in panel['rows']}, helper.assumed_available)
    pd.testing.assert_series_equal(own.own_feature_ready, h.named_ewma_eligible, check_names=False)
    return pd.concat([h, own], axis=1), parent


def mature_dates(h, cutoff):
    dates = base.mature_dates(h, cutoff)
    subset = rows_at(h, dates)
    return subset.loc[subset.own_feature_ready, 'date'].dt.strftime('%Y-%m-%d').tolist()


def plan(h, split):
    folds, readiness = [], []
    for parent in split['folds']:
        past = mature_dates(h, parent['origins'][0])[-189:]
        inner = [past[i:i+63] for i in range(0, len(past), 63)]
        jobs = [j for i, block in enumerate(inner) for j in base.control_jobs(h, block, f'inner_{i}')]
        jobs += base.control_jobs(h, parent['origins'], 'outer')
        for job in jobs:
            original_train = job['train_dates']
            job['train_dates'] = mature_dates(h, job['cutoff'])
            job['excluded_parent_train_dates'] = [d for d in original_train if d not in set(job['train_dates'])]
            job['origin_slots'] = job['origins']
            test = rows_at(h, job['origin_slots'])
            job['origins'] = test.loc[test.own_feature_ready, 'date'].dt.strftime('%Y-%m-%d').tolist()
        minimum = min((len(j['train_dates']) for j in jobs), default=0)
        supported = len(past) == 189 and minimum >= 500 and all(j['origins'] for j in jobs)
        readiness.append({'year': parent['year'], 'supported': bool(supported),
                          'minimum_train': minimum, 'validation_labels': len(past)})
        if supported:
            outer = rows_at(h, parent['origins'])
            folds.append({'year': parent['year'], 'origins': parent['origins'], 'inner': inner, 'jobs': jobs,
                          'forecast_origins': outer.loc[outer.own_feature_ready, 'date'].dt.strftime('%Y-%m-%d').tolist(),
                          'eligible_outer': int((outer.named_eligible & outer.own_feature_ready).sum())})
    return {'profile': PROFILE, 'recipe': RECIPE, 'arms': list(ARMS), 'folds': folds, 'readiness': readiness,
            'fit_budget': {'market': 2*sum(len(f['jobs']) for f in folds), 'synthetic': 0,
                           'processes': 1, 'threads': 2, 'session_minutes': 30},
            'primary': 'raw CT-history minus own-contract-history HAR QLIKE on same named risk',
            'secondary': 'both raw/selected versus unchanged own-contract EWMA; largest1pct loss sensitivity',
            'feature_formula': 'log mean last1/5/22 observed exact adjacent-Cotton-date pair squares of origin second contract',
            'missing_policy': 'No imputed returns; retain246ordinal slots; no forecast for source-unready slots in BOTH arms',
            'selection': 'same3x63 past-only named QLIKE, weights0/.25/.5/.75/1, smallest-weight tie',
            'baseline': 'unchanged own-contract EWMA lambda.94,22observed pairs, first-square initialization',
            'clock': 'date+1day00:15UTC; archive calendar+1day00:00UTC/final vintage ASSUMED',
            'learning_control': 'Reuse native HAR only if learner bytes/dependencies and numerical replay match; no new synthetic fit',
            'deliberate_repeat_reason': 'Both arms named labels; change only HAR input process on NEW common mature population',
            'gate_evaluation_allowed': False, 'independent_holdout': False, 'release_allowed': False,
            'source_timestamp_verified': False, 'first_vintage_verified': False}


def projection(h, dates, arm, *, training):
    if arm not in ARMS:
        raise ValueError('Unknown feature arm')
    rows = rows_at(h, dates).copy()
    if not rows.own_feature_ready.all():
        raise ValueError('No future-fill or missing own-history imputation')
    if arm == 'named':
        for k in (1, 5, 22):
            rows[f'har_{k}'] = rows[f'own_har_{k}']
    rows['variance_target_1'] = rows.named_variance
    if training and (not rows.named_eligible.all() or not np.isfinite(rows.variance_target_1).all()):
        raise ValueError('Common finite named mature labels required')
    return rows


def output(h, fold, arm, records):
    # Reuse verified selection logic only on actual forecast rows, then restore
    # missing ordinal slots explicitly; never compress the statistical cohort.
    valid_fold = {**fold, 'origins': fold['forecast_origins']}
    result = base.output(h, valid_fold, arm, records)
    by_date = {r['date']: r for r in result['records']}
    entries = []
    for row in rows_at(h, fold['origins']).itertuples():
        day = row.date.strftime('%Y-%m-%d')
        if day in by_date:
            entry = {**by_date[day], 'forecast_available': True}
        else:
            entry = {'date': day, 'target_date': row.target_date_1.strftime('%Y-%m-%d'),
                     'contract': row.named_contract if row.named_eligible else None,
                     'eligible': False, 'named_actual': float(row.named_variance) if row.named_eligible else None,
                     'baseline': None, 'raw': None, 'selected': None, 'forecast_available': False,
                     'named_reason': row.named_reason, 'ewma_observations': int(row.named_ewma_observations)}
        entry.update(own_22_session_span=int(row.own_22_session_span) if row.own_feature_ready else None,
                     own_latest_pair_age=int(row.own_latest_pair_age) if row.own_feature_ready else None)
        entries.append(entry)
    return {**result, 'records': entries, 'feature_arm': arm}


def prepare(repo, folder, evidence):
    repo, folder, evidence = map(Path, (repo, folder, evidence))
    if folder.exists():
        raise ValueError('Immutable namespace exists')
    with writer(folder):
        for p in (evidence/'execution/inputs').iterdir():
            if p.is_file():
                copy_immutable(p, folder/'inputs'/p.name)
        for name, path in {'label-ready.json': evidence/'execution/ready.json',
                           'label-history.parquet': evidence/'execution/history.parquet',
                           'label-report.json': evidence/'execution/reports/result.json',
                           'label-proof.json': repo/'research/evidence/named-risk-label-result-20261012.json',
                           'parent-source.json': evidence/'execution/source-manifest.json'}.items():
            copy_immutable(path, folder/'inputs'/name)
        # Old native control stays separately bound to old source, not reused as
        # a training cache. Copy only immutable completed receipt/payload files.
        for section in ('completed', 'payloads'):
            for p in (evidence/'execution/learning-ledger'/section).rglob('*'):
                if p.is_file():
                    copy_immutable(p, folder/'learning-reference'/p.relative_to(evidence/'execution/learning-ledger'))
        h, parent = load_inputs(folder/'inputs')
        design = plan(h, parent['identity']['split'])
        if not design['folds']:
            raise ValueError('No fixed-rule fold supported; preserve readiness evidence')
        validate(repo/'research')
        reg, trials = load_registry(repo/'research')
        related = check(reg, trials, family='har', horizon=1)
        identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'],
                    'design': design, 'design_id': content_id(design), 'split': parent['identity']['split'],
                    'inputs_sha256': {p.relative_to(folder).as_posix(): digest(p)
                                     for section in ('inputs', 'learning-reference') for p in (folder/section).rglob('*') if p.is_file()},
                    'research_data_id': frame_identity(h, list(h)), 'python': sys.version.split()[0],
                    'lock_sha256': digest(repo/'ml/uv.lock'),
                    'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'pyarrow', 'scikit-learn')}}
        h.to_parquet(folder/'history.parquet', index=False)
        freeze_record(folder/'source-manifest.json', research_source_identity(repo))
        freeze_record(folder/'history-check.json', {'related_records': [r['id'] for r in related['experiments']], 'new_fits': 0})
        freeze_record(folder/'decision-contract.json', design)
        freeze_record(folder/'ready.json', {'identity': identity, 'history_sha256': digest(folder/'history.parquet')})
    return design


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder/'ready.json')
    identity = ready['identity']
    if (identity['profile'] != PROFILE or identity['source_id'] != research_source_identity(repo)['source_id']
            or identity['python'] != sys.version.split()[0] or identity['lock_sha256'] != digest(repo/'ml/uv.lock')
            or identity['versions'] != {p: importlib.metadata.version(p) for p in identity['versions']}):
        raise ValueError('Source/environment changed; new namespace required')
    for name, expected in identity['inputs_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Frozen input/reference hash mismatch')
    h, parent = load_inputs(folder/'inputs')
    design = plan(h, parent['identity']['split'])
    if (design != identity['design'] or content_id(design) != identity['design_id']
            or design != read_record(folder/'decision-contract.json')
            or digest(folder/'history.parquet') != ready['history_sha256']
            or frame_identity(h, list(h)) != identity['research_data_id']):
        raise ValueError('Data/cohort/design changed')
    pd.testing.assert_frame_equal(h, pd.read_parquet(folder/'history.parquet'), check_exact=True)
    return ready, h


def validate_learning(folder, repo):
    folder, repo = Path(folder), Path(repo)
    source = read_record(folder/'inputs/parent-source.json')
    parent = read_record(folder/'inputs/label-ready.json')['identity']
    if source['source_id'] != parent['source_id'] or source['source_id'] != content_id(source['files']):
        raise ValueError('Prior learner source identity mismatch')
    for name in LEARNER_FILES:
        if digest(repo/name) != source['files'][name]:
            raise ValueError('Learner changed; old control cannot certify it')
    if (parent['python'] != sys.version.split()[0]
            or parent['versions'] != {p: importlib.metadata.version(p) for p in parent['versions']}
            or parent['lock_sha256'] != digest(repo/'ml/uv.lock')):
        raise ValueError('Prior learner environment changed')
    receipts = list((folder/'learning-reference/completed').glob('*.json'))
    if len(receipts) != 1:
        raise ValueError('Exact prior HAR synthetic receipt required')
    record = read_record(receipts[0])
    if (record['identity']['source_id'] != source['source_id'] or
            record['specification']['role'] != 'synthetic_noiseless_log_har_control'):
        raise ValueError('Wrong prior control')
    for name, expected in record['files'].items():
        if not safe_member(name) or digest(folder/'learning-reference'/name) != expected:
            raise ValueError('Prior native control hash mismatch')
    model = json.loads((folder/'learning-reference'/record['payload_root']/'variance-model.json').read_bytes())
    if model['kind'] != 'log_har' or model['features'] != RECIPE['features']:
        raise ValueError('Wrong prior family/features')
    x = np.random.default_rng(42).uniform(-10, -4, size=(1200, 3))
    actual = np.exp(-1 + x[1000:] @ np.array([.2, .4, .1]))
    predicted = np.exp(np.column_stack([np.ones(200), x[1000:]]) @ np.array(model['coef'])) * model['smearing']
    error = float(np.max(np.abs(predicted/actual-1)))
    np.testing.assert_allclose(predicted, record['result']['variance_predictions'], rtol=1e-12)
    if error > 1e-10:
        raise ValueError('Prior numerical learning control failed')
    return {'status': 'passed_reverified_native', 'prior_fit_id': record['experiment_id'],
            'new_synthetic_fits': 0, 'maximum_relative_error': error, 'learner_files': list(LEARNER_FILES), 'market_skill': False}


def run(folder, repo, authorization, minutes):
    from cottonlens_ml.research.engine import Experiment
    from cottonlens_ml.runtime_guard import require_training
    require_training('har', 'cpu')
    folder = Path(folder)
    ready, h = verify(folder, repo)
    design = ready['identity']['design']
    auth = read_record(Path(authorization))
    if (auth.get('ready_sha256') != digest(folder/'ready.json') or auth.get('source_id') != ready['identity']['source_id']
            or auth.get('design_id') != ready['identity']['design_id'] or auth.get('fresh_restore_verified') is not True
            or auth.get('fit_budget') != design['fit_budget'] or not 0 < minutes <= 30):
        raise ValueError('Frozen recovery/authorization/budget mismatch')
    deadline = time.monotonic() + minutes*60
    with writer(folder):
        freeze_record(folder/'learning-control-reuse.json', validate_learning(folder, repo))
        experiment = Experiment(folder, repo=repo)
        used, keys = fit_consumption(experiment.ledger.root)
        if used != len(keys) or used > design['fit_budget']['market']:
            raise ValueError('Failed/excess fits; no silent retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('Checkpoint budget pause')
            if count[0] >= design['fit_budget']['market']:
                raise ValueError('Fit ceiling exceeded')
            count[0] += 1

        experiment.before_compute = before_compute
        try:
            for fold in design['folds']:
                for arm in ARMS:
                    records = []
                    for job in fold['jobs']:
                        train = projection(h, job['train_dates'], arm, training=True)
                        test = projection(h, job['origins'], arm, training=False)
                        records.append(experiment.fit({**RECIPE, 'feature_arm': arm}, train, None, test,
                            f'history-{fold["year"]}-{job["role"]}-{job["bucket"]}', iterations=1))
                    freeze_record(folder/'risk-outputs'/f'{arm}-{fold["year"]}.json', output(h, fold, arm, records))
        except FitBudgetReached:
            return {'status': 'planned_pause', 'market_fits': count[0]}
    return {'status': 'completed', 'market_fits': count[0], 'synthetic_fits': 0}


def compare(folder, repo):
    folder = Path(folder)
    ready, h = verify(folder, repo)
    learning = validate_learning(folder, repo)
    ledger = Ledger(folder/'ledger', ready['identity'])
    paired, native_ids = [], set()
    for fold in ready['identity']['design']['folds']:
        outputs = {}
        for arm in ARMS:
            records = []
            for job in fold['jobs']:
                train = projection(h, job['train_dates'], arm, training=True)
                test = projection(h, job['origins'], arm, training=False)
                spec = {'recipe': {**RECIPE, 'feature_arm': arm},
                        'role': f'history-{fold["year"]}-{job["role"]}-{job["bucket"]}', 'iterations': 1,
                        'train_dates': job['train_dates'], 'test_dates': job['origins'], 'validation_dates': [],
                        'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}
                key = ledger.key(spec)
                record = read_record(ledger.root/'completed'/f'{key}.json')
                if record['identity'] != ready['identity'] or record['specification'] != spec or record['experiment_id'] != key:
                    raise ValueError('Native identity/recipe mismatch')
                if not record['files']:
                    raise ValueError('Native payload missing')
                for name, expected in record['files'].items():
                    if not safe_member(name) or digest(ledger.root/name) != expected:
                        raise ValueError('Native payload hash mismatch')
                model = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_bytes())
                if model['kind'] != 'log_har' or model['features'] != RECIPE['features']:
                    raise ValueError('Wrong native family/features')
                values = np.maximum(np.exp(np.column_stack([np.ones(len(test)), test[RECIPE['features']]])
                                           @ np.array(model['coef'])) * model['smearing'], FLOOR)
                np.testing.assert_allclose(values, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
                records.append(record)
                native_ids.add(key)
            expected = output(h, fold, arm, records)
            if expected != read_record(folder/'risk-outputs'/f'{arm}-{fold["year"]}.json'):
                raise ValueError('Prediction envelope or past selection mismatch')
            outputs[arm] = expected
        for ct, own in zip(outputs['ct']['records'], outputs['named']['records'], strict=True):
            for key in ('date', 'target_date', 'contract', 'eligible', 'named_actual', 'baseline', 'forecast_available'):
                if ct[key] != own[key]:
                    raise ValueError('Paired origin/target/availability mismatch')
            row = {'year': fold['year'], 'date': ct['date'], 'target_date': ct['target_date'], 'contract': ct['contract'],
                   'eligible': ct['eligible'], 'forecast_available': ct['forecast_available'], 'actual': ct['named_actual'],
                   'baseline_variance': ct['baseline'], 'ct_raw': ct['raw'], 'named_raw': own['raw'],
                   'ct_selected': ct['selected'], 'named_selected': own['selected'],
                   'own_22_session_span': ct['own_22_session_span'], 'own_latest_pair_age': ct['own_latest_pair_age']}
            if ct['eligible']:
                for key, value in [('baseline', ct['baseline']), ('ct_raw', ct['raw']), ('named_raw', own['raw']),
                                   ('ct_selected', ct['selected']), ('named_selected', own['selected'])]:
                    row[key+'_loss'] = float(qlike(np.array([ct['named_actual']]), np.array([value]))[0])
            paired.append(row)
    frame = pd.DataFrame(paired)
    valid = frame.loc[frame.eligible]
    spec = importlib.util.spec_from_file_location('fixed_blocks', folder/'inputs/bootstrap.py')
    blocks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(blocks)
    comparisons = {}
    for name, a, b in [('feature_mechanism', 'ct_raw', 'named_raw'), ('named_raw_vs_ewma', 'baseline', 'named_raw'),
                       ('ct_raw_vs_ewma', 'baseline', 'ct_raw'), ('named_selected_vs_ewma', 'baseline', 'named_selected')]:
        a, b = a+'_loss', b+'_loss'
        offset = max(0., 1-min(valid[a].min(), valid[b].min()))
        sample = frame[['year']].copy()
        sample['a'] = np.where(frame.eligible, frame[a].fillna(0)+offset, 0)
        sample['b'] = np.where(frame.eligible, frame[b].fillna(0)+offset, 0)
        sample['weight'] = frame.eligible.astype(int)
        parts = [g[['a', 'b', 'weight']].to_numpy() for _, g in sample.groupby('year')]
        intervals = {}
        for block in (20, 60):
            v = blocks.weighted_bootstrap(parts, block=block)
            intervals[str(block)] = {k: v[k] for k in ('eligible', 'ordinal_slots', 'block', 'replicates',
                                                      'paired_difference', 'difference_ci_95')}
        trimmed = valid.drop(valid.nlargest(max(1, math.ceil(len(valid)*.01)), 'baseline_loss').index)
        comparisons[name] = {'delta': float((valid[a]-valid[b]).mean()), 'bootstrap': intervals,
                             'trim_top1pct_ewma_loss_delta': float((trimmed[a]-trimmed[b]).mean())}
    mechanism = comparisons['feature_mechanism']['bootstrap']
    positive = all(v['difference_ci_95'][0] > 0 for v in mechanism.values())
    negative = all(v['difference_ci_95'][1] <= 0 for v in mechanism.values())
    skill = comparisons['named_raw_vs_ewma']
    report = {'profile': PROFILE, 'new_fits_during_report': 0, 'native_market_fits_verified': len(native_ids),
              'retained_origins': len(frame), 'forecast_available': int(frame.forecast_available.sum()),
              'eligible_origins': len(valid), 'comparisons': comparisons,
              'classification': 'DECISIVE_POSITIVE' if positive else 'DECISIVE_NEGATIVE' if negative else 'INCONCLUSIVE',
              'classification_scope': 'Conditional one-year input-process mechanism only',
              'conditional_raw_followup_candidate': all(v['difference_ci_95'][0] > 0 for v in skill['bootstrap'].values())
                  and skill['trim_top1pct_ewma_loss_delta'] > 0,
              'learner_control': learning, 'eight_year_gate_allowed': False, 'independent_holdout': False,
              'release_allowed': False, 'source_timestamp_verified': False, 'first_vintage_verified': False}
    destination = folder/'reports'
    destination.mkdir(exist_ok=True)
    if (destination/'paired-losses.csv').exists():
        raise ValueError('Immutable report already exists')
    frame.to_csv(destination/'paired-losses.csv', index=False, lineterminator='\n')
    report['paired_losses_sha256'] = digest(destination/'paired-losses.csv')
    freeze_record(destination/'result.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--authorization', type=Path)
    p.add_argument('--max-minutes', type=float, default=30)
    a = p.parse_args()
    if a.stage == 'compare':
        print(compare(a.root, a.repo))
        return
    if a.stage == 'pilot':
        if a.authorization is None:
            p.error('pilot requires authorization')
        print(run(a.root, a.repo, a.authorization, a.max_minutes))
        return
    if a.stage == 'prepare':
        if a.evidence is None:
            p.error('prepare requires restored label evidence')
        design = prepare(a.repo, a.root, a.evidence)
    else:
        ready, _ = verify(a.root, a.repo)
        design = ready['identity']['design']
    print({'new_fits': 0, 'years': [f['year'] for f in design['folds']], 'budget': design['fit_budget']})


if __name__ == '__main__':
    main()
