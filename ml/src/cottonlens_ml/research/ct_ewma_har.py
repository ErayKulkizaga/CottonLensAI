"""One fixed full-year CT variance candidate; existing QLIKE HAR and Ledger.

This is conditional historical proxy research, not named-contract, price or
trading skill. No asymmetric feature, source search or EWMA lambda tuning.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.volatility import FLOOR, fit_predict, qlike
from cottonlens_ml.runtime_guard import require_training

PROFILE = 'ct-ewma-har-full-year-t1-v1'
FEATURES = ['har_1', 'har_5', 'har_22']
RECIPE = {'family': 'har', 'horizon': 1, 'device': 'cpu', 'window': 1,
          'features': FEATURES, 'training_loss': 'qlike', 'parameters': 4,
          'feature_formula': 'har_k-log(ct_ewma)', 'target_formula': 'variance_target_1/ct_ewma',
          'forecast_formula': 'ct_ewma*exp(intercept+relative_har@coef)', 'seed': 42, 'profile': PROFILE}
VERSIONS = ('numpy', 'pandas', 'scipy')


def transform(original, training):
    rows = original.copy()
    baseline = rows.ct_ewma.to_numpy(float)
    if not np.isfinite(baseline).all() or (baseline < FLOOR).any():
        raise ValueError('Finite positive origin CT EWMA required')
    for name in FEATURES:
        rows[name] = rows[name]-np.log(baseline)
    if training:
        rows['variance_target_1'] = rows.variance_target_1/baseline
    return rows


def fit_offset(train, test, workspace):
    require_training('har', 'cpu')
    tr, te = transform(train, True), transform(test, False)
    ratio_root = workspace/'ratio-learner'; ratio_root.mkdir()
    result = fit_predict(tr, tr, te, RECIPE, ratio_root)
    model = json.loads((ratio_root/'variance-model.json').read_bytes())
    ratio = np.exp(np.column_stack([np.ones(len(te)), te[FEATURES]])@model['coef'])
    if not np.isfinite(ratio).all() or (ratio < FLOOR).any():
        raise ValueError('No silent ratio prediction clipping')
    np.testing.assert_array_equal(ratio, result['variance_predictions'])
    predicted = test.ct_ewma.to_numpy(float)*ratio
    if not np.isfinite(predicted).all() or (predicted < FLOOR).any():
        raise ValueError('No silent absolute variance clipping')
    model.update(kind='ewma_offset_har', recipe=RECIPE, baseline_in_model='ct_ewma at origin',
                 effective_parameters=4, ratio_smearing=1.)
    (workspace/'variance-model.json').write_text(json.dumps(model)+'\n')
    return {**result, 'variance_predictions': predicted.tolist(),
            'origins': test.date.dt.strftime('%Y-%m-%d').tolist()}


def history_from(path):
    h = pd.read_parquet(path)
    if h.date.duplicated().any() or not h.date.is_monotonic_increasing or h.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('Unique chronological pre2024 CT history required')
    h = h.copy()
    h['ct_ewma'] = np.maximum(h.daily_variance_proxy.ewm(alpha=.06, adjust=False).mean(), FLOOR)
    return h


def prepare(repo, folder, readiness, history, predictions, bootstrap):
    repo, folder, readiness = map(Path, (repo, folder, readiness))
    if folder.exists():
        raise ValueError('Preserve old/incomplete namespace')
    validate(repo/'research')
    registry, trials = load_registry(repo/'research')
    related = check(registry, trials, family='har', horizon=1, query='qlike')
    with writer(folder):
        inputs = {'history.parquet': history, 'reference.csv': predictions, 'bootstrap.py': bootstrap}
        inputs.update({name: readiness/name for name in ('contract.json', 'result.json', 'verification.json', 'jobs.json', 'split.json')})
        for name, source in inputs.items():
            copy_immutable(Path(source), folder/'inputs'/name)
        parent = json.loads((folder/'inputs/result.json').read_text())
        proof = json.loads((folder/'inputs/verification.json').read_text())
        if (parent['new_fits'] != 0 or parent['training_allowed'] or proof['status'] != 'verified'
                or proof['past_only_jobs'] != 169 or proof['ordinal_origins'] != 2006):
            raise ValueError('Verified eight-year temporal readiness required')
        h = history_from(folder/'inputs/history.parquet')
        ref = pd.read_csv(folder/'inputs/reference.csv').query('horizon==1').reset_index(drop=True)
        jobs = json.loads((folder/'inputs/jobs.json').read_text())
        split = json.loads((folder/'inputs/split.json').read_text())
        outer = [d for f in split['folds'] for d in f['origins']]
        if outer != ref.date.tolist() or len(outer) != 2006 or len(jobs) != 169:
            raise ValueError('Exact original risk cohort/jobs required')
        selected = h.set_index('date').loc[pd.to_datetime(outer)]
        np.testing.assert_allclose(selected.ct_ewma, ref.ewma_variance, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(selected.variance_target_1, ref.actual_variance, rtol=1e-12, atol=1e-14)
        design = {'profile': PROFILE, 'recipe': RECIPE, 'jobs': jobs, 'split': split,
                  'fit_budget': {'market': 169, 'synthetic': 1, 'processes': 1, 'threads': 2, 'session_minutes': 30},
                  'primary': 'selected variance versus original EWMA; raw fixed candidate secondary',
                  'selection': 'past189/3x63 arithmetic variance mixture; weights0,.25,.5,.75,1; smallest tie',
                  'gate': 'selected baseline-minus-candidate QLIKE both20/60CI lower>0;>=6/8positive years;top1pctbaseline-loss trim>0',
                  'scope': 'seen historical CT squared-return proxy; not named or price skill',
                  'no_search': True, 'source_clock_verified': False, 'independent_holdout': False,
                  'release_allowed': False, 'deliberate_repeat_reason': 'Completed QLIKE/offset controls use named2023 with554minimum training. This single fixed candidate uses original CT8year population,minimum1191. No pure-loss or information-contribution claim.'}
        manifest = research_source_identity(repo)
        h.to_parquet(folder/'history.parquet', index=False)
        identity = {'profile': PROFILE, 'source_id': manifest['source_id'], 'design': design,
                    'design_id': content_id(design), 'research_data_id': frame_identity(h, list(h)),
                    'split_id': content_id(split), 'python': sys.version.split()[0],
                    'versions': {p: importlib.metadata.version(p) for p in VERSIONS},
                    'lock_sha256': digest(repo/'ml/uv.lock'),
                    'inputs_sha256': {name: digest(folder/'inputs'/name) for name in inputs}}
        for name in manifest['files']:
            copy_immutable(repo/name, folder/'source-snapshot'/name)
        freeze_record(folder/'source-manifest.json', manifest)
        freeze_record(folder/'history-check.json', {'status': related['status'],
                      'related_experiment_ids': [r['id'] for r in related['experiments']],
                      'related_fit_recipe_ids': [r['id'] for r in related['fit_recipes']],
                      'no_match_is_not_novelty': True})
        freeze_record(folder/'decision-contract.json', design)
        freeze_record(folder/'ready.json', {'identity': identity, 'history_sha256': digest(folder/'history.parquet')})
    return {'status': 'prepared', 'profile': PROFILE, 'source_id': identity['source_id'],
            'design_id': identity['design_id'], 'data_id': identity['research_data_id'], 'fit_budget': design['fit_budget']}


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder/'ready.json'); identity = ready['identity']
    if (identity['profile'] != PROFILE or identity['source_id'] != research_source_identity(repo)['source_id']
            or identity['python'] != sys.version.split()[0] or identity['lock_sha256'] != digest(repo/'ml/uv.lock')
            or identity['versions'] != {p: importlib.metadata.version(p) for p in VERSIONS}):
        raise ValueError('Frozen source/environment mismatch')
    for name, expected in identity['inputs_sha256'].items():
        if digest(folder/'inputs'/name) != expected:
            raise ValueError('Frozen input/reference changed')
    h = history_from(folder/'inputs/history.parquet')
    if (frame_identity(h, list(h)) != identity['research_data_id'] or digest(folder/'history.parquet') != ready['history_sha256']
            or content_id(identity['design']) != identity['design_id'] or read_record(folder/'decision-contract.json') != identity['design']):
        raise ValueError('Frozen history/design mismatch')
    pd.testing.assert_frame_equal(h, pd.read_parquet(folder/'history.parquet'), check_exact=True)
    dates = h.date.dt.strftime('%Y-%m-%d')
    for job in identity['design']['jobs']:
        cut = pd.Timestamp(job['cutoff'])
        eligible = (h.date.lt(cut) & h.target_date_5.lt(cut) & h.date.ge(h.date.iloc[119])
                    & np.isfinite(h.variance_target_1))
        if job['train_dates'] != dates.loc[eligible].tolist() or len(job['train_dates']) < 500:
            raise ValueError('Mature training changed')
    return ready, h


def projection(h, dates):
    rows = h.set_index('date', drop=False).loc[pd.to_datetime(dates)].reset_index(drop=True)
    return rows[['date', 'ct_ewma', *FEATURES, 'variance_target_1']].copy()


def specification(job, train):
    return {'recipe': RECIPE, 'role': f'ct-offset-{job["year"]}-{job["role"]}-{job["bucket"]}',
            'iterations': 1, 'train_dates': job['train_dates'], 'test_dates': job['origins'],
            'validation_dates': [], 'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}


def ledger_identity(identity):
    # Bind the full immutable plan by design_id without copying hundreds of
    # thousands of job dates into each of the169 completed native records.
    return {k: v for k, v in identity.items() if k != 'design'}


def run(folder, repo, authorization, minutes=30):
    require_training('har', 'cpu')
    folder = Path(folder); ready, h = verify(folder, repo); identity = ready['identity']; design = identity['design']
    authorized = read_record(Path(authorization))
    if (authorized.get('ready_sha256') != digest(folder/'ready.json') or authorized.get('source_id') != identity['source_id']
            or authorized.get('design_id') != identity['design_id'] or authorized.get('fit_budget') != design['fit_budget']
            or authorized.get('fresh_restore_verified') is not True or not 0 < minutes <= 30):
        raise ValueError('Exact preregistration recovery/source/budget authorization required')
    deadline = time.monotonic()+60*minutes
    with writer(folder):
        learning = Ledger(folder/'learning/ledger', {**ledger_identity(identity), 'role': 'synthetic_not_market'})
        used, keys = fit_consumption(learning.root)
        if used != len(keys) or used > 1:
            raise ValueError('Failed/excess synthetic fit; no retry')
        rng = np.random.default_rng(42)
        baseline = np.exp(rng.uniform(-12, -4, 1200)); x = rng.uniform(-3, 3, (1200, 3))
        fixture = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1200), 'ct_ewma': baseline})
        for column, name in enumerate(FEATURES):
            fixture[name] = np.log(baseline)+x[:, column]
        fixture['variance_target_1'] = baseline*np.exp(.1+x@np.array([.2, .1, -.1]))
        smoke = learning.run({'recipe': RECIPE, 'role': 'synthetic_ct_offset_har', 'kind': 'known_signal'},
                             lambda path: fit_offset(fixture.iloc[:1000], fixture.iloc[1000:], path))
        np.testing.assert_allclose(smoke['result']['variance_predictions'], fixture.variance_target_1.iloc[1000:], rtol=1e-5)
        freeze_record(folder/'learning-control.json', {'status': 'passed', 'fit_id': smoke['experiment_id'], 'market_skill': False})
        ledger = Ledger(folder/'ledger', ledger_identity(identity)); used, keys = fit_consumption(ledger.root)
        if used != len(keys) or used > design['fit_budget']['market']:
            raise ValueError('Failed/excess market fit; no retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('Checkpoint session boundary')
            if count[0] >= design['fit_budget']['market']:
                raise ValueError('Frozen market budget exhausted')
            count[0] += 1

        try:
            for job in design['jobs']:
                train, test = projection(h, job['train_dates']), projection(h, job['origins'])
                ledger.run(specification(job, train), lambda path, tr=train, te=test: fit_offset(tr, te, path),
                           before_compute=before_compute)
        except FitBudgetReached:
            return {'status': 'planned_pause', 'market_fits': count[0], 'synthetic_fits': 1}
    return {'status': 'completed', 'market_fits': design['fit_budget']['market'], 'synthetic_fits': 1}


def replay_native(ledger, job, train, test):
    spec = specification(job, train); key = ledger.key(spec)
    record = read_record(ledger.root/'completed'/f'{key}.json')
    if record['identity'] != ledger.identity or record['specification'] != spec or record['experiment_id'] != key:
        raise ValueError('Native fit identity mismatch')
    for name, expected in record['files'].items():
        if not safe_member(name) or digest(ledger.root/name) != expected:
            raise ValueError('Native payload changed')
    model = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_bytes())
    if model['kind'] != 'ewma_offset_har' or model['recipe'] != RECIPE or model['ratio_smearing'] != 1.:
        raise ValueError('Native variance model contract mismatch')
    tr, te = transform(train, True), transform(test, False)
    eta = np.column_stack([np.ones(len(tr)), tr[FEATURES]])@model['coef']
    ratio = tr.variance_target_1.to_numpy()*np.exp(-eta)
    optimization = model['optimization']
    z = (tr[FEATURES].to_numpy()-optimization['training_feature_mean'])/optimization['training_feature_scale']
    gradient = np.column_stack([np.ones(len(tr)), z]).T@(1-ratio)/len(tr)
    if (not optimization['success'] or np.max(np.abs(gradient)) > 1.001e-6
            or not math.isclose(float(np.mean(eta+ratio)), optimization['final_training_qlike'], abs_tol=1e-10)):
        raise ValueError('Native optimizer certificate failed')
    predicted = test.ct_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(te)), te[FEATURES]])@model['coef'])
    np.testing.assert_allclose(predicted, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
    if record['result']['origins'] != job['origins']:
        raise ValueError('Native prediction origin mismatch')
    return predicted, key, float(np.max(np.abs(gradient)))


def compare(folder, repo):
    folder = Path(folder); ready, h = verify(folder, repo); identity = ready['identity']; design = identity['design']
    ledger = Ledger(folder/'ledger', ledger_identity(identity))
    loader = importlib.util.spec_from_file_location('ct_fixed_bootstrap', folder/'inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(loader); loader.loader.exec_module(bootstrap)
    rows, keys, certificates = [], set(), []
    for fold in design['split']['folds']:
        inner, outer = {}, {}
        for job in (j for j in design['jobs'] if j['year'] == fold['year']):
            train, test = projection(h, job['train_dates']), projection(h, job['origins'])
            predictions, key, gradient = replay_native(ledger, job, train, test)
            keys.add(key); certificates.append(gradient)
            destination = inner if job['role'].startswith('inner_') else outer
            for day, predicted in zip(job['origins'], predictions, strict=True):
                if day in destination:
                    raise ValueError('Duplicate folded prediction')
                destination[day] = float(predicted)
        if len(inner) != 189 or list(outer) != fold['origins']:
            raise ValueError('Complete historical inner/outer forecast coverage required')
        validation = projection(h, list(inner))
        candidate = np.array(list(inner.values())); baseline = validation.ct_ewma.to_numpy(); actual = validation.variance_target_1.to_numpy()
        scored = [(float(np.mean(qlike(actual, w*candidate+(1-w)*baseline))), w) for w in (0., .25, .5, .75, 1.)]
        selected_weight = min(scored)[1]
        for day, raw in outer.items():
            original = h.loc[h.date.eq(pd.Timestamp(day))].iloc[0]
            actual, baseline = float(original.variance_target_1), float(original.ct_ewma)
            selected = selected_weight*raw+(1-selected_weight)*baseline
            rows.append({'date': day, 'target_date': pd.Timestamp(original.target_date_1).strftime('%Y-%m-%d'),
                         'year': fold['year'], 'horizon': 1, 'actual_variance': actual, 'ewma_variance': baseline,
                         'raw_variance': raw, 'selected_variance': selected, 'selected_weight': selected_weight,
                         'baseline_loss': float(qlike([actual], [baseline])[0]),
                         'raw_loss': float(qlike([actual], [raw])[0]), 'selected_loss': float(qlike([actual], [selected])[0])})
    if len(keys) != 169 or len(rows) != 2006:
        raise ValueError('Expected complete native fit/cohort counts')
    frame = pd.DataFrame(rows); ref = pd.read_csv(folder/'inputs/reference.csv').query('horizon==1').reset_index(drop=True)
    if frame.date.tolist() != ref.date.tolist() or frame.target_date.tolist() != ref.target_date.tolist():
        raise ValueError('Original cohort endpoint identity changed')
    np.testing.assert_allclose(frame.ewma_variance, ref.ewma_variance, rtol=1e-12, atol=1e-14)
    comparisons = {}
    for arm in ('raw', 'selected'):
        a, b = 'baseline_loss', arm+'_loss'; offset = max(0., 1-min(frame[a].min(), frame[b].min()))
        parts = [np.column_stack([g[a]+offset, g[b]+offset, np.ones(len(g))]) for _, g in frame.groupby('year')]
        intervals = {str(block): bootstrap.weighted_bootstrap(parts, block=block)['difference_ci_95'] for block in (20, 60)}
        yearly = {str(year): float((g[a]-g[b]).mean()) for year, g in frame.groupby('year')}
        trim = frame.drop(frame.nlargest(math.ceil(len(frame)*.01), a).index)
        gain = float((frame[a]-frame[b]).mean()); trimmed = float((trim[a]-trim[b]).mean())
        passed = all(ci[0] > 0 for ci in intervals.values()) and sum(v > 0 for v in yearly.values()) >= 6 and trimmed > 0
        comparisons[arm] = {'baseline_minus_candidate_gain': gain, 'gain_ci': intervals, 'yearly_gain': yearly,
                            'positive_years': sum(v > 0 for v in yearly.values()), 'trim_top1pct_gain': trimmed,
                            'fixed_risk_gate_passed': bool(passed)}
    report = {'profile': PROFILE, 'new_fits_during_report': 0, 'native_market_fits': len(keys),
              'optimizer_gradient_inf_max': max(certificates), 'retained_origins': len(frame),
              'comparisons': comparisons, 'selected_risk_gate_passed': comparisons['selected']['fixed_risk_gate_passed'],
              'price_gates_passed': False, 'independent_holdout': False, 'source_clock_verified': False,
              'release_allowed': False, 'scope': design['scope']}
    dest = folder/'reports'; dest.mkdir(exist_ok=True)
    if (dest/'paired-losses.csv').exists():
        raise ValueError('Preserve existing comparison')
    frame.to_csv(dest/'paired-losses.csv', index=False, lineterminator='\n'); report['paired_losses_sha256'] = digest(dest/'paired-losses.csv')
    freeze_record(dest/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    parser.add_argument('--repo', type=Path, required=True); parser.add_argument('--root', type=Path, required=True)
    for name in ('readiness', 'history', 'predictions', 'bootstrap', 'authorization'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.repo, args.root, args.readiness, args.history, args.predictions, args.bootstrap)
    elif args.stage == 'verify':
        ready, _ = verify(args.root, args.repo)
        result = {'status': 'verified', 'new_fits': 0, 'profile': PROFILE, 'fit_budget': ready['identity']['design']['fit_budget']}
    elif args.stage == 'pilot':
        result = run(args.root, args.repo, args.authorization)
    else:
        result = compare(args.root, args.repo)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
