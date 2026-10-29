"""One capacity-matched CT signed/unsigned forecast contrast on existing Ledger."""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import ct_ewma_har as parent
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.volatility import FLOOR, fit_predict, qlike
from cottonlens_ml.runtime_guard import require_training

PROFILE = 'ct-signed-unsigned-shock-forecast-t1-v1'
ARMS = ('unsigned', 'signed')
NAMES = [*parent.FEATURES, 'origin_shock']
VERSIONS = ('numpy', 'pandas', 'scipy', 'pyarrow')
HELPERS = ('research/ct_shock_control.py', 'research/verify_ct_shock_control.py',
           'research/tests/test_ct_shock_control.py', 'research/verify_risk_weak_teacher.py',
           'research/verify_ct_signed_risk_score.py')


def source(repo):
    body = {'ml': research_source_identity(repo), 'helpers': {n: digest(repo/n) for n in HELPERS}}
    return {**body, 'source_id': content_id(body)}


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('Only two frozen shock representations are allowed')
    return {**parent.RECIPE, 'profile': PROFILE, 'arm': arm, 'parameters': 5,
            'features': NAMES, 'variance_features': NAMES,
            'extra_feature': ('r_t^2/origin_EWMA' if arm == 'unsigned' else 'sign(r_t)*r_t^2/origin_EWMA'),
            'feature_formula': '3relativeHAR plus one declared origin shock',
            'forecast_formula': 'origin_EWMA*exp(intercept+4features@coef)'}


def attach(history):
    h = history.copy()
    returns = np.log(h.cotton_close/h.cotton_close.shift())
    np.testing.assert_allclose(returns, h.cotton_ret_1, rtol=1e-10, atol=1e-14, equal_nan=True)
    baseline = np.maximum(returns.pow(2).ewm(alpha=.06, adjust=False).mean(), FLOOR)
    np.testing.assert_allclose(baseline, h.ct_ewma, rtol=1e-12, atol=1e-14, equal_nan=True)
    h['unsigned_shock'] = returns.pow(2)/baseline
    h['signed_shock'] = np.sign(returns)*h.unsigned_shock
    return h


def project(h, dates, arm):
    recipe(arm)
    rows = h.set_index(h.date.dt.strftime('%Y-%m-%d')).loc[dates]
    result = parent.projection(h, dates)
    result['origin_shock'] = rows[f'{arm}_shock'].to_numpy(float)
    if not np.isfinite(result[['ct_ewma', *NAMES]].to_numpy()).all():
        raise ValueError('Do not drop or impute feature-unready origins')
    return result


def relative(rows, training):
    transformed = parent.transform(rows, training)
    if not np.isfinite(transformed[NAMES].to_numpy()).all():
        raise ValueError('Finite declared four-feature design required')
    return transformed


def fit(train, test, workspace, arm):
    require_training('har', 'cpu')
    tr, te = relative(train, True), relative(test, False)
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(tr)), tr[NAMES]])) != 5:
        raise ValueError('Five-parameter design must have full rank')
    ratio_root = workspace/'ratio-learner'; ratio_root.mkdir()
    result = fit_predict(tr, tr, te, recipe(arm), ratio_root)
    model = json.loads((ratio_root/'variance-model.json').read_bytes())
    if model['features'] != NAMES or len(model['coef']) != 5:
        raise ValueError('Declared fourth feature was not consumed by the shared learner')
    ratio = np.exp(np.column_stack([np.ones(len(te)), te[NAMES]])@model['coef'])
    np.testing.assert_array_equal(ratio, result['variance_predictions'])
    predicted = test.ct_ewma.to_numpy()*ratio
    if (not np.isfinite(predicted).all() or (predicted < FLOOR).any() or (ratio < FLOOR).any()):
        raise ValueError('No silent variance output clipping')
    model.update(kind=PROFILE, recipe=recipe(arm), effective_parameters=5, ratio_smearing=1.)
    (workspace/'variance-model.json').write_text(json.dumps(model)+'\n')
    return {**result, 'variance_predictions': predicted.tolist()}


def specification(job, train, arm):
    return {'recipe': recipe(arm), 'role': f'{arm}-{job["year"]}-{job["role"]}-{job["bucket"]}',
            'iterations': 1, 'train_dates': job['train_dates'], 'test_dates': job['origins'],
            'validation_dates': [], 'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}


def prepare(repo, root, reference):
    if root.exists():
        raise ValueError('Preserve existing/incomplete namespace')
    validate(repo/'research')
    previous, h = parent.verify(reference, reference/'source-snapshot')
    proof_path = repo/'research/evidence/ct-ewma-har-result-20261014.json'
    proof = json.loads(proof_path.read_bytes())
    if (digest(reference/'reports/result.json') != proof['result_sha256']
            or digest(reference/'reports/paired-losses.csv') != proof['paired_losses_sha256']):
        raise ValueError('Registered CT reference changed')
    h = attach(h); jobs = previous['identity']['design']['jobs']
    split = previous['identity']['design']['split']
    if len(jobs) != 169 or sum(len(f['origins']) for f in split['folds']) != 2006:
        raise ValueError('Exact existing jobs and origin cohort required')
    for job in jobs:
        for arm in ARMS:
            project(h, job['train_dates'], arm); project(h, job['origins'], arm)
    related = check(*load_registry(repo/'research'), family='har', horizon=1, query='signed shock')
    design = {'profile': PROFILE, 'recipes': {a: recipe(a) for a in ARMS},
              'reference_design_id': previous['identity']['design_id'],
              'fit_budget': {'market': 338, 'synthetic': 1, 'processes': 1, 'threads': 2, 'session_minutes': 30},
              'primary_candidate': 'signed selected variance versus unchanged EWMA0.94',
              'mechanism': 'signed raw versus unsigned raw at equal5parameter capacity; no OOS arm selection',
              'secondary': 'raw and selected unsigned; raw signed; unchanged4parameter native control',
              'gate': 'positiveQLIKEgain;both20/60CI lower>0;>=6/8positiveyears;top1pctEWMA-loss trim>0',
              'selection': 'same past189/3x63 QLIKE arithmetic mixture weights0,.25,.5,.75,1; smallest tie',
              'optimizer': 'same shared QLIKE/BFGS max500 gtol1e-6, train-only standardization, OLS+smearing init; no retry/fallback',
              'unchanged': '2006pre2024origins,targets,EWMA,expandingtrain,5sessionmaturity,21refit;reuse169nativecontrols',
              'bootstrap': {'replicates': 10000, 'seed': 42, 'blocks': [20, 60], 'preserve_year_boundaries': True},
              'prior': 'Signed ex-post covariance and conditional derivative already inspected; inconclusive is not no-signal proof. No learned equal-capacity shock forecast contrast found.',
              'independent_holdout': False, 'historical_availability_verified': False,
              'first_vintage_verified': False, 'multiplicity_adjusted': False,
              'release_allowed': False, 'new_dataset_downloads': False, 'no_search': True,
              'scope': 'seen CT continuous-price squared-return proxy, not clean named-futures or trading skill'}
    with writer(root):
        shutil.copytree(reference, root/'reference')
        shutil.copyfile(proof_path, root/'parent-proof.json')
        h.to_parquet(root/'history.parquet', index=False)
        manifest = source(repo)
        for name in {**manifest['ml']['files'], **manifest['helpers']}:
            target = root/'source-snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/name, target)
        freeze_record(root/'source-manifest.json', manifest)
        freeze_record(root/'decision-contract.json', design)
        freeze_record(root/'history-check.json', {'status': related['status'],
                      'related': [r['id'] for r in related['experiments']], 'no_match_is_not_novelty': True})
        identity = {'profile': PROFILE, 'source_id': manifest['source_id'], 'design_id': content_id(design),
                    'research_data_id': frame_identity(h, list(h)),
                    'split': {'parent_split_id': previous['identity']['split_id']},
                    'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in VERSIONS},
                    'lock_sha256': digest(repo/'ml/uv.lock')}
        files = {p.relative_to(root).as_posix(): digest(p) for p in (root/'reference').rglob('*') if p.is_file()}
        files.update({n: digest(root/n) for n in ('parent-proof.json', 'history.parquet')})
        freeze_record(root/'ready.json', {'identity': identity, 'frozen_hashes': files})
    return {'status': 'prepared', **identity, 'fit_budget': design['fit_budget']}


def verify(root, repo):
    ready = read_record(root/'ready.json'); identity = ready['identity']
    design = read_record(root/'decision-contract.json')
    if (source(repo) != read_record(root/'source-manifest.json') or identity['source_id'] != source(repo)['source_id']
            or identity['design_id'] != content_id(design) or identity['profile'] != PROFILE
            or identity['python'] != sys.version.split()[0]
            or identity['versions'] != {p: importlib.metadata.version(p) for p in VERSIONS}
            or identity['lock_sha256'] != digest(repo/'ml/uv.lock')):
        raise ValueError('Frozen source/environment/design changed')
    for name, checksum in ready['frozen_hashes'].items():
        if not safe_member(name) or digest(root/name) != checksum:
            raise ValueError('Frozen input/native reference changed')
    old, history = parent.verify(root/'reference', root/'reference/source-snapshot')
    h = attach(history)
    pd.testing.assert_frame_equal(h, pd.read_parquet(root/'history.parquet'), check_exact=True)
    if frame_identity(h, list(h)) != identity['research_data_id'] or old['identity']['design_id'] != design['reference_design_id']:
        raise ValueError('Frozen causal features/reference design changed')
    return ready, design, old, h


def run(root, repo, authorization, minutes=30):
    require_training('har', 'cpu')
    ready, design, old, h = verify(root, repo); identity = ready['identity']
    auth = read_record(authorization)
    if (auth.get('ready_sha256') != digest(root/'ready.json') or auth.get('source_id') != identity['source_id']
            or auth.get('design_id') != identity['design_id'] or auth.get('fresh_restore_verified') is not True
            or auth.get('fit_budget') != design['fit_budget'] or not 0 < minutes <= 30):
        raise ValueError('Exact source/preregistration/restore/budget authorization required')
    deadline = time.monotonic()+minutes*60
    with writer(root):
        learning = Ledger(root/'learning/ledger', {**identity, 'role': 'synthetic_not_market'})
        used, keys = fit_consumption(learning.root)
        if used != len(keys) or used > 1:
            raise ValueError('Failed/excess synthetic attempt; no retry')
        rng = np.random.default_rng(42); x = rng.uniform(-2, 2, (1200, 4))
        b = np.exp(rng.uniform(-12, -4, 1200))
        fixture = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1200), 'ct_ewma': b,
                                'origin_shock': x[:, 3]})
        for i, name in enumerate(parent.FEATURES):
            fixture[name] = np.log(b)+x[:, i]
        fixture['variance_target_1'] = b*np.exp(.1+x@np.array([.2, .1, -.1, -.15]))
        control = learning.run({'recipe': recipe('signed'), 'role': 'synthetic_five_parameter_shock', 'kind': 'known_signal'},
                               lambda p: fit(fixture.iloc[:1000], fixture.iloc[1000:], p, 'signed'))
        np.testing.assert_allclose(control['result']['variance_predictions'], fixture.variance_target_1.iloc[1000:], rtol=1e-5)
        control_body = {'status': 'native_known_mean_reverified', 'fit_id': control['experiment_id'], 'market_skill': False}
        if (root/'learning-control.json').exists():
            if read_record(root/'learning-control.json') != control_body:
                raise ValueError('Changed native learning control')
        else:
            freeze_record(root/'learning-control.json', control_body)
        ledger = Ledger(root/'ledger', identity); used, keys = fit_consumption(ledger.root)
        if used != len(keys) or used > 338:
            raise ValueError('Failed/excess market attempt; no retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('Checkpoint session boundary')
            if count[0] >= 338:
                raise ValueError('Frozen market fit budget exhausted')
            count[0] += 1

        try:
            for job in old['identity']['design']['jobs']:
                for arm in ARMS:
                    train, test = project(h, job['train_dates'], arm), project(h, job['origins'], arm)
                    ledger.run(specification(job, train, arm), lambda p, tr=train, te=test, a=arm: fit(tr, te, p, a),
                               before_compute=before_compute)
        except FitBudgetReached:
            return {'status': 'planned_pause', 'market_fits': count[0], 'synthetic_fits': 1}
    return {'status': 'completed', 'market_fits': count[0], 'synthetic_fits': 1}


def native(ledger, job, train, test, arm):
    key = ledger.key(specification(job, train, arm))
    record = read_record(ledger.root/'completed'/f'{key}.json')
    if record['identity'] != ledger.identity or record['specification'] != specification(job, train, arm) or record['experiment_id'] != key:
        raise ValueError('Native fit identity differs')
    for name, checksum in record['files'].items():
        if not safe_member(name) or digest(ledger.root/name) != checksum:
            raise ValueError('Native model payload changed')
    m = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_bytes())
    if m['recipe'] != recipe(arm) or m['features'] != NAMES or m['effective_parameters'] != 5:
        raise ValueError('Native model did not consume declared features')
    tr, te = relative(train, True), relative(test, False); x = tr[NAMES].to_numpy()
    eta = np.column_stack([np.ones(len(tr)), x])@m['coef']; ratio = tr.variance_target_1.to_numpy()*np.exp(-eta)
    o = m['optimization']; z = (x-o['training_feature_mean'])/o['training_feature_scale']
    gradient = np.column_stack([np.ones(len(tr)), z]).T@(1-ratio)/len(tr)
    if (not o['success'] or np.max(np.abs(gradient)) > 1.001e-6
            or not math.isclose(float(np.mean(eta+ratio)), o['final_training_qlike'], abs_tol=1e-10)):
        raise ValueError('Uncertified optimizer')
    prediction = test.ct_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(te)), te[NAMES]])@m['coef'])
    np.testing.assert_allclose(prediction, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
    if record['result']['origins'] != job['origins']:
        raise ValueError('Native forecast origins changed')
    return prediction


def folded_output(h, design, predict):
    outputs, weights = {}, {}
    for fold in design['split']['folds']:
        inner, outer = {}, {}
        for job in (j for j in design['jobs'] if j['year'] == fold['year']):
            prediction = predict(job)
            dest = inner if job['role'].startswith('inner_') else outer
            for day, value in zip(job['origins'], prediction, strict=True):
                if day in dest:
                    raise ValueError('Duplicate folded prediction')
                dest[day] = float(value)
        if len(inner) != 189 or list(outer) != fold['origins']:
            raise ValueError('Exact validation/origin coverage required')
        rows = h.set_index(h.date.dt.strftime('%Y-%m-%d')).loc[list(inner)]
        scores = [(float(qlike(rows.variance_target_1, w*np.array(list(inner.values()))+(1-w)*rows.ct_ewma).mean()), w)
                  for w in (0., .25, .5, .75, 1.)]
        weight = min(scores)[1]; weights[str(fold['year'])] = weight
        for day, value in outer.items():
            baseline = float(h.loc[h.date.eq(pd.Timestamp(day)), 'ct_ewma'].iloc[0])
            outputs[day] = (value, weight*value+(1-weight)*baseline, weight)
    return outputs, weights


def statistics(frame, a, b, bootstrap):
    offset = max(0., 1-min(frame[a].min(), frame[b].min()))
    parts = [np.column_stack([g[a]+offset, g[b]+offset, np.ones(len(g))]) for _, g in frame.groupby('year')]
    ci = {str(k): bootstrap.weighted_bootstrap(parts, block=k)['difference_ci_95'] for k in (20, 60)}
    yearly = {str(y): float((g[a]-g[b]).mean()) for y, g in frame.groupby('year')}
    trim = frame.drop(frame.nlargest(math.ceil(len(frame)*.01), 'baseline_loss').index)
    gain, trimmed = float((frame[a]-frame[b]).mean()), float((trim[a]-trim[b]).mean())
    return {'gain': gain, 'gain_ci': ci, 'yearly_gain': yearly, 'positive_years': sum(v > 0 for v in yearly.values()),
            'trim_top1pct_gain': trimmed, 'fixed_risk_gate_passed': bool(gain > 0 and all(v[0] > 0 for v in ci.values())
                                                                     and sum(v > 0 for v in yearly.values()) >= 6 and trimmed > 0)}


def compare(root, repo):
    ready, _contract, old, h = verify(root, repo)
    used, keys = fit_consumption(root/'ledger')
    if used != 338 or len(keys) != 338 or not (root/'learning-control.json').exists():
        raise ValueError('Require exact complete native market/learning fits')
    ledger = Ledger(root/'ledger', ready['identity'])
    old_ledger = Ledger(root/'reference/ledger', parent.ledger_identity(old['identity']))
    design = old['identity']['design']; outputs = {}; weights = {}
    for arm in ARMS:
        outputs[arm], weights[arm] = folded_output(h, design, lambda j, a=arm: native(
            ledger, j, project(h, j['train_dates'], a), project(h, j['origins'], a), a))
    original, _ = folded_output(h, design, lambda j: parent.replay_native(old_ledger, j,
        parent.projection(h, j['train_dates']), parent.projection(h, j['origins']))[0])
    frame = pd.read_csv(root/'reference/reports/paired-losses.csv', float_precision='round_trip')
    origins = [d for f in design['split']['folds'] for d in f['origins']]
    if frame.date.tolist() != origins or len(frame) != 2006:
        raise ValueError('No cohort intersection/drop')
    for column, k in (('raw_variance', 0), ('selected_variance', 1), ('selected_weight', 2)):
        np.testing.assert_allclose(frame[column], [original[d][k] for d in origins], rtol=1e-12, atol=1e-14)
    for arm in ARMS:
        for name, k in (('raw', 0), ('selected', 1), ('weight', 2)):
            frame[f'{arm}_{name}'] = [outputs[arm][d][k] for d in origins]
        for kind in ('raw', 'selected'):
            frame[f'{arm}_{kind}_loss'] = qlike(frame.actual_variance, frame[f'{arm}_{kind}'])
    loader = importlib.util.spec_from_file_location('frozen_shock_bootstrap', root/'reference/inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(loader); loader.loader.exec_module(bootstrap)
    contrasts = {f'{a}_{k}_vs_EWMA': ('baseline_loss', f'{a}_{k}_loss') for a in ARMS for k in ('raw', 'selected')}
    contrasts.update({f'signed_{k}_vs_unsigned': (f'unsigned_{k}_loss', f'signed_{k}_loss') for k in ('raw', 'selected')})
    comparisons = {name: statistics(frame, a, b, bootstrap) for name, (a, b) in contrasts.items()}
    result = {'profile': PROFILE, 'fitted_source_id': ready['identity']['source_id'], 'new_fits_during_report': 0,
              'native_market_fits': 338, 'synthetic_fits': 1, 'reused_control_fits': 169, 'retained_origins': len(frame),
              'comparisons': comparisons, 'past_selected_weights': weights,
              'primary_gate_passed': comparisons['signed_selected_vs_EWMA']['fixed_risk_gate_passed'],
              'independent_holdout': False, 'historical_availability_verified': False,
              'first_vintage_verified': False, 'release_allowed': False, 'market_skill': False,
              'scope': 'seen historical conditional CT proxy forecast comparison; no independent forward success'}
    destination = root/'reports'; destination.mkdir(exist_ok=True)
    if (destination/'result.json').exists() or (destination/'paired-losses.csv').exists():
        raise ValueError('Preserve completed/partial report')
    frame.to_csv(destination/'paired-losses.csv', index=False, lineterminator='\n')
    result['paired_losses_sha256'] = digest(destination/'paired-losses.csv')
    freeze_record(destination/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    for name in ('root', 'repo'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('reference', 'authorization'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--minutes', type=float, default=30)
    args = parser.parse_args()
    if args.stage == 'prepare':
        output = prepare(args.repo, args.root, args.reference)
    elif args.stage == 'verify':
        verify(args.root, args.repo); output = {'status': 'verified', 'new_fits': 0}
    elif args.stage == 'pilot':
        output = run(args.root, args.repo, args.authorization, args.minutes)
    else:
        output = compare(args.root, args.repo)
    print(json.dumps(output))
