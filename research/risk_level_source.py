"""Capacity-matched own-level versus DXY-state risk control; fixed historical cohort."""

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
from cottonlens_ml.research.volatility import (
    FLOOR,
    InvalidVarianceFit,
    qlike,
    qlike_value_gradient,
)
from cottonlens_ml.runtime_guard import require_training

from research import ams_named_ewma_har as parent

PROFILE = 'risk-level-source-t1-v1'
ARMS = ('own_level', 'dxy_state')
HELPERS = ('research/risk_level_source.py', 'research/tests/test_risk_level_source.py',
           'research/verify_risk_level_source.py', 'research/verify_risk_weak_teacher.py',
           'research/ams_named_ewma_har.py', 'research/ams_named_readiness.py',
           'research/ams_upland_futures.py')
VERSIONS = ('numpy', 'pandas', 'scipy', 'pyarrow')


def source(repo):
    body = {'ml': research_source_identity(repo), 'helpers': {p: digest(repo/p) for p in HELPERS}}
    return {**body, 'source_id': content_id(body)}


def recipe(arm):
    if arm not in ARMS:
        raise ValueError('Unknown fixed arm')
    return {**parent.learner.RECIPE, 'profile': PROFILE, 'parameters': 5,
            'features': ['relative_har_1', 'relative_har_5', 'relative_har_22',
                         'log_named_ewma' if arm == 'own_level' else 'log_dxy_rv5'],
            'arm': arm, 'extra_feature': 'free log origin named EWMA' if arm == 'own_level'
            else 'log mean last5 raw DXY observation squared returns; assumed-clock asof',
            'feature_formula': '3relativeHAR plus one declared log-state',
            'forecast_formula': 'named_ewma*exp(intercept+4features@coef)'}


def dxy_state(market, dates, decisions=None):
    """Compute on source observations FIRST; then inclusive assumed-clock as-of."""
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Unique chronological Cotton origins required')
    quotes = market.loc[market.series.eq('dxy'), ['date', 'close']].sort_values('date').copy()
    quotes['date'] = pd.to_datetime(quotes.date)
    if (not len(quotes) or quotes.date.isna().any() or quotes.date.duplicated().any()
            or not quotes.date.eq(quotes.date.dt.normalize()).all()):
        raise ValueError('Unique date-only DXY source observations required')
    # Preserve invalid observations in the rolling window; never skip them to find a usable past bar.
    prices = quotes.close.where(np.isfinite(quotes.close) & quotes.close.gt(0))
    quotes['dxy_rv5'] = np.log(prices/prices.shift()).pow(2).rolling(5, min_periods=5).mean()
    quotes['window_start'] = quotes.date.shift(5)
    clocks = pd.DatetimeIndex(pd.to_datetime(quotes.date, utc=True)+pd.Timedelta(days=1))
    decision = (pd.DatetimeIndex(pd.to_datetime(dates, utc=True))+pd.Timedelta(days=1, minutes=15)
                if decisions is None else pd.DatetimeIndex(pd.to_datetime(decisions, utc=True)))
    if len(decision) != len(dates) or decision.hasnans:
        raise ValueError('One valid decision clock per Cotton origin required')
    indices = clocks.searchsorted(decision, side='right')-1
    usable = indices >= 0
    picked = quotes.iloc[np.clip(indices, 0, len(quotes)-1)].reset_index(drop=True)
    selected_dates = picked.date.where(usable)
    age = pd.Series(np.arange(len(dates))-(dates.searchsorted(pd.DatetimeIndex(selected_dates), side='right')-1))
    age = age.where(usable)
    ready = usable & age.between(0, 3).to_numpy()
    return pd.DataFrame({'date': dates, 'dxy_rv5': picked.dxy_rv5.where(ready),
                         'dxy_state_source_date': selected_dates,
                         'dxy_state_available_at': pd.Series(clocks.take(np.clip(indices, 0, len(quotes)-1))).where(usable),
                         'dxy_state_window_start': picked.window_start.where(usable),
                         'dxy_state_age': age, 'decision_at': decision})


def features(rows, arm):
    recipe(arm)
    b = rows.named_ewma.to_numpy(float)
    x = rows[['har_1', 'har_5', 'har_22']].to_numpy(float)-np.log(b)[:, None]
    extra = np.log(b) if arm == 'own_level' else np.log(np.maximum(rows.dxy_rv5.to_numpy(float), FLOOR))
    x = np.column_stack([x, extra])
    if not np.isfinite(x).all() or not np.isfinite(b).all() or (b < FLOOR).any():
        raise InvalidVarianceFit('Finite past-only features and positive EWMA required')
    return x


def fit(train, test, workspace, arm):
    """Same log-link QLIKE solver/initialization; exactly one additional coefficient."""
    from scipy.optimize import minimize
    require_training('har', 'cpu')
    raw, xt = features(train, arm), features(test, arm)
    actual = (train.variance_target_1/train.named_ewma).to_numpy(float)
    if not np.isfinite(actual).all() or (actual < 0).any() or not (actual > 0).any():
        raise InvalidVarianceFit('Nonnegative mature ratios with positive mass required')
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.column_stack([np.ones(len(train)), (raw-mean)/scale])
    if np.linalg.matrix_rank(design) != 5:
        raise InvalidVarianceFit('Five-parameter design must have full rank')
    y = np.log(np.maximum(actual, FLOOR))
    initial = np.linalg.lstsq(design, y, rcond=None)[0]
    initial[0] += np.log(np.exp(y-design@initial).mean())
    initial_loss, _ = qlike_value_gradient(initial, design, actual)
    fitted = minimize(qlike_value_gradient, initial, args=(design, actual), jac=True,
                      method='BFGS', options={'maxiter': 500, 'gtol': 1e-6})
    value, gradient = qlike_value_gradient(fitted.x, design, actual)
    if (not fitted.success or not np.isfinite(fitted.x).all() or np.max(np.abs(gradient)) > 1e-6
            or value > initial_loss+1e-10 or (design@fitted.x < np.log(FLOOR)).any()):
        raise InvalidVarianceFit('QLIKE optimization not certified; no fallback/retry')
    coef = fitted.x.copy(); coef[1:] /= scale; coef[0] -= mean@coef[1:]
    prediction = test.named_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), xt])@coef)
    if not np.isfinite(prediction).all() or (prediction < FLOOR).any():
        raise InvalidVarianceFit('No silent variance output clipping')
    model = {'kind': PROFILE, 'recipe': recipe(arm), 'coef': coef.tolist(), 'effective_parameters': 5,
             'optimization': {'success': bool(fitted.success), 'method': 'BFGS', 'iterations': int(fitted.nit),
                              'gradient_inf_norm': float(np.max(np.abs(gradient))),
                              'initial_training_qlike': initial_loss, 'final_training_qlike': value,
                              'training_feature_mean': mean.tolist(), 'training_feature_scale': scale.tolist()}}
    (workspace/'variance-model.json').write_text(json.dumps(model), encoding='utf-8')
    return {'origins': test.date.dt.strftime('%Y-%m-%d').tolist(), 'variance_predictions': prediction.tolist()}


def spec(job, train, arm):
    return {'recipe': recipe(arm), 'role': f'{arm}-{job["role"]}-{job["bucket"]}', 'iterations': 1,
            'train_dates': job['train_dates'], 'test_dates': job['origins'], 'validation_dates': [],
            'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}


def project(history, states, dates, training):
    rows = parent.projection(history, dates, training=training)
    indexed = states.set_index('date').loc[rows.date]
    for name in states.columns.drop('date'):
        rows[name] = indexed[name].to_numpy()
    if not np.isfinite(rows.dxy_rv5).all() or (rows.dxy_rv5 < 0).any():
        raise ValueError('Do not drop/impute source-unready origins')
    return rows


def prepare(repo, root, reference, market):
    repo, root, reference, market = map(Path, (repo, root, reference, market))
    if root.exists():
        raise ValueError('Preserve existing namespace')
    validate(repo/'research')
    old, h = parent.verify(reference, repo)
    related = check(*load_registry(repo/'research'), family='har', horizon=1, query='qlike')
    folds = old['identity']['design']['folds']
    states = dxy_state(pd.read_parquet(market), h.date)
    origin_dates = [d for f in folds for d in f['origins']]
    forecast_dates = [d for f in folds for d in f['forecast_origins']]
    train_dates = sorted({d for f in folds for j in f['jobs'] for d in j['train_dates']})
    eval_state = states.set_index('date').loc[pd.to_datetime(origin_dates)]
    if not np.isfinite(eval_state.dxy_rv5).all():
        raise ValueError('Preserve all ordinal source slots including original no-forecast slots')
    for dates, training in [(forecast_dates, False), (train_dates, True)]:
        rows = project(h, states, dates, training)
        if (rows.dxy_state_available_at > rows.decision_at).any():
            raise ValueError('Source clock after decision')
    if len(origin_dates) != 2006 or sum(len(f['jobs']) for f in folds) != 174:
        raise ValueError('Original ordinal cohort and native job schedule required')
    design = {'profile': PROFILE, 'folds': folds, 'recipes': {a: recipe(a) for a in ARMS},
              'fit_budget': {'market': 348, 'synthetic': 1, 'processes': 1, 'threads': 2, 'session_minutes': 30},
              'primary': 'raw own_level minus dxy_state QLIKE; raw arms versus unchanged EWMA and frozen four-parameter control',
              'secondary': 'each past-selected arm with original3x63 QLIKE shrinkage; no between-arm OOS selection',
              'risk_gate': 'both20/60CI lower>0,>=6/8positiveyears,top1pct EWMA-loss trim>0',
              'interpretation': 'A isolates free own-level coefficient; B adds DXY state at same capacity, not timing-only or causal source proof',
              'clock': 'DXY source observation date+1day00UTC assumed <= origin+1day00:15UTC; maxage3 Cotton observations',
              'dxy_state': 'mean5 squared log returns on raw DXY observation calendar, then asof; log floor1e-12',
              'optimizer': 'same OLS+smearing initialization, standardized train-only, BFGS max500 gtol1e-6; no fallback',
              'unchanged': ['native174four-param fits', 'origin/target/reference', '5session maturity', '21refit', '189validation', 'shrink0/.25/.5/.75/1'],
              'new_data_downloaded': False, 'historical_availability_verified': False,
              'first_vintage_verified': False, 'independent_holdout': False, 'release_allowed': False,
              'no_search': True, 'deliberate_repeat_reason': 'Existing controls used constrained four-parameter relative HAR and no DXY risk state; test one coefficient restriction against one source at equal capacity.'}
    with writer(root):
        for p in reference.rglob('*'):
            if p.is_file():
                copy_immutable(p, root/'reference'/p.relative_to(reference))
        copy_immutable(market, root/'inputs/market.parquet')
        states.to_parquet(root/'inputs/dxy-state.parquet', index=False)
        manifest = source(repo)
        for name in [*manifest['ml']['files'], *manifest['helpers']]:
            copy_immutable(repo/name, root/'source-snapshot'/name)
        freeze_record(root/'history-check.json', {'related': [r['id'] for r in related['experiments']],
                                                 'related_recipes': len(related['fit_recipes']), 'new_fits': 0})
        freeze_record(root/'source-manifest.json', manifest)
        freeze_record(root/'decision-contract.json', design)
        identity = {'profile': PROFILE, 'source_id': manifest['source_id'], 'design_id': content_id(design),
                    'research_data_id': content_id({'history': old['identity']['research_data_id'], 'market': digest(market)}),
                    'split': {'design_id': content_id(folds)}, 'python': sys.version.split()[0],
                    'versions': {p: importlib.metadata.version(p) for p in VERSIONS}, 'lock_sha256': digest(repo/'ml/uv.lock')}
        frozen = {p.relative_to(root).as_posix(): digest(p) for section in ('inputs', 'reference')
                  for p in (root/section).rglob('*') if p.is_file()}
        freeze_record(root/'ready.json', {'identity': identity, 'frozen_hashes': frozen,
                                         'origin_count': 2006, 'unique_training_origins': len(train_dates)})
    return {'status': 'prepared', **identity, 'fit_budget': design['fit_budget']}


def verify(root, repo):
    root, repo = Path(root), Path(repo)
    ready = read_record(root/'ready.json'); i = ready['identity']
    design = read_record(root/'decision-contract.json')
    if (i['profile'] != PROFILE or source(repo) != read_record(root/'source-manifest.json')
            or i['source_id'] != source(repo)['source_id'] or content_id(design) != i['design_id']
            or i['python'] != sys.version.split()[0] or i['lock_sha256'] != digest(repo/'ml/uv.lock')
            or i['versions'] != {p: importlib.metadata.version(p) for p in VERSIONS}):
        raise ValueError('Frozen source/design/environment mismatch')
    for name, checksum in ready['frozen_hashes'].items():
        if not safe_member(name) or digest(root/name) != checksum:
            raise ValueError(f'Frozen source input mismatch: {name}')
    _, h = parent.verify(root/'reference', repo)
    states = pd.read_parquet(root/'inputs/dxy-state.parquet')
    pd.testing.assert_frame_equal(states, dxy_state(pd.read_parquet(root/'inputs/market.parquet'), h.date), check_exact=True)
    return ready, design, h, states


def run(root, repo, authorization, minutes):
    require_training('har', 'cpu')
    ready, design, h, states = verify(root, repo)
    auth = read_record(authorization)
    if (auth.get('ready_sha256') != digest(root/'ready.json') or auth.get('source_id') != ready['identity']['source_id']
            or auth.get('design_id') != ready['identity']['design_id'] or auth.get('fresh_restore_verified') is not True
            or auth.get('fit_budget') != design['fit_budget'] or not 0 < minutes <= 30):
        raise ValueError('Exact frozen preregistration/recovery/budget required')
    deadline = time.monotonic()+minutes*60
    with writer(root):
        learning = Ledger(root/'learning/ledger', {**ready['identity'], 'role': 'synthetic_not_market'})
        used, keys = fit_consumption(learning.root)
        if used != len(keys) or used > 1:
            raise ValueError('Failed/excess synthetic attempt; no retry')
        rng = np.random.default_rng(42); x = rng.uniform(-2, 2, (1200, 4))
        b = np.exp(-8+x[:, 3])
        fixture = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1200), 'named_ewma': b})
        for j, k in enumerate((1, 5, 22)):
            fixture[f'har_{k}'] = np.log(b)+x[:, j]
        fixture['variance_target_1'] = b*np.exp(.1+features(fixture, 'own_level')@np.array([.2, .1, -.1, .08]))
        smoke = learning.run({'recipe': recipe('own_level'), 'role': 'synthetic_five_parameter_known_signal', 'kind': 'known_signal'},
                             lambda p: fit(fixture.iloc[:1000], fixture.iloc[1000:], p, 'own_level'))
        np.testing.assert_allclose(smoke['result']['variance_predictions'], fixture.variance_target_1.iloc[1000:], rtol=1e-5)
        control = {'status': 'passed', 'fit_id': smoke['experiment_id'], 'market_skill': False}
        if (root/'learning-control.json').exists():
            if read_record(root/'learning-control.json') != control:
                raise ValueError('Changed learning control')
        else:
            freeze_record(root/'learning-control.json', control)
        ledger = Ledger(root/'ledger', ready['identity'])
        used, keys = fit_consumption(ledger.root)
        if used != len(keys) or used > 348:
            raise ValueError('Failed/excess market attempt; no silent retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('Checkpoint boundary')
            if count[0] >= 348:
                raise ValueError('Frozen market fit budget exhausted')
            count[0] += 1

        try:
            for arm in ARMS:
                for fold in design['folds']:
                    for job in fold['jobs']:
                        train = project(h, states, job['train_dates'], True)
                        test = project(h, states, job['origins'], False)
                        ledger.run(spec(job, train, arm), lambda p, tr=train, te=test, a=arm: fit(tr, te, p, a),
                                   before_compute=before_compute)
        except FitBudgetReached:
            return {'status': 'planned_pause', 'market_fits': count[0], 'synthetic_fits': 1}
    return {'status': 'completed', 'market_fits': count[0], 'synthetic_fits': 1}


def native(ledger, job, train, test, arm):
    key = ledger.key(spec(job, train, arm)); r = read_record(ledger.root/'completed'/f'{key}.json')
    if r['identity'] != ledger.identity or r['specification'] != spec(job, train, arm) or r['experiment_id'] != key:
        raise ValueError('Native identity/spec mismatch')
    for name, checksum in r['files'].items():
        if not safe_member(name) or digest(ledger.root/name) != checksum:
            raise ValueError('Native model hash mismatch')
    m = read_record(ledger.root/r['payload_root']/'variance-model.json'); o = m['optimization']
    if m['recipe'] != recipe(arm) or m['effective_parameters'] != 5 or m['kind'] != PROFILE:
        raise ValueError('Wrong five-parameter recipe')
    x = features(train, arm); eta = np.column_stack([np.ones(len(train)), x])@m['coef']
    ratio = (train.variance_target_1/train.named_ewma).to_numpy()*np.exp(-eta)
    z = (x-o['training_feature_mean'])/o['training_feature_scale']
    grad = np.column_stack([np.ones(len(train)), z]).T@(1-ratio)/len(train)
    if not o['success'] or np.max(np.abs(grad)) > 1.001e-6 or not math.isclose(float(np.mean(eta+ratio)), o['final_training_qlike'], abs_tol=1e-10):
        raise ValueError('Uncertified native optimizer')
    predicted = test.named_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), features(test, arm)])@m['coef'])
    np.testing.assert_allclose(predicted, r['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
    if r['result']['origins'] != job['origins']:
        raise ValueError('Native origin mismatch')
    return r


def statistics(frame, a, b, bootstrap):
    valid = frame.eligible; clean = frame.loc[valid]
    offset = max(0., 1-min(clean[a].min(), clean[b].min()))
    s = pd.DataFrame({'year': frame.year, 'a': np.where(valid, frame[a].fillna(0)+offset, 0),
                      'b': np.where(valid, frame[b].fillna(0)+offset, 0), 'weight': valid.astype(int)})
    parts = [g[['a', 'b', 'weight']].to_numpy() for _, g in s.groupby('year')]
    ci = {str(k): bootstrap.weighted_bootstrap(parts, block=k)['difference_ci_95'] for k in (20, 60)}
    yearly = {str(y): float((g[a]-g[b]).mean()) for y, g in clean.groupby('year')}
    trimmed = clean.drop(clean.nlargest(math.ceil(len(clean)*.01), 'baseline_loss').index)
    trim = float((trimmed[a]-trimmed[b]).mean()); gain = float((clean[a]-clean[b]).mean())
    return {'gain': gain, 'gain_ci': ci, 'yearly_gain': yearly, 'positive_years': sum(v > 0 for v in yearly.values()),
            'trim_top1pct_gain': trim, 'fixed_risk_gate_passed': bool(gain > 0 and all(v[0] > 0 for v in ci.values())
                                                                 and sum(v > 0 for v in yearly.values()) >= 6 and trim > 0)}


def compare(root, repo):
    ready, design, h, states = verify(root, repo)
    ledger = Ledger(root/'ledger', ready['identity']); outputs = {}
    used, keys = fit_consumption(ledger.root)
    if used != 348 or len(keys) != 348 or not (root/'learning-control.json').is_file():
        raise ValueError('Exact completed market and synthetic fits required')
    previous = read_record(root/'reference/ready.json')
    oldledger = Ledger(root/'reference/ledger', parent.ledger_identity(previous['identity']))
    for arm in (*ARMS, 'four_parameter'):
        rows, selections = [], {}
        for fold in design['folds']:
            records = []
            for job in fold['jobs']:
                if arm == 'four_parameter':
                    train = parent.projection(h, job['train_dates'], training=True)
                    test = parent.projection(h, job['origins'], training=False)
                    r, _ = parent.replay_native(oldledger, job, train, test)
                else:
                    train = project(h, states, job['train_dates'], True)
                    test = project(h, states, job['origins'], False)
                    r = native(ledger, job, train, test, arm)
                records.append(r)
            out = parent.parent.output(h, fold, 'named', records)
            selections[str(fold['year'])] = out['selection']
            rows.extend({**r, 'year': fold['year']} for r in out['records'])
        outputs[arm] = (pd.DataFrame(rows), selections)
    frame = outputs['four_parameter'][0].copy()
    for arm in ARMS:
        candidate = outputs[arm][0]
        for name in ('date', 'target_date', 'contract', 'eligible', 'named_actual', 'baseline'):
            pd.testing.assert_series_equal(frame[name], candidate[name], check_names=False, check_exact=True)
        for name in ('raw', 'selected'):
            frame[f'{arm}_{name}'] = candidate[name]
    if len(frame) != 2006 or int(frame.eligible.sum()) != 1989:
        raise ValueError('No cohort changes or silent intersection allowed')
    metadata = states.set_index('date').loc[pd.to_datetime(frame.date)]
    for name in states.columns.drop('date'):
        frame[name] = metadata[name].to_numpy()
    names = ['baseline', 'raw', 'selected', *[f'{a}_{k}' for a in ARMS for k in ('raw', 'selected')]]
    for name in names:
        frame[name+'_loss'] = np.nan
        frame.loc[frame.eligible, name+'_loss'] = qlike(frame.loc[frame.eligible, 'named_actual'], frame.loc[frame.eligible, name])
    loader = importlib.util.spec_from_file_location('frozen_level_bootstrap', root/'reference/inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(loader); loader.loader.exec_module(bootstrap)
    comparisons = {}
    for arm in ARMS:
        for kind in ('raw', 'selected'):
            for control in ('baseline', 'raw' if kind == 'raw' else 'selected'):
                key = f'{arm}_{kind}_vs_{control}'
                comparisons[key] = statistics(frame, control+'_loss', f'{arm}_{kind}_loss', bootstrap)
    comparisons['dxy_raw_vs_own_raw'] = statistics(frame, 'own_level_raw_loss', 'dxy_state_raw_loss', bootstrap)
    comparisons['dxy_selected_vs_own_selected'] = statistics(frame, 'own_level_selected_loss', 'dxy_state_selected_loss', bootstrap)
    report = {'profile': PROFILE, 'new_fits_during_report': 0, 'native_market_fits': 348, 'synthetic_fits': 1,
              'reused_four_parameter_fits': 174, 'retained_origins': 2006, 'paired_origins': 1989,
              'comparisons': comparisons, 'selections': {a: outputs[a][1] for a in outputs},
              'independent_holdout': False, 'historical_availability_verified': False, 'first_vintage_verified': False,
              'release_allowed': False, 'no_automatic_winner_selection': True}
    dest = root/'reports'; dest.mkdir(exist_ok=True)
    if (dest/'paired-losses.csv').exists():
        raise ValueError('Immutable report exists')
    frame.to_csv(dest/'paired-losses.csv', index=False, lineterminator='\n')
    report['paired_losses_sha256'] = digest(dest/'paired-losses.csv')
    freeze_record(dest/'result.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    for name in ('repo', 'root'):
        p.add_argument('--'+name, required=True, type=Path)
    for name in ('reference', 'market', 'authorization'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--minutes', type=float, default=30)
    a = p.parse_args()
    if a.stage == 'prepare':
        result = prepare(a.repo, a.root, a.reference, a.market)
    elif a.stage == 'verify':
        ready, design, _, _ = verify(a.root, a.repo)
        result = {'status': 'verified', 'source_id': ready['identity']['source_id'], 'fit_budget': design['fit_budget']}
    elif a.stage == 'pilot':
        result = run(a.root, a.repo, a.authorization, a.minutes)
    else:
        result = compare(a.root, a.repo)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
