"""Bounded price-only median pilot using the existing temporal engine and ledger."""
import argparse
import importlib.metadata
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.full_year import chunks
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.path_pilot import output, run
from cottonlens_ml.research.protocol import (
    Preprocessor,
    Target,
    full_year_manifest,
    inputs,
    rows_at,
)
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.statistics import paired_bootstrap

PROFILE = 'price-only-median-t1-v1'
FEATURES = ['mae_ret1', 'mae_ret5', 'mae_ret20', 'mae_vol20', 'mae_z20']
PARAMS = {'quantile': .5, 'alpha': .01, 'fit_intercept': True, 'solver': 'highs',
          'solver_options': {'threads': 1, 'time_limit': 120.}}
GROUPS = ('ridge', 'quantile')
MARKET_SHA = '23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c'


def features(history):
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique chronological price history required')
    p = history.cotton_close
    if not np.isfinite(p).all() or p.le(0).any():
        raise ValueError('Finite positive Cotton prices required')
    frame = history.copy()
    for lag in (1, 5, 20):
        frame[f'mae_ret{lag}'] = np.log(p / p.shift(lag))
    frame['mae_vol20'] = frame.mae_ret1.rolling(20).std()
    std = p.rolling(20).std()
    frame['mae_z20'] = (p - p.rolling(20).mean()) / std.where(std > 0)
    return frame.replace([np.inf, -np.inf], np.nan)


def validate_quantile(spec):
    if (spec.get('research_profile') != PROFILE or spec.get('params') != PARAMS
            or spec.get('horizon') != 1 or spec.get('task') != 'price'
            or spec.get('target') != 'price_delta' or spec.get('window') != 1
            or spec.get('features') != FEATURES or spec.get('device') != 'cpu'):
        raise ValueError('Only the frozen price-only Quantile recipe is admitted')


def recipe(design, group, h):
    if group not in GROUPS or h != 1:
        raise ValueError('Fixed T+1 pilot only')
    return {'family': group, 'params': PARAMS if group == 'quantile' else {'alpha': 1.},
            'research_profile': PROFILE, 'horizon': 1, 'task': 'price', 'device': 'cpu',
            'seed': 42, 'target': 'price_delta', 'years': None, 'cadence': 21,
            'window': 1, 'features': FEATURES}


def validate_design(registration):
    d = registration['design']
    if (registration['design_id'] != content_id(d) or d['profile'] != PROFILE
            or d['groups'] != list(GROUPS) or d['features'] != FEATURES
            or d['quantile_params'] != PARAMS or d['budget'] != {'market': 338, 'synthetic': 4}
            or d['total_minutes'] != 60 or d['split']['purge_observations'] != 5
            or d['primary'] != 'selected_quantile'
            or d['gate'] != {'gain_pct': 5., 'direction_pct': 53., 'year_wins': 6}
            or d['bootstrap'] != {'blocks': [20, 60], 'repetitions': 10000, 'seed': 42}
            or d['split']['refit_cadence'] != 21
            or [f['year'] for f in d['split']['folds']] != list(range(2016, 2024))
            or sum(len(f['origins']) for f in d['split']['folds']) != 2006):
        raise ValueError('Frozen median design changed')
    return d


def load_reference(reference, market):
    ready, history = verify_history(reference)
    if digest(market) != MARKET_SHA or digest(market) != ready['identity']['raw_audit']['market_sha256']:
        raise ValueError('Raw market checksum mismatch')
    raw = pd.read_parquet(market)
    cotton = raw.loc[raw.series.eq('cotton')].sort_values('date').set_index('date')
    if cotton.index.has_duplicates or cotton.close.isna().any() or cotton.close.le(0).any():
        raise ValueError('Invalid raw Cotton observations')
    aligned = cotton.reindex(history.date)
    if not np.array_equal(aligned.close.to_numpy(), history.cotton_close.to_numpy()):
        raise ValueError('Raw/frozen Cotton price mismatch')
    for h in (1, 5):
        expected = np.log(cotton.close.shift(-h) / cotton.close).reindex(history.date)
        dates = pd.Series(cotton.index, index=cotton.index).shift(-h).reindex(history.date)
        if (not np.allclose(expected, history[f'target_return_{h}'], rtol=0, atol=1e-12, equal_nan=True)
                or not np.array_equal(dates.to_numpy(), history[f'target_date_{h}'].to_numpy())):
            raise ValueError('Raw/frozen target or target-date mismatch')
    if full_year_manifest(history) != ready['identity']['split']:
        raise ValueError('Original full-year cohort changed')
    return ready, history, aligned.reset_index(drop=True)


def write_csv(path, frame):
    data = frame.to_csv(index=False, lineterminator='\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Conflicting analysis output; preserve and use a new namespace')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def loss_arrays(frame, field='predicted_return', horizon=1):
    if frame.date.duplicated().any() or not np.isfinite(
            frame[['cotton_close', f'target_return_{horizon}', field]]).all().all():
        raise ValueError('Finite, unique, aligned prediction rows required')
    p, y, pred = (frame[c].to_numpy(dtype=float) for c in
                  ('cotton_close', f'target_return_{horizon}', field))
    return p * np.abs(np.expm1(y)), p * np.abs(np.expm1(y) - np.expm1(pred))


def annotate_windows(frame, history, annotation, horizon):
    """Ex-post descriptive windows; unknown observations never become non-events."""
    annotation = annotation.set_index('date')
    for name in ('wasde_event', 'contract_transition'):
        if not annotation[name].isin(['yes', 'no', 'unknown']).all():
            raise ValueError('Explicit yes/no/unknown annotation required')
        values = annotation[name].to_dict()
        labels = []
        for index in history.date.searchsorted(frame.date):
            window = [values.get(d.strftime('%Y-%m-%d'), 'unknown')
                      for d in history.date.iloc[index+1:index+horizon+1]]
            if len(window) != horizon:
                raise ValueError('Incomplete descriptive horizon')
            if name == 'contract_transition' and 'unknown' in window:
                labels.append('unknown')  # A verified whole contract path is required.
            else:
                labels.append('yes' if 'yes' in window else ('unknown' if 'unknown' in window else 'no'))
        frame[name] = labels
    return frame


def audit(repo, folder, reference, market, evidence_roots, annotations=None):
    """Zero fits. Exact hashes/origins precede descriptive, past-defined regimes."""
    ready, history, raw = load_reference(reference, market)
    history = features(history)
    manifest = ready['identity']['split']
    registry, _trials = load_registry(repo / 'research')
    inventory = []
    for rec in registry['records']:
        present = absent = mismatch = 0
        for name, expected in rec.get('payloads', {}).items():
            found = next((r / name for r in evidence_roots if (r / name).is_file()), None)
            if found is None:
                absent += 1
            elif digest(found) == expected:
                present += 1
            else:
                mismatch += 1
        inventory.append({'id': rec['id'], 'profile': rec.get('profile'), 'horizon': rec.get('horizon'),
                          'classification': rec.get('classification'), 'coverage': rec.get('coverage'),
                          'verified_local_payloads': present, 'not_in_supplied_local_roots': absent,
                          'mismatched_local_payloads': mismatch})
    if any(r['mismatched_local_payloads'] for r in inventory):
        raise ValueError('Registered payload checksum mismatch')
    annotation = None
    if annotations:
        annotation = pd.read_csv(annotations, keep_default_na=False)
        if annotation.date.duplicated().any():
            raise ValueError('Duplicate explanatory annotation dates')
    rows = []
    for h in (1, 5):
        for fold in manifest['folds']:
            frame = rows_at(history, fold['origins']).copy()
            frame['volatility_group'] = 'unknown'
            frame['trend_group'] = 'unknown'
            frame['large_realized_move'] = False
            frame['trend_strength'] = frame.mae_ret20.abs() / (frame.mae_vol20 * np.sqrt(20))
            for chunk in chunks(history, fold['origins']):
                from cottonlens_ml.research.review import training_rows
                past = training_rows(history, chunk.date.min(), {}, None)
                mask = frame.date.isin(chunk.date)
                for column, field in [('mae_vol20', 'volatility_group'), ('trend_strength', 'trend_group')]:
                    values = (past.mae_ret20.abs() / (past.mae_vol20 * np.sqrt(20))
                              if column == 'trend_strength' else past[column])
                    cuts = values.dropna().quantile([1/3, 2/3]).to_numpy()
                    valid = frame.loc[mask, column].notna()
                    labels = np.array(['low', 'middle', 'high'])[np.searchsorted(cuts, frame.loc[mask, column])]
                    frame.loc[mask, field] = np.where(valid, labels, 'unknown')
                threshold = past[f'target_return_{h}'].abs().quantile(.9)
                frame.loc[mask, 'large_realized_move'] = frame.loc[mask, f'target_return_{h}'].abs() > threshold
            frame['horizon'], frame['year'] = h, fold['year']
            frame['target_date'] = frame[f'target_date_{h}']
            frame['signed_error'] = frame.cotton_close * np.expm1(frame[f'target_return_{h}'])
            frame['naive_error'] = frame.signed_error.abs()
            frame['month'] = frame.date.dt.strftime('%Y-%m')
            frame['wasde_event'], frame['contract_transition'] = 'unknown', 'unknown'
            if annotation is not None:
                frame = annotate_windows(frame, history, annotation, h)
            for name in ('open', 'high', 'low', 'volume'):
                frame['raw_' + name] = raw.set_axis(history.date).loc[frame.target_date, name].to_numpy()
            frame['target_close'] = frame.cotton_close * np.exp(frame[f'target_return_{h}'])
            frame['outside_high_low'] = (frame.target_close < frame.raw_low - 1e-5) | (frame.target_close > frame.raw_high + 1e-5)
            rows.append(frame[['date', 'target_date', 'horizon', 'year', 'month', 'cotton_close',
                               'naive_error', 'signed_error', 'volatility_group', 'trend_group',
                               'large_realized_move', 'wasde_event', 'contract_transition', 'outside_high_low', *FEATURES]])
    combined = pd.concat(rows, ignore_index=True)
    tables = []
    for dimension in ('year', 'month', 'volatility_group', 'trend_group', 'large_realized_move',
                      'wasde_event', 'contract_transition', 'outside_high_low'):
        for (h, group), f in combined.groupby(['horizon', dimension]):
            tables.append({'horizon': int(h), 'dimension': dimension, 'group': str(group), 'count': len(f),
                           'mae': float(f.naive_error.mean()), 'median_absolute_error': float(f.naive_error.median()),
                           'p90': float(f.naive_error.quantile(.9)), 'p99': float(f.naive_error.quantile(.99))})
    correlations, acf = [], []
    for h, f in combined.groupby('horizon'):
        for field in ('signed_error', 'naive_error'):
            for lag in (1, 5, 20):
                value = f[field].autocorr(lag)
                acf.append({'horizon': int(h), 'field': field, 'lag': lag, 'correlation': float(value) if np.isfinite(value) else None})
            for feature in FEATURES:
                value = f[feature].corr(f[field], method='spearman')
                correlations.append({'horizon': int(h), 'field': field, 'feature': feature,
                                     'spearman': float(value) if np.isfinite(value) else None})
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / 'naive-origins.csv', combined)
    write_csv(folder / 'naive-summary.csv', pd.DataFrame(tables))
    write_csv(folder / 'evidence-inventory.csv', pd.DataFrame(inventory))
    body = {'status': 'verified', 'new_fits': 0, 'market_sha256': digest(market),
            'reference_ready_sha256': digest(reference / 'ready.json'), 'history_sha256': ready['history_sha256'],
            'split_id': manifest['split_id'], 'origins_per_horizon': 2006, 'registry': validate(repo/'research'),
            'acf': acf, 'associations': correlations, 'annotations_sha256': digest(annotations) if annotations else None,
            'limits': ['Reviewed history, not independent holdout.', 'Associations and ex-post movements are descriptive, not forecast skill.',
                       'Unknown contract/event annotations remain unknown; daily source dates do not prove first-vintage/availability.',
                       'Missing local copies are not automatically missing remote evidence. Legacy missing LSTM/native evidence is not regenerated.'],
            'files': {p.name: digest(p) for p in folder.glob('*.csv')}}
    freeze_record(folder / 'audit.json', body)
    return body


def prepare(repo, folder, reference, market, audit_root):
    ready, history, _ = load_reference(reference, market)
    audited = read_record(audit_root / 'audit.json')
    if (audited['status'] != 'verified' or audited['history_sha256'] != ready['history_sha256']
            or audited['market_sha256'] != digest(market)):
        raise ValueError('Matching completed zero-fit audit required')
    for name, checksum in audited['files'].items():
        if Path(name).name != name or digest(audit_root/name) != checksum:
            raise ValueError('Audit evidence changed')
    expanded = features(history)
    split = ready['identity']['split']
    count = 2 * sum(sum(len(chunks(expanded, b['origins'])) for b in f['inner'])
                    + len(chunks(expanded, f['origins'])) for f in split['folds'])
    if count != 338:
        raise ValueError('Original fit budget changed')
    design = {'profile': PROFILE, 'groups': list(GROUPS), 'features': FEATURES, 'quantile_params': PARAMS,
              'split': split, 'budget': {'market': count, 'synthetic': 4}, 'total_minutes': 60,
              'audit_sha256': digest(audit_root/'audit.json'), 'market_sha256': digest(market),
              'reference_ready_sha256': digest(reference/'ready.json'),
              'feature_data_id': frame_identity(expanded, list(expanded.columns)),
              'primary': 'selected_quantile', 'gate': {'gain_pct': 5., 'direction_pct': 53., 'year_wins': 6},
              'bootstrap': {'blocks': [20, 60], 'repetitions': 10000, 'seed': 42},
              'novelty': 'Five price-only covariates with conditional linear median and same-input Ridge; not a repeat of 24-column MAE trees or scalar calibration. Covariate formulas overlap prior research; no claim of entirely new information.',
              'ablation_if_continue': ['remove_ret5_ret20', 'remove_vol20', 'remove_z20']}
    registration = {'design_id': content_id(design), 'design': design}
    validate_design(registration)
    registry, trials = load_registry(repo/'research')
    proposals = [check(registry, trials, proposal={'scope_id': content_id(design), 'recipe': recipe(design, g, 1)}) for g in GROUPS]
    if any(p['status'] == 'REPEAT_FIT_RECIPE' for p in proposals):
        raise ValueError('Exact frozen recipe already completed; use existing evidence')
    identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'],
                'design_id': registration['design_id'], 'design': design, 'split': split,
                'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in
                    ('numpy', 'pandas', 'scipy', 'scikit-learn')}, 'execution': {'device': 'cpu', 'threads': 2}}
    with writer(folder):
        if (folder/'ready.json').exists():
            old, _ = verify_history(folder)
            if old['identity'] != identity:
                raise ValueError('Frozen source, design or environment changed')
            return old
        copy_immutable(reference/'ready.json', folder/'reference/ready.json')
        copy_immutable(reference/'history.parquet', folder/'reference/history.parquet')
        copy_immutable(audit_root/'audit.json', folder/'audit/audit.json')
        for name in audited['files']:
            copy_immutable(audit_root/name, folder/'audit'/name)
        expanded.to_parquet(folder/'history.parquet', index=False)
        freeze_record(folder/'preregistered.json', registration)
        freeze_record(folder/'novelty-check.json', {'statuses': [p['status'] for p in proposals],
            'related_record_ids': [[r['id'] for r in p['experiments']] for p in proposals]})
        result = {'identity': identity, 'history_sha256': digest(folder/'history.parquet')}
        freeze_record(folder/'ready.json', result)
    return read_record(folder/'ready.json')


def verify_linear(record, ledger, history):
    """Safe JSON native replay, independent of the fitting path; never retrain."""
    from cottonlens_ml.research.reproduction import verified_native_predictions
    expected = record['specification']
    saved, _ = verified_native_predictions(ledger, record['experiment_id'], expected)
    root = ledger.root / record['payload_root']
    adapter = json.loads((root/'adapter.json').read_bytes())
    train, test = (rows_at(history, expected[k]) for k in ('train_dates', 'test_dates'))
    if not train.target_date_5.lt(test.date.min()).all():
        raise ValueError('Immature training labels')
    processor = Preprocessor.fit(train, adapter['spec']['features'])
    target = Target.fit(train, 1, 'price_delta')
    if processor.as_dict() != adapter['processor'] or target.as_dict() != adapter['target']:
        raise ValueError('Native state differs from train-only reconstruction')
    model = json.loads((root/adapter['model_file']).read_bytes())
    dtype = model.get('dtype', 'float32')
    predicted = inputs(history, test, processor, 1) @ np.asarray(model['coef'], dtype=dtype) + np.asarray(model['intercept'], dtype=dtype)
    replay = target.inverse(predicted, test.cotton_close.to_numpy())
    difference = float(np.max(np.abs(replay-saved)))
    if difference > 1e-7:
        raise ValueError('Native linear prediction mismatch')
    return difference


def control_frames():
    rng = np.random.default_rng(1701)
    frame = pd.DataFrame(rng.normal(size=(800, 5)), columns=FEATURES)
    frame['date'] = pd.bdate_range('2000-01-03', periods=len(frame))
    frame['cotton_close'] = 80.
    frame['cotton_session_index'] = np.arange(len(frame))
    for kind in ('known_signal', 'negative_shuffled'):
        delta = 2 * frame.mae_ret1.to_numpy() if kind == 'known_signal' else rng.normal(size=len(frame))
        for h in (1, 5):
            frame[f'target_return_{h}'] = np.log1p(delta/80.)
            frame[f'target_date_{h}'] = frame.date.shift(-h)
        yield kind, frame.copy()


def verify_controls(experiment):
    body = read_record(experiment.root/'learning-controls.json')
    if body['status'] != 'passed' or len(body['checks']) != 4:
        raise ValueError('Failed/incomplete learning gate')
    expected = [(kind, family) for kind in ('known_signal', 'negative_shuffled') for family in GROUPS]
    if [(c['kind'], c['family']) for c in body['checks']] != expected:
        raise ValueError('Learning gate families changed')
    frames = dict(control_frames())
    for c in body['checks']:
        rec = read_record(experiment.ledger.root/'completed'/f'{c["fit_id"]}.json')
        verify_linear(rec, experiment.ledger, frames[c['kind']])
        test = rows_at(frames[c['kind']], rec['specification']['test_dates'])
        naive, error = loss_arrays(test.assign(predicted_return=rec['result']['predictions']))
        gain = float(1-error.mean()/naive.mean())
        if abs(gain-c['gain']) > 1e-12 or not (gain >= .5 if c['kind'] == 'known_signal' else gain < .2):
            raise ValueError('Learning gate does not match native controls')
    return body


def learning_controls(experiment):
    path = experiment.root/'learning-controls.json'
    if path.exists():
        return verify_controls(experiment)
    from cottonlens_ml.research.models import fit_predict
    checks = []
    for kind, frame in control_frames():
        for family in GROUPS:
            spec = recipe(experiment.identity['design'], family, 1)
            train, test = frame.iloc[:600].copy(), frame.iloc[610:790].copy()
            specification = {'kind': 'synthetic_not_market', 'recipe': spec, 'control': kind,
                             'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(),
                             'test_dates': test.date.dt.strftime('%Y-%m-%d').tolist(),
                             'seed': 1701}
            rec = experiment.ledger.run(specification, lambda workspace, frame=frame, train=train, test=test, spec=spec: fit_predict(
                frame, train, None, test, spec, workspace, iterations=1))
            naive, error = loss_arrays(test.assign(predicted_return=rec['result']['predictions']))
            gain = 1-error.mean()/naive.mean()
            checks.append({'kind': kind, 'family': family, 'fit_id': rec['experiment_id'], 'gain': float(gain),
                           'passed': bool(gain >= .5 if kind == 'known_signal' else gain < .2)})
    body = {'status': 'passed' if all(r['passed'] for r in checks) else 'failed', 'synthetic_fits': 4, 'checks': checks}
    freeze_record(path, body)
    if body['status'] != 'passed':
        raise ValueError('Learning controls failed; no market fit')
    return body


def pilot(repo, folder, minutes):
    from cottonlens_ml.research.engine import Experiment
    if not 0 < minutes <= 60:
        raise ValueError('Positive bounded session required')
    with writer(folder):
        exp = Experiment(folder, repo=repo)
        validate_design({'design_id': exp.identity['design_id'], 'design': exp.identity['design']})
        learning_controls(exp)
        used = sum(read_record(p)['seconds'] for p in (folder/'sessions').glob('*.json'))
        allowance = min(minutes*60, 3600-used)
        if allowance < 120:
            raise FitBudgetReached('Cumulative 60-minute budget exhausted')
        start = time.monotonic()
        def before():
            if time.monotonic()-start > allowance-120:
                raise FitBudgetReached('Insufficient time remaining for a bounded solver fit')
            if len(list((folder/'ledger/completed').glob('*.json'))) >= 342:
                raise FitBudgetReached('Frozen fit budget exhausted')
        try:
            return run(exp, allowance/60, group_names=GROUPS, namespace='median', recipe_fn=recipe,
                       validate_fn=validate_design, before_fit=before, horizons=(1,))
        finally:
            freeze_record(folder/'sessions'/f'{uuid.uuid4().hex}.json', {'seconds': time.monotonic()-start})


def verify(folder):
    from cottonlens_ml.research.ledger import Ledger
    ready, history = verify_history(folder)
    design = validate_design({'design_id': ready['identity']['design_id'], 'design': ready['identity']['design']})
    parent = read_record(folder/'reference/ready.json')
    if (digest(folder/'reference/ready.json') != design['reference_ready_sha256']
            or digest(folder/'reference/history.parquet') != parent['history_sha256']
            or frame_identity(history, list(history.columns)) != design['feature_data_id']
            or frame_identity(features(pd.read_parquet(folder/'reference/history.parquet')), list(history.columns)) != design['feature_data_id']
            or full_year_manifest(history) != design['split']):
        raise ValueError('Frozen data/features or canonical cohort changed')
    ledger = Ledger(folder/'ledger', ready['identity'])
    from types import SimpleNamespace
    verify_controls(SimpleNamespace(root=folder, ledger=ledger))
    expected_ids, maximum = [], 0.
    frames = {g: [] for g in GROUPS}
    expected_outputs = {f'{g}-t1-year{f["year"]}.json' for g in GROUPS for f in design['split']['folds']}
    if {p.name for p in (folder/'median-outputs').glob('*.json')} != expected_outputs:
        raise ValueError('Incomplete/unregistered annual output cohort')
    from cottonlens_ml.research.engine import Experiment
    for fold in design['split']['folds']:
        for group in GROUPS:
            name = f'{group}-t1-year{fold["year"]}.json'
            rec, frame = output(folder, name, fold, group, 1, design, history, namespace='median', recipe_fn=recipe)
            frame['year'], frame['group'] = fold['year'], group
            frame['target_date'] = rows_at(history, fold['origins']).target_date_1.dt.strftime('%Y-%m-%d').to_numpy()
            frame['horizon'] = 1
            frames[group].append(frame)
            inner_scores = []
            for i, block in enumerate(fold['inner']):
                raw = []
                for j, test in enumerate(chunks(history, block['origins'])):
                    from cottonlens_ml.research.review import training_rows
                    train = training_rows(history, test.date.min(), recipe(design, group, 1), None)
                    spec = Experiment.fit_specification(recipe(design, group, 1), train, None, test,
                                                       f'inner-{fold["year"]}-{i}-{j}', 1)
                    spec['repeat_reason'] = None
                    key = ledger.key(spec)
                    record = read_record(ledger.root/'completed'/f'{key}.json')
                    maximum = max(maximum, verify_linear(record, ledger, history))
                    expected_ids.append(key)
                    raw.extend(record['result']['predictions'])
                test = rows_at(history, block['origins'])
                inner_scores.append([float(loss_arrays(test.assign(predicted_return=w*np.asarray(raw)))[1].mean()
                                           /loss_arrays(test.assign(predicted_return=0.))[0].mean()) for w in (0., .25, .5, .75, 1.)])
            selected = int(np.argmin(np.mean(inner_scores, axis=0)))
            if rec['weight'] != (0., .25, .5, .75, 1.)[selected] or abs(rec['inner_score']-np.mean(inner_scores, axis=0)[selected]) > 1e-12:
                raise ValueError('Selection does not match past-only native predictions')
            predictions = []
            for j, test in enumerate(chunks(history, fold['origins'])):
                train = training_rows(history, test.date.min(), recipe(design, group, 1), None)
                spec = Experiment.fit_specification(recipe(design, group, 1), train, None, test,
                                                   f'path-outer-{group}-{fold["year"]}-{j}', 1)
                spec['repeat_reason'] = None
                key = ledger.key(spec)
                record = read_record(ledger.root/'completed'/f'{key}.json')
                maximum = max(maximum, verify_linear(record, ledger, history))
                expected_ids.append(key)
                predictions.extend(record['result']['predictions'])
            if not np.array_equal(frame.raw_predicted_return, predictions):
                raise ValueError('Output differs from native fit receipts')
    all_ids = {p.stem for p in (folder/'ledger/completed').glob('*.json')}
    controls = read_record(folder/'learning-controls.json')
    if len(set(expected_ids)) != 338 or all_ids != set(expected_ids) | {r['fit_id'] for r in controls['checks']}:
        raise ValueError('Unexpected/incomplete fit cohort')
    receipt = {'status': 'verified', 'market_fits': 338, 'synthetic_fits': 4, 'max_native_return_difference': maximum,
               'fit_ids_sha256': content_id(expected_ids), 'ready_sha256': digest(folder/'ready.json')}
    freeze_record(folder/'verification.json', receipt)
    return receipt, frames


def compare(folder):
    _receipt, frames = verify(folder)
    results = {}
    for group, yearly in frames.items():
        frame = pd.concat(yearly, ignore_index=True)
        modes = {}
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            naive, error = loss_arrays(frame, field)
            parts = [np.column_stack(loss_arrays(f, field)) for f in yearly]
            intervals = {str(b): paired_bootstrap(parts, block=b, repetitions=10000, seed=42,
                                                  normalization=float(naive.mean())) for b in (20, 60)}
            annual = [{'year': int(f.year.iloc[0]), 'gain_pct': float(100*(1-loss_arrays(f, field)[1].mean()/loss_arrays(f, field)[0].mean()))} for f in yearly]
            trim = np.argsort(-naive, kind='stable')[:int(np.ceil(.01*len(frame)))]
            keep = np.ones(len(frame), bool)
            keep[trim] = False
            modes[mode] = {'count': len(frame), 'mae': float(error.mean()), 'naive_mae': float(naive.mean()),
                           'gain_pct': float(100*(1-error.mean()/naive.mean())), 'median_absolute_error': float(np.median(error)),
                           'direction_pct': float(100*np.mean(np.sign(frame[field]) == np.sign(frame.target_return_1))),
                           'active_rate_pct': float(100*frame[field].ne(0).mean()), 'year_wins': sum(r['gain_pct'] > 0 for r in annual),
                           'years': annual, 'paired': intervals,
                           'gain_ci_pct': {b: (100*np.asarray(ci['difference_ci_95'])/naive.mean()).tolist()
                                           for b, ci in intervals.items()},
                           'trim_gain_pct': float(100*(1-error[keep].mean()/naive[keep].mean()))}
        results[group] = modes
    selected = results['quantile']['selected']
    contrast = {str(b): paired_bootstrap([
        np.column_stack([loss_arrays(a)[1], loss_arrays(c)[1]])
        for a, c in zip(frames['ridge'], frames['quantile'], strict=True)],
        block=b, repetitions=10000, seed=42,
        normalization=results['ridge']['selected']['mae']) for b in (20, 60)}
    passed = (selected['gain_pct'] >= 5 and selected['direction_pct'] >= 53 and selected['year_wins'] >= 6
              and all(selected['paired'][str(b)]['difference_ci_95'][0] > 0 for b in (20, 60)))
    combined = pd.concat([f for fs in frames.values() for f in fs], ignore_index=True)
    annotations = pd.read_csv(folder/'audit/naive-origins.csv', float_precision='round_trip')
    annotations = annotations.loc[annotations.horizon.eq(1)].set_index('date')
    regimes = []
    for group, fs in frames.items():
        frame = pd.concat(fs, ignore_index=True)
        if set(frame.date) != set(annotations.index) or annotations.index.has_duplicates:
            raise ValueError('Diagnostic and prediction cohorts differ')
        aligned = annotations.loc[frame.date]
        if aligned.target_date.tolist() != frame.target_date.tolist():
            raise ValueError('Diagnostic and prediction target dates differ')
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            naive, error = loss_arrays(frame, field)
            if not np.allclose(naive, aligned.naive_error, rtol=0, atol=1e-12):
                raise ValueError('Diagnostic and prediction targets differ')
            for dimension in ('month', 'volatility_group', 'trend_group', 'wasde_event', 'contract_transition', 'large_realized_move'):
                for value in aligned[dimension].unique():
                    mask = aligned[dimension].eq(value).to_numpy()
                    regimes.append({'group': group, 'mode': mode, 'dimension': dimension, 'value': str(value),
                                    'count': int(mask.sum()), 'naive_mae': float(naive[mask].mean()),
                                    'mae': float(error[mask].mean()), 'gain_pct': float(100*(1-error[mask].mean()/naive[mask].mean()))})
    write_csv(folder/'reports/regime-comparisons.csv', pd.DataFrame(regimes))
    write_csv(folder/'reports/predictions.csv', combined)
    write_csv(folder/'reports/paired-losses.csv', pd.DataFrame([
        {'date': r.date, 'target_date': r.target_date, 'group': g, 'year': r.year,
         'naive_error': n, 'model_error': e, 'paired_gain': n-e}
        for g, fs in frames.items() for f in fs
        for r, n, e in zip(f.itertuples(), *loss_arrays(f), strict=True)]))
    body = {'status': 'complete', 'decision': 'CONTINUE' if passed else 'STOP', 'primary': 'selected_quantile',
            'arms': results, 'selected_quantile_vs_ridge': contrast,
            'verification_sha256': digest(folder/'verification.json'),
            'predictions_sha256': digest(folder/'reports/predictions.csv'), 'release_allowed': False,
            'scope': 'Repeatedly reviewed historical CT=F proxy, not independent validation. Fixed recipe only; no universal no-signal claim.'}
    freeze_record(folder/'reports/result.json', body)
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'prepare', 'pilot', 'compare', 'verify', 'status'))
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--market-file', type=Path)
    parser.add_argument('--audit-root', type=Path)
    parser.add_argument('--evidence-root', action='append', type=Path, default=[])
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--max-minutes', type=float, default=15.)
    args = parser.parse_args(argv)
    if args.stage in ('audit', 'prepare') and (not args.reference_root or not args.market_file):
        parser.error('Explicit --reference-root and --market-file are required')
    if args.stage == 'prepare' and not args.audit_root:
        parser.error('Matching --audit-root is required before preparation')
    if args.stage == 'audit':
        result = audit(args.repo, args.folder, args.reference_root, args.market_file, args.evidence_root, args.annotations)
    elif args.stage == 'prepare':
        result = prepare(args.repo, args.folder, args.reference_root, args.market_file, args.audit_root)
    elif args.stage == 'pilot':
        result = pilot(args.repo, args.folder, args.max_minutes)
    elif args.stage == 'compare':
        result = compare(args.folder)
    elif args.stage == 'verify':
        result = verify(args.folder)[0]
    else:
        result = {'prepared': (args.folder/'ready.json').exists(),
                  'outputs': len(list((args.folder/'median-outputs').glob('*.json')))}
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
