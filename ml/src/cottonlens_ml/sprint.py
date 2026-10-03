"""Bounded, staged Colab research. R2 remains immutable; no fits outside Colab."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cottonlens_ml.checkpoints import (
    completed_checkpoint,
    identity_hash,
    save_completed_checkpoint,
)
from cottonlens_ml.code_identity import digest, source_identity
from cottonlens_ml.cohort import content_id, verify_cohort
from cottonlens_ml.config import FEATURE_NAMES, PipelinePaths
from cottonlens_ml.evaluation import evaluate, price_mae_metric
from cottonlens_ml.features import select_feature_rows
from cottonlens_ml.preflight import (
    experiment_directory,
    load_frozen_data,
    verify_target_contract,
)
from cottonlens_ml.protocol import experiment_identity
from cottonlens_ml.runtime_guard import require_colab_training
from cottonlens_ml.walkforward import summarize

VERSION = 'tabular-sprint-v1'
R2 = 'corrected-v2-day1-r2'
R2_READINESS = 'e153054a49965a3e176aedc4105280232dcda62ac81cfac5c9889cd31bb92b82'
DEVELOPMENT_ENDS = ('2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31')
COTTON = [n for n in FEATURE_NAMES if not n.startswith(('dxy_', 'wti_', 'cotton_dxy_', 'cotton_wti_', 'cotton_volume'))]
GROUPS = {'full': FEATURE_NAMES, 'cotton': COTTON,
          'cotton_macro': [n for n in FEATURE_NAMES if not n.startswith('cotton_volume')]}


def read_record(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    body = {k: v for k, v in value.items() if k != 'record_id'}
    if value.get('record_id') != content_id(body):
        raise ValueError(f'Corrupt completed record: {path}')
    return body


def freeze_record(path: Path, body: dict) -> dict:
    """Single writer; atomic publication, never replace completed evidence."""
    if path.exists():
        existing = read_record(path)
        if existing != body:
            raise ValueError(f'Frozen record changed: {path}')
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.pending')
    temporary.write_text(json.dumps({**body, 'record_id': content_id(body)}, indent=2, allow_nan=False), encoding='utf-8')
    temporary.rename(path)
    return body


@contextmanager
def writer(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.writer-lock'
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError('Another sprint writer or interrupted session exists; confirm it stopped before removing its lock') from exc
    owner = {'token': uuid.uuid4().hex, 'pid': os.getpid(), 'host': socket.gethostname(),
             'started_at': datetime.now(UTC).isoformat()}
    try:
        (lock / 'owner.json').write_text(json.dumps(owner), encoding='utf-8')
        yield
    finally:
        owner_path = lock / 'owner.json'
        if owner_path.exists() and json.loads(owner_path.read_text(encoding='utf-8')) == owner:
            owner_path.unlink()
            lock.rmdir()


def development_cohort(history: pd.DataFrame) -> list[dict]:
    available = select_feature_rows(history, FEATURE_NAMES)
    available = available.loc[available.target_date_5 < pd.Timestamp('2022-01-01')]
    result = []
    for number, end in enumerate(DEVELOPMENT_ENDS, 1):
        rows = available.loc[available.date <= pd.Timestamp(end)].tail(126)
        if len(rows) != 126 or rows.date.min().year != pd.Timestamp(end).year:
            raise ValueError('Development anchor lacks 126 common origins within its year')
        result.append({'fold': number, 'end_anchor': end,
                       'origins': rows.date.dt.strftime('%Y-%m-%d').tolist()})
    return result


def recipes() -> list[dict]:
    return [{'depth': depth, 'objective': objective, 'features': 'full', 'years': None}
            for depth in (1, 2, 3) for objective in ('reg:squarederror', 'reg:absoluteerror')]


def recipe_key(recipe: dict) -> tuple:
    return (recipe['depth'], len(GROUPS[recipe['features']]), content_id(recipe))


def fit_rows(history: pd.DataFrame, recipe: dict, cutoff) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    rows = select_feature_rows(history, GROUPS[recipe['features']])
    rows = rows.loc[(rows.date < cutoff) & (rows.target_date_5 < cutoff)]
    if recipe['years']:
        rows = rows.loc[rows.date >= cutoff - pd.DateOffset(years=recipe['years'])]
    if len(rows) < 126:
        raise ValueError('Insufficient past-only fitting history')
    return rows.copy()


def partition(history, recipe, cutoff):
    past = fit_rows(history, recipe, cutoff)
    validation = past.tail(126).copy()
    train = past.loc[(past.date < validation.date.min()) & (past.target_date_5 < validation.date.min())].copy()
    if len(train) < 126:
        raise ValueError('Insufficient purged training history')
    return train, validation


def price_weights(train: pd.DataFrame) -> np.ndarray:
    prices = train.cotton_close.to_numpy(dtype=float)
    if not len(prices) or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Weights require finite positive training prices')
    return prices / prices.mean()


def configuration(recipe, trees=1200):
    return {'n_estimators': trees, 'max_depth': recipe['depth'], 'learning_rate': 0.03,
            'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_weight': 20,
            'reg_lambda': 10, 'reg_alpha': 1, 'random_state': 42, 'n_jobs': 2,
            'objective': recipe['objective']}


def model_inputs(rows, recipe):
    """Keep the public 24-column order; unused inputs are constant zero.

    XGBoost cannot split on constants. The saved native booster therefore accepts
    the full schema without a custom Python runtime or a dimensionality change.
    Its embedded attribute records the ordered active columns.
    """
    result = rows[FEATURE_NAMES].copy()
    inactive = [n for n in FEATURE_NAMES if n not in GROUPS[recipe['features']]]
    result[inactive] = 0.0
    return result


def fit_model(train, validation, recipe, horizon, folder, *, trees=None, fresh=False):
    require_colab_training()
    import xgboost as xgb

    settings = {'sprint': VERSION, 'recipe': recipe, 'horizon': horizon, 'trees': trees,
                'phase': 'tuning' if trees is None else 'refit'}
    identity = experiment_identity(train, *([] if validation is None else [validation]), settings=settings)
    path = folder / f'{identity_hash(identity)}.json'
    model = xgb.XGBRegressor()
    if not fresh and completed_checkpoint(path, identity):
        model.load_model(path)
    else:
        config = configuration(recipe, trees or 1200)
        kwargs = {}
        if trees is None:
            config.update(early_stopping_rounds=50, eval_metric=price_mae_metric(validation.cotton_close.to_numpy()))
            kwargs['eval_set'] = [(model_inputs(validation, recipe), validation[f'target_return_{horizon}'])]
        if recipe['objective'] == 'reg:absoluteerror':
            kwargs['sample_weight'] = price_weights(train)
        model = xgb.XGBRegressor(**config)
        model.fit(model_inputs(train, recipe), train[f'target_return_{horizon}'], verbose=False, **kwargs)
        model.get_booster().set_attr(cottonlens_active_features=json.dumps(GROUPS[recipe['features']]))
        # Native 24-input runtime parity: inactive columns must have zero importance.
        inactive = set(FEATURE_NAMES) - set(GROUPS[recipe['features']])
        if inactive.intersection(model.get_booster().get_score()):
            raise ValueError('Inactive feature unexpectedly used by a tree')
        save_completed_checkpoint(path, identity, model.save_model)
    return model


def refit_schedule(history, test, cadence):
    if cadence not in (21, 126):
        raise ValueError('Only the predeclared 21/126 cadence comparison is allowed')
    start = int(test.cotton_session_index.iloc[0])
    buckets = ((test.cotton_session_index - start) // 21) if cadence == 21 else pd.Series(0, index=test.index)
    dates = history.set_index('cotton_session_index').date
    return [(pd.Timestamp(dates.loc[start + int(bucket) * 21]), test.loc[buckets.eq(bucket)].copy())
            for bucket in sorted(buckets.unique())]


def score_run(history, folds, recipe, horizon, cadence, folder, *, fresh=False):
    """Fit one recipe on four frozen blocks; settings never use audit outcomes."""
    started = time.monotonic()
    paired, fold_records = [], []
    for entry in folds:
        test = history.set_index('date', drop=False).loc[pd.to_datetime(entry['origins'])].reset_index(drop=True)
        train, validation = partition(history, recipe, test.date.min())
        tuning = fit_model(train, validation, recipe, horizon, folder, fresh=fresh)
        trees = int(tuning.best_iteration) + 1
        predictions, chunk_records = [], []
        for cutoff, chunk in refit_schedule(history, test, cadence):
            refit = fit_rows(history, recipe, cutoff)
            model = fit_model(refit, None, recipe, horizon, folder, trees=trees, fresh=fresh)
            predicted = model.predict(model_inputs(chunk, recipe)).astype(float)
            if not np.isfinite(predicted).all():
                raise ValueError('Non-finite prediction')
            predictions.extend(predicted.tolist())
            chunk_records.append({'cutoff': cutoff.isoformat(), 'fit_rows': len(refit),
                                  'last_label': refit.target_date_5.max().isoformat(), 'origins': len(chunk)})
        predicted = np.asarray(predictions)
        candidate = SimpleNamespace(name='XGBoost', horizon=horizon, predictions=predicted,
                                    parameters={'fit_cutoff': test.date.min().isoformat()})
        paired.append((test, candidate))
        metrics = evaluate(test.cotton_close.to_numpy(), test[f'target_return_{horizon}'].to_numpy(), predicted)
        naive = evaluate(test.cotton_close.to_numpy(), test[f'target_return_{horizon}'].to_numpy(), np.zeros(len(test)))
        natural = select_feature_rows(history, GROUPS[recipe['features']])
        natural = natural.loc[natural.date.between(test.date.min(), test.date.max())]
        fold_records.append({'fold': entry['fold'], 'metrics': metrics, 'naive': naive,
                             'coverage': {'natural_origins_in_scored_date_range': len(natural),
                                          'scored_common_origins': len(test),
                                          'additional_origins_not_scored': len(natural.loc[~natural.date.isin(test.date)])},
                             'trees': trees, 'validation_origins': validation.date.dt.strftime('%Y-%m-%d').tolist(),
                             'refits': chunk_records, 'predictions': predicted.tolist(), 'origins': entry['origins']})
        print(f"T+{horizon} depth={recipe['depth']} {recipe['objective']} {recipe['features']} cadence={cadence} fold={entry['fold']} MAE={metrics['mae']:.5f}", flush=True)
        if entry['fold'] == 1:
            print(f'Pilot timing: first fold {time.monotonic() - started:.1f}s including checkpoint IO', flush=True)
    aggregate = summarize(paired)
    # Per-origin cutoff must reflect each scheduled refit, not the block start.
    for record in aggregate['origin_records']:
        chunks = fold_records[record['fold'] - 1]['refits']
        record['fit_cutoff'] = max(c['cutoff'] for c in chunks if c['cutoff'] <= record['origin'])
    return {'recipe': recipe, 'horizon': horizon, 'cadence': cadence,
            'aggregate': aggregate, 'folds': fold_records,
            'fold_wins': sum(f['metrics']['mae'] < f['naive']['mae'] for f in fold_records),
            'elapsed_seconds': time.monotonic() - started}


def baseline_runs(history, folds, horizon):
    require_colab_training()
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    collected = {name: [] for name in ('Naive', 'Ridge', 'MedianReturn', 'PastMajorityDirection')}
    for entry in folds:
        test = history.set_index('date', drop=False).loc[pd.to_datetime(entry['origins'])].reset_index(drop=True)
        past = fit_rows(history, recipes()[0], test.date.min())
        y = past[f'target_return_{horizon}'].to_numpy()
        scaler = StandardScaler().fit(past[FEATURE_NAMES])
        ridge = Ridge(alpha=1.0).fit(scaler.transform(past[FEATURE_NAMES]), y)
        predictions = {'Naive': np.zeros(len(test)),
                       'Ridge': ridge.predict(scaler.transform(test[FEATURE_NAMES])),
                       'MedianReturn': np.full(len(test), np.median(y)),
                       'PastMajorityDirection': np.full(len(test), 1e-6 if (y > 0).sum() > (y < 0).sum() else -1e-6)}
        for name, predicted in predictions.items():
            collected[name].append((test, SimpleNamespace(name=name, horizon=horizon, predictions=predicted,
                                                          parameters={'fit_cutoff': test.date.min().isoformat()})))
    return {name: {**summarize(pairs), 'role': 'direction_only' if name == 'PastMajorityDirection' else 'reference_only'}
            for name, pairs in collected.items()}


def best_result(results):
    if not results:
        raise ValueError('No completed candidate')
    return min(results, key=lambda r: (r['aggregate']['mae'], *recipe_key(r['recipe'])))


def refinement_allowed(result):
    return result['aggregate']['relative_mae_improvement_pct'] > 0 and result['fold_wins'] >= 3


def cadence_wins(old, new):
    return (new['aggregate']['mae'] <= old['aggregate']['mae'] * .99
            and new['aggregate']['directional_accuracy'] >= old['aggregate']['directional_accuracy']
            and sum(b['metrics']['mae'] < a['metrics']['mae'] for a, b in zip(old['folds'], new['folds'], strict=True)) >= 3)


def gate(result):
    aggregate = result['aggregate']
    passes = (aggregate['relative_mae_improvement_pct'] >= 5 and result['fold_wins'] >= 3
              and aggregate['directional_accuracy'] >= (53 if result['horizon'] == 1 else 55))
    return {'selected': 'XGBoost' if passes else 'Naive', 'passes': passes,
            'mae_improvement_pct': aggregate['relative_mae_improvement_pct'],
            'directional_accuracy': aggregate['directional_accuracy'], 'fold_wins': result['fold_wins'],
            'evidence_role': 'seen_historical_development', 'audit_role': 'descriptive_only'}


def prepare(repo, root, name, smoke_path):
    code = source_identity(repo)
    smoke = json.loads(smoke_path.read_text(encoding='utf-8'))
    if smoke.get('status') != 'passed' or smoke.get('source_id') != code['source_id'] or smoke['environment']['lock_sha256'] != digest(repo / 'ml/uv.lock'):
        raise ValueError('Pass smoke for this exact source and lock first')
    old = json.loads((root / 'experiments' / R2 / 'ready.json').read_text(encoding='utf-8'))
    old_payload = {k: v for k, v in old.items() if k != 'readiness_id'}
    if old['readiness_id'] != R2_READINESS or content_id(old_payload) != R2_READINESS:
        raise ValueError('Expected frozen R2 readiness evidence')
    _, history = load_frozen_data(old['data_identity'])
    verify_target_contract(history)
    verify_cohort(old['cohort_identity'], history, old['data_identity']['data_id'])
    directory = experiment_directory(PipelinePaths(root), name)
    if name == R2:
        raise ValueError('Never write into the R2 experiment')
    with writer(directory):
        existing = read_record(directory / 'ready.json') if (directory / 'ready.json').exists() else None
        payload = {'protocol': VERSION, 'code_identity': code, 'data_identity': old['data_identity'],
                   'benchmark_cohort': old['cohort_identity'], 'development_folds': development_cohort(history),
                   'r2_readiness_id': R2_READINESS, 'smoke_sha256': digest(smoke_path), 'smoke_path': str(smoke_path),
                   'created_at': existing['created_at'] if existing else datetime.now(UTC).isoformat(),
                   'search_deadline_hours': 48, 'recipes': recipes(), 'seed': 42}
        freeze_record(directory / 'ready.json', payload)
    print(json.dumps({'status': 'ready', 'experiment': name, 'development_origins': 504,
                      'benchmark_origins': 504, 'new_training_started': False}, indent=2))


def load_ready(repo, root, name):
    directory = experiment_directory(PipelinePaths(root), name)
    ready = read_record(directory / 'ready.json')
    if ready['protocol'] != VERSION or ready['code_identity'] != source_identity(repo):
        raise ValueError('Sprint source changed; retain results and use a new experiment')
    if digest(Path(ready['smoke_path'])) != ready['smoke_sha256']:
        raise ValueError('Sprint smoke evidence changed')
    _, history = load_frozen_data(ready['data_identity'])
    verify_target_contract(history)
    verify_cohort(ready['benchmark_cohort'], history, ready['data_identity']['data_id'])
    return directory, ready, history


def check_search_open(directory, ready):
    if (directory / 'locked.json').exists():
        raise ValueError('Search closed: candidate already locked')
    if (datetime.now(UTC) - datetime.fromisoformat(ready['created_at'])).total_seconds() >= 48 * 3600:
        raise ValueError('48-hour search deadline reached; use lock with completed evidence')


def run_stage(repo, root, name, stage):
    require_colab_training()
    directory, ready, history = load_ready(repo, root, name)
    # Limit all discovery to pre-2022 rows, even though the snapshot contains audit.
    development = history.loc[history.date < '2022-01-01'].copy()
    with writer(directory):
        if stage in ('search', 'refine', 'cadence'):
            check_search_open(directory, ready)
        if stage in ('search', 'refine') and any(directory.glob('cadence-t*.json')):
            raise ValueError('Recipe search closed once cadence comparison starts')
        if stage == 'benchmark' and not (directory / 'locked.json').is_file():
            raise ValueError('Lock before accessing benchmark')

        def trial(recipe, horizon, cadence=126, *, benchmark=False, fresh=False):
            key = content_id({'recipe': recipe, 'horizon': horizon, 'cadence': cadence})
            group = 'reproduction' if fresh else ('benchmark' if benchmark else 'development')
            path = directory / group / f'{key}.json'
            if path.exists():
                return read_record(path)
            if not benchmark:
                check_search_open(directory, ready)
            result = score_run(history if benchmark else development,
                               ready['benchmark_cohort']['folds'] if benchmark else ready['development_folds'],
                               recipe, horizon, cadence, directory / 'checkpoints' / group, fresh=fresh)
            result['readiness_id'] = content_id(ready)
            freeze_record(path, result)
            return result

        def initial(h):
            results = []
            for recipe in recipes():
                key = content_id({'recipe': recipe, 'horizon': h, 'cadence': 126})
                path = directory / 'development' / f'{key}.json'
                if path.exists():
                    results.append(read_record(path))
            return results

        if stage == 'search':
            # First recipe/fold logs provide the measured pilot, not a full pipeline rerun.
            for h in (1, 5):
                baseline_path = directory / f'baselines-t{h}.json'
                if not baseline_path.exists():
                    freeze_record(baseline_path, baseline_runs(development, ready['development_folds'], h))
                for recipe in recipes():
                    trial(recipe, h)
        elif stage == 'refine':
            for h in (1, 5):
                results = initial(h)
                if len(results) != 6:
                    raise ValueError('Complete six initial recipes before refinement')
                best = best_result(results)
                if refinement_allowed(best):
                    base = best['recipe']
                    for change in ({'years': 5}, {'features': 'cotton'}, {'features': 'cotton_macro'}):
                        trial({**base, **change}, h)
                else:
                    freeze_record(directory / f'refinement-skipped-t{h}.json', {'reason': 'initial_gain_or_fold_gate_failed'})
        elif stage in ('cadence', 'lock'):
            locked = {}
            for h in (1, 5):
                if not initial(h):
                    raise ValueError('At least one completed initial recipe per horizon required')
                results = [read_record(p) for p in (directory / 'development').glob('*.json')]
                best = best_result([r for r in results if r['horizon'] == h and r['cadence'] == 126])
                cadence_path = directory / f'cadence-t{h}.json'
                if stage == 'cadence' and not cadence_path.exists():
                    challenger = trial(best['recipe'], h, 21)
                    freeze_record(cadence_path, {'recipe': best['recipe'], 'selected': 21 if cadence_wins(best, challenger) else 126,
                                                 'rationale': '21 is an unvalidated approximately-monthly hypothesis; 1%/3-fold/nonworse-direction rule'})
                cadence = read_record(cadence_path) if cadence_path.exists() else {'recipe': best['recipe'], 'selected': 126}
                if cadence['recipe'] != best['recipe']:
                    raise ValueError('Recipe changed after cadence decision')
                locked[str(h)] = {'recipe': best['recipe'], 'cadence': cadence['selected'],
                                  'initial_completed': len(initial(h)), 'development_mae': best['aggregate']['mae']}
            if stage == 'lock':
                freeze_record(directory / 'locked.json', {'readiness_id': content_id(ready), 'horizons': locked,
                              'policy': 'one candidate per horizon; benchmark cannot reopen search'})
        elif stage in ('benchmark', 'reproduce'):
            locked = read_record(directory / 'locked.json')
            decisions = {}
            for h in (1, 5):
                chosen = locked['horizons'][str(h)]
                if stage == 'reproduce' and not (directory / 'selection.json').exists():
                    raise ValueError('Complete benchmark before independent reproduction')
                result = trial(chosen['recipe'], h, chosen['cadence'], benchmark=True, fresh=stage == 'reproduce')
                decisions[str(h)] = gate(result)
                if stage == 'reproduce':
                    key = content_id({'recipe': chosen['recipe'], 'horizon': h, 'cadence': chosen['cadence']})
                    original = read_record(directory / 'benchmark' / f'{key}.json')
                    delta = max(float(np.max(np.abs(np.asarray(a['predictions']) - b['predictions'])))
                                for a, b in zip(original['folds'], result['folds'], strict=True))
                    freeze_record(directory / f'reproduction-t{h}.json', {'max_abs_log_return_difference': delta,
                                  'status': 'passed' if delta <= 1e-6 else 'failed', 'fresh_fits': True})
                    if delta > 1e-6:
                        raise ValueError('Reproduction exceeds 1e-6; no verified release allowed')
            if stage == 'benchmark':
                freeze_record(directory / 'selection.json', {'horizons': decisions, 'audit_used': False})
        elif stage == 'release':
            release_stage(directory, ready, history)
        else:
            raise ValueError(f'Unknown stage: {stage}')
    write_report(directory)


def release_stage(directory, ready, history):
    """Export only after benchmark selection and a genuinely fresh reproduction."""
    from cottonlens_ml.export import export_release
    from cottonlens_ml.training import Candidate, _fit_metadata
    from cottonlens_ml.walkforward import audit_split

    pending = directory / 'release.json'
    if pending.exists():
        value = read_record(pending)
        if digest(Path(value['path'])) != value['zip_sha256']:
            raise ValueError('Pending release changed')
        print('Pending release:', value['path'], flush=True)
        return
    locked = read_record(directory / 'locked.json')
    decisions = read_record(directory / 'selection.json')['horizons']
    for h in (1, 5):
        receipt = read_record(directory / f'reproduction-t{h}.json')
        if receipt['status'] != 'passed' or receipt['fresh_fits'] is not True:
            raise ValueError('Fresh reproduction must pass before release')
    market, _ = load_frozen_data(ready['data_identity'])
    modeling = select_feature_rows(history, FEATURE_NAMES)
    audit = audit_split(modeling)['test']
    walk = {'protocol': VERSION, 'evaluation_status': 'seen_historical_development',
            'folds': [], 'aggregate': {}, 'feature_ablation': [], 'feature_groups': GROUPS,
            'cftc_candidate': 'excluded: actual publication times unverified',
            'cadence_by_horizon': {h: c['cadence'] for h, c in locked['horizons'].items()}}
    all_candidates, deployed, selected, deployment_selected, selection_audit = [], [], {}, {}, {}
    for h in (1, 5):
        chosen = locked['horizons'][str(h)]
        recipe = chosen['recipe']
        key = content_id({'recipe': recipe, 'horizon': h, 'cadence': chosen['cadence']})
        benchmark = read_record(directory / 'benchmark' / f'{key}.json')
        trees = benchmark['folds'][-1]['trees']
        walk['aggregate'][f'XGBoost-T+{h}'] = benchmark['aggregate']
        naive_pairs = []
        for i, entry in enumerate(ready['benchmark_cohort']['folds']):
            rows = history.set_index('date', drop=False).loc[pd.to_datetime(entry['origins'])].reset_index(drop=True)
            train, validation = partition(history, recipe, rows.date.min())
            if len(walk['folds']) <= i:
                walk['folds'].append({'fold': i + 1, 'train_end': str(train.date.max().date()),
                    'validation_end': str(validation.date.max().date()), 'test_start': entry['origins'][0],
                    'test_end': entry['origins'][-1], 'sample_count': 126, 'metrics': {}, 'experiments': {}})
            fold = benchmark['folds'][i]
            walk['folds'][i]['metrics'].update({f'XGBoost-T+{h}': fold['metrics'], f'Naive-T+{h}': fold['naive']})
            walk['folds'][i]['experiments'][f'XGBoost-T+{h}'] = {
                'recipe': recipe, 'trees': fold['trees'], 'refits': fold['refits'],
                'validation_origins': fold['validation_origins'], 'trials': []}
            naive_pairs.append((rows, SimpleNamespace(name='Naive', horizon=h, predictions=np.zeros(len(rows)), parameters={})))
        walk['aggregate'][f'Naive-T+{h}'] = summarize(naive_pairs)

        def candidate_at(cutoff, role, test, recipe=recipe, h=h, trees=trees, chosen=chosen):
            refit = fit_rows(history, recipe, cutoff)
            model = fit_model(refit, None, recipe, h, directory / 'checkpoints' / role, trees=trees)
            config = {'recipe': recipe, 'trees': trees, 'active_features': GROUPS[recipe['features']],
                      'input_adapter': '24_columns_inactive_constant_during_fit_no_tree_splits', 'cadence': chosen['cadence']}
            identity = experiment_identity(refit, settings={'config': config, 'role': role, 'horizon': h, 'cutoff': str(cutoff)})
            predicted = model.predict(model_inputs(test, recipe)) if len(test) else np.zeros(0)
            scores = evaluate(test.cotton_close.to_numpy(), test[f'target_return_{h}'].to_numpy(), predicted) if len(test) else {}
            return Candidate('XGBoost', h, model, predicted, scores,
                             parameters=_fit_metadata(refit, cutoff, identity, config, role=role))

        tree = candidate_at(audit.date.min(), 'evaluation_backtest', audit)
        naive = Candidate('Naive', h, None, np.zeros(len(audit)),
                          evaluate(audit.cotton_close.to_numpy(), audit[f'target_return_{h}'].to_numpy(), np.zeros(len(audit))),
                          parameters={'model_role': 'evaluation_backtest', 'fit_cutoff': audit.date.min().isoformat()})
        live_tree = candidate_at(history.date.max(), 'deployment_live', audit.iloc[:0])
        live_naive = Candidate('Naive', h, None, np.zeros(0), {},
                               parameters={'model_role': 'deployment_live', 'fit_cutoff': history.date.max().isoformat()})
        all_candidates.extend([naive, tree])
        deployed.extend([live_naive, live_tree])
        name = decisions[str(h)]['selected']
        selected[h] = tree if name == 'XGBoost' else naive
        deployment_selected[h] = live_tree if name == 'XGBoost' else live_naive
        selection_audit[h] = {'locked_candidate': name, 'selected': name,
                             'fold_wins_vs_naive': {'XGBoost': benchmark['fold_wins']},
                             'xgboost_walkforward_gate': decisions[str(h)],
                             'lstm_walkforward_gate': {'status': 'not_in_sprint'}, 'lstm_beats_xgboost': False,
                             'historical_audit': {'Naive': naive.metrics, 'XGBoost': tree.metrics}}
    smoke = json.loads(Path(ready['smoke_path']).read_text(encoding='utf-8'))
    evidence = {'code_identity': ready['code_identity'], 'data_identity': ready['data_identity'],
                'cohort_identity': ready['benchmark_cohort'], 'data_quality': history.attrs['data_quality'],
                'environment_smoke': smoke, 'protocol_identity': {'version': VERSION, 'locked': locked,
                    'reproduction': {str(h): read_record(directory / f'reproduction-t{h}.json') for h in (1, 5)},
                    'audit_role': 'seen_historical_audit_descriptive_only_no_selection_or_veto'}}
    release_root = directory / 'artifacts/releases'
    release_root.mkdir(parents=True, exist_ok=True)
    release = export_release(release_root, history, audit, market, all_candidates, selected, selection_audit, walk,
                             deployment_selected=deployment_selected, deployment_candidates=deployed, evidence_identity=evidence)
    freeze_record(pending, {'path': str(release), 'zip_sha256': digest(release), 'status': 'pending_runtime_validation'})
    print('Pending release:', release, flush=True)


def write_report(directory):
    lines = ['CottonLens staged tabular research', 'Historical development evidence; no independent holdout claim.',
             '2024+ audit is excluded from search and selection.', 'Positive gain means lower MAE than Naive.', '']
    for h in (1, 5):
        path = directory / f'baselines-t{h}.json'
        if path.exists():
            lines.append(f'T+{h} DEVELOPMENT REFERENCES')
            for name, a in read_record(path).items():
                direction = 'not applicable (flat persistence)' if name == 'Naive' else f"{a['directional_accuracy']:.3f}%"
                lines.append(f"{name}: MAE={a['mae']:.5f} direction={direction} role={a['role']}")
    for group in ('development', 'benchmark'):
        lines.append(group.upper())
        for path in sorted((directory / group).glob('*.json')):
            r = read_record(path)
            a = r['aggregate']
            lines.append(f"T+{r['horizon']} {r['recipe']} cadence={r['cadence']} MAE={a['mae']:.5f} gain={a['relative_mae_improvement_pct']:.3f}% direction={a['directional_accuracy']:.3f}% wins={r['fold_wins']}/4 seconds={r['elapsed_seconds']:.1f}")
            for fold in r['folds']:
                lines.append(f"  Fold {fold['fold']}: MAE={fold['metrics']['mae']:.5f} Naive={fold['naive']['mae']:.5f} direction={fold['metrics']['directional_accuracy']:.3f}% trees={fold['trees']} coverage={fold.get('coverage', {})}")
    for name in ('locked.json', 'selection.json', 'reproduction-t1.json', 'reproduction-t5.json'):
        if (directory / name).exists():
            lines.extend(['', name, json.dumps(read_record(directory / name), indent=2)])
    report = '\n'.join(lines) + '\n'
    path = directory / ('summary-' + content_id({'text': report})[:16] + '.txt')
    if not path.exists():
        path.write_text(report, encoding='utf-8')
    print(report, flush=True)
    print('REPORT:', path, flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--drive-root', type=Path, required=True)
    parser.add_argument('--experiment', default='tabular-sprint-v1')
    parser.add_argument('--stage', choices=('prepare', 'search', 'refine', 'cadence', 'lock', 'benchmark', 'reproduce', 'release', 'report'), required=True)
    parser.add_argument('--smoke-report', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not args.smoke_report:
            parser.error('prepare requires --smoke-report')
        prepare(args.repo, args.drive_root, args.experiment, args.smoke_report)
    elif args.stage == 'report':
        directory, _, _ = load_ready(args.repo, args.drive_root, args.experiment)
        write_report(directory)
    else:
        directory = experiment_directory(PipelinePaths(args.drive_root), args.experiment)
        started = datetime.now(UTC).isoformat()
        try:
            run_stage(args.repo, args.drive_root, args.experiment, args.stage)
        except Exception as exc:
            if args.experiment != R2 and (directory / 'ready.json').exists():
                freeze_record(directory / 'attempts' / f'{uuid.uuid4().hex}.json', {
                    'stage': args.stage, 'started_at': started, 'status': 'failed',
                    'error_type': type(exc).__name__, 'traceback': traceback.format_exc()})
            raise
        else:
            freeze_record(directory / 'attempts' / f'{uuid.uuid4().hex}.json', {
                'stage': args.stage, 'started_at': started, 'status': 'completed'})


if __name__ == '__main__':
    main()
