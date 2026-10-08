"""Bounded matched Ridge/return-path stages on the existing Experiment/Ledger."""
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.full_year import (
    SHRINKAGE,
    chunks,
    inner_price,
    json_predictions,
    predict_chunks,
)
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.return_path import (
    PATH_COLUMNS,
    PATH_FEATURES,
    add_return_path,
)
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.statistics import bh_adjust, paired_bootstrap

PROFILE = 'return-path-pilot-v1'
GROUPS = ('base', 'base_plus_return_path')


def recipe(design, group, h):
    settings = design['recipes']
    return {'family': settings['family'], 'params': {'alpha': settings['alpha']},
        'horizon': h, 'task': 'price', 'device': 'cpu', 'seed': settings['seed'],
        'target': settings['target'], 'years': settings['years'], 'cadence': settings['cadence'],
        'window': settings['window'], 'features': design['groups'][group]}


def validate_design(registration):
    design = registration['design']
    if (registration['design_id'] != content_id(design) or design['profile'] != PROFILE
            or design['groups'] != {'base': list(FEATURE_NAMES), 'base_plus_return_path': PATH_FEATURES}
            or design['new_columns'] != PATH_COLUMNS
            or design['recipes'] != {'family': 'ridge', 'alpha': 10., 'target': 'scaled_log',
                'years': None, 'seed': 42, 'cadence': 21, 'window': 1}
            or design['shrinkage_weights'] != list(SHRINKAGE)
            or design['split']['purge_observations'] != 5
            or design['split']['refit_cadence'] != 21
            or [f['year'] for f in design['split']['folds']] != list(range(2016, 2024))
            or design['price_gate'] != {'gain_pct': 5., 'direction_pct': {'1': 53., '5': 55.}, 'year_wins': 6}
            or design['automatic_release'] is not False or design['audit_2024_used'] is not False):
        raise ValueError('Frozen matched return-path design changed')
    return design


def prepare(repo, folder, reference, *, profile=PROFILE, validate_fn=validate_design,
            registration_name='return-path-preregistration.json', device='cpu'):
    registration = read_record(reference.parent / registration_name)
    design = validate_fn(registration)
    old, history = verify_history(reference)
    if (digest(reference / 'ready.json') != design['reference_ready_sha256']
            or digest(reference / 'history.parquet') != design['history_sha256']
            or old['identity']['split'] != design['split']
            or digest(repo / 'ml/src/cottonlens_ml/research/return_path.py') != design['feature_helper_sha256']):
        raise ValueError('Frozen reference or feature implementation changed')
    expanded = add_return_path(history)
    if frame_identity(expanded, list(expanded.columns)) != design['feature_data_id']:
        raise ValueError('Frozen feature data changed')
    budget = 4 * (sum(len(chunks(history, b['origins'])) for f in design['split']['folds'] for b in f['inner'])
        + sum(len(chunks(history, f['origins'])) for f in design['split']['folds']))
    if design['recipes']['family'] == 'xgboost':
        budget += 4 * sum(len(f['inner']) for f in design['split']['folds'])
    if budget != design['fit_budget']['total'] or design['fit_budget']['annual_outputs'] != 32:
        raise ValueError('Frozen work budget changed')
    identity = {'profile': profile, 'source_id': research_source_identity(repo)['source_id'],
        'design_id': registration['design_id'], 'design': design, 'split': design['split'],
        'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in
            ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost')},
        'lock_sha256': digest(repo / 'ml/uv.lock'), 'execution': {'device': device, 'threads': 2}}
    with writer(folder):
        if (folder / 'ready.json').exists():
            ready = read_record(folder / 'ready.json')
            if ready['identity'] != identity or digest(folder / 'history.parquet') != ready['history_sha256']:
                raise ValueError('Frozen path code/environment/data changed; preserve the experiment')
            return ready
        copy_immutable(reference / 'ready.json', folder / 'reference/ready.json')
        copy_immutable(reference / 'history.parquet', folder / 'reference/history.parquet')
        pending = folder / 'history.pending.parquet'
        expanded.to_parquet(pending, index=False)
        copy_immutable(pending, folder / 'history.parquet')
        pending.unlink()
        freeze_record(folder / 'preregistered.json', registration)
        ready = {'identity': identity, 'history_sha256': digest(folder / 'history.parquet'),
            'created_at': datetime.now(UTC).isoformat()}
        freeze_record(folder / 'ready.json', ready)
    return ready


def selected_decision(path, spec):
    decision = read_record(path)
    chosen = decision['selected']
    options = spec if isinstance(spec,list) else [spec]
    if (decision['selection_used_outer'] is not False or chosen['recipe'] not in options
            or chosen['weight'] not in SHRINKAGE or type(chosen['iterations']) is not int
            or not 1 <= chosen['iterations'] <= chosen['recipe'].get('max_iterations',1)
            or not np.isfinite(chosen['inner_score'])):
        raise ValueError('Frozen path selection changed; no silent retraining')
    if isinstance(spec,list):
        candidates = decision['candidates']
        if [c['recipe'] for c in candidates] != options:
            raise ValueError('Frozen statistical candidates changed')
        completed = []
        for candidate in candidates:
            if candidate['status']=='complete':
                result = candidate['selected']
                if (result['recipe']!=candidate['recipe'] or result['iterations']!=1
                        or result['weight'] not in SHRINKAGE or not np.isfinite(result['inner_score'])):
                    raise ValueError('Invalid completed statistical candidate')
                completed.append(result)
            elif candidate['status']!='numerical_failure' or candidate['recipe']['family']!='arima':
                raise ValueError('Incomplete candidate cannot be selected')
        if not completed or min(completed,key=lambda c:c['inner_score']) != chosen:
            raise ValueError('Statistical choice differs from past-only score/tie break')
    return chosen


def output(folder, name, fold, group, h, design, history, *, namespace='path', recipe_fn=recipe):
    record = read_record(folder / f'{namespace}-outputs' / name)
    path = folder / f'{namespace}-decisions' / name
    chosen = selected_decision(path, recipe_fn(design, group, h))
    frame = pd.DataFrame(record['records'])
    actual = rows_at(history, fold['origins'])
    fields = ['cotton_close', f'target_return_{h}', 'predicted_return', 'raw_predicted_return',
        'median_return', 'past_majority_sign']
    if (record['year'] != fold['year'] or record['horizon'] != h or record['group'] != group
            or record['design_id'] != content_id(design) or record['decision_sha256'] != digest(path)
            or record['recipe'] != chosen['recipe'] or record['weight'] != chosen['weight']
            or record['inner_score'] != chosen['inner_score'] or frame.date.tolist() != fold['origins']
            or not np.isfinite(frame[fields]).all().all()
            or not np.array_equal(frame.cotton_close, actual.cotton_close)
            or not np.array_equal(frame[f'target_return_{h}'], actual[f'target_return_{h}'])
            or not np.array_equal(frame.predicted_return, chosen['weight'] * frame.raw_predicted_return)):
        raise ValueError('Frozen path output changed; no silent retraining')
    return record, frame


def run(experiment, max_minutes, *, group_names=GROUPS, namespace='path', recipe_fn=recipe, validate_fn=validate_design,
        selection_fn=None, record_frame_fn=None, output_fn=output, before_fit=None, horizons=(1, 5)):
    if not 0 < max_minutes <= 60:
        raise ValueError('Path session budget must be positive and at most 60 minutes')
    design = experiment.identity['design']
    validate_fn({'design_id': experiment.identity['design_id'], 'design': design})
    deadline = time.monotonic() + max_minutes * 60

    def before():
        if time.monotonic() >= deadline:
            raise FitBudgetReached('Planned path pause')
        if before_fit is not None:
            before_fit()

    experiment.before_compute = before
    try:
        for fold in experiment.identity['split']['folds']:
            for h in horizons:
                for group in group_names:
                    name = f'{group}-t{h}-year{fold["year"]}.json'
                    marker = experiment.root / f'{namespace}-outputs' / name
                    if marker.exists():
                        output_fn(experiment.root, name, fold, group, h, design, experiment.history,
                            namespace=namespace, recipe_fn=recipe_fn)
                        print(f'CACHE {namespace} {group} {fold["year"]} T+{h}; no new fits', flush=True)
                        continue
                    spec = recipe_fn(design, group, h)
                    path = experiment.root / f'{namespace}-decisions' / name
                    print(f'STAGE {namespace} {group} year={fold["year"]} T+{h}', flush=True)
                    if not path.exists():
                        selection = ({'selected':inner_price(experiment,spec,fold)} if selection_fn is None
                            else selection_fn(experiment,spec,fold))
                        freeze_record(path, {**selection, 'selection_used_outer': False})
                    chosen = selected_decision(path, spec)
                    spec = chosen['recipe']
                    test = rows_at(experiment.history, fold['origins']).copy()
                    raw = predict_chunks(experiment, spec, fold['origins'], f'path-outer-{group}-{fold["year"]}',chosen['iterations'])
                    test['raw_predicted_return'], test['predicted_return'] = raw, chosen['weight'] * raw
                    test['median_return'], test['past_majority_sign'] = np.nan, np.nan
                    for block in chunks(experiment.history, fold['origins']):
                        past = experiment.train_rows(block.date.min(), spec)
                        if 'asset' in past:
                            past = past.loc[past.asset.eq('cotton')]
                        signs, counts = np.unique(np.sign(past[f'target_return_{h}']), return_counts=True)
                        mask = test.date.isin(block.date)
                        test.loc[mask, 'median_return'] = float(past[f'target_return_{h}'].median())
                        test.loc[mask, 'past_majority_sign'] = float(signs[np.argmax(counts)])
                    test['date'] = test.date.dt.strftime('%Y-%m-%d')
                    payload = test[['date', 'cotton_close', f'target_return_{h}',
                        'predicted_return', 'raw_predicted_return', 'median_return', 'past_majority_sign']].copy()
                    if record_frame_fn is not None:
                        payload = record_frame_fn(test, payload, group, h, chosen)
                    freeze_record(marker, {'year': fold['year'], 'horizon': h, 'group': group,
                        'design_id': experiment.identity['design_id'], 'decision_sha256': digest(path),
                        'recipe': spec, 'weight': chosen['weight'], 'inner_score': chosen['inner_score'],
                        'records': json_predictions(payload)})
                    if getattr(experiment, 'mirror', None):
                        experiment.mirror.publish_metadata([marker.relative_to(experiment.root).as_posix(),
                            path.relative_to(experiment.root).as_posix()])
                    print(f'SAVED {namespace} {group} {fold["year"]} T+{h}', flush=True)
    except FitBudgetReached:
        print('PILOT planned_pause: path checkpoint preserved; resume same profile', flush=True)
        return {'status': 'planned_pause', 'saved_outputs': len(list((experiment.root / f'{namespace}-outputs').glob('*.json')))}
    return {'status': 'complete', 'saved_outputs': len(group_names) * len(horizons) * len(experiment.identity['split']['folds'])}


def compare(folder, repetitions=10000, *, group_names=GROUPS, namespace='path', recipe_fn=recipe,
            validate_fn=validate_design, hypothesis_prefix='L', scope='reused historical research; no independent holdout claim'):
    ready, history = verify_history(folder)
    design = validate_fn({'design_id': ready['identity']['design_id'], 'design': ready['identity']['design']})
    if len(list((folder / f'{namespace}-outputs').glob('*.json'))) != 32:
        return {'status': 'pending', 'required_outputs': 32}
    horizons, hypotheses = {}, {}
    for h in (1, 5):
        comparisons, inner, arms = [], [], {g: {'losses': [], 'direction': [], 'wins': 0, 'flat': 0} for g in group_names}
        for fold in design['split']['folds']:
            records, errors = {}, {}
            for group in group_names:
                name = f'{group}-t{h}-year{fold["year"]}.json'
                record, frame = output(folder, name, fold, group, h, design, history,
                    namespace=namespace, recipe_fn=recipe_fn)
                a, c, p = (frame[k].to_numpy() for k in (f'target_return_{h}', 'cotton_close', 'predicted_return'))
                naive = c * np.abs(np.expm1(a))
                errors[group] = c * np.abs(np.expm1(a) - np.expm1(p))
                arm = arms[group]
                arm['losses'].append(np.column_stack([naive, errors[group]]))
                arm['direction'].append(np.column_stack([np.sign(a) == np.sign(p), np.sign(a) == frame.past_majority_sign]).astype(float))
                arm['wins'] += int(errors[group].mean() < naive.mean())
                arm['flat'] += int(frame.predicted_return.eq(0).sum())
                records[group] = record
            comparisons.append(np.column_stack([errors[group_names[0]], errors[group_names[1]]]))
            inner.append((records[group_names[0]]['inner_score'], records[group_names[1]]['inner_score']))
        scores = {}
        for group, arm in arms.items():
            losses, directions = np.concatenate(arm['losses']), np.concatenate(arm['direction'])
            gain = float(100 * (1 - losses[:, 1].mean() / losses[:, 0].mean()))
            direction = float(100 * directions[:, 0].mean())
            scores[group] = {'count': len(losses), 'price_mae': float(losses[:, 1].mean()), 'naive_mae_gain_pct': gain,
                'direction_pct': direction, 'majority_direction_pct': float(100 * directions[:, 1].mean()),
                'flat_count': arm['flat'], 'fold_wins': arm['wins'],
                'price_thresholds_passed': bool(gain >= 5 and direction >= (53 if h == 1 else 55) and arm['wins'] >= 6),
                'versus_naive': {str(b): paired_bootstrap(arm['losses'], block=b, repetitions=repetitions,
                    normalization=float(losses[:, 0].mean())) for b in (10, 20, 40)},
                'direction_vs_majority': paired_bootstrap(arm['direction'], repetitions=repetitions)}
        ci = {str(b): paired_bootstrap(comparisons, block=b, repetitions=repetitions,
            normalization=scores[group_names[0]]['price_mae']) for b in (10, 20, 40)}
        inner_gain = float(100 * (1 - np.mean([i[1] for i in inner]) / np.mean([i[0] for i in inner])))
        inner_wins = sum(new < old for old, new in inner)
        challenger = scores[group_names[1]]
        horizons[str(h)] = {'arms': scores, 'inner_gain_vs_base_pct': inner_gain, 'inner_year_wins': inner_wins,
            'paired_vs_base': ci, 'research_priority_signal': bool(inner_gain >= .5 and inner_wins >= 5
                and challenger['naive_mae_gain_pct'] > 0 and challenger['fold_wins'] >= 5)}
        hypotheses[f'{hypothesis_prefix}{h}'] = {'p_two_sided': ci['20']['centered_bootstrap_p_two_sided']}
    for key, q in zip(hypotheses, bh_adjust([p['p_two_sided'] for p in hypotheses.values()]), strict=True):
        hypotheses[key]['bh_adjusted_p'] = q
    body = {'status': 'complete', 'scope': scope,
        'design_id': ready['identity']['design_id'], 'horizons': horizons, 'hypotheses': hypotheses,
        'release_allowed': False, 'ready_sha256': digest(folder / 'ready.json'),
        'output_sha256': {p.name: digest(p) for p in sorted((folder / f'{namespace}-outputs').glob('*.json'))}}
    freeze_record(folder / 'reports' / f'{namespace}-{content_id(body)[:16]}.json', body)
    return body


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    from cottonlens_ml.research.mirror import Mirror
    folder = root_path(args.drive_root, args.experiment)
    if args.stage not in ('status', 'prepare', 'pilot-plan', 'pilot', 'compare', 'report'):
        raise ValueError('Return-path supports bounded pilot/report stages; no automatic release')
    mirror = Mirror(folder, root_path(args.mirror_root, args.experiment)) if args.mirror_root else None
    if mirror:
        mirror.hydrate_metadata()
        if args.stage == 'pilot' and len(list((folder / 'path-outputs').glob('*.json'))) != 32:
            mirror.hydrate()
    if args.stage == 'prepare':
        if not args.reference_root:
            raise ValueError('Frozen return-path input packet required')
        ready = prepare(args.repo, folder, args.reference_root)
        if mirror:
            mirror.publish_metadata(['ready.json', 'history.parquet', 'preregistered.json',
                'reference/ready.json', 'reference/history.parquet'])
        result = {'status': 'prepared', 'design_id': ready['identity']['design_id'], 'fit_budget': ready['identity']['design']['fit_budget']}
    elif args.stage == 'status':
        result = {'prepared': (folder / 'ready.json').exists(), 'writer_lock_present': (folder / '.writer-lock').exists(),
            'saved_outputs': len(list((folder / 'path-outputs').glob('*.json'))), 'required_outputs': 32,
            'ledger': Ledger(folder / 'ledger', {}).summary(), 'model_payloads_verified_by_status': False}
    elif args.stage == 'pilot-plan':
        result = {'fit_budget': read_record(folder / 'ready.json')['identity']['design']['fit_budget'], 'required_outputs': 32}
    elif args.stage in ('compare', 'report'):
        result = compare(folder)
        if mirror and result['status'] == 'complete':
            mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder / 'reports').glob('*.json')])
    else:
        with writer(folder):
            experiment = Experiment(folder, repo=args.repo)
            if mirror:
                experiment.mirror, experiment.after_fit = mirror, mirror.fit
            result = run(experiment, args.max_minutes)
    print(json.dumps(result, indent=2))
