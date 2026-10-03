"""One fixed three-year price hypothesis using the existing Experiment/Ledger."""
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.full_year import (
    chunks,
    inner_price,
    json_predictions,
    predict_chunks,
    price_recipes,
)
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.statistics import bh_adjust, paired_bootstrap

PROFILE = 'recency-pilot-v1'


def validate_design(design):
    body = {k: v for k, v in design.items() if k != 'design_id'}
    if content_id(body) != design['design_id'] or design['profile'] != PROFILE:
        raise ValueError('Recency design identity changed')
    recipes = {str(h): [{**r, 'years': 3} for r in price_recipes(h)] for h in (1, 5)}
    if (design['recipes'] != recipes or design['control_retraining_allowed'] is not False
            or design['history_calendar_years'] != 3):
        raise ValueError('Only the preregistered three-year recipes are allowed')
    if (design['horizons'] != [1, 5] or len(design['split']['folds']) != 8
            or design['split']['purge_observations'] != 5
            or design['split']['refit_cadence'] != 21):
        raise ValueError('Frozen horizons/split/purge/cadence changed')


def verify_controls(folder, design):
    if digest(folder / 'ready.json') != design['control_ready_sha256']:
        raise ValueError('Full-history control readiness changed')
    ready, history = verify_history(folder)
    if ready['history_sha256'] != design['history_sha256'] or ready['identity']['split'] != design['split']:
        raise ValueError('Control snapshot or frozen origins changed')
    expected = {f't{h}-year{f["year"]}.json' for f in design['split']['folds'] for h in (1, 5)}
    if set(design['control_outputs_sha256']) != expected:
        raise ValueError('Exactly sixteen immutable controls required')
    for fold in design['split']['folds']:
        rows = history.set_index('date').loc[pd.to_datetime(fold['origins'])]
        for h in (1, 5):
            name = f't{h}-year{fold["year"]}.json'
            path = folder / 'pilot-full-year' / name
            if digest(path) != design['control_outputs_sha256'][name]:
                raise ValueError('Control prediction checksum changed')
            record = read_record(path)
            frame = pd.DataFrame(record['records'])
            decision = read_record(folder / 'decisions/full-year' / name)
            if (record['year'] != fold['year'] or record['horizon'] != h
                    or frame.date.tolist() != fold['origins']
                    or record['decision_id'] != content_id(decision)
                    or decision['price']['recipe'] not in price_recipes(h)
                    or not np.isfinite(frame.predicted_return).all()
                    or not np.array_equal(frame.cotton_close, rows.cotton_close)
                    or not np.array_equal(frame[f'target_return_{h}'], rows[f'target_return_{h}'])):
                raise ValueError('Control decisions, targets or origin order changed')
    return history


def copy_immutable(source, target):
    content = source.read_bytes()
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError('Conflicting preserved experiment input')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name('.' + target.name + '.pending')
    pending.write_bytes(content)
    pending.replace(target)


def completed_output(folder, name, fold, h, design):
    record = read_record(folder / 'recency-outputs' / name)
    path = folder / 'recency-decisions' / name
    chosen = read_record(path)['selected']
    frame = pd.DataFrame(record['records'])
    if (record['design_id'] != design['design_id'] or record['year'] != fold['year']
            or record['horizon'] != h or frame.date.tolist() != fold['origins']
            or record['decision_sha256'] != digest(path) or record['recipe'] != chosen['recipe']
            or record['weight'] != chosen['weight'] or record['inner_score'] != chosen['inner_score']
            or record['recipe'] not in design['recipes'][str(h)]
            or not np.isfinite(frame[['cotton_close', f'target_return_{h}',
                'raw_predicted_return', 'predicted_return', 'median_return', 'past_majority_sign']]).all().all()
            or not np.array_equal(frame.predicted_return, record['weight'] * frame.raw_predicted_return)):
        raise ValueError('Completed recency output or frozen decision changed; no silent retraining')
    return record, frame


def prepare(repo, folder, reference):
    design = json.loads((reference.parent / 'recency-preregistration.json').read_text())
    validate_design(design)
    history = verify_controls(reference, design)
    identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'],
        'design_id': design['design_id'], 'design': design, 'split': design['split'],
        'recipes': design['recipes'], 'python': sys.version.split()[0],
        'versions': {p: importlib.metadata.version(p) for p in
            ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost')},
        'lock_sha256': digest(repo / 'ml/uv.lock'), 'execution': {'device': 'cpu', 'threads': 2},
        'history_policy': 'three calendar years before each fit cutoff; available labels only'}
    actual_budget = 2 * (6 * sum(len(chunks(history, b['origins'])) for f in design['split']['folds']
        for b in f['inner']) + sum(len(f['inner']) for f in design['split']['folds'])
        + sum(len(chunks(history, f['origins'])) for f in design['split']['folds']))
    if actual_budget != design['fit_budget']['maximum_total_new_fits']:
        raise ValueError('Preregistered work budget disagrees with frozen origin ordinals')
    with writer(folder):
        if (folder / 'ready.json').exists():
            old = read_record(folder / 'ready.json')
            if old['identity'] != identity or digest(folder / 'history.parquet') != design['history_sha256']:
                raise ValueError('Frozen recency code/environment/data changed; preserve the old experiment')
            verify_controls(folder / 'controls', design)
            return old
        copy_immutable(reference / 'history.parquet', folder / 'history.parquet')
        for name in ['ready.json', 'history.parquet',
                *['pilot-full-year/' + n for n in design['control_outputs_sha256']],
                *['decisions/full-year/' + n for n in design['control_outputs_sha256']]]:
            copy_immutable(reference / name, folder / 'controls' / name)
        freeze_record(folder / 'preregistered.json', design)
        ready = {'identity': identity, 'history_sha256': design['history_sha256'],
            'created_at': datetime.now(UTC).isoformat()}
        freeze_record(folder / 'ready.json', ready)
        return ready


def run(experiment, max_minutes):
    if not 0 < max_minutes <= 240:
        raise ValueError('CPU session budget must be positive and no more than four hours')
    design = experiment.identity['design']
    verify_controls(experiment.root / 'controls', design)
    deadline = time.monotonic() + max_minutes * 60

    def before():
        if time.monotonic() >= deadline:
            raise FitBudgetReached('Planned recency pause; resume the same experiment')

    experiment.before_compute = before
    try:
        for fold in experiment.identity['split']['folds']:
            for h in (1, 5):
                name = f't{h}-year{fold["year"]}.json'
                marker = experiment.root / 'recency-outputs' / name
                if marker.exists():
                    completed_output(experiment.root, name, fold, h, design)
                    print(f'CACHE recency {fold["year"]} T+{h}; no new fits', flush=True)
                    continue
                decision_path = experiment.root / 'recency-decisions' / name
                print(f'STAGE recency year={fold["year"]} T+{h} history=3y', flush=True)
                if decision_path.exists():
                    chosen = read_record(decision_path)
                else:
                    choices = [inner_price(experiment, spec, fold) for spec in experiment.identity['recipes'][str(h)]]
                    selected = min(enumerate(choices), key=lambda p: (p[1]['inner_score'], p[0]))[1]
                    chosen = {'selected': selected, 'all_candidates': choices, 'selection_used_outer': False}
                    freeze_record(decision_path, chosen)
                selected = chosen['selected']
                test = rows_at(experiment.history, fold['origins']).copy()
                raw = predict_chunks(experiment, selected['recipe'], fold['origins'],
                    f'recency-outer-{fold["year"]}', selected['iterations'])
                test['raw_predicted_return'], test['predicted_return'] = raw, selected['weight'] * raw
                test['median_return'], test['past_majority_sign'] = np.nan, np.nan
                for block in chunks(experiment.history, fold['origins']):
                    past = experiment.train_rows(block.date.min(), selected['recipe'])
                    signs, counts = np.unique(np.sign(past[f'target_return_{h}']), return_counts=True)
                    mask = test.date.isin(block.date)
                    test.loc[mask, 'median_return'] = float(past[f'target_return_{h}'].median())
                    test.loc[mask, 'past_majority_sign'] = float(signs[np.argmax(counts)])
                test['date'] = test.date.dt.strftime('%Y-%m-%d')
                freeze_record(marker, {'year': fold['year'], 'horizon': h, 'design_id': design['design_id'],
                    'decision_sha256': digest(decision_path), 'inner_score': selected['inner_score'],
                    'weight': selected['weight'], 'recipe': selected['recipe'],
                    'records': json_predictions(test[['date', 'cotton_close', f'target_return_{h}',
                        'predicted_return', 'raw_predicted_return', 'median_return', 'past_majority_sign']])})
                if getattr(experiment, 'mirror', None):
                    experiment.mirror.enqueue([marker.relative_to(experiment.root).as_posix(),
                        decision_path.relative_to(experiment.root).as_posix()])
                    experiment.mirror.flush()
                print(f'SAVED recency {fold["year"]} T+{h}', flush=True)
    except FitBudgetReached:
        print('PILOT planned_pause: recency checkpoint preserved; resume same profile', flush=True)
        return {'status': 'planned_pause', 'saved_outputs': len(list((experiment.root / 'recency-outputs').glob('*.json')))}
    return {'status': 'complete', 'saved_outputs': 16, 'control_fits': 0}


def compare(folder, repetitions=10000):
    ready = read_record(folder / 'ready.json')
    design = ready['identity']['design']
    verify_controls(folder / 'controls', design)
    if digest(folder / 'history.parquet') != ready['history_sha256']:
        raise ValueError('Recency history changed')
    if len(list((folder / 'recency-outputs').glob('*.json'))) != 16:
        return {'status': 'pending', 'required_outputs': 16}
    horizons, hypotheses = {}, {}
    for h in (1, 5):
        paired, baseline_paired, direction_paired, inner, frames = [], [], [], [], []
        naive_wins = 0
        for fold in design['split']['folds']:
            name = f't{h}-year{fold["year"]}.json'
            record, frame = completed_output(folder, name, fold, h, design)
            original = read_record(folder / 'controls/pilot-full-year' / name)
            control = pd.DataFrame(original['records'])
            if (frame.date.tolist() != fold['origins'] or record['year'] != fold['year'] or record['horizon'] != h
                    or not np.array_equal(frame.cotton_close, control.cotton_close)
                    or not np.array_equal(frame[f'target_return_{h}'], control[f'target_return_{h}'])):
                raise ValueError('Common control origins/targets changed')
            a, c, p = (frame[k].to_numpy() for k in (f'target_return_{h}', 'cotton_close', 'predicted_return'))
            if not np.isfinite(np.column_stack([a, c, p])).all():
                raise ValueError('Nonfinite completed predictions')
            naive = c * np.abs(np.expm1(a))
            error = c * np.abs(np.exp(a) - np.exp(p))
            old_error = c * np.abs(np.exp(a) - np.exp(control.predicted_return))
            paired.append(np.column_stack([naive, error]))
            baseline_paired.append(np.column_stack([old_error, error]))
            direction_paired.append(np.column_stack([np.sign(a) == np.sign(p),
                np.sign(a) == frame.past_majority_sign]).astype(float))
            naive_wins += int(error.mean() < naive.mean())
            inner.append((original['inner_score'], record['inner_score']))
            frames.append(frame)
        losses = np.concatenate(paired)
        old = np.concatenate(baseline_paired)
        frame = pd.concat(frames, ignore_index=True)
        gain = float(100 * (1 - losses[:, 1].mean() / losses[:, 0].mean()))
        direction = float(100 * np.mean(np.sign(frame[f'target_return_{h}']) == np.sign(frame.predicted_return)))
        inner_gain = float(100 * (1 - np.mean([p[1] for p in inner]) / np.mean([p[0] for p in inner])))
        inner_wins = sum(new < original for original, new in inner)
        baseline_ci = paired_bootstrap(baseline_paired, repetitions=repetitions,
            normalization=float(old[:, 0].mean()))
        hypotheses[f'R{h}'] = {'p_two_sided': baseline_ci['centered_bootstrap_p_two_sided']}
        horizons[str(h)] = {'naive_mae_gain_pct': gain, 'direction_pct': direction,
            'fold_wins': naive_wins, 'flat_count': int(frame.predicted_return.eq(0).sum()),
            'inner_gain_vs_full_history_pct': inner_gain, 'inner_year_wins': inner_wins,
            'research_priority_signal': bool(inner_gain >= .5 and inner_wins >= 5 and gain > 0 and naive_wins >= 5),
            'price_thresholds_passed': bool(gain >= 5 and direction >= (53 if h == 1 else 55) and naive_wins >= 6),
            'versus_full_history': baseline_ci,
            'versus_naive': {str(b): paired_bootstrap(paired, block=b, repetitions=repetitions,
                normalization=float(losses[:, 0].mean())) for b in (10, 20, 40)},
            'direction_vs_majority': paired_bootstrap(direction_paired, repetitions=repetitions)}
    for key, q in zip(hypotheses, bh_adjust([p['p_two_sided'] for p in hypotheses.values()]), strict=True):
        hypotheses[key]['bh_adjusted_p'] = q
    body = {'status': 'complete', 'scope': 'reused historical research; no independent holdout claim',
        'horizons': horizons, 'hypotheses': hypotheses, 'design_id': design['design_id'],
        'release_allowed': False, 'control_fits': 0, 'ready_sha256': digest(folder / 'ready.json'),
        'output_sha256': {p.name: digest(p) for p in sorted((folder / 'recency-outputs').glob('*.json'))}}
    freeze_record(folder / 'reports' / f'recency-{content_id(body)[:16]}.json', body)
    return body


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    from cottonlens_ml.research.mirror import Mirror
    folder = root_path(args.drive_root, args.experiment)
    if args.stage not in ('status', 'prepare', 'pilot-plan', 'pilot', 'compare', 'report'):
        raise ValueError('Recency is a bounded research pilot; no automatic lock/export')
    mirror = Mirror(folder, root_path(args.mirror_root, args.experiment)) if args.mirror_root else None
    if mirror:
        metadata_only = args.stage != 'pilot'
        if not (metadata_only and (folder / 'ready.json').exists()
                and len(list((folder / 'recency-outputs').glob('*.json'))) == 16):
            print('Restoring recency metadata' if metadata_only else 'Restoring recency checkpoints', flush=True)
            mirror.hydrate(metadata_only=metadata_only)
    if args.stage == 'prepare':
        if not args.reference_root:
            raise ValueError('Frozen control/input packet required')
        ready = prepare(args.repo, folder, args.reference_root)
        if mirror:
            mirror.enqueue(['ready.json', 'history.parquet', 'preregistered.json',
                *[p.relative_to(folder).as_posix() for p in (folder / 'controls').rglob('*') if p.is_file()]])
            mirror.flush()
        result = {'status': 'prepared', 'design_id': ready['identity']['design_id'], 'control_fits': 0}
    elif args.stage == 'status':
        result = {'prepared': (folder / 'ready.json').exists(), 'writer_lock_present': (folder / '.writer-lock').exists(),
            'saved_outputs': len(list((folder / 'recency-outputs').glob('*.json'))), 'required_outputs': 16,
            'ledger': Ledger(folder / 'ledger', {}).summary(), 'control_fits': 0}
    elif args.stage == 'pilot-plan':
        ready = read_record(folder / 'ready.json')
        result = {'fit_budget': ready['identity']['design']['fit_budget'], 'required_outputs': 16, 'control_fits': 0}
    elif args.stage in ('compare', 'report'):
        result = compare(folder)
        if mirror:
            mirror.enqueue([p.relative_to(folder).as_posix() for p in (folder / 'reports').glob('*.json')])
            mirror.flush()
    else:
        with writer(folder):
            experiment = Experiment(folder, repo=args.repo)
            if mirror:
                experiment.mirror = mirror
                experiment.after_fit = mirror.fit
            result = run(experiment, args.max_minutes)
    print(json.dumps(result, indent=2))
