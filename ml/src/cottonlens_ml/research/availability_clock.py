"""Preregistered availability-assumption sensitivity; existing Ridge/Ledger engine."""
import importlib.metadata
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.features import AVAILABILITY_ALIGNMENT, build_feature_history
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.full_year import SHRINKAGE, chunks
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    freeze_record,
    writer,
)
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.statistics import paired_bootstrap

PROFILE = 'availability-clock-pilot-v1'
GROUPS = ('control', 'available')
CROSS = [n for n in FEATURE_NAMES if n.startswith(('dxy_', 'wti_', 'cotton_dxy_', 'cotton_wti_'))]
FEATURES = {'control': list(FEATURE_NAMES),
            'available': ['available_' + n if n in CROSS else n for n in FEATURE_NAMES]}
RECIPE = {'family': 'ridge', 'alpha': 1., 'target': 'scaled_log',
          'years': None, 'seed': 42, 'cadence': 21, 'window': 1}
BUDGET = {'inner': 288, 'outer': 388, 'total': 676, 'annual_outputs': 32, 'prediction_rows': 8024}
GATE = {'gain_pct': 5., 'direction_pct': {'1': 53., '5': 55.}, 'year_wins': 6}
ASSUMPTION = 'source bar date + 1 day 00:00 UTC; not verified publication, vintage or ingestion'
RULES = {'primary': 'T+1 selected available versus control price MAE', 'secondary': 'T+5',
         'bootstrap': {'blocks': [20, 60], 'repetitions': 10000, 'seed': 42},
         'sensitivity': 'remove ceil(1% * n) largest Naive errors, deterministic date tie break',
         'independent_holdout': False, 'automatic_release': False}


def require_same_targets(left, right):
    """No silent intersection, reordering or tolerance for different realized labels."""
    fields = ['date', 'target_date', 'horizon', 'cotton_close', 'actual_return']
    for frame in (left, right):
        if not set(fields).issubset(frame) or frame.empty or frame[fields].isna().any().any():
            raise ValueError('Complete comparison keys and targets required')
        if frame.duplicated(['date', 'horizon']).any():
            raise ValueError('Duplicate comparison origins')
        if not np.isfinite(frame[['cotton_close', 'actual_return', 'horizon']]).all().all():
            raise ValueError('Nonfinite comparison targets')
    for field in fields:
        a, b = left[field].reset_index(drop=True), right[field].reset_index(drop=True)
        if field in ('date', 'target_date'):
            a, b = pd.to_datetime(a, utc=True), pd.to_datetime(b, utc=True)
        if not np.array_equal(a.to_numpy(), b.to_numpy()):
            raise ValueError(f'Unmatched comparison {field}; intersection is forbidden')


def expanded_history(history, market):
    market = market.loc[pd.to_datetime(market.date) < pd.Timestamp('2024-01-01')].copy()
    empty_cftc = pd.DataFrame(columns=['available_date', 'cftc_managed_money_net'])
    control = build_feature_history(market, empty_cftc)
    treatment = build_feature_history(market, empty_cftc, alignment_policy=AVAILABILITY_ALIGNMENT)
    if not np.array_equal(control.date.to_numpy(), history.date.to_numpy()):
        raise ValueError('Reference Cotton observations differ')
    if not np.array_equal(control.cotton_close, history.cotton_close):
        raise ValueError('Reference Cotton prices differ')
    for feature in FEATURE_NAMES:
        if not np.allclose(control[feature], history[feature], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f'Control feature differs from frozen reference: {feature}')
    result = history.copy()
    for feature in CROSS:
        result['available_' + feature] = treatment[feature].to_numpy()
    for source in ('dxy', 'wti'):
        for field in ('source_date', 'available_at', 'age_sessions', 'stale'):
            result[f'available_{source}_{field}'] = treatment[f'{source}_{field}'].to_numpy()
    result['clock_decision_time'] = pd.to_datetime(result.date, utc=True) + pd.Timedelta(days=1, minutes=15)
    result.attrs = {}
    return result


def validate_design(registration):
    d = registration['design']
    if (registration['design_id'] != content_id(d) or d['profile'] != PROFILE
            or d['groups'] != FEATURES or d['recipes'] != RECIPE or d['fit_budget'] != BUDGET
            or d['shrinkage_weights'] != list(SHRINKAGE) or d['price_gate'] != GATE
            or d['rules'] != RULES or d['availability_assumption'] != ASSUMPTION
            or d['alignment_policy'] != AVAILABILITY_ALIGNMENT
            or d['decision_time'] != 'Cotton source date + 1 day 00:15 UTC'
            or d['audit_2024_used'] is not False or d['automatic_release'] is not False
            or d['split']['purge_observations'] != 5 or d['split']['refit_cadence'] != 21
            or [f['year'] for f in d['split']['folds']] != list(range(2016, 2024))
            or sum(len(f['origins']) for f in d['split']['folds']) != 2006
            or any(len(f['inner']) != 3 or any(len(b['origins']) != 63 for b in f['inner'])
                   for f in d['split']['folds'])):
        raise ValueError('Frozen availability-clock design changed')
    return d


def prepare(repo, folder, reference, market_file):
    old, history = verify_history(reference)
    market_sha = digest(market_file)
    if old['identity'].get('raw_audit', {}).get('market_sha256') != market_sha:
        raise ValueError('Raw market checksum differs from frozen reference')
    expanded = expanded_history(history, pd.read_parquet(market_file))
    split = old['identity']['split']
    inner = 4 * sum(len(chunks(history, b['origins'])) for f in split['folds'] for b in f['inner'])
    outer = 4 * sum(len(chunks(history, f['origins'])) for f in split['folds'])
    if (inner, outer) != (288, 388):
        raise ValueError('Fit budget differs from approved 676 fits')
    for fold in split['folds']:
        origins = rows_at(history, fold['origins'])
        if origins.target_date_5.ge(pd.Timestamp('2024-01-01')).any():
            raise ValueError('Evaluation would use 2024 labels')
    design = {'profile': PROFILE, 'groups': FEATURES, 'recipes': RECIPE, 'fit_budget': BUDGET,
        'split': split, 'shrinkage_weights': list(SHRINKAGE), 'price_gate': GATE, 'rules': RULES,
        'reference_ready_sha256': digest(reference / 'ready.json'),
        'reference_history_sha256': digest(reference / 'history.parquet'),
        'market_sha256': market_sha, 'alignment_policy': AVAILABILITY_ALIGNMENT,
        'availability_assumption': ASSUMPTION, 'decision_time': 'Cotton source date + 1 day 00:15 UTC',
        'feature_data_id': frame_identity(expanded, list(expanded.columns)),
        'automatic_release': False, 'audit_2024_used': False}
    registration = {'design_id': content_id(design), 'design': design}
    validate_design(registration)
    source = research_source_identity(repo)
    identity = {'profile': PROFILE, 'source_id': source['source_id'], **registration, 'split': split,
        'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in
            ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'threadpoolctl', 'pyarrow')},
        'lock_sha256': digest(repo / 'ml/uv.lock'), 'execution': {'device': 'cpu', 'threads': 2}}
    with writer(folder):
        if (folder / 'ready.json').exists():
            ready, _ = verify_history(folder)
            if ready['identity'] != identity:
                raise ValueError('Code/data/environment/policy changed; use a new namespace')
            return ready
        copy_immutable(reference / 'ready.json', folder / 'reference/ready.json')
        copy_immutable(reference / 'history.parquet', folder / 'reference/history.parquet')
        expanded.to_parquet(folder / 'history.parquet', index=False)
        freeze_record(folder / 'source-manifest.json', source)
        freeze_record(folder / 'preregistered.json', registration)
        ready = {'identity': identity, 'history_sha256': digest(folder / 'history.parquet'),
                 'created_at': datetime.now(UTC).isoformat()}
        freeze_record(folder / 'ready.json', ready)
    return ready


def record_frame(test, payload, group, h, chosen):
    payload['target_date'] = pd.to_datetime(test[f'target_date_{h}']).dt.strftime('%Y-%m-%d')
    payload['horizon'], payload['actual_return'] = h, test[f'target_return_{h}']
    payload['selected_weight'] = chosen['weight']
    payload['decision_time'] = pd.to_datetime(test.clock_decision_time, utc=True).map(lambda x: x.isoformat())
    prefix = 'available_' if group == 'available' else ''
    for source in ('dxy', 'wti'):
        for field in ('source_date', 'available_at'):
            values = pd.to_datetime(test[f'{prefix}{source}_{field}'], utc=True)
            payload[f'{source}_{field}'] = values.map(lambda x: None if pd.isna(x) else x.isoformat())
        payload[f'{source}_stale'] = test[f'{prefix}{source}_stale']
    return payload


def output(folder, name, fold, group, h, design, history, **kwargs):
    record, frame = path_pilot.output(folder, name, fold, group, h, design, history, **kwargs)
    actual = rows_at(history, fold['origins']).copy()
    expected = record_frame(actual, pd.DataFrame({'date': fold['origins']}, index=actual.index),
                            group, h, {'weight': record['weight']}).reset_index(drop=True)
    expected['cotton_close'] = actual.cotton_close.to_numpy()
    require_same_targets(frame, expected)
    for field in expected.columns:
        if field not in ('date', 'cotton_close') and not np.array_equal(frame[field], expected[field]):
            raise ValueError(f'Frozen availability-clock output changed: {field}')
    return record, frame


def run(experiment, max_minutes):
    if not 0 < max_minutes <= 30:
        raise ValueError('Availability-clock sessions are limited to 30 minutes')
    # Count unique attempted computations too, so a failed computation cannot silently expand the budget.
    keys = {p.stem for p in (experiment.root / 'ledger/completed').glob('*.json')}
    keys |= {p.parent.name for p in (experiment.root / 'ledger/local-work').glob('*/locally-completed.json')}
    failed = len(list((experiment.root / 'ledger/attempts').glob('*.json')))
    count = [len(keys) + failed]

    def before_fit():
        if count[0] >= BUDGET['total']:
            raise FitBudgetReached('Approved 676-fit budget exhausted; no additional fit authorized')
        count[0] += 1

    return path_pilot.run(experiment, max_minutes, group_names=GROUPS, namespace='clock',
        validate_fn=validate_design, record_frame_fn=record_frame, output_fn=output, before_fit=before_fit)


def _metrics(frame, prediction):
    a, p, c = frame.actual_return.to_numpy(), frame[prediction].to_numpy(), frame.cotton_close.to_numpy()
    naive, error = c * np.abs(np.expm1(a)), c * np.abs(np.exp(a) - np.exp(p))
    return naive, error, {'count': len(frame), 'price_mae': float(error.mean()),
        'naive_mae': float(naive.mean()), 'naive_gain_pct': float(100 * (1 - error.mean() / naive.mean())),
        'direction_pct': float(100 * np.mean(np.sign(a) == np.sign(p))),
        'active_rate_pct': float(100 * np.mean(p != 0)), 'flat_count': int(np.sum(p == 0)),
        'active_direction_pct': float(100 * np.mean(np.sign(a[p != 0]) == np.sign(p[p != 0]))) if np.any(p != 0) else None}


def _interval(parts, repetitions):
    baseline = float(np.concatenate(parts)[:, 0].mean())
    result = {}
    for block in (20, 60):
        item = paired_bootstrap(parts, block=block, repetitions=repetitions, seed=42, normalization=baseline)
        item['gain_ci_pct'] = (100 * np.array(item['difference_ci_95']) / baseline).tolist()
        result[str(block)] = item
    return result


def decision(horizon):
    arm = horizon['arms']['available']['selected']
    paired = horizon['selected_vs_control']
    positive = all(paired[str(b)]['difference_ci_95'][0] > 0 for b in (20, 60))
    if arm['naive_gain_pct'] >= 5 and arm['direction_pct'] >= 53 and arm['year_wins'] >= 6 and positive:
        return 'strong_timing_candidate_verify_real_availability_and_prospective_recipe'
    if positive:
        return 'timing_contribution_below_practical_gates_no_new_search'
    if all(arm['versus_naive'][str(b)]['gain_ci_pct'][1] < 5 for b in (20, 60)):
        return 'timing_does_not_rescue_fixed_recipe_keep_naive_stop_expansion'
    return 'inconclusive_no_automatic_training'


def compare(folder, repetitions=10000):
    ready, history = verify_history(folder)
    design = validate_design({'design_id': ready['identity']['design_id'], 'design': ready['identity']['design']})
    if len(list((folder / 'clock-outputs').glob('*.json'))) != 32:
        return {'status': 'pending', 'required_outputs': 32}
    horizons, all_rows = {}, []
    for h in (1, 5):
        frames = {g: [] for g in GROUPS}
        for fold in design['split']['folds']:
            for group in GROUPS:
                _, frame = output(folder, f'{group}-t{h}-year{fold["year"]}.json', fold, group, h,
                    design, history, namespace='clock')
                frame['year'], frame['group'] = fold['year'], group
                frames[group].append(frame)
        combined = {g: pd.concat(f, ignore_index=True) for g, f in frames.items()}
        require_same_targets(combined['control'], combined['available'])
        all_rows.extend(combined.values())
        arms = {}
        naive = _metrics(combined['control'], 'predicted_return')[0]
        keep = np.ones(len(naive), bool)
        keep[np.argsort(-naive, kind='stable')[:int(np.ceil(.01 * len(naive)))]] = False
        for group in GROUPS:
            arms[group] = {}
            for mode, field in (('selected', 'predicted_return'), ('raw', 'raw_predicted_return')):
                _, _, summary = _metrics(combined[group], field)
                years, parts = {}, []
                for frame in frames[group]:
                    baseline, errors, score = _metrics(frame, field)
                    years[str(frame.year.iloc[0])] = score
                    parts.append(np.column_stack([baseline, errors]))
                summary.update(years=years, year_wins=sum(v['naive_gain_pct'] > 0 for v in years.values()),
                    versus_naive=_interval(parts, repetitions),
                    without_top_1pct_naive_errors=_metrics(combined[group].loc[keep], field)[2])
                arms[group][mode] = summary
        comparisons = {}
        for mode, field in (('selected', 'predicted_return'), ('raw', 'raw_predicted_return')):
            parts = [np.column_stack([_metrics(a, field)[1], _metrics(b, field)[1]])
                     for a, b in zip(frames['control'], frames['available'], strict=True)]
            comparisons[mode + '_vs_control'] = _interval(parts, repetitions)
            # Keep original year boundaries after removing the same extreme origins in both arms.
            trimmed = {g: combined[g].loc[keep] for g in GROUPS}
            trimmed_parts = [np.column_stack([
                _metrics(trimmed['control'].loc[trimmed['control'].year.eq(year)], field)[1],
                _metrics(trimmed['available'].loc[trimmed['available'].year.eq(year)], field)[1]])
                for year in range(2016, 2024)]
            comparisons[mode + '_vs_control_without_top_1pct'] = _interval(trimmed_parts, repetitions)
        horizons[str(h)] = {'arms': arms, **comparisons}
    body = {'status': 'complete', 'design_id': ready['identity']['design_id'], 'horizons': horizons,
        'decision': decision(horizons['1']), 'release_allowed': False,
        'scope': 'availability-assumption sensitivity on repeatedly reviewed history; no independent holdout',
        'limits': 'Fixed Ridge/information recipe only. Constant zero-weight losses do not establish no source signal.',
        'ready_sha256': digest(folder / 'ready.json'),
        'output_sha256': {p.name: digest(p) for p in sorted((folder / 'clock-outputs').glob('*.json'))}}
    # The records above are verified before exporting a convenient prediction-level view.
    rows = pd.concat(all_rows, ignore_index=True)
    export = folder / 'reports/outer-predictions.csv'
    export.parent.mkdir(exist_ok=True)
    encoded = rows.to_csv(index=False, lineterminator='\n').encode()
    if export.exists() and export.read_bytes() != encoded:
        raise ValueError('Frozen prediction export changed')
    if not export.exists():
        export.write_bytes(encoded)
    body['predictions_sha256'] = digest(export)
    freeze_record(folder / 'reports' / f'clock-{content_id(body)[:16]}.json', body)
    return body


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder = root_path(args.drive_root, args.experiment)
    if args.mirror_root:
        raise ValueError('This bounded local pilot has no remote mirror; local checkpoints remain resumable')
    if args.stage == 'prepare':
        if not args.reference_root or not args.market_file:
            raise ValueError('Explicit frozen reference and --market-file required')
        ready = prepare(args.repo, folder, args.reference_root, args.market_file)
        result = {'status': 'prepared', 'design_id': ready['identity']['design_id'], 'fit_budget': BUDGET}
    elif args.stage in ('status', 'pilot-plan'):
        result = {'prepared': (folder / 'ready.json').exists(), 'fit_budget': BUDGET,
            'saved_outputs': len(list((folder / 'clock-outputs').glob('*.json'))),
            'completed_fits': len(list((folder / 'ledger/completed').glob('*.json')))}
    elif args.stage == 'pilot':
        with writer(folder):
            result = run(Experiment(folder, repo=args.repo), args.max_minutes)
    elif args.stage in ('compare', 'report'):
        with writer(folder):
            result = compare(folder)
    else:
        raise ValueError('Availability-clock supports prepare/pilot-plan/pilot/compare/report/status only')
    print(json.dumps(result, indent=2))
