"""One frozen price-delta loss contrast on the existing Experiment/Ledger."""
import importlib.metadata
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.availability_clock import require_same_targets
from cottonlens_ml.research.diagnostics import (
    VALIDATION_POLICY,
    control_failures,
    synthetic_control,
)
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.models import fit_predict
from cottonlens_ml.research.nass_regional_execution import fit_consumption
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import training_rows, verify_history
from cottonlens_ml.research.statistics import paired_bootstrap

PROFILE = 'paired-price-loss-control-v1'
ARMS = {'squared': 'reg:squarederror', 'absolute': 'reg:absoluteerror'}
SEEDS = (17, 42, 101)
VERSIONS = ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl', 'xgboost')
REGISTRATION_PROOF = 'paired-price-loss-preregistration-20261010.json'
RULES = {'primary': 'absolute vs squared same-origin paired price-MAE',
         'baseline': 'same-origin zero-return Naive; no fit',
         'bootstrap': {'blocks': [20, 60], 'replicates': 10000, 'seed': 42, 'within_period': True},
         'gates': {'naive_mae_gain_pct': 5, 'direction_pct': 53, 'periods_won': 6},
         'selection': 'none; original past-chosen parameters and per-seed iterations fixed in both arms',
         'ensemble': 'mean of the three log-return forecasts; identical in both arms',
         'shrinkage': None, 'independent_holdout': False, 'historical_source_admitted': False,
         'automatic_release': False,
         'learning_controls': {'policy': VALIDATION_POLICY, 'losses': list(ARMS.values()),
                               'known_signal_min_gain': .5, 'memorization_min_gain': .9,
                               'negative_shift_max_gain': .2, 'fits': 6, 'market_evidence': False},
         'limitations': ['Reviewed historical periods; not independent holdout.',
                        'Original parameters/iterations were chosen under mixed targets/losses; inference is conditional on these recipes.',
                        'Original source clock/vintage assumptions remain unverified.',
                        'A negative contrast is not universal absence of information or failure of all MAE models.']}


def integrity():
    if not __debug__:
        raise RuntimeError('Scientific integrity assertions must remain enabled')


def paired_recipe(original, arm):
    if arm not in ARMS or original['family'] != 'xgboost' or original['horizon'] != 1:
        raise ValueError('Only fixed paired T+1 XGBoost recipes admitted')
    return {**original, 'params': dict(original['params']), 'features': list(original['features']),
            'device': 'cpu', 'target': 'price_delta', 'loss': ARMS[arm],
            'hypothesis': 'fixed price-delta loss contrast; no new selection'}


def training_cohort(history, dates, cutoff, identity, coverage_start):
    train = training_rows(history, cutoff, {'years': None}, coverage_start)
    if (train.date.dt.strftime('%Y-%m-%d').tolist() != dates
            or frame_identity(train, list(train)) != identity
            or train.target_date_1.ge(cutoff).any() or train.target_date_5.ge(cutoff).any()):
        raise ValueError('Original training cohort/maturity differs; never take an intersection')
    return train


def make_design(ready, history, report, receipts):
    integrity()
    folds = ready['identity']['split']['folds']
    assert report['source_id'] == ready['identity']['source_id']
    assert len(folds) == len(report['years']) == 8
    assert len(ready['identity']['groups']['base']) == 24
    periods, all_dates = [], []
    for fold, chosen in zip(folds, report['years'], strict=True):
        assert fold['year'] == chosen['year'] and len(fold['origins']) == 126
        test = rows_at(history, fold['origins'])
        assert test.target_date_1.lt('2024-01-01').all() and test.target_date_5.lt('2024-01-01').all()
        seeds = []
        for selected, key in zip(chosen['chosen']['seeds'], chosen['outer_fit_ids'], strict=True):
            receipt = receipts[key]
            spec, original = receipt['specification'], selected['recipe']
            assert receipt['identity'] == ready['identity']
            assert receipt['experiment_id'] == key == content_id({'identity': receipt['identity'], 'specification': spec})
            assert spec['recipe'] == original and original['features'] == ready['identity']['groups']['base']
            assert spec['iterations'] == selected['iterations'] > 0
            assert spec['test_dates'] == receipt['result']['origins'] == fold['origins']
            assert spec['validation_dates'] == [] and original['years'] is None and original['window'] == 1
            training_cohort(history, spec['train_dates'], test.date.min(), spec['train_identity'],
                            ready['identity']['split'].get('coverage_start'))
            seeds.append({'reference_fit_id': key, 'seed': original['seed'], 'iterations': spec['iterations'],
                          'original_recipe': original, 'train_dates': spec['train_dates'], 'train_identity': spec['train_identity']})
        assert tuple(s['seed'] for s in seeds) == SEEDS
        periods.append({'year': fold['year'], 'origins': fold['origins'], 'seeds': seeds})
        all_dates.extend(fold['origins'])
    assert len(all_dates) == len(set(all_dates)) == 1008
    trees = sum(2 * s['iterations'] for p in periods for s in p['seeds'])
    if trees != 2688:
        raise ValueError('Approved fixed tree budget changed')
    return {'profile': PROFILE, 'periods': periods, 'rules': RULES, 'fit_budget': 48, 'tree_budget': trees,
            'synthetic_control_fit_budget': 6, 'unique_origins': 1008, 'prediction_rows': 2016,
            'seed_prediction_rows': 6048, 'reference_source_id': report['source_id'],
            'clock': ready['identity']['split']['decision_time'], 'processes': 1, 'threads': 2,
            'session_minutes': 30, 'deliberate_repeat_reason': 'MAE was tried in adaptive mixed recipes; this repeats fixed recipes to isolate loss, without selecting from OOS.'}


def prepare(repo, folder, reference):
    integrity()
    repo, folder, reference = map(Path, (repo, folder, reference))
    validate(repo / 'research')
    proof_path = repo / 'research/evidence/legacy-loss-lineage-20261010.json'
    proof = json.loads(proof_path.read_bytes())
    report_path = reference / 'analysis/lineage-report.json'
    if digest(report_path) != proof['report_sha256'] or not proof['independent_checks']['exact_report_and_csv_reproduction']:
        raise ValueError('Verified completed legacy lineage required')
    report = read_record(report_path)
    inputs_root = reference / 'inputs'
    for name, expected in report['input_hashes'].items():
        if not safe_member(name) or digest(inputs_root / name) != expected:
            raise ValueError('Original consumed evidence changed')
    legacy = inputs_root / 'output/claude-audit/drive/experiments/research-v2-tf-placement'
    parent, history = verify_history(legacy)
    ids = [key for y in report['years'] for key in y['outer_fit_ids']]
    receipts = {key: read_record(legacy / 'ledger/completed' / (key + '.json')) for key in ids}
    design = make_design(parent, history, report, receipts)
    source = research_source_identity(repo)
    inputs = {'reference-ready.json': legacy / 'ready.json', 'reference-report.json': report_path,
              'lineage-proof.json': proof_path, 'history.parquet': legacy / 'history.parquet',
              **{'receipts/' + k + '.json': legacy / 'ledger/completed' / (k + '.json') for k in ids}}
    hashes = {n: digest(p) for n, p in inputs.items()}
    identity = {'profile': PROFILE, 'source_id': source['source_id'], 'design': design,
                'design_id': content_id(design), 'split': parent['identity']['split'], 'input_sha256': hashes,
                'research_data_id': hashes['history.parquet'],
                'python': sys.version.split()[0], 'versions': {n: importlib.metadata.version(n) for n in VERSIONS},
                'lock_sha256': digest(repo / 'ml/uv.lock'), 'release_allowed': False,
                'historical_source_admitted': False}
    registry, trials = load_registry(repo / 'research')
    related = check(registry, trials, family='xgboost', horizon=1)
    lookups = []
    for p in design['periods']:
        for seed in p['seeds']:
            for arm in ARMS:
                lookup = check(registry, trials, proposal={'scope_id': content_id(identity),
                                                          'recipe': paired_recipe(seed['original_recipe'], arm)})
                if lookup['exact_fit_recipes']:
                    raise ValueError('Exact frozen recipe already fitted')
                lookups.append({'year': p['year'], 'seed': seed['seed'], 'arm': arm, 'status': lookup['status']})
    with writer(folder):
        if any(p.name != '.writer-lock' for p in folder.iterdir()):
            raise ValueError('Preserve existing preparation; use new namespace')
        for name, path in inputs.items():
            copy_immutable(path, folder / 'inputs' / name)
        copy_immutable(legacy / 'history.parquet', folder / 'history.parquet')
        for name in source['files']:
            copy_immutable(repo / name, folder / 'source-snapshot' / name)
        freeze_record(folder / 'source-manifest.json', source)
        freeze_record(folder / 'history-check.json', {'related_experiments': [r['id'] for r in related['experiments']],
                      'related_fit_recipes': len(related['fit_recipes']), 'proposals': lookups, 'new_fits': 0})
        freeze_record(folder / 'preregistered.json', {'identity_id': content_id(identity), 'identity': identity,
                      'created_at': datetime.now(UTC).isoformat(), 'completed_experiment': False, 'new_fits': 0})
        if source != research_source_identity(repo) or any(digest(path) != hashes[n] for n, path in inputs.items()):
            raise ValueError('Source/input changed during preparation')
        freeze_record(folder / 'ready.json', {'identity': identity, 'history_sha256': hashes['history.parquet']})
    verify_execution(folder)


def verify_execution(folder):
    integrity()
    folder = Path(folder)
    ready, history = verify_history(folder)
    identity = ready['identity']
    if identity['profile'] != PROFILE or identity['design_id'] != content_id(identity['design']):
        raise ValueError('Prepared design identity changed')
    for name, expected in identity['input_sha256'].items():
        if not safe_member(name) or digest(folder / 'inputs' / name) != expected:
            raise ValueError('Prepared original input changed')
    reg = read_record(folder / 'preregistered.json')
    if reg['identity'] != identity or reg['identity_id'] != content_id(identity) or reg['new_fits'] != 0:
        raise ValueError('Preregistration differs from execution')
    parent = read_record(folder / 'inputs/reference-ready.json')
    report = read_record(folder / 'inputs/reference-report.json')
    receipts = {key: read_record(folder / 'inputs/receipts' / (key + '.json'))
                for y in report['years'] for key in y['outer_fit_ids']}
    if make_design(parent, history, report, receipts) != identity['design']:
        raise ValueError('Scientific design differs from original receipts')
    return ready, history


def verify_registration(repo, folder, ready):
    proof = json.loads((Path(repo) / 'research/evidence' / REGISTRATION_PROOF).read_bytes())
    if (proof['ready_sha256'] != digest(Path(folder) / 'ready.json')
            or proof['preregistered_sha256'] != digest(Path(folder) / 'preregistered.json')
            or proof['identity_id'] != content_id(ready['identity']) or proof['market_fits'] != 0
            or proof['fit_budget'] != 48 or proof['tree_budget'] != 2688):
        raise ValueError('Published zero-fit preregistration differs; no market fitting')


def learning_controls(experiment):
    root = experiment.root / 'learning-control-ledger'
    consumed, reusable = fit_consumption(root)
    if consumed > 6 or consumed != len(reusable):
        raise ValueError('Failed learning fit cannot be silently retried')
    ledger = Ledger(root, {**experiment.identity, 'role': 'synthetic_controls_not_market_evidence'})
    controls = {}
    for arm, loss in ARMS.items():
        measured = {}
        for kind in ('known_signal', 'negative_block_shift', 'small_subset_overfit'):
            frame, train, _, test = synthetic_control(kind)
            spec = {'family': 'xgboost', 'horizon': 1, 'task': 'price', 'device': 'cpu',
                    'features': list(experiment.identity['design']['periods'][0]['seeds'][0]['original_recipe']['features']),
                    'seed': 42, 'window': 1, 'target': 'price_delta', 'loss': loss,
                    'params': {'max_depth': 8, 'eta': .15, 'min_child_weight': 1, 'alpha': 0, 'lambda': .01}}
            iterations = 700 if kind == 'small_subset_overfit' else 250
            record = ledger.run({'role': 'synthetic_control_not_market_evidence', 'arm': arm, 'kind': kind,
                                 'recipe': spec, 'iterations': iterations},
                                lambda w, f=frame, tr=train, te=test, s=spec, it=iterations:
                                fit_predict(f, tr, None, te, s, w, iterations=it))
            naive = np.mean(test.cotton_close.to_numpy() * np.abs(np.expm1(test.target_return_1.to_numpy())))
            measured['xgboost/' + kind] = {'relative_mae_gain': 1 - record['result']['metrics']['mae'] / naive,
                                         'fit_id': record['experiment_id']}
        failures = control_failures(measured, ['xgboost'])
        if failures:
            freeze_record(experiment.root / 'learning-control-failed.json', {'arm': arm, 'failures': failures, 'controls': measured})
            raise ValueError('; '.join(failures))
        controls[arm] = measured
    freeze_record(experiment.root / 'learning-control-passed.json', {'controls': controls, 'synthetic_fits': 6, 'market_evidence': False})


def run(experiment, minutes):
    if not 0 < minutes <= 30:
        raise ValueError('One process/two threads; at most 30 minute session')
    ready, _ = verify_execution(experiment.root)
    verify_registration(experiment.repo_for_registration, experiment.root, ready)
    deadline = time.monotonic() + minutes * 60
    learning_controls(experiment)
    consumed, reusable = fit_consumption(experiment.root / 'ledger')
    if consumed > 48 or consumed != len(reusable):
        raise ValueError('Failed or excess market fits; no silent retry')
    count = [consumed]

    def before_compute():
        if count[0] >= 48:
            raise ValueError('48-fit budget exhausted')
        count[0] += 1

    experiment.before_compute = before_compute
    for period in experiment.identity['design']['periods']:
        test = rows_at(experiment.history, period['origins'])
        for arm in ARMS:
            ids, forecasts = [], []
            for seed in period['seeds']:
                if time.monotonic() >= deadline:
                    return {'status': 'planned_pause', 'completed_market_fits': count[0]}
                train = training_cohort(experiment.history, seed['train_dates'], test.date.min(), seed['train_identity'],
                                        experiment.identity['split'].get('coverage_start'))
                spec = paired_recipe(seed['original_recipe'], arm)
                record = experiment.fit(spec, train, None, test, f'paired-{period["year"]}-{arm}-{seed["seed"]}',
                                        iterations=seed['iterations'])
                if record['result']['origins'] != period['origins'] or record['result']['iterations'] != seed['iterations']:
                    raise ValueError('Frozen fit origin/iteration mismatch')
                ids.append(record['experiment_id']); forecasts.append(record['result']['predictions'])
            predicted = np.asarray(forecasts, float)
            if predicted.shape != (3, 126) or not np.isfinite(predicted).all():
                raise ValueError('Incomplete/nonfinite seed forecasts')
            freeze_record(experiment.root / 'paired-outputs' / f'{period["year"]}-{arm}.json', {
                'year': period['year'], 'arm': arm, 'origins': period['origins'],
                'target_dates': test.target_date_1.dt.strftime('%Y-%m-%d').tolist(),
                'fit_ids': ids, 'predicted_return': predicted.mean(axis=0).tolist()})
    return {'status': 'completed', 'market_fits': count[0], 'synthetic_fits': 6}


def metric(frame):
    close, actual, predicted = (frame[n].to_numpy(dtype=float) for n in ('cotton_close', 'actual_return', 'predicted_return'))
    if not np.isfinite([close, actual, predicted]).all() or (close <= 0).any():
        raise ValueError('Finite positive-domain aligned predictions required')
    naive, error = close * np.abs(np.expm1(actual)), close * np.abs(np.exp(actual) - np.exp(predicted))
    if naive.mean() <= 0:
        raise ValueError('Relative MAE requires a nonzero Naive loss')
    return np.column_stack([naive, error]), {'price_mae': float(error.mean()), 'naive_mae': float(naive.mean()),
        'naive_gain_pct': float(100 * (1 - error.mean() / naive.mean())),
        'direction_pct': float(100 * np.mean(np.sign(actual) == np.sign(predicted))),
        'active_rate_pct': float(100 * np.mean(predicted != 0))}


def interval(parts):
    normalization = float(np.concatenate(parts)[:, 0].mean())
    result = {}
    for b in (20, 60):
        value = paired_bootstrap(parts, block=b, repetitions=10000, seed=42, normalization=normalization)
        value['gain_ci_pct'] = (100 * np.asarray(value['difference_ci_95']) / normalization).tolist()
        result[str(b)] = value
    return result


def decision(arms, paired):
    a = arms['absolute']
    positive = all(paired[str(b)]['difference_ci_95'][0] > 0 for b in (20, 60))
    if positive and a['naive_gain_pct'] >= 5 and a['direction_pct'] >= 53 and a['period_wins'] >= 6:
        return 'STRONG_FIXED_LOSS_CANDIDATE_REQUIRES_PROSPECTIVE_VALIDATION'
    if positive:
        return 'LIMITED_LOSS_CONTRIBUTION_PRACTICAL_GOAL_UNMET'
    if all(arms[arm]['versus_naive'][str(b)]['gain_ci_pct'][1] < 5 for arm in ARMS for b in (20, 60)):
        return 'BOTH_FIXED_LOSSES_BELOW_PRACTICAL_GOAL'
    return 'INCONCLUSIVE_NO_AUTOMATIC_SEARCH'


def compare(experiment):
    verify_execution(experiment.root)
    passed = read_record(experiment.root / 'learning-control-passed.json')
    if passed['synthetic_fits'] != 6 or passed['market_evidence'] is not False or set(passed['controls']) != set(ARMS):
        raise ValueError('All six separately indexed synthetic controls required')
    control_records = Ledger(experiment.root / 'learning-control-ledger', {}).results()
    if len(control_records) != 6:
        raise ValueError('Incomplete learning-control payloads')
    control_by_id = {r['experiment_id']: r for r in control_records}
    for measured in passed['controls'].values():
        failures = control_failures(measured, ['xgboost'])
        if failures:
            raise ValueError('; '.join(failures))
        for value in measured.values():
            record = control_by_id[value['fit_id']]
            if record['identity'] != {**experiment.identity, 'role': 'synthetic_controls_not_market_evidence'}:
                raise ValueError('Learning-control identity mismatch')
            for name, expected in record['files'].items():
                if not safe_member(name) or digest(experiment.root / 'learning-control-ledger' / name) != expected:
                    raise ValueError('Learning-control payload checksum mismatch')
    records = experiment.ledger.results()
    if len(records) != 48 or (experiment.root / 'learning-control-failed.json').exists():
        raise ValueError('Incomplete/failed experiment is not a market result')
    by_id = {r['experiment_id']: r for r in records}
    frames, seed_frames, used = [], [], set()
    for period in experiment.identity['design']['periods']:
        test = rows_at(experiment.history, period['origins'])
        for arm in ARMS:
            output = read_record(experiment.root / 'paired-outputs' / f'{period["year"]}-{arm}.json')
            vectors = []
            for seed, key in zip(period['seeds'], output['fit_ids'], strict=True):
                record = by_id[key]
                spec = record['specification']
                if (record['identity'] != experiment.identity or record['experiment_id'] != experiment.ledger.key(spec)
                        or spec['recipe'] != paired_recipe(seed['original_recipe'], arm)
                        or spec['iterations'] != seed['iterations'] or spec['validation_dates']
                        or spec['role'] != f'paired-{period["year"]}-{arm}-{seed["seed"]}'
                        or spec['repeat_reason'] is not None or record['result']['iterations'] != seed['iterations']
                        or spec['train_dates'] != seed['train_dates'] or spec['train_identity'] != seed['train_identity']
                        or spec['test_dates'] != period['origins'] or record['result']['origins'] != period['origins']):
                    raise ValueError('Fit receipt differs from registered plan')
                for name, expected in record['files'].items():
                    if not safe_member(name) or digest(experiment.root / 'ledger' / name) != expected:
                        raise ValueError('Saved fitted payload checksum mismatch')
                used.add(key); vectors.append(record['result']['predictions'])
                seed_frames.append(pd.DataFrame({'date': period['origins'], 'arm': arm, 'year': period['year'],
                                                  'seed': seed['seed'], 'predicted_return': vectors[-1], 'fit_id': key}))
            predicted = np.asarray(vectors, float).mean(axis=0)
            if (output['origins'] != period['origins'] or output['target_dates'] != test.target_date_1.dt.strftime('%Y-%m-%d').tolist()
                    or not np.array_equal(predicted, output['predicted_return'])):
                raise ValueError('Saved ensemble changed')
            frames.append(pd.DataFrame({'date': period['origins'], 'target_date': output['target_dates'],
                'year': period['year'], 'arm': arm, 'horizon': 1, 'cotton_close': test.cotton_close,
                'actual_return': test.target_return_1, 'predicted_return': predicted}))
    if len(used) != 48:
        raise ValueError('Duplicate/unused market fit')
    rows = pd.concat(frames, ignore_index=True)
    controls, candidates = (rows.loc[rows.arm == a].reset_index(drop=True) for a in ARMS)
    require_same_targets(controls, candidates)
    parts, arms = {}, {}
    for arm in ARMS:
        frame = rows.loc[rows.arm == arm]
        parts[arm], per_period = [], []
        for year, group in frame.groupby('year', sort=True):
            errors, measured = metric(group)
            parts[arm].append(errors); per_period.append({'year': int(year), **measured})
        _, measured = metric(frame)
        order = np.lexsort((frame.date.to_numpy(), -metric(frame)[0][:, 0]))
        keep = np.ones(len(frame), bool); keep[order[:int(np.ceil(.01 * len(frame)))]] = False
        arms[arm] = {**measured, 'periods': per_period, 'period_wins': sum(p['naive_gain_pct'] > 0 for p in per_period),
                     'versus_naive': interval(parts[arm]), 'without_top_1pct_naive_errors': metric(frame.iloc[np.flatnonzero(keep)])[1]}
    paired = interval([np.column_stack([a[:, 1], b[:, 1]]) for a, b in zip(parts['squared'], parts['absolute'], strict=True)])
    root = experiment.root / 'reports'
    for name, data in [('outer-predictions.csv', rows), ('seed-predictions.csv', pd.concat(seed_frames, ignore_index=True))]:
        path, encoded = root / name, data.to_csv(index=False, lineterminator='\n').encode()
        root.mkdir(exist_ok=True)
        if path.exists() and path.read_bytes() != encoded:
            raise ValueError('Frozen prediction export changed')
        if not path.exists():
            path.write_bytes(encoded)
    report = {'profile': PROFILE, 'identity_id': content_id(experiment.identity), 'unique_origins': 1008,
              'market_fits': 48, 'synthetic_fits': 6, 'ensemble_rows': 2016, 'seed_rows': 6048,
              'arms': arms, 'absolute_vs_squared': paired, 'decision': decision(arms, paired),
              'predictions_sha256': digest(root / 'outer-predictions.csv'), 'seed_predictions_sha256': digest(root / 'seed-predictions.csv'),
              'independent_holdout': False, 'historical_source_admitted': False, 'automatic_release': False}
    freeze_record(root / 'comparison.json', report)
    return report


def dispatch(args):
    integrity()
    from cottonlens_ml.research.engine import Experiment
    folder = Path(args.drive_root) / 'experiments' / args.experiment
    if args.stage == 'prepare':
        if args.reference_root is None:
            raise ValueError('Explicit restored legacy lineage archive required')
        prepare(args.repo, folder, args.reference_root)
        print('Prepared frozen 48-fit / 2688-tree paired loss experiment; zero fits.')
        return
    ready, _ = verify_execution(folder)
    experiment = Experiment(folder, repo=args.repo)
    experiment.repo_for_registration = args.repo
    if args.stage in ('status', 'pilot-plan', 'report'):
        print({'design_id': ready['identity']['design_id'], 'market_fit_budget': 48, 'tree_budget': 2688,
               'synthetic_fit_budget': 6, 'ledger': experiment.ledger.summary()})
        return
    with writer(folder):
        if args.stage == 'pilot':
            print(run(experiment, args.max_minutes))
        elif args.stage == 'compare':
            print(compare(experiment)['decision'])
        else:
            raise ValueError('Only prepare/status/pilot-plan/pilot/compare/report permitted')
