"""One fixed nonlinear control on the frozen named-contract cohort; no search."""
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research import named_label_control as parent
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.diagnostics import (
    VALIDATION_POLICY,
    control_failures,
    synthetic_control,
)
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.models import fit_predict
from cottonlens_ml.research.recency import copy_immutable

PROFILE = 'named-capacity-control-t5-v1'
TREES = 100
RECIPE = {**parent.RECIPE, 'family': 'xgboost',
          'params': {'max_depth': 2, 'eta': .03, 'min_child_weight': 20, 'alpha': 0, 'lambda': 1},
          'max_iterations': TREES}
KINDS = ('known_signal', 'known_nonlinear', 'negative_block_shift', 'small_subset_overfit')
CORE = ('ml/src/cottonlens_ml/research/protocol.py', 'ml/src/cottonlens_ml/research/models.py',
        'ml/src/cottonlens_ml/research/named_label_control.py', 'ml/src/cottonlens_ml/research/contract_curve.py',
        'ml/src/cottonlens_ml/config.py', 'ml/src/cottonlens_ml/evaluation.py',
        'ml/src/cottonlens_ml/features.py', 'ml/src/cottonlens_ml/cohort.py')


def design(parent_ready):
    original = parent_ready['identity']['design']
    if original['profile'] != parent.PROFILE or original['recipe'] != parent.RECIPE:
        raise ValueError('Only the completed matched named-label parent is authorized')
    return {'profile': PROFILE, 'recipe': RECIPE, 'iterations': TREES, 'folds': original['folds'],
            'parent_design_id': content_id(original), 'reference_arm': 'named',
            'fit_budget': {'market': sum(len(f['jobs']) for f in original['folds']), 'synthetic': 4,
                           'threads': 2, 'processes': 1, 'session_minutes': 30},
            'primary': 'RAW fixed XGBoost vs frozen named Ridge, paired same-contract price-MAE',
            'secondary': 'Naive price-MAE/direction; past-only shrinkage; no raw/selected switch',
            'selection': original['selection'], 'clock': original['clock'],
            'bootstrap': {'repetitions': 10000, 'seed': 42, 'blocks': [20, 60],
                          'ordinal_slots_preserved': True, 'missing_weight': 0, 'selection_adjusted': False},
            'decision_rules': {'candidate': 'Both raw Ridge-minus-XGB CIs lower>0 and raw XGB Naive gain>=5%; research candidate only',
                               'below_goal': 'Both raw XGB-vs-Naive gain CI uppers<5%; fixed shallow nonlinear recipe does not rescue practical goal',
                               'otherwise': 'INCONCLUSIVE; no automatic extra fit/grid'},
            'gate_evaluation_allowed': False, 'release_allowed': False,
            'source_timestamp_verified': False, 'first_vintage_verified': False,
            'deliberate_repeat_reason': 'Old native XGBoost uses CT label/core24; matched named label/core28 capacity untested. Isolate family,not another CT grid.',
            'tree_count_reason': '100 trees frozen before fitting, conservative single boosting budget; no early stop or iteration search. Nested63-label early stopping would reduce earliest528 training below500.'}


def verify_parent(folder, repo):
    from cottonlens_ml.research.engine import Experiment
    p = Path(folder) / 'parent'
    snapshot = p / 'source-snapshot'
    for name, checksum in read_record(Path(folder) / 'parent-files.json')['files'].items():
        if not safe_member(name) or digest(p / name) != checksum:
            raise ValueError('Frozen completed parent changed')
    for name in CORE:
        if digest(Path(repo) / name) != digest(snapshot / name):
            raise ValueError('Shared feature/target/fitting implementation changed; not a family-only comparison')
    ready, history = parent.verify(p, snapshot)
    experiment = Experiment(p, repo=snapshot)
    references = {str(f['year']): parent.verify_output(experiment, history, f, 'named')
                  for f in ready['identity']['design']['folds']}
    return ready, history, references


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder / 'ready.json')
    identity = ready['identity']
    if identity['profile'] != PROFILE or research_source_identity(repo)['source_id'] != identity['source_id']:
        raise ValueError('Source/profile changed; new namespace required')
    actual = {p: importlib.metadata.version(p) for p in identity['versions']}
    if actual != identity['versions'] or sys.version.split()[0] != identity['python']:
        raise ValueError('Frozen environment changed')
    if digest(repo / 'ml/uv.lock') != identity['lock_sha256']:
        raise ValueError('Dependency lock changed')
    if digest(folder / 'parent-files.json') != identity['parent_files_sha256']:
        raise ValueError('Parent inventory changed')
    old, history, references = verify_parent(folder, repo)
    expected = design(old)
    if (identity['design'] != expected or identity['design_id'] != content_id(expected)
            or identity['parent_ready_sha256'] != digest(folder / 'parent/ready.json')
            or digest(folder / 'history.parquet') != ready['history_sha256']
            or frame_identity(history, list(history)) != identity['research_data_id']):
        raise ValueError('Frozen cohort/target/design mismatch')
    pd.testing.assert_frame_equal(history, pd.read_parquet(folder / 'history.parquet'), check_exact=True)
    return ready, history, references


def prepare(repo, folder, registration_root, decision_contract):
    repo, folder, registration_root, decision_contract = map(Path, (repo, folder, registration_root, decision_contract))
    ready, _, _ = verify(registration_root, repo)
    proof = read_record(repo / 'research/evidence/named-capacity-preregistration-20261010.json')
    if (proof['ready_sha256'] != digest(registration_root / 'ready.json')
            or proof['decision_contract_sha256'] != digest(decision_contract)
            or read_record(decision_contract) != ready['identity']['design'] or proof['new_fits'] != 0):
        raise ValueError('Published preregistration differs')
    with writer(folder):
        if (folder / 'ready.json').exists():
            old, _, _ = verify(folder, repo)
            if old != ready:
                raise ValueError('Existing identity differs; preserve cache')
            return ready
        if any(p.name != '.writer-lock' for p in folder.iterdir()):
            raise ValueError('Preserve partial preparation')
        for p in registration_root.rglob('*'):
            if p.is_file():
                copy_immutable(p, folder / p.relative_to(registration_root))
        copy_immutable(decision_contract, folder / 'decision-contract.json')
    verify(folder, repo)
    return ready


def synthetic(kind):
    if kind not in KINDS:
        raise ValueError('Unknown synthetic control')
    frame, train, _, test = synthetic_control('known_signal' if kind == 'known_nonlinear' else kind)
    rng = np.random.default_rng(703)
    for name in RECIPE['features']:
        if name not in frame:
            frame[name] = rng.normal(size=len(frame))
    # Preserve modified/shifted synthetic labels; only copy added feature fields.
    for name in RECIPE['features']:
        train[name], test[name] = frame.loc[train.index, name], frame.loc[test.index, name]
    if kind == 'known_nonlinear':
        frame['target_return_5'] = .02 * (frame[RECIPE['features'][0]] ** 2 - 1)
        train['target_return_5'], test['target_return_5'] = frame.loc[train.index, 'target_return_5'], frame.loc[test.index, 'target_return_5']
    recipe, iterations = RECIPE, TREES
    if kind == 'small_subset_overfit':
        recipe = {**RECIPE, 'params': {'max_depth': 8, 'eta': .15, 'min_child_weight': 1, 'alpha': 0, 'lambda': .01},
                  'max_iterations': 400}
        iterations = 400
    return frame, train, test, recipe, iterations


def learning(experiment, *, deadline=None, repo=None):
    root = experiment.root / 'capacity-learning-ledger'
    used, _ = fit_consumption(root)
    if used > 4:
        raise ValueError('Synthetic budget exceeded')
    identity = {k: experiment.identity[k] for k in ('source_id', 'versions', 'python')}
    identity.update(kind='synthetic_learning_only', policy=VALIDATION_POLICY, profile=PROFILE)
    ledger, measurements, keys = Ledger(root, identity), {}, {}
    count = [used]
    def before():
        if deadline is not None and time.monotonic() >= deadline:
            raise FitBudgetReached('Synthetic session time exhausted; checkpoint retained')
        if count[0] >= 4:
            raise FitBudgetReached('Synthetic budget consumed; no automatic replacement')
        if repo is not None and research_source_identity(repo)['source_id'] != experiment.identity['source_id']:
            raise ValueError('Source changed during learning controls')
        count[0] += 1
    for kind in KINDS:
        frame, train, test, recipe, iterations = synthetic(kind)
        record = ledger.run({'role': kind, 'recipe': recipe, 'iterations': iterations, 'seed': 703},
                            lambda work, frame=frame, train=train, test=test, recipe=recipe, iterations=iterations:
                                fit_predict(frame, train, None, test, recipe, work, iterations=iterations), before_compute=before)
        predicted = np.asarray(record['result']['predictions'])
        baseline = evaluate(test.cotton_close.to_numpy(), test.target_return_5.to_numpy(), np.zeros(len(test)))['mae']
        error = evaluate(test.cotton_close.to_numpy(), test.target_return_5.to_numpy(), predicted)['mae']
        measurements[f'xgboost/{kind}'] = {'relative_mae_gain': 1-error/baseline}
        keys[kind] = record['experiment_id']
    failures = control_failures(measurements, ('xgboost',))
    nonlinear = measurements['xgboost/known_nonlinear']['relative_mae_gain']
    if not np.isfinite(nonlinear) or nonlinear < .5:
        failures.append('Frozen shallow recipe fails strong nonlinear signal')
    freeze_record(experiment.root / 'capacity-learning.json', {'status': 'failed' if failures else 'passed',
                  'policy': VALIDATION_POLICY, 'measurements': measurements, 'failures': failures,
                  'fit_ids': keys, 'market_evidence': False, 'weak_signal_power_claimed': False})
    if failures:
        raise ValueError('Learning control failed; no market fit or automatic recipe change')


def verify_output(experiment, history, fold):
    records = []
    for job in fold['jobs']:
        train = parent.projection(history, job['train_dates'], 'named')
        expected = {'recipe': RECIPE, 'role': f'xgboost-{fold["year"]}-{job["role"]}-{job["bucket"]}',
                    'iterations': TREES, 'train_dates': job['train_dates'], 'validation_dates': [],
                    'test_dates': job['origins'], 'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}
        key = experiment.ledger.key(expected)
        record = read_record(experiment.ledger.root / 'completed' / f'{key}.json')
        if record['identity'] != experiment.identity or record['specification'] != expected or record['experiment_id'] != key:
            raise ValueError('Unexpected fit identity/recipe/cutoff')
        if not record['files'] or record['result']['iterations'] != TREES:
            raise ValueError('Native payload missing or fixed tree count changed')
        for name, checksum in record['files'].items():
            if not safe_member(name) or digest(experiment.ledger.root / name) != checksum:
                raise ValueError('Native payload changed')
        records.append(record)
    expected = parent.make_output(history, fold, 'xgboost', experiment.identity['design'], records)
    if read_record(experiment.root / 'capacity-outputs' / f'xgboost-{fold["year"]}.json') != expected:
        raise ValueError('Output/selection differs from native receipts')
    return expected


def run(experiment, repo, max_minutes):
    if not 0 < max_minutes <= 30:
        raise ValueError('CPU session must be <=30 minutes')
    ready, history, _ = verify(experiment.root, repo)
    deadline = time.monotonic() + max_minutes*60
    try:
        learning(experiment, deadline=deadline, repo=repo)
    except FitBudgetReached:
        return {'status': 'planned_pause', 'phase': 'learning', 'market_fits': 0}
    used, _ = fit_consumption(experiment.root / 'ledger')
    count = [used]
    def before():
        if time.monotonic() >= deadline or count[0] >= ready['identity']['design']['fit_budget']['market']:
            raise FitBudgetReached('Frozen fit/time budget; checkpoint retained')
        if research_source_identity(repo)['source_id'] != experiment.identity['source_id']:
            raise ValueError('Source changed during execution')
        count[0] += 1
    experiment.before_compute = before
    try:
        for fold in ready['identity']['design']['folds']:
            path = experiment.root / 'capacity-outputs' / f'xgboost-{fold["year"]}.json'
            if path.exists():
                verify_output(experiment, history, fold)
                continue
            records = []
            for job in fold['jobs']:
                train = parent.projection(history, job['train_dates'], 'named')
                test = parent.projection(history, job['origins'], 'named', test=True)
                records.append(experiment.fit(RECIPE, train, None, test,
                               f'xgboost-{fold["year"]}-{job["role"]}-{job["bucket"]}', iterations=TREES))
            freeze_record(path, parent.make_output(history, fold, 'xgboost', ready['identity']['design'], records))
            verify_output(experiment, history, fold)
    except FitBudgetReached:
        return {'status': 'planned_pause', 'completed_market_fits': len(experiment.ledger.results())}
    if len(experiment.ledger.results()) != ready['identity']['design']['fit_budget']['market']:
        raise ValueError('Exact fit count required')
    verify(experiment.root, repo)
    return {'status': 'complete', 'market_fits': len(experiment.ledger.results()), 'synthetic_fits': 4,
            'reference_ridge_fits': 0, 'skill_claimed': False}


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder = root_path(args.drive_root, args.experiment)
    if args.mirror_root:
        raise ValueError('No remote mirror or model promotion')
    if args.stage == 'prepare':
        if not args.registration_root or not args.decision_contract:
            raise ValueError('Explicit preregistration and contract required')
        ready = prepare(args.repo, folder, args.registration_root, args.decision_contract)
        result = {'status': 'prepared', 'fit_budget': ready['identity']['design']['fit_budget']}
    elif args.stage in ('status', 'pilot-plan'):
        ready, _, _ = verify(folder, args.repo)
        result = {'status': 'ready', 'fit_budget': ready['identity']['design']['fit_budget'],
                  'completed_fits': len(list((folder / 'ledger/completed').glob('*.json')))}
    elif args.stage == 'pilot':
        with writer(folder):
            result = run(Experiment(folder, repo=args.repo), args.repo, args.max_minutes)
    else:
        raise ValueError('Only prepare/status/pilot-plan/pilot; independent no-fit report required')
    print(json.dumps(result, indent=2))
