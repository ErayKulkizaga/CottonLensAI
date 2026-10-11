"""One same-contract history numeric/mask contrast on frozen named cohorts."""
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import named_capacity_control as capacity
from cottonlens_ml.research import named_history as source
from cottonlens_ml.research import named_label_control as labels
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.recency import copy_immutable

PROFILE = 'named-history-source-control-t5-v1'
ARMS = ('numeric', 'mask')
RECIPE = {**capacity.RECIPE, 'features': [*capacity.RECIPE['features'], *source.FEATURES]}
TREES = capacity.TREES
KINDS = capacity.KINDS


def design(parent_ready):
    old = parent_ready['identity']['design']
    if old['profile'] != capacity.PROFILE or old['recipe'] != capacity.RECIPE:
        raise ValueError('Completed frozen named-capacity parent required')
    return {'profile': PROFILE, 'recipe': RECIPE, 'iterations': TREES, 'arms': list(ARMS),
            'folds': old['folds'], 'parent_design_id': content_id(old),
            'feature_policy': source.POLICY,
            'fit_budget': {**old['fit_budget'], 'market': 2 * sum(len(f['jobs']) for f in old['folds'])},
            'selection': old['selection'], 'clock': old['clock'], 'bootstrap': old['bootstrap'],
            'primary': 'RAW mask-minus-numeric paired same-contract price-MAE; same 32 fields/missingness',
            'secondary': 'Numeric and mask vs Naive; past-only shrinkage, direction, top1pct sensitivity',
            'decision_rules': {
                'candidate': 'Both RAW mask-minus-numeric CI lowers>0, numeric Naive gain>=5% and direction>=55%; research candidate only',
                'contribution_only': 'Both RAW contrast CI lowers>0 but practical point goal not met; no automatic search',
                'below_goal': 'Both numeric RAW vs Naive CI uppers<5%; this fixed source/recipe does not rescue goal',
                'otherwise': 'INCONCLUSIVE; no automatic extra fit/grid'},
            'gate_evaluation_allowed': False, 'release_allowed': False,
            'source_timestamp_verified': False, 'first_vintage_verified': False,
            'deliberate_repeat_reason': 'Completed named-label/capacity studies retain CT price history. New same-contract histories were not fit; compare identical-width numeric/mask arms, not a new model/grid.'}


def classify(summaries, intervals):
    positive = all(intervals[f'mask_minus_numeric_raw_{b}']['difference_ci_95'][0] > 0 for b in (20, 60))
    below = all(intervals[f'numeric_raw_vs_naive_{b}']['gain_ci_95_pct'][1] < 5 for b in (20, 60))
    useful = summaries['numeric_raw']['gain_pct'] >= 5 and summaries['numeric_raw']['direction_pct'] >= 55
    if positive:
        decision = 'NAMED_HISTORY_CANDIDATE_RESEARCH_ONLY' if useful else 'NAMED_HISTORY_CONTRIBUTION_BELOW_POINT_GOAL'
    else:
        decision = 'FIXED_NAMED_HISTORY_BELOW_PRACTICAL_GOAL' if below else 'INCONCLUSIVE_NAMED_HISTORY_CONTROL'
    return {'decision': decision, 'source_contribution_supported': positive,
            'numeric_practical_effect_excluded': below, 'practical_point_thresholds_met': useful}


def attach(history, quotes):
    if any(f in history for f in source.FEATURES):
        raise ValueError('History fields already attached')
    result = history.copy()
    features, _ = source.build(history, quotes)
    for f in source.FEATURES:
        result[f] = features[f].to_numpy()
    pd.testing.assert_frame_equal(result[list(history)], history, check_exact=True)
    return result


def arm_history(history, arm):
    if arm not in ARMS or not set(source.FEATURES).issubset(history):
        raise ValueError('Exact source arm and attached fields required')
    result = history.copy()
    if arm == 'mask':
        result[list(source.FEATURES)] = source.mask(result[list(source.FEATURES)])
    return result


def verify_parent(folder, repo):
    folder, repo = Path(folder), Path(repo)
    parent = folder / 'parent'
    for name, checksum in read_record(folder / 'parent-files.json')['files'].items():
        if not safe_member(name) or digest(parent / name) != checksum:
            raise ValueError('Frozen completed parent changed')
    ready, history, _ = capacity.verify(parent, parent / 'source-snapshot')
    for name in capacity.CORE:
        if digest(repo / name) != digest(parent / 'source-snapshot' / name):
            raise ValueError('Shared target/feature/fitting code changed')
    proof = read_record(folder / 'readiness-proof.json')
    if (proof['status'] != 'completed_named_history_readiness_no_skill_test'
            or proof['policy'] != source.POLICY or proof['parent_ready_sha256'] != digest(parent / 'ready.json')):
        raise ValueError('Readiness policy/parent mismatch')
    delivery = read_record(folder / 'readiness-delivery.json')
    if delivery['status'] != 'fresh_github_named_history_readiness_verified' or delivery['source_id'] != proof['source_id']:
        raise ValueError('Fresh published readiness required')
    quotes = json.loads((parent / 'parent/inputs/quotes.json').read_bytes())['rows']
    enhanced = attach(history, quotes)
    # Recreate the published numeric bytes; parsing CSV must not change float precision.
    encoded = enhanced[list(source.FEATURES)].assign(date=enhanced.date.dt.strftime('%Y-%m-%d')).to_csv(index=False, lineterminator='\n').encode()
    import hashlib
    if hashlib.sha256(encoded).hexdigest() != proof['numeric_sha256']:
        raise ValueError('Published history feature bytes changed')
    return ready, enhanced


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder / 'ready.json')
    identity = ready['identity']
    if identity['profile'] != PROFILE or research_source_identity(repo)['source_id'] != identity['source_id']:
        raise ValueError('Source/profile changed; new namespace required')
    if (sys.version.split()[0] != identity['python']
            or {p: importlib.metadata.version(p) for p in identity['versions']} != identity['versions']
            or digest(repo / 'ml/uv.lock') != identity['lock_sha256']):
        raise ValueError('Frozen environment changed')
    for name in ('parent-files.json', 'readiness-proof.json', 'readiness-delivery.json'):
        if digest(folder / name) != identity['input_hashes'][name]:
            raise ValueError('Frozen input changed')
    old, history = verify_parent(folder, repo)
    expected = design(old)
    if (identity['design'] != expected or identity['design_id'] != content_id(expected)
            or identity['research_data_id'] != frame_identity(history, list(history))
            or digest(folder / 'history.parquet') != ready['history_sha256']):
        raise ValueError('Frozen source/cohort/design differs')
    pd.testing.assert_frame_equal(history, pd.read_parquet(folder / 'history.parquet'), check_exact=True)
    return ready, history


def prepare(repo, folder, registration_root, decision_contract):
    repo, folder, registration_root, decision_contract = map(Path, (repo, folder, registration_root, decision_contract))
    ready, _ = verify(registration_root, repo)
    proof = read_record(repo / 'research/evidence/named-history-source-preregistration-20261011.json')
    if (proof['ready_sha256'] != digest(registration_root / 'ready.json')
            or proof['decision_contract_sha256'] != digest(decision_contract)
            or read_record(decision_contract) != ready['identity']['design'] or proof['new_fits'] != 0):
        raise ValueError('Published preregistration differs')
    with writer(folder):
        if (folder / 'ready.json').exists():
            old, _ = verify(folder, repo)
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
    frame, train, test, original, trees = capacity.synthetic(kind)
    rng = np.random.default_rng(1703)
    for name in source.FEATURES:
        frame[name] = rng.normal(size=len(frame))
        train[name], test[name] = frame.loc[train.index, name], frame.loc[test.index, name]
    recipe = {**original, 'features': RECIPE['features']}
    if kind != 'small_subset_overfit':
        x = frame[source.FEATURES[0]]
        signal = .02 * (x**2 - 1) if kind == 'known_nonlinear' else .02 * x
        frame['target_return_5'] = signal
        train['target_return_5'] = signal.loc[train.index].to_numpy()
        test['target_return_5'] = signal.loc[test.index].to_numpy()
        if kind == 'negative_block_shift':
            train['target_return_5'] = np.roll(train.target_return_5.to_numpy(), 137)
    return frame, train, test, recipe, trees


def verify_output(experiment, history, fold, arm):
    h = arm_history(history, arm)
    records = []
    for job in fold['jobs']:
        train = labels.projection(h, job['train_dates'], 'named')
        expected = {'recipe': RECIPE, 'role': f'{arm}-{fold["year"]}-{job["role"]}-{job["bucket"]}',
                    'iterations': TREES, 'train_dates': job['train_dates'], 'validation_dates': [],
                    'test_dates': job['origins'], 'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}
        key = experiment.ledger.key(expected)
        record = read_record(experiment.ledger.root / 'completed' / f'{key}.json')
        if record['identity'] != experiment.identity or record['specification'] != expected or record['experiment_id'] != key:
            raise ValueError('Unexpected source-arm fit identity/recipe/cutoff')
        if not record['files'] or record['result']['iterations'] != TREES:
            raise ValueError('Native payload missing or tree count changed')
        for name, checksum in record['files'].items():
            if not safe_member(name) or digest(experiment.ledger.root / name) != checksum:
                raise ValueError('Native payload changed')
        records.append(record)
    expected = labels.make_output(h, fold, arm, experiment.identity['design'], records)
    if read_record(experiment.root / 'history-outputs' / f'{arm}-{fold["year"]}.json') != expected:
        raise ValueError('Output/selection differs from native receipts')
    return expected


def run(experiment, repo, max_minutes):
    if not 0 < max_minutes <= 30:
        raise ValueError('CPU session must be <=30 minutes')
    ready, history = verify(experiment.root, repo)
    deadline = time.monotonic() + max_minutes * 60
    try:
        capacity.learning(experiment, deadline=deadline, repo=repo, synthetic_fn=synthetic, profile=PROFILE)
    except FitBudgetReached:
        return {'status': 'planned_pause', 'phase': 'learning', 'market_fits': 0}
    used, _ = fit_consumption(experiment.root / 'ledger')
    budget = ready['identity']['design']['fit_budget']['market']
    if used > budget:
        raise ValueError('Market attempt budget exceeded')
    count = [used]
    def before():
        if time.monotonic() >= deadline or count[0] >= budget:
            raise FitBudgetReached('Frozen fit/time budget; checkpoints retained')
        if research_source_identity(repo)['source_id'] != experiment.identity['source_id']:
            raise ValueError('Source changed during execution')
        count[0] += 1
    experiment.before_compute = before
    original = experiment.history
    try:
        for fold in ready['identity']['design']['folds']:
            for arm in ARMS:
                path = experiment.root / 'history-outputs' / f'{arm}-{fold["year"]}.json'
                if path.exists():
                    verify_output(experiment, history, fold, arm)
                    continue
                h = arm_history(history, arm)
                # inputs() reads Experiment.history, even for window1. Mask the full
                # input calendar, not just the projected train/test DataFrames.
                experiment.history = h
                records = []
                for job in fold['jobs']:
                    train = labels.projection(h, job['train_dates'], 'named')
                    test = labels.projection(h, job['origins'], 'named', test=True)
                    records.append(experiment.fit(RECIPE, train, None, test,
                                   f'{arm}-{fold["year"]}-{job["role"]}-{job["bucket"]}', iterations=TREES))
                freeze_record(path, labels.make_output(h, fold, arm, ready['identity']['design'], records))
                verify_output(experiment, history, fold, arm)
    except FitBudgetReached:
        return {'status': 'planned_pause', 'completed_market_fits': len(experiment.ledger.results())}
    finally:
        experiment.history = original
    if len(experiment.ledger.results()) != budget:
        raise ValueError('Exact fit count required')
    verify(experiment.root, repo)
    return {'status': 'complete', 'market_fits': budget, 'synthetic_fits': 4, 'skill_claimed': False}


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
        ready, _ = verify(folder, args.repo)
        result = {'status': 'ready', 'fit_budget': ready['identity']['design']['fit_budget'],
                  'completed_fits': len(list((folder / 'ledger/completed').glob('*.json')))}
    elif args.stage == 'pilot':
        with writer(folder):
            result = run(Experiment(folder, repo=args.repo), args.repo, args.max_minutes)
    else:
        raise ValueError('Only prepare/status/pilot-plan/pilot; independent frozen no-fit report required')
    print(json.dumps(result, indent=2))
