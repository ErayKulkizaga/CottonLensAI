"""One protected-clock CFTC content contrast; existing named engine, no grid.

Historical annual values and report+7/L1 alignment are exploratory assumptions,
not first-vintage or historical publication certification.
"""
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import cftc_exploration as source
from cottonlens_ml.research import named_capacity_control as capacity
from cottonlens_ml.research import named_history_control as shared
from cottonlens_ml.research import named_label_control as labels
from cottonlens_ml.research import named_weak_execution as weak
from cottonlens_ml.research.ledger import read_record, writer
from cottonlens_ml.research.recency import copy_immutable

PROFILE = 'named-cftc-source-control-t5-v1'
ARMS = ('numeric', 'mask')
POSITION = tuple(f'cftc_{field}_L1' for field in source.FIELDS)
FIELDS = ('cftc_age_L1', 'cftc_missing_L1', *POSITION)
RECIPE = {**shared.RECIPE, 'features': [*shared.RECIPE['features'], *FIELDS]}
TREES, KINDS = shared.TREES, shared.KINDS
VERSIONS = weak.VERSIONS
INPUTS = ('named/ready.json', 'named/history.parquet', 'inputs/positions-2010-2023.csv',
          'inputs/positions-2010-2023.manifest.json', 'result/candidate.json',
          'result/candidate-history.parquet', 'result/audit.json', 'controls.json')
AUDIT_PROOF = 'research/evidence/named-cftc-information-audit-20261012.json'
RELEASE_PROOF = 'research/evidence/named-cftc-information-audit-release-20261012.json'
DELIVERY_PROOF = 'research/evidence/named-cftc-information-audit-delivery-20261012.json'
PREREGISTRATION = 'research/evidence/named-cftc-source-preregistration-20261012.json'
PREREGISTRATION_DELIVERY = 'research/evidence/named-cftc-source-preregistration-delivery-20261012.json'


def design(parent_ready):
    original = parent_ready['identity']['design']
    folds = original['folds']
    if (original['profile'] != weak.PROFILE or original['recipe'] != shared.RECIPE
            or [f['year'] for f in folds] != [2023] or len(folds[0]['origins']) != 246
            or sum(len(f['jobs']) for f in folds) != 22):
        raise ValueError('Frozen real named 32-field cohort with 22 jobs required')
    return {'profile': PROFILE, 'recipe': RECIPE, 'iterations': TREES, 'arms': list(ARMS),
            'folds': folds, 'parent_design_id': content_id(original),
            'fit_budget': {'market': 44, 'synthetic': 4, 'threads': 2, 'processes': 1, 'session_minutes': 30},
            'feature_policy': 'Protected report+7 calendar days, next Cotton observation, L1; carry<=10 extra observations. Mask only three finite position values; retain age/missing/NaNs.',
            'selection': 'Latest 189 mature labels, 3x63; equal mean relative price-MAE; smallest-weight tie; [0,.25,.5,.75,1]',
            'clock': 'Unchanged parent Cotton date+1 day00:15UTC; market/AMS calendar availability and final vintage assumed. CFTC report+7/L1 is not certified publication.',
            'bootstrap': {'repetitions': 10000, 'seed': 42, 'blocks': [20, 60],
                          'ordinal_slots_preserved': True, 'missing_weight': 0, 'selection_adjusted': False},
            'primary': 'RAW mask-minus-numeric paired named-contract price-MAE; same 37 fields and missingness',
            'secondary': 'Both vs named Naive; past-only shrinkage, direction, active rate, top1pct sensitivity',
            'decision_rules': {
                'candidate': 'Both raw contribution CI lowers>0, raw numeric Naive gain>=5% and direction>=55%; research candidate only',
                'contribution_only': 'Both raw contribution CI lowers>0 but practical point goal not met; no automatic search',
                'below_goal': 'Both numeric RAW vs Naive gain CI uppers<5%; this source/clock/recipe does not rescue goal',
                'otherwise': 'INCONCLUSIVE; no automatic extra fit/grid'},
            'gate_evaluation_allowed': False, 'release_allowed': False,
            'source_timestamp_verified': False, 'first_vintage_verified': False,
            'historical_source_admitted': False, 'independent_holdout': False,
            'deliberate_repeat_reason': 'Old CFTC tested CT=F labels/core24 with Ridge-XGB selection. Completed named core32 excludes CFTC. Test protected L1 content at fixed named target/model; no lag/model grid or clock optimization.'}


def arm_history(history, arm):
    if arm not in ARMS or not set(FIELDS).issubset(history):
        raise ValueError('Exact CFTC arm and attached fields required')
    result = history.copy()
    if arm == 'mask':
        for name in POSITION:
            result[name] = result[name].where(~np.isfinite(result[name]), 0.)
    return result


def attach(history, rows):
    if any(name in history for name in FIELDS):
        raise ValueError('Protected CFTC fields already attached')
    full = source.add_positions(history, rows)
    result = full[[*history.columns, *FIELDS]].copy()
    pd.testing.assert_frame_equal(result[list(history)], history, check_exact=True)
    return result


def verify_inputs(folder, repo):
    """Trust only exact members bound to the freshly recovered private audit."""
    folder, repo = Path(folder), Path(repo)
    release, proof, delivery = [read_record(repo / name) for name in (RELEASE_PROOF, AUDIT_PROOF, DELIVERY_PROOF)]
    inventory_path = folder / 'inputs-inventory.json'
    if (digest(inventory_path) != release['inventory_sha256']
            or delivery['status'] != 'fresh_github_cftc_audit_native_and_scalar_recovery_verified'
            or delivery['fresh_restore']['archive_sha256'] != release['archive']['sha256']
            or not delivery['fresh_restore']['exact_audit_candidate_controls']
            or delivery['historical_source_admitted'] or delivery['first_vintage_verified']):
        raise ValueError('Fresh CFTC audit recovery binding required')
    inventory = json.loads(inventory_path.read_bytes())
    for name in INPUTS:
        entry = inventory[name]
        path = folder / 'inputs' / name
        if digest(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError('Frozen CFTC input changed')
    for name in (*capacity.CORE, 'ml/src/cottonlens_ml/research/cftc_exploration.py'):
        if digest(repo / name) != inventory['source-snapshot/' + name]['sha256']:
            raise ValueError('Shared source/target/fitting code changed')
    audit, candidate = [read_record(folder / 'inputs/result' / name) for name in ('audit.json', 'candidate.json')]
    if (digest(folder / 'inputs/result/audit.json') != proof['audit_sha256']
            or digest(folder / 'inputs/result/candidate-history.parquet') != proof['candidate_sha256']
            or digest(folder / 'inputs/controls.json') != proof['controls_sha256']
            or audit['named_target_source_effect_tested_by_old_experiment']
            or candidate['status'] != 'exploratory_alignment_only_not_admitted_or_preregistered'
            or candidate['release_allowed'] or candidate['new_fits'] != 0):
        raise ValueError('Audit/candidate scope changed')
    parent = read_record(folder / 'inputs/named/ready.json')
    history = pd.read_parquet(folder / 'inputs/named/history.parquet')
    if (digest(folder / 'inputs/named/history.parquet') != parent['history_sha256']
            or candidate['parent_history_sha256'] != parent['history_sha256']
            or parent['identity']['research_data_id'] != frame_identity(history, list(history))
            or parent['identity']['design_id'] != content_id(parent['identity']['design'])):
        raise ValueError('Actual real named parent history changed')
    rows, _ = source.read_positions(folder / 'inputs/inputs/positions-2010-2023.csv')
    full = source.add_positions(history, rows)
    pd.testing.assert_frame_equal(full, pd.read_parquet(folder / 'inputs/result/candidate-history.parquet'), check_exact=True)
    expected = labels.plan(history, {'folds': [{'year': f['year'], 'origins': f['origins']}
                                            for f in parent['identity']['design']['folds']]})
    if expected['folds'] != parent['identity']['design']['folds']:
        raise ValueError('Frozen maturity/cohort/refit jobs changed')
    for fold in expected['folds']:
        for job in fold['jobs']:
            weak.validate_job(history, job)
    return parent, attach(history, rows)


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder / 'ready.json')
    identity = ready['identity']
    if (identity['profile'] != PROFILE or research_source_identity(repo)['source_id'] != identity['source_id']
            or identity['python'] != sys.version.split()[0]
            or identity['versions'] != {name: importlib.metadata.version(name) for name in VERSIONS}
            or digest(repo / 'ml/uv.lock') != identity['lock_sha256']):
        raise ValueError('Frozen source/environment changed; new namespace required')
    required = {'inputs-inventory.json', *('inputs/' + name for name in INPUTS)}
    if (set(identity['input_hashes']) != required
            or set(identity['evidence_hashes']) != {AUDIT_PROOF, RELEASE_PROOF, DELIVERY_PROOF}):
        raise ValueError('Exact frozen input/evidence scope required')
    for name, checksum in identity['evidence_hashes'].items():
        if digest(repo / name) != checksum:
            raise ValueError('Published parent evidence changed')
    for name, checksum in identity['input_hashes'].items():
        if not safe_member(name) or digest(folder / name) != checksum:
            raise ValueError('Frozen input changed')
    old, history = verify_inputs(folder, repo)
    fixed = design(old)
    if (identity['design'] != fixed or identity['design_id'] != content_id(fixed)
            or identity['research_data_id'] != frame_identity(history, list(history))
            or ready['history_sha256'] != digest(folder / 'history.parquet')
            or read_record(folder / 'decision-contract.json') != fixed
            or identity['historical_source_admitted'] or identity['release_allowed']):
        raise ValueError('Frozen source/cohort/design differs')
    pd.testing.assert_frame_equal(history, pd.read_parquet(folder / 'history.parquet'), check_exact=True)
    return ready, history


def synthetic(kind):
    frame, train, test, original, trees = shared.synthetic(kind)
    rng = np.random.default_rng(1703)
    for name in FIELDS:
        frame[name] = rng.normal(size=len(frame))
        train[name], test[name] = frame.loc[train.index, name], frame.loc[test.index, name]
    recipe = {**original, 'features': RECIPE['features']}
    if kind != 'small_subset_overfit':
        x = frame[POSITION[0]]
        signal = .02 * (x**2 - 1) if kind == 'known_nonlinear' else .02 * x
        frame['target_return_5'] = signal
        train['target_return_5'], test['target_return_5'] = signal.loc[train.index], signal.loc[test.index]
        if kind == 'negative_block_shift':
            train['target_return_5'] = np.roll(train.target_return_5.to_numpy(), 137)
    return frame, train, test, recipe, trees


def verify_output(experiment, history, fold, arm):
    return shared.verify_output(experiment, history, fold, arm, control=sys.modules[__name__])


def run(experiment, repo, max_minutes):
    publication_check(experiment.root, repo)
    return shared.run(experiment, repo, max_minutes, control=sys.modules[__name__])


def classify(summaries, intervals):
    result = shared.classify(summaries, intervals)
    result['decision'] = result['decision'].replace('NAMED_HISTORY', 'NAMED_CFTC')
    return result


def publication_check(folder, repo):
    """No fit or prepare before exact preregistration was recovered from GitHub."""
    folder, repo = Path(folder), Path(repo)
    proof, delivery = [read_record(repo / name) for name in (PREREGISTRATION, PREREGISTRATION_DELIVERY)]
    ready = read_record(folder / 'ready.json')
    if (proof['source_id'] != ready['identity']['source_id']
            or proof['ready_sha256'] != digest(folder / 'ready.json')
            or proof['decision_contract_sha256'] != digest(folder / 'decision-contract.json')
            or proof['design_id'] != ready['identity']['design_id'] or proof['new_fits'] != 0
            or delivery['status'] != 'fresh_github_named_cftc_preregistration_verified'
            or delivery['preregistration_sha256'] != digest(repo / PREREGISTRATION)
            or delivery['ready_sha256'] != proof['ready_sha256']
            or delivery['source_id'] != proof['source_id'] or not delivery['exact_members']
            or delivery['new_fits'] != 0):
        raise ValueError('Fresh published preregistration differs')


def prepare(repo, folder, registration_root, decision_contract):
    repo, folder, registration_root, decision_contract = map(Path, (repo, folder, registration_root, decision_contract))
    ready, _ = verify(registration_root, repo)
    publication_check(registration_root, repo)
    proof = read_record(repo / PREREGISTRATION)
    if (proof['ready_sha256'] != digest(registration_root / 'ready.json')
            or proof['decision_contract_sha256'] != digest(decision_contract)
            or read_record(decision_contract) != ready['identity']['design'] or proof['new_fits'] != 0):
        raise ValueError('Published preregistration differs')
    with writer(folder):
        if (folder / 'ready.json').exists():
            if verify(folder, repo)[0] != ready:
                raise ValueError('Existing identity differs; preserve cache')
            return ready
        if any(p.name != '.writer-lock' for p in folder.iterdir()):
            raise ValueError('Preserve partial preparation')
        for p in registration_root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                copy_immutable(p, folder / p.relative_to(registration_root))
    verify(folder, repo)
    return ready


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
        raise ValueError('Only prepare/status/pilot-plan/pilot; frozen independent no-fit report required')
    print(json.dumps(result, indent=2))
