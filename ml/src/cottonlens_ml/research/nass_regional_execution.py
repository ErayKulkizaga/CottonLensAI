"""Execute the frozen NASS Texas T+1 design on Experiment/Ledger only."""
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import availability_clock as clock
from cottonlens_ml.research import nass_regional_pilot as pilot
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research import wasde_regional_execution as shared
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.wasde_regional_pilot import completed_files

PROFILE = pilot.PROFILE
NAMESPACE = 'nass-regional'
EVIDENCE = 'nass-regional-preregistration-20261009.json'
recipe = shared.recipe


def verify_registration(root, contract_path, evidence):
    root, contract_path = Path(root), Path(contract_path)
    complete = completed_files(root)
    reg, contract = read_record(root / 'preregistered.json'), read_record(contract_path)
    if (digest(root / 'preregistered.json') != evidence['preregistered_sha256']
            or digest(root / 'complete.json') != evidence['complete_sha256']
            or digest(contract_path) != evidence['decision_contract_sha256']
            or reg['registration_id'] != evidence['registration_id']
            or reg['registration_id'] != content_id(reg['identity'])
            or complete['registration_id'] != reg['registration_id']
            or reg['profile'] != PROFILE or reg['fits'] != 0 or reg['completed_experiment'] is not False
            or reg['availability_verified'] is not False or reg['model_eligible'] is not False
            or contract != {'registration_id': reg['registration_id'], 'rules': reg['identity']['design']['rules'],
                            'completed_experiment': False, 'fits': 0, 'automatic_release': False}):
        raise ValueError('NASS preregistration/contract differs from registered evidence')
    for name, expected in reg['identity']['input_sha256'].items():
        if digest(root / 'inputs' / name) != expected:
            raise ValueError('Preregistered input changed')
    _, parent = verify_history(root / 'inputs/reference')
    panel_path = root / 'inputs/candidate/panel.json'
    numeric = json.loads((root / 'inputs/evidence/nass-regional-audit-20261009.json').read_bytes())
    if digest(panel_path) != numeric['panel_sha256']:
        raise ValueError('Panel differs from registered numeric audit')
    reports = pilot.report_features(json.loads(panel_path.read_bytes()))
    expanded, design, _ = pilot.make_design(parent, reports)
    if design != reg['identity']['design']:
        raise ValueError('Frozen scientific design changed')
    pd.testing.assert_frame_equal(expanded, pd.read_parquet(root / 'history.parquet'), check_exact=True)
    return reg, expanded, contract


def prepare(repo, folder, registration_root, contract_path):
    repo, folder, registration_root, contract_path = map(Path, (repo, folder, registration_root, contract_path))
    validate(repo / 'research')
    evidence_path = repo / 'research/evidence' / EVIDENCE
    evidence = json.loads(evidence_path.read_bytes())
    reg, _, _ = verify_registration(registration_root, contract_path, evidence)
    design, source = reg['identity']['design'], research_source_identity(repo)
    identity = {'profile': PROFILE, 'source_id': source['source_id'], 'design': design,
                'design_id': content_id(design), 'split': design['split'], 'registration_id': reg['registration_id'],
                'research_data_id': design['feature_data_id'], 'registered_evidence_sha256': digest(evidence_path),
                'registration_complete_sha256': digest(registration_root / 'complete.json'),
                'decision_contract_sha256': digest(contract_path), 'python': sys.version.split()[0],
                'versions': {p: importlib.metadata.version(p) for p in
                    ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl')},
                'lock_sha256': digest(repo / 'ml/uv.lock'),
                'execution': {'device': 'cpu', 'threads': 2, 'processes': 1, 'session_minutes': 30},
                'availability_verified': False, 'historical_source_admitted': False, 'release_allowed': False}
    if any(identity[k] != reg['identity'][k] for k in ('python', 'versions', 'lock_sha256')):
        raise ValueError('Preregistered environment changed; no fitting')
    registry, trials = load_registry(repo / 'research')
    scope = {'identity_id': content_id(identity), 'source_id': identity['source_id'],
             'data_id': identity['research_data_id'], 'split_id': content_id(identity['split']), 'profile': PROFILE}
    lookups = []
    for group in design['groups']:
        result = check(registry, trials, proposal={'scope_id': content_id(scope), 'recipe': recipe(design, group, 1)})
        if result['exact_fit_recipes']:
            raise ValueError('Exact frozen recipe already fitted')
        lookups.append({'group': group, 'status': result['status'], 'exact_matches': 0,
                        'related_fit_recipes': len(result['fit_recipes'])})
    with writer(folder):
        if (folder / 'ready.json').exists():
            ready, _ = verify_execution(folder)
            if ready['identity'] != identity:
                raise ValueError('Code/data/environment changed; choose new namespace')
            return ready
        if any(p.name != '.writer-lock' for p in folder.iterdir()):
            raise ValueError('Preserve incomplete execution; choose new namespace')
        for name in [*read_record(registration_root / 'complete.json')['files'], 'complete.json']:
            copy_immutable(registration_root / name, folder / 'preregistration' / name)
        for path, name in [(contract_path, 'decision-contract.json'), (evidence_path, 'registered-evidence.json'),
                           (registration_root / 'history.parquet', 'history.parquet')]:
            copy_immutable(path, folder / name)
        freeze_record(folder / 'source-manifest.json', source)
        for name, expected in source['files'].items():
            copy_immutable(repo / name, folder / 'source-snapshot' / name)
            if digest(folder / 'source-snapshot' / name) != expected:
                raise ValueError('Execution source changed during copy')
        freeze_record(folder / 'history-check.json', {'lookups': lookups, 'fits': 0})
        if (source != research_source_identity(repo) or digest(evidence_path) != identity['registered_evidence_sha256']
                or digest(contract_path) != identity['decision_contract_sha256']):
            raise ValueError('Input/code changed during preparation')
        ready = {'identity': identity, 'history_sha256': digest(folder / 'history.parquet'),
                 'created_at': datetime.now(UTC).isoformat()}
        freeze_record(folder / 'ready.json', ready)
    verify_execution(folder)
    return ready


def verify_execution(folder):
    folder = Path(folder)
    ready, history = verify_history(folder)
    identity = ready['identity']
    if (identity['profile'] != PROFILE or identity['availability_verified'] is not False
            or identity['historical_source_admitted'] is not False or identity['release_allowed'] is not False
            or digest(folder / 'registered-evidence.json') != identity['registered_evidence_sha256']
            or digest(folder / 'decision-contract.json') != identity['decision_contract_sha256']
            or digest(folder / 'preregistration/complete.json') != identity['registration_complete_sha256']):
        raise ValueError('Execution identity/proof changed')
    evidence = json.loads((folder / 'registered-evidence.json').read_bytes())
    reg, expected, _ = verify_registration(folder / 'preregistration', folder / 'decision-contract.json', evidence)
    if (identity['registration_id'] != reg['registration_id'] or identity['design'] != reg['identity']['design']
            or identity['design_id'] != content_id(identity['design']) or identity['split'] != reg['identity']['design']['split']):
        raise ValueError('Execution design differs from registration')
    pd.testing.assert_frame_equal(history, expected, check_exact=True)
    return ready, history


def record_frame(test, payload, group, h, chosen):
    delay = int(group[-1])
    payload['horizon'], payload['actual_return'], payload['selected_weight'] = h, test.target_return_1, chosen['weight']
    payload['target_date'] = pd.to_datetime(test.target_date_1).dt.strftime('%Y-%m-%d')
    payload['decision_time'] = pd.to_datetime(test.nass_decision_time, utc=True).map(lambda t: t.isoformat())
    for field in ('report_date', 'week_ending', 'assumed_available_at'):
        values = pd.to_datetime(test[f'nass_D{delay}_{field}'], utc=True)
        payload[field] = values.map(lambda t: None if pd.isna(t) else t.isoformat())
    for field in ('report_sha256', 'age_sessions', 'unavailable'):
        payload[field] = test[f'nass_D{delay}_{field}']
    # Unknown provenance is nullable metadata, never a missing prediction/label.
    payload['age_sessions'] = payload.age_sessions.astype(object).where(payload.age_sessions.notna(), None)
    for name in pilot.REGIONAL:
        payload['known_' + name] = test[f'numeric_D{delay}_{name}'].notna().astype(int)
    return payload


def output(folder, name, fold, group, h, design, history, *, namespace=NAMESPACE, recipe_fn=recipe):
    saved, frame = path_pilot.output(folder, name, fold, group, h, design, history,
                                     namespace=namespace, recipe_fn=recipe_fn)
    actual = rows_at(history, fold['origins']).copy()
    expected = record_frame(actual, pd.DataFrame({'date': fold['origins']}, index=actual.index),
                            group, h, {'weight': saved['weight']}).reset_index(drop=True)
    expected['cotton_close'] = actual.cotton_close.to_numpy()
    clock.require_same_targets(frame, expected)
    for field in expected:
        a, b = frame[field], expected[field]
        if not np.array_equal(a.isna(), b.isna()) or not np.array_equal(a.loc[a.notna()], b.loc[b.notna()]):
            raise ValueError('Frozen NASS output changed: ' + field)
    return saved, frame


def learning_control(experiment):
    root = experiment.root / 'learning-control-ledger'
    consumed, reusable = fit_consumption(root)
    if consumed > 1 or (consumed == 1 and not reusable):
        raise ValueError('Synthetic fit attempt already consumed; no automatic extra fit')
    shared.learning_control(experiment)


def fit_consumption(root):
    """Count computed fits once even if only checkpoint publication failed."""
    keys = {p.stem for p in (root / 'completed').glob('*.json')}
    for directory in ('local-work', 'synthetic-work'):
        keys |= {p.parent.name for p in (root / directory).glob('*/locally-completed.json')}
    failed = [read_record(p)['experiment_id'] for p in (root / 'attempts').glob('*.json')]
    return len(keys) + sum(key not in keys for key in failed), keys


def run(experiment, max_minutes):
    if not 0 < max_minutes <= 30:
        raise ValueError('NASS CPU session must be positive and at most 30 minutes')
    verify_execution(experiment.root)
    learning_control(experiment)
    total = experiment.identity['design']['fit_budget']['total']
    consumed, _ = fit_consumption(experiment.root / 'ledger')
    count = [consumed]
    def before_fit():
        if count[0] >= total:
            raise FitBudgetReached(f'Approved NASS {total}-fit budget exhausted')
        count[0] += 1
    def validate_design(reg):
        if reg['design_id'] != content_id(reg['design']) or reg['design'] != experiment.identity['design']:
            raise ValueError('NASS design changed')
    return path_pilot.run(experiment, max_minutes, group_names=tuple(pilot.groups()), namespace=NAMESPACE,
                          horizons=(1,), recipe_fn=recipe, validate_fn=validate_design,
                          record_frame_fn=record_frame, output_fn=output, before_fit=before_fit)


def verify_fit_payloads(folder, identity):
    return shared.verify_fit_payloads(folder, identity, namespace=NAMESPACE)


def decision(arms, paired):
    verdict = shared.decision(arms, paired)
    if verdict == 'CONDITIONAL_CANDIDATE' and arms['numeric_D0']['selected']['year_wins'] < 6:
        return 'INCREMENTAL_BUT_BELOW_GOAL'
    return verdict


def compare(folder, repetitions=10000):
    return shared.compare(folder, repetitions, namespace=NAMESPACE, profile=PROFILE,
                          verify_fn=verify_execution, output_fn=output, decision_fn=decision,
                          verify_fits_fn=verify_fit_payloads)


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder = root_path(args.drive_root, args.experiment)
    if args.mirror_root:
        raise ValueError('NASS local pilot has no remote mirror')
    if args.stage == 'prepare':
        if not args.registration_root or not args.decision_contract:
            raise ValueError('Explicit registration-root and decision-contract required')
        ready = prepare(args.repo, folder, args.registration_root, args.decision_contract)
        result = {'status': 'prepared', 'source_id': ready['identity']['source_id'],
                  'registration_id': ready['identity']['registration_id'], 'fit_budget': ready['identity']['design']['fit_budget']}
    elif args.stage in ('status', 'pilot-plan'):
        ready, _ = verify_execution(folder)
        result = {'fit_budget': ready['identity']['design']['fit_budget'],
                  'completed_fits': len(list((folder / 'ledger/completed').glob('*.json'))),
                  'saved_outputs': len(list((folder / f'{NAMESPACE}-outputs').glob('*.json')))}
    elif args.stage == 'pilot':
        with writer(folder):
            result = run(Experiment(folder, repo=args.repo), args.max_minutes)
    elif args.stage in ('compare', 'report'):
        with writer(folder):
            result = compare(folder)
    else:
        raise ValueError('NASS cannot search, lock, export or release')
    print(json.dumps(result, indent=2))
