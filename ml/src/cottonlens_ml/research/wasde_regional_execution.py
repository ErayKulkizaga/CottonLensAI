"""Execute only the registered regional WASDE ablation on existing CPU/Ledger paths."""
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research import availability_clock as clock
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research import wasde_regional_pilot as pilot
from cottonlens_ml.research.diagnostics import (
    VALIDATION_POLICY,
    control_failures,
    synthetic_control,
)
from cottonlens_ml.research.full_year import chunks
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.models import fit_predict
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import training_rows, verify_history

PROFILE = pilot.PROFILE
NAMESPACE = 'regional'
EVIDENCE = 'wasde-regional-preregistration-20261008.json'


def verify_registration(root, contract_path, evidence):
    """Bind values/design to the registered public proof, not a self-updated cache."""
    root, contract_path = Path(root), Path(contract_path)
    complete = pilot.completed_files(root)
    reg = read_record(root / 'preregistered.json')
    contract = json.loads(contract_path.read_text('utf-8'))
    if (digest(root / 'preregistered.json') != evidence['preregistered_sha256']
            or digest(root / 'complete.json') != evidence['complete_sha256']
            or digest(contract_path) != evidence['decision_contract_sha256']
            or reg['registration_id'] != content_id(reg['identity'])
            or reg['registration_id'] != evidence['registration_id']
            or complete['registration_id'] != reg['registration_id']
            or reg['profile'] != PROFILE or reg['completed_experiment'] is not False
            or reg['availability_verified'] is not False or reg['fits'] != 0
            or contract['registration_id'] != reg['registration_id']
            or contract['preregistered_sha256'] != digest(root / 'preregistered.json')
            or contract['fits_completed'] != 0 or contract['frozen_before_market_fits'] is not True
            or contract['historical_source_admitted'] is not False or contract['automatic_release'] is not False):
        raise ValueError('Preregistration/decision contract differs from registered evidence')
    reference = root / 'inputs/reference'
    _, parent = verify_history(reference)
    reports = pilot.candidate_rows(root / 'inputs/candidate', root / 'inputs/numeric_audit', root / 'inputs/csv_audit')
    expanded, design, _ = pilot.make_design(parent, reports)
    if design != reg['identity']['design']:
        raise ValueError('Frozen scientific design changed; new execution is not a new research recipe')
    history = pd.read_parquet(root / 'history.parquet')
    pd.testing.assert_frame_equal(expanded, history, check_exact=True)
    return reg, history, contract


def prepare(repo, folder, registration_root, contract_path):
    repo, folder, registration_root, contract_path = map(Path, (repo, folder, registration_root, contract_path))
    validate(repo / 'research')
    evidence_path = repo / 'research/evidence' / EVIDENCE
    evidence = json.loads(evidence_path.read_text('utf-8'))
    reg, _, _ = verify_registration(registration_root, contract_path, evidence)
    design = reg['identity']['design']
    source = research_source_identity(repo)
    identity = {'profile': PROFILE, 'source_id': source['source_id'], 'design': design,
                'design_id': content_id(design), 'split': design['split'], 'registration_id': reg['registration_id'],
                'research_data_id': design['feature_data_id'],
                'registered_evidence_sha256': digest(evidence_path),
                'registration_complete_sha256': digest(registration_root / 'complete.json'),
                'decision_contract_sha256': digest(contract_path),
                'python': sys.version.split()[0], 'versions': {name: importlib.metadata.version(name) for name in
                    ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl')},
                'lock_sha256': digest(repo / 'ml/uv.lock'),
                'execution': {'device': 'cpu', 'threads': 2, 'processes': 1, 'session_minutes': 30},
                'availability_verified': False, 'historical_source_admitted': False, 'release_allowed': False}
    if any(identity[name] != reg['identity'][name] for name in ('python', 'versions', 'lock_sha256')):
        raise ValueError('Preregistered environment changed; no fitting')
    registry, trials = load_registry(repo / 'research')
    scope = {'identity_id': content_id(identity), 'source_id': identity['source_id'],
             'data_id': identity['research_data_id'], 'split_id': content_id(identity['split']), 'profile': PROFILE}
    lookups = []
    for group in design['groups']:
        result = check(registry, trials, proposal={'scope_id': content_id(scope), 'recipe': recipe(design, group, 1)})
        if result['exact_fit_recipes']:
            raise ValueError('Exact frozen recipe already fitted; deliberate repeat review required')
        lookups.append({'group': group, 'status': result['status'], 'exact_matches': 0,
                        'related_fit_recipes': len(result['fit_recipes']), 'limits': result['limits']})
    with writer(folder):
        if (folder / 'ready.json').exists():
            ready, _ = verify_execution(folder)
            if ready['identity'] != identity:
                raise ValueError('Code/data/environment/contract changed; choose a new namespace')
            return ready
        if any(path.name != '.writer-lock' for path in folder.iterdir()):
            raise ValueError('Preserve incomplete execution preparation; choose a new namespace')
        for name in [*read_record(registration_root / 'complete.json')['files'], 'complete.json']:
            copy_immutable(registration_root / name, folder / 'preregistration' / name)
        copy_immutable(contract_path, folder / 'decision-contract.json')
        copy_immutable(evidence_path, folder / 'registered-evidence.json')
        copy_immutable(registration_root / 'history.parquet', folder / 'history.parquet')
        freeze_record(folder / 'source-manifest.json', source)
        for name, expected in source['files'].items():
            copy_immutable(repo / name, folder / 'source-snapshot' / name)
            if digest(folder / 'source-snapshot' / name) != expected:
                raise ValueError('Execution source changed during preparation')
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
        raise ValueError('Execution proof/contract changed')
    evidence = json.loads((folder / 'registered-evidence.json').read_text('utf-8'))
    reg, expected, _ = verify_registration(folder / 'preregistration', folder / 'decision-contract.json', evidence)
    if (identity['registration_id'] != reg['registration_id'] or identity['design'] != reg['identity']['design']
            or identity['split'] != reg['identity']['design']['split']
            or identity['design_id'] != content_id(identity['design'])):
        raise ValueError('Execution design differs from preregistration')
    pd.testing.assert_frame_equal(history, expected, check_exact=True)
    return ready, history


def recipe(design, group, h):
    if h != 1:
        raise ValueError('Only preregistered T+1 is authorized')
    return {**design['recipe'], 'features': design['groups'][group]}


def record_frame(test, payload, group, h, chosen):
    delay = int(group[-1])
    payload['horizon'], payload['actual_return'], payload['selected_weight'] = h, test.target_return_1, chosen['weight']
    payload['target_date'] = pd.to_datetime(test.target_date_1).dt.strftime('%Y-%m-%d')
    payload['decision_time'] = pd.to_datetime(test.regional_decision_time, utc=True).map(lambda value: value.isoformat())
    for field in ('report_date', 'assumed_available_at'):
        values = pd.to_datetime(test[f'regional_D{delay}_{field}'], utc=True)
        payload[field] = values.map(lambda value: None if pd.isna(value) else value.isoformat())
    for field in ('age_sessions', 'unavailable'):
        payload[field] = test[f'regional_D{delay}_{field}']
    return payload


def output(folder, name, fold, group, h, design, history, *, namespace=NAMESPACE, recipe_fn=recipe):
    record, frame = path_pilot.output(folder, name, fold, group, h, design, history,
                                     namespace=namespace, recipe_fn=recipe_fn)
    actual = rows_at(history, fold['origins']).copy()
    expected = record_frame(actual, pd.DataFrame({'date': fold['origins']}, index=actual.index),
                            group, h, {'weight': record['weight']}).reset_index(drop=True)
    expected['cotton_close'] = actual.cotton_close.to_numpy()
    clock.require_same_targets(frame, expected)
    for field in expected:
        a, b = frame[field], expected[field]
        if not np.array_equal(a.isna(), b.isna()) or not np.array_equal(a.loc[a.notna()], b.loc[b.notna()]):
            raise ValueError(f'Frozen regional output changed: {field}')
    return record, frame


def run(experiment, max_minutes):
    if not 0 < max_minutes <= 30:
        raise ValueError('Regional CPU session must be positive and at most 30 minutes')
    verify_execution(experiment.root)
    learning_control(experiment)
    total = experiment.identity['design']['fit_budget']['total']
    keys = {path.stem for path in (experiment.root / 'ledger/completed').glob('*.json')}
    keys |= {path.parent.name for path in (experiment.root / 'ledger/local-work').glob('*/locally-completed.json')}
    count = [len(keys) + len(list((experiment.root / 'ledger/attempts').glob('*.json')))]

    def before_fit():
        if count[0] >= total:
            raise FitBudgetReached('Approved regional 424-fit budget exhausted; no extra computation')
        count[0] += 1

    def validate_design(reg):
        if reg['design_id'] != content_id(reg['design']) or reg['design'] != experiment.identity['design']:
            raise ValueError('Regional design changed')

    return path_pilot.run(experiment, max_minutes, group_names=tuple(pilot.groups()), namespace=NAMESPACE,
                          horizons=(1,), recipe_fn=recipe, validate_fn=validate_design,
                          record_frame_fn=record_frame, output_fn=output, before_fit=before_fit)


def learning_control(experiment, *, horizon=1, recipe_fn=recipe):
    """One separately indexed synthetic fit; never consumes or masquerades as OOS fits."""
    frame, train, _, test = synthetic_control('known_signal')
    spec = recipe_fn(experiment.identity['design'], 'numeric_D0', horizon)
    rng = np.random.default_rng(703)
    for name in spec['features']:
        if name not in frame:
            frame[name] = rng.normal(size=len(frame))
    train, test = frame.loc[train.index].copy(), frame.loc[test.index].copy()
    identity = {key: experiment.identity[key] for key in ('source_id', 'python', 'versions')}
    identity.update(kind='synthetic_learning_control', policy=VALIDATION_POLICY)
    ledger = Ledger(experiment.root / 'learning-control-ledger', identity)
    record = ledger.run({'role': 'synthetic_known_signal_not_market_evidence', 'recipe': spec, 'synthetic_seed': 703},
                        lambda workspace: fit_predict(frame, train, None, test, spec, workspace, iterations=1))
    naive = evaluate(test.cotton_close.to_numpy(), test[f'target_return_{horizon}'].to_numpy(), np.zeros(len(test)))['mae']
    controls = {'ridge/known_signal': {'relative_mae_gain': 1 - record['result']['metrics']['mae'] / naive}}
    failures = control_failures(controls, ('ridge',))  # Re-evaluate, never trust cached passed prose.
    freeze_record(experiment.root / 'learning-control.json', {'status': 'failed' if failures else 'passed',
                  'policy': VALIDATION_POLICY, 'controls': controls, 'fits': 1, 'market_evidence': False,
                  'fit_record': record['experiment_id'], 'failures': failures})
    if failures:
        raise ValueError('Ridge synthetic learning control failed; no market fitting')


def verify_learning_evidence(folder, identity, *, horizon=1):
    control = read_record(folder / 'learning-control.json')
    record = read_record(folder / f'learning-control-ledger/completed/{control["fit_record"]}.json')
    if (control['policy'] != VALIDATION_POLICY or control['market_evidence'] is not False
            or control['status'] != 'passed' or control_failures(control['controls'], ('ridge',))
            or record['identity']['source_id'] != identity['source_id'] or not record['files']
            or record['specification']['recipe']['horizon'] != horizon):
        raise ValueError('Learning evidence missing, failed or belongs to different source')
    for name, expected in record['files'].items():
        if not safe_member(name) or digest(folder / 'learning-control-ledger' / name) != expected:
            raise ValueError('Learning checkpoint changed')
    _, _, _, test = synthetic_control('known_signal')
    actual, close = test[f'target_return_{horizon}'].to_numpy(), test.cotton_close.to_numpy()
    error = evaluate(close, actual, np.asarray(record['result']['predictions']))['mae']
    baseline = evaluate(close, actual, np.zeros(len(test)))['mae']
    gain = 1 - error / baseline
    if gain < .5 or gain != control['controls']['ridge/known_signal']['relative_mae_gain']:
        raise ValueError('Learning score differs from stored predictions')


def verify_fit_payloads(folder, identity, *, namespace=NAMESPACE, horizon=1, recipe_fn=recipe):
    history = pd.read_parquet(folder / 'history.parquet')
    paths = sorted((folder / 'ledger/completed').glob('*.json'))
    if len(paths) != identity['design']['fit_budget']['total']:
        raise ValueError(f'All {identity["design"]["fit_budget"]["total"]} fit receipts required before scientific comparison')
    records = {}
    allowed = [recipe_fn(identity['design'], group, horizon) for group in identity['design']['groups']]
    for path in paths:
        record = read_record(path)
        spec = record['specification']
        if (record['identity'] != identity or record['experiment_id'] != path.stem
                or path.stem != content_id({'identity': identity, 'specification': spec})
                or spec['recipe'] not in allowed or not record['files']
                or len(record['result']['predictions']) != len(spec['test_dates'])):
            raise ValueError('Fit identity mismatch')
        test = rows_at(history, spec['test_dates'])
        train = training_rows(history, test.date.min(), spec['recipe'], identity['split'].get('coverage_start'))
        if (spec['train_dates'] != train.date.dt.strftime('%Y-%m-%d').tolist()
                or spec['validation_dates'] or spec['train_identity'] != frame_identity(train, list(train))):
            raise ValueError('Fit training cohort/maturity differs from frozen history')
        for name, expected in record['files'].items():
            if not safe_member(name) or digest(folder / 'ledger' / name) != expected:
                raise ValueError('Fit payload corrupted; no completed scientific result')
        key = (spec['role'], content_id(spec['recipe']))
        if key in records:
            raise ValueError('Duplicate fit role/recipe')
        records[key] = record
    for fold in identity['split']['folds']:
        for group in identity['design']['groups']:
            frame = pd.DataFrame(read_record(folder / f'{namespace}-outputs/{group}-t{horizon}-year{fold["year"]}.json')['records'])
            predicted = []
            for index, block in enumerate(chunks(history, fold['origins'])):
                key = (f'path-outer-{group}-{fold["year"]}-{index}', content_id(recipe_fn(identity['design'], group, horizon)))
                saved = records[key]
                if saved['specification']['test_dates'] != block.date.dt.strftime('%Y-%m-%d').tolist():
                    raise ValueError('Outer fit origin mismatch')
                predicted.extend(saved['result']['predictions'])
            if not np.array_equal(frame.raw_predicted_return, predicted):
                raise ValueError('Outer predictions differ from completed fit receipts')


def decision(arms, paired):
    positive = all(paired[f'selected_D{delay}'][str(block)]['difference_ci_95'][0] > 0
                   for delay in (0, 1) for block in (20, 60))
    main = arms['numeric_D0']['selected']
    if positive and main['naive_gain_pct'] >= 5 and main['direction_pct'] >= 53:
        return 'CONDITIONAL_CANDIDATE'
    if all(arms[f'numeric_D{delay}']['selected']['versus_naive'][str(block)]['gain_ci_pct'][1] < 5
           for delay in (0, 1) for block in (20, 60)):
        return 'FIXED_RECIPE_BELOW_PRACTICAL_GOAL'
    return 'INCREMENTAL_BUT_BELOW_GOAL' if positive else 'INCONCLUSIVE_OR_DELAY_SENSITIVE'


def compare(folder, repetitions=10000, *, namespace=NAMESPACE, profile=None,
            verify_fn=None, output_fn=None, decision_fn=None, verify_fits_fn=None,
            horizon=1, verify_learning_fn=None):
    folder = Path(folder)
    ready, history = (verify_fn or verify_execution)(folder)
    design = ready['identity']['design']
    if design['recipe']['horizon'] != horizon:
        raise ValueError('Comparison horizon differs from registered recipe')
    expected = {f'{group}-t{horizon}-year{fold["year"]}.json' for group in design['groups'] for fold in design['split']['folds']}
    paths = list((folder / f'{namespace}-outputs').glob('*.json'))
    if {path.name for path in paths} - expected:
        raise ValueError('Unregistered output exists')
    if {path.name for path in paths} != expected:
        return {'status': 'pending', 'complete_outputs': len(paths), 'required_outputs': len(expected)}
    (verify_learning_fn or verify_learning_evidence)(folder, ready['identity'])
    (verify_fits_fn or verify_fit_payloads)(folder, ready['identity'])
    frames = {group: [] for group in design['groups']}
    for fold in design['split']['folds']:
        for group in design['groups']:
            _, frame = (output_fn or output)(folder, f'{group}-t{horizon}-year{fold["year"]}.json', fold, group, horizon, design, history)
            frame['year'], frame['group'] = fold['year'], group
            frames[group].append(frame)
    combined = {group: pd.concat(values, ignore_index=True) for group, values in frames.items()}
    anchor = combined['mask_D0']
    for frame in combined.values():
        clock.require_same_targets(anchor, frame)
    naive = clock._metrics(anchor, 'predicted_return')[0]
    keep = np.ones(len(anchor), bool)
    keep[np.argsort(-naive, kind='stable')[:int(np.ceil(.01 * len(naive)))]] = False
    arms, paired = {}, {}
    for group in design['groups']:
        arms[group] = {}
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            _, _, score = clock._metrics(combined[group], field)
            years, parts = {}, []
            for frame in frames[group]:
                baseline, error, annual = clock._metrics(frame, field)
                years[str(frame.year.iloc[0])] = annual
                parts.append(np.column_stack([baseline, error]))
            conditional = combined[group].loc[combined[group].age_sessions.between(0, 4)]
            score.update(years=years, year_wins=sum(value['naive_gain_pct'] > 0 for value in years.values()),
                         versus_naive=clock._interval(parts, repetitions),
                         without_top_1pct_naive_errors=clock._metrics(combined[group].loc[keep], field)[2],
                         source_age_0_4=clock._metrics(conditional, field)[2] if len(conditional) else None)
            arms[group][mode] = score
    for delay in (0, 1):
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            left, right = frames[f'mask_D{delay}'], frames[f'numeric_D{delay}']
            parts = [np.column_stack([clock._metrics(a, field)[1], clock._metrics(b, field)[1]])
                     for a, b in zip(left, right, strict=True)]
            paired[f'{mode}_D{delay}'] = clock._interval(parts, repetitions)
            trimmed, annual = [], {}
            offset = 0
            for a, b, matrix in zip(left, right, parts, strict=True):
                mask = keep[offset:offset + len(a)]
                trimmed.append(matrix[mask])
                annual[str(a.year.iloc[0])] = float(100 * (1 - matrix[:, 1].mean() / matrix[:, 0].mean()))
                offset += len(a)
            paired[f'{mode}_D{delay}_trimmed'] = clock._interval(trimmed, repetitions)
            paired[f'{mode}_D{delay}_annual_gain_pct'] = annual
    rows = pd.concat(combined.values(), ignore_index=True)
    export = folder / 'reports/outer-predictions.csv'
    export.parent.mkdir(exist_ok=True)
    encoded = rows.to_csv(index=False, lineterminator='\n').encode()
    if export.exists() and export.read_bytes() != encoded:
        raise ValueError('Frozen prediction export changed')
    if not export.exists():
        export.write_bytes(encoded)
    body = {'status': 'complete', 'profile': profile or PROFILE, 'registration_id': ready['identity']['registration_id'],
            'ready_sha256': digest(folder / 'ready.json'), 'decision_contract_sha256': ready['identity']['decision_contract_sha256'],
            'fits': design['fit_budget']['total'], 'origin_count': len(anchor), 'prediction_rows': len(rows),
            'arms': arms, 'paired': paired, 'decision': (decision_fn or decision)(arms, paired),
            'availability_verified': False, 'gate_evaluated': False, 'release_allowed': False,
            'independent_holdout': False, 'primary': design['rules']['primary'],
            'predictions_sha256': digest(export), 'output_sha256': {path.name: digest(path) for path in paths}}
    freeze_record(folder / 'reports' / f'{namespace}-{content_id(body)[:16]}.json', body)
    return body


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder = root_path(args.drive_root, args.experiment)
    if args.mirror_root:
        raise ValueError('Regional local pilot has no remote mirror')
    if args.stage == 'prepare':
        if not args.registration_root or not args.decision_contract:
            raise ValueError('Explicit --registration-root and --decision-contract required')
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
        raise ValueError('Regional pilot cannot search, lock, export or release')
    print(json.dumps(result, indent=2))
