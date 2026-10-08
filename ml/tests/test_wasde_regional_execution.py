"""Synthetic execution and corruption regressions; CI never fits market data."""
import copy
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research import wasde_regional_execution as execute
from cottonlens_ml.research import wasde_regional_pilot as pilot
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.sources.wasde_regional import LEVELS, add_revisions


@pytest.fixture
def packet(tmp_path, monkeypatch):
    repo, registration, folder = [tmp_path / name for name in ('repo', 'registration', 'execution')]
    (repo / 'ml').mkdir(parents=True)
    (repo / 'ml/uv.lock').write_text('synthetic lock')
    (repo / 'research/evidence').mkdir(parents=True)
    counts = {2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 251}
    dates = pd.DatetimeIndex(np.concatenate([pd.bdate_range(f'{year}-01-01', f'{year}-12-31')[:counts.get(year, 250)]
                                           for year in range(2015, 2024)]))
    index = np.arange(len(dates))
    parent = pd.DataFrame({'date': dates, 'cotton_session_index': index, 'cotton_close': 80 + np.sin(index / 10)})
    for i, name in enumerate(FEATURE_NAMES):
        parent[name] = np.sin(index / (i + 2))
    for h in (1, 5):
        parent[f'target_return_{h}'] = np.log(parent.cotton_close.shift(-h) / parent.cotton_close)
        parent[f'target_date_{h}'] = parent.date.shift(-h)
    reports = pd.DataFrame({'report_date': pd.date_range('2016-01-01', '2023-12-01', freq='MS') + pd.Timedelta(days=11)})
    reports['crop_year'] = reports.report_date.dt.year
    for i, name in enumerate(LEVELS):
        reports[name] = 1 + i + np.arange(len(reports)) / 100
    reports = add_revisions(reports)
    monkeypatch.setattr(pilot, 'candidate_rows', lambda *args: reports)
    expanded, design, _ = pilot.make_design(parent, reports)
    assert design['fit_budget']['total'] == 424
    reference = registration / 'inputs/reference'
    reference.mkdir(parents=True)
    parent.to_parquet(reference / 'history.parquet', index=False)
    freeze_record(reference / 'ready.json', {'identity': {'profile': 'full-year-v1'},
                                           'history_sha256': digest(reference / 'history.parquet')})
    expanded.to_parquet(registration / 'history.parquet', index=False)
    identity = {'design': design, 'source_id': 'prior-synthetic-source', 'python': sys.version.split()[0],
                'versions': {name: importlib.metadata.version(name) for name in
                             ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl')},
                'lock_sha256': digest(repo / 'ml/uv.lock')}
    reg = {'profile': pilot.PROFILE, 'identity': identity, 'registration_id': content_id(identity),
           'completed_experiment': False, 'fits': 0, 'availability_verified': False}
    freeze_record(registration / 'preregistered.json', reg)
    freeze_record(registration / 'complete.json', {'completed': True, 'fits': 0, 'model_eligible': False,
                  'registration_id': reg['registration_id'],
                  'files': {path.relative_to(registration).as_posix(): digest(path) for path in registration.rglob('*') if path.is_file()}})
    contract = tmp_path / 'decision-contract.json'
    contract.write_text(json.dumps({'registration_id': reg['registration_id'], 'preregistered_sha256': digest(registration / 'preregistered.json'),
                                   'fits_completed': 0, 'frozen_before_market_fits': True,
                                   'historical_source_admitted': False, 'automatic_release': False}))
    evidence = repo / 'research/evidence' / execute.EVIDENCE
    evidence.write_text(json.dumps({'preregistered_sha256': digest(registration / 'preregistered.json'),
                                   'complete_sha256': digest(registration / 'complete.json'),
                                   'registration_id': reg['registration_id'], 'decision_contract_sha256': digest(contract)}))
    (repo / 'research/registry.json').write_text(json.dumps({'schema': 1, 'records': [],
                    'local_evidence': [{'path': 'evidence/' + execute.EVIDENCE, 'sha256': digest(evidence)}]}))
    (repo / 'research/trials.json').write_text(json.dumps({'schema': 1, 'records': []}))
    execute.prepare(repo, folder, registration, contract)
    return repo, folder, registration, contract


def test_prepare_preserves_registration_and_rejects_environment_change(packet, monkeypatch):
    repo, folder, registration, contract = packet
    old = {path: digest(path) for path in registration.rglob('*') if path.is_file()}
    ready = digest(folder / 'ready.json')
    execute.prepare(repo, folder, registration, contract)
    assert digest(folder / 'ready.json') == ready
    assert all(digest(path) == expected for path, expected in old.items())
    monkeypatch.setattr(execute.sys, 'version', '0.0.0 changed')
    with pytest.raises(ValueError, match='environment changed'):
        execute.prepare(repo, folder, registration, contract)


@pytest.mark.parametrize('name', ['decision-contract.json', 'history.parquet', 'preregistration/complete.json'])
def test_input_corruption_blocks_execution(packet, name):
    repo, folder, _, _ = packet
    path = folder / name
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        execute.run(Experiment(folder, repo=repo), 30)


def test_bound_preregistration_rejects_self_rechecksummed_replacement(packet):
    repo, _, registration, contract = packet
    reg = read_record(registration / 'preregistered.json')
    reg.pop('record_id', None)
    reg['identity']['design']['recipe']['params']['alpha'] = 10.
    reg['registration_id'] = content_id(reg['identity'])
    path = registration / 'preregistered.json'
    path.unlink()  # synthetic source fixture only
    freeze_record(path, reg)
    complete_path = registration / 'complete.json'
    complete = read_record(complete_path)
    complete['files']['preregistered.json'] = digest(path)
    complete['registration_id'] = reg['registration_id']
    complete_path.unlink()
    freeze_record(complete_path, complete)
    evidence = json.loads((repo / 'research/evidence' / execute.EVIDENCE).read_text())
    with pytest.raises(ValueError):
        execute.verify_registration(registration, contract, evidence)


def test_resume_all_four_arms_t1_only_and_report_replays_prices(packet, monkeypatch):
    repo, folder, _, _ = packet
    calls = []

    def inner(exp, spec, fold):
        calls.append((spec['horizon'], fold['year']))
        if len(calls) == 2:
            raise FitBudgetReached('synthetic pause')
        return {'recipe': spec, 'weight': .5, 'iterations': 1, 'inner_score': 1.}

    monkeypatch.setattr(execute, 'learning_control', lambda exp: None)
    monkeypatch.setattr(path_pilot, 'inner_price', inner)
    monkeypatch.setattr(path_pilot, 'predict_chunks', lambda exp, spec, dates, role, count: np.full(len(dates), .001))
    monkeypatch.setattr(execute, 'verify_learning_evidence', lambda *args: None)
    monkeypatch.setattr(execute, 'verify_fit_payloads', lambda *args: None)  # no estimator runs in this synthetic replay
    exp = Experiment(folder, repo=repo)
    assert execute.run(exp, 30)['status'] == 'planned_pause'
    assert execute.compare(folder, 20)['status'] == 'pending'
    assert execute.run(exp, 30)['saved_outputs'] == 20
    assert len(calls) == 21 and all(h == 1 for h, _ in calls)
    result = execute.compare(folder, 20)
    assert result['prediction_rows'] == 5016 and not result['release_allowed']
    rows = pd.read_csv(folder / 'reports/outer-predictions.csv')
    for group, modes in result['arms'].items():
        frame = rows.loc[rows.group.eq(group)]
        actual_price = frame.cotton_close * np.exp(frame.actual_return)
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            error = (actual_price - frame.cotton_close * np.exp(frame[field])).abs().mean()
            assert modes[mode]['price_mae'] == pytest.approx(error)
    before = digest(folder / 'reports/outer-predictions.csv')
    assert execute.compare(folder, 20) == result
    assert digest(folder / 'reports/outer-predictions.csv') == before
    execute.run(exp, 30)
    assert len(calls) == 21
    marker = next((folder / 'regional-outputs').glob('*.json'))
    record = read_record(marker)
    record.pop('record_id', None)
    record['records'][0]['target_date'] = '2020-01-01'
    marker.unlink()  # valid checksum but invalid synthetic target
    freeze_record(marker, record)
    with pytest.raises(ValueError, match='Unmatched'):
        execute.run(exp, 30)


def test_no_scientific_completion_without_all_fit_receipts(packet):
    _, folder, _, _ = packet
    with pytest.raises(ValueError, match='424 fit receipts'):
        execute.verify_fit_payloads(folder, read_record(folder / 'ready.json')['identity'])


def test_fit_budget_and_t5_are_blocked(packet, monkeypatch):
    _, folder, _, _ = packet
    completed = folder / 'ledger/completed'
    completed.mkdir(parents=True)
    for i in range(424):
        (completed / f'{i}.json').touch()
    def shared(*args, **kwargs):
        assert kwargs['horizons'] == (1,)
        kwargs['before_fit']()
    monkeypatch.setattr(execute, 'learning_control', lambda exp: None)
    monkeypatch.setattr(path_pilot, 'run', shared)
    with pytest.raises(FitBudgetReached, match='424-fit'):
        execute.run(SimpleNamespace(root=folder, identity=read_record(folder / 'ready.json')['identity']), 30)
    with pytest.raises(ValueError, match='30 minutes'):
        execute.run(SimpleNamespace(), 31)
    with pytest.raises(ValueError, match=r'Only preregistered T\+1'):
        execute.recipe({}, 'numeric_D0', 5)


def test_decisions_follow_frozen_precedence_not_raw_scores():
    score = {'naive_gain_pct': 5., 'direction_pct': 53.,
             'versus_naive': {str(block): {'gain_ci_pct': [1., 6.]} for block in (20, 60)}}
    arms = {f'numeric_D{delay}': {'selected': copy.deepcopy(score), 'raw': {'naive_gain_pct': 100}}
            for delay in (0, 1)}
    paired = {f'selected_D{delay}': {str(block): {'difference_ci_95': [.01, .1]} for block in (20, 60)}
              for delay in (0, 1)}
    assert execute.decision(arms, paired) == 'CONDITIONAL_CANDIDATE'
    arms['numeric_D0']['selected']['direction_pct'] = 52.
    assert execute.decision(arms, paired) == 'INCREMENTAL_BUT_BELOW_GOAL'
    paired['selected_D1']['60']['difference_ci_95'][0] = -.1
    assert execute.decision(arms, paired) == 'INCONCLUSIVE_OR_DELAY_SENSITIVE'
    for arm in arms.values():
        for bound in arm['selected']['versus_naive'].values():
            bound['gain_ci_pct'][1] = 4.9
    assert execute.decision(arms, paired) == 'FIXED_RECIPE_BELOW_PRACTICAL_GOAL'


def test_launcher_forwards_pinned_cpu_paths_and_caps_session(tmp_path, monkeypatch):
    import runpy
    namespace = runpy.run_path(str(Path(__file__).parents[1] / 'full_year_cpu.py'))
    env = tmp_path / 'cpu'
    python = env / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []
    monkeypatch.setattr(sys, 'argv', ['full_year_cpu.py', 'pilot', '--profile', pilot.PROFILE,
        '--cpu-environment', str(env), '--drive-root', str(tmp_path / 'drive'),
        '--registration-root', str(tmp_path / 'registration'), '--decision-contract', str(tmp_path / 'contract'),
        '--max-minutes', '60'])
    monkeypatch.setattr(namespace['subprocess'], 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    namespace['main']()
    command, kwargs = calls[0]
    assert command[0] == str(python)
    assert command[command.index('--max-minutes') + 1] == '30.0'
    assert command[command.index('--drive-root') + 1] == str(tmp_path / 'drive')
    assert '--registration-root' in command and '--decision-contract' in command
    assert kwargs['env']['COTTONLENS_ALLOW_LOCAL_CPU_TABULAR'] == '1'
    assert kwargs['env']['OMP_NUM_THREADS'] == '2'


def test_correctly_identified_fit_with_corrupt_payload_is_rejected(packet):
    _, folder, _, _ = packet
    identity = read_record(folder / 'ready.json')['identity']
    identity['design']['fit_budget']['total'] = 1  # direct validator unit, not an execution identity
    history = pd.read_parquet(folder / 'history.parquet')
    spec = execute.recipe(identity['design'], 'numeric_D0', 1)
    date = identity['split']['folds'][0]['origins'][0]
    train = execute.training_rows(history, pd.Timestamp(date), spec, identity['split']['coverage_start'])
    from cottonlens_ml.cohort import frame_identity
    specification = {'recipe': spec, 'role': 'synthetic-payload-check', 'repeat_reason': None,
                     'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(), 'validation_dates': [],
                     'test_dates': [date], 'train_identity': frame_identity(train, list(train))}
    key = content_id({'identity': identity, 'specification': specification})
    payload = folder / 'ledger/payloads/synthetic.txt'
    payload.parent.mkdir(parents=True)
    payload.write_text('unchanged')
    freeze_record(folder / f'ledger/completed/{key}.json', {'identity': identity, 'specification': specification,
        'experiment_id': key, 'result': {'predictions': [.001]}, 'files': {'payloads/synthetic.txt': digest(payload)}})
    payload.write_text('corrupted')
    with pytest.raises(ValueError, match='payload corrupted'):
        execute.verify_fit_payloads(folder, identity)


@pytest.mark.parametrize('factor', [1., -1.])
def test_learning_control_uses_separate_ledger_and_rechecks_predictions(packet, monkeypatch, factor):
    repo, folder, _, _ = packet
    calls = []
    def fit(frame, train, validation, test, spec, workspace, **kwargs):
        calls.append(1)
        (workspace / 'synthetic-model.txt').write_text('synthetic fixture')
        prediction = factor * test.target_return_1.to_numpy()
        return {'predictions': prediction.tolist(),
                'metrics': execute.evaluate(test.cotton_close.to_numpy(), test.target_return_1.to_numpy(), prediction)}
    monkeypatch.setattr(execute, 'fit_predict', fit)
    exp = Experiment(folder, repo=repo)
    if factor == 1:
        execute.learning_control(exp)
        execute.verify_learning_evidence(folder, exp.identity)
        execute.learning_control(exp)
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match='control failed'):
            execute.learning_control(exp)
        with pytest.raises(ValueError, match='Learning evidence'):
            execute.verify_learning_evidence(folder, exp.identity)
    assert len(list((folder / 'learning-control-ledger/completed').glob('*.json'))) == 1
    assert not list((folder / 'ledger/completed').glob('*.json'))


def test_prepare_rejects_exact_registered_fit_scope(packet):
    repo, folder, registration, contract = packet
    identity = read_record(folder / 'ready.json')['identity']
    scope = {'identity_id': content_id(identity), 'source_id': identity['source_id'],
             'data_id': identity['research_data_id'], 'split_id': content_id(identity['split']), 'profile': pilot.PROFILE}
    spec = execute.recipe(identity['design'], 'numeric_D0', 1)
    record = {'scope': scope, 'scope_id': content_id(scope), 'recipes': [spec], 'kind': 'market_fit_receipt',
              'completed_fits': 1, 'horizon': 1, 'model_families': ['ridge']}
    record['id'] = content_id({'scope_id': record['scope_id'], 'recipe': spec, 'kind': record['kind']})
    (repo / 'research/trials.json').write_text(json.dumps({'schema': 1, 'records': [record]}))
    with pytest.raises(ValueError, match='already fitted'):
        execute.prepare(repo, folder, registration, contract)
