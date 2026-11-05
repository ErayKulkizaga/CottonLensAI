"""Synthetic execution regressions: no market fits in CI."""
import copy
import importlib.metadata
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import nass_regional_execution as execute
from cottonlens_ml.research import nass_regional_pilot as pilot
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from test_nass_regional_pilot import panel


@pytest.fixture
def packet(tmp_path):
    repo, registration, folder = [tmp_path / n for n in ('repo', 'registration', 'execution')]
    (repo / 'ml').mkdir(parents=True)
    (repo / 'ml/uv.lock').write_text('synthetic lock')
    (repo / 'research/evidence').mkdir(parents=True)
    counts = {2016: 250, 2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 251}
    dates = pd.DatetimeIndex(np.concatenate([pd.bdate_range(f'{y}-01-01', f'{y}-12-31')[:counts.get(y, 250)]
                                            for y in range(2010, 2024)]))
    index = np.arange(len(dates))
    parent = pd.DataFrame({'date': dates, 'cotton_session_index': index, 'cotton_close': 80 + np.sin(index / 10)})
    for i, name in enumerate(FEATURE_NAMES):
        parent[name] = np.sin(index / (i + 2))
    for h in (1, 5):
        parent[f'target_return_{h}'] = np.log(parent.cotton_close.shift(-h) / parent.cotton_close)
        parent[f'target_date_{h}'] = parent.date.shift(-h)
    reference = registration / 'inputs/reference'
    reference.mkdir(parents=True)
    parent.to_parquet(reference / 'history.parquet', index=False)
    freeze_record(reference / 'ready.json', {'identity': {'profile': 'full-year-v1'},
                                           'history_sha256': digest(reference / 'history.parquet')})
    candidate = registration / 'inputs/candidate/panel.json'
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(panel()))
    audit = registration / 'inputs/evidence/nass-regional-audit-20261009.json'
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({'panel_sha256': digest(candidate)}))
    expanded, design, _ = pilot.make_design(parent, pilot.report_features(panel()))
    assert design['fit_budget']['total'] == 676
    expanded.to_parquet(registration / 'history.parquet', index=False)
    identity = {'design': design, 'source_id': 'prior-synthetic-source', 'python': sys.version.split()[0],
                'input_sha256': {p.relative_to(registration / 'inputs').as_posix(): digest(p)
                                 for p in (registration / 'inputs').rglob('*') if p.is_file()},
                'versions': {n: importlib.metadata.version(n) for n in
                             ('numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'threadpoolctl')},
                'lock_sha256': digest(repo / 'ml/uv.lock')}
    reg = {'profile': pilot.PROFILE, 'identity': identity, 'registration_id': content_id(identity),
           'completed_experiment': False, 'fits': 0, 'availability_verified': False, 'model_eligible': False}
    freeze_record(registration / 'preregistered.json', reg)
    freeze_record(registration / 'complete.json', {'completed': True, 'fits': 0, 'model_eligible': False,
                  'registration_id': reg['registration_id'],
                  'files': {p.relative_to(registration).as_posix(): digest(p)
                            for p in registration.rglob('*') if p.is_file()}})
    contract = tmp_path / 'decision-contract.json'
    freeze_record(contract, {'registration_id': reg['registration_id'], 'rules': design['rules'],
                            'completed_experiment': False, 'fits': 0, 'automatic_release': False})
    evidence = repo / 'research/evidence' / execute.EVIDENCE
    evidence.write_text(json.dumps({'preregistered_sha256': digest(registration / 'preregistered.json'),
                                   'complete_sha256': digest(registration / 'complete.json'),
                                   'registration_id': reg['registration_id'], 'decision_contract_sha256': digest(contract)}))
    (repo / 'research/registry.json').write_text(json.dumps({'schema': 1, 'records': [],
                    'local_evidence': [{'path': 'evidence/' + execute.EVIDENCE, 'sha256': digest(evidence)}]}))
    (repo / 'research/trials.json').write_text(json.dumps({'schema': 1, 'records': []}))
    execute.prepare(repo, folder, registration, contract)
    return repo, folder, registration, contract


def test_prepare_preserves_inputs_and_rejects_changed_environment(packet, monkeypatch):
    repo, folder, registration, contract = packet
    old = {p: digest(p) for p in registration.rglob('*') if p.is_file()}
    ready = digest(folder / 'ready.json')
    execute.prepare(repo, folder, registration, contract)
    assert digest(folder / 'ready.json') == ready
    assert all(digest(p) == value for p, value in old.items())
    monkeypatch.setattr(execute.sys, 'version', '0.0.0 changed')
    with pytest.raises(ValueError, match='environment changed'):
        execute.prepare(repo, folder, registration, contract)


@pytest.mark.parametrize('name', ['decision-contract.json', 'history.parquet', 'preregistration/complete.json'])
def test_corruption_blocks_fitting(packet, name):
    repo, folder, _, _ = packet
    path = folder / name
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        execute.run(Experiment(folder, repo=repo), 30)


def test_resumes_four_arms_and_checks_all_targets_provenance(packet, monkeypatch):
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
    monkeypatch.setattr(execute.shared, 'verify_learning_evidence', lambda *args: None)
    monkeypatch.setattr(execute, 'verify_fit_payloads', lambda *args: None)
    exp = Experiment(folder, repo=repo)
    assert execute.run(exp, 30)['status'] == 'planned_pause'
    assert execute.compare(folder, 20)['required_outputs'] == 32
    assert execute.run(exp, 30)['saved_outputs'] == 32
    result = execute.compare(folder, 20)
    assert len(calls) == 33 and all(h == 1 for h, _ in calls)
    assert result['prediction_rows'] == 8024 and result['profile'] == pilot.PROFILE
    rows = pd.read_csv(folder / 'reports/outer-predictions.csv')
    for group, modes in result['arms'].items():
        frame = rows.loc[rows.group.eq(group)]
        assert len(frame) == 2006
        for mode, field in [('selected', 'predicted_return'), ('raw', 'raw_predicted_return')]:
            mae = (frame.cotton_close * (np.exp(frame.actual_return) - np.exp(frame[field]))).abs().mean()
            assert modes[mode]['price_mae'] == pytest.approx(mae)
    before = digest(folder / 'reports/outer-predictions.csv')
    assert execute.compare(folder, 20) == result
    assert digest(folder / 'reports/outer-predictions.csv') == before
    execute.run(exp, 30)
    assert len(calls) == 33
    marker = next((folder / 'nass-regional-outputs').glob('*.json'))
    record = read_record(marker)
    record['records'][0]['report_sha256'] = 'forged'
    marker.unlink()  # synthetic fixture only
    freeze_record(marker, record)
    with pytest.raises(ValueError, match='Frozen NASS output changed'):
        execute.run(exp, 30)


def test_completion_requires_all_676_fit_receipts(packet):
    _, folder, _, _ = packet
    with pytest.raises(ValueError, match='676 fit receipts'):
        execute.verify_fit_payloads(folder, read_record(folder / 'ready.json')['identity'])


def test_budget_checkpoint_publication_and_failed_compute_count_once(tmp_path):
    root = tmp_path / 'ledger'
    freeze_record(root / 'local-work/key/locally-completed.json', {'synthetic': True})
    freeze_record(root / 'attempts/one.json', {'experiment_id': 'key'})
    freeze_record(root / 'attempts/two.json', {'experiment_id': 'failed-compute'})
    assert execute.fit_consumption(root) == (2, {'key'})
    freeze_record(root / 'completed/key.json', {'synthetic': True})
    assert execute.fit_consumption(root) == (2, {'key'})


def test_budget_no_t5_or_extra_synthetic_fit(packet, monkeypatch):
    _, folder, _, _ = packet
    completed = folder / 'ledger/completed'
    completed.mkdir(parents=True)
    for i in range(676):
        (completed / f'{i}.json').touch()
    def engine(*args, **kwargs):
        assert kwargs['horizons'] == (1,)
        kwargs['before_fit']()
    monkeypatch.setattr(execute, 'learning_control', lambda exp: None)
    monkeypatch.setattr(path_pilot, 'run', engine)
    with pytest.raises(FitBudgetReached, match='676-fit'):
        execute.run(SimpleNamespace(root=folder, identity=read_record(folder / 'ready.json')['identity']), 30)
    with pytest.raises(ValueError, match='30 minutes'):
        execute.run(SimpleNamespace(), 31)
    with pytest.raises(ValueError, match=r'Only preregistered T\+1'):
        execute.recipe({}, 'numeric_D0', 5)


def test_failed_synthetic_attempt_cannot_trigger_second_fit(tmp_path, monkeypatch):
    freeze_record(tmp_path / 'learning-control-ledger/attempts/one.json', {'experiment_id': 'failed'})
    monkeypatch.setattr(execute.shared, 'learning_control', lambda exp: pytest.fail('extra fit'))
    with pytest.raises(ValueError, match='already consumed'):
        execute.learning_control(SimpleNamespace(root=tmp_path))


def test_candidate_requires_six_year_wins_and_delay_robustness():
    score = {'naive_gain_pct': 5., 'direction_pct': 53., 'year_wins': 6,
             'versus_naive': {str(b): {'gain_ci_pct': [1., 6.]} for b in (20, 60)}}
    arms = {f'numeric_D{d}': {'selected': copy.deepcopy(score)} for d in (0, 1)}
    paired = {f'selected_D{d}': {str(b): {'difference_ci_95': [.01, .1]} for b in (20, 60)} for d in (0, 1)}
    assert execute.decision(arms, paired) == 'CONDITIONAL_CANDIDATE'
    arms['numeric_D0']['selected']['year_wins'] = 5
    assert execute.decision(arms, paired) == 'INCREMENTAL_BUT_BELOW_GOAL'
    paired['selected_D1']['60']['difference_ci_95'][0] = -.1
    assert execute.decision(arms, paired) == 'INCONCLUSIVE_OR_DELAY_SENSITIVE'


def test_launcher_selects_profile_and_caps_session(tmp_path, monkeypatch):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / 'full_year_cpu.py'))
    env = tmp_path / 'cpu'
    python = env / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    python.parent.mkdir(parents=True); python.touch()
    calls = []
    monkeypatch.setattr(sys, 'argv', ['full_year_cpu.py', 'pilot', '--profile', pilot.PROFILE,
        '--cpu-environment', str(env), '--drive-root', str(tmp_path / 'drive'), '--max-minutes', '60'])
    monkeypatch.setattr(namespace['subprocess'], 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    namespace['main']()
    command, kwargs = calls[0]
    assert command[command.index('--max-minutes') + 1] == '30.0'
    assert command[command.index('--experiment') + 1] == 'research-nass-regional-t1-pilot-v1'
    assert kwargs['env']['OMP_NUM_THREADS'] == '2'
    assert kwargs['env']['COTTONLENS_ALLOW_LOCAL_CPU_TABULAR'] == '1'
