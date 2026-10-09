"""Semi-synthetic calibration, identity and bounded execution; no market fits."""
import copy
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import weak_signal as study
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.protocol import full_year_manifest


@pytest.fixture(scope='module')
def parent():
    counts = {2016: 250, 2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252, 2022: 251, 2023: 251}
    dates = pd.DatetimeIndex(np.concatenate([pd.bdate_range(f'{y}-01-01', f'{y}-12-31')[:counts.get(y, 250)]
                                            for y in range(2010, 2024)]))
    i = np.arange(len(dates))
    frame = pd.DataFrame({'date': dates, 'cotton_session_index': i, 'cotton_close': 80 + np.sin(i / 10),
                          'nass_decision_time': dates.tz_localize('UTC') + pd.Timedelta(days=1, minutes=15)})
    for j, name in enumerate(study.reference_groups()[study.GROUP]):
        frame[name] = np.sin(i / (j + 2))
    frame['cotton_volatility_20'] = .02 + .005 * np.cos(i / 20)
    frame.loc[i % 4 == 0, study.SOURCE] = np.nan
    for h in (1, 5):
        frame[f'target_return_{h}'] = np.log(frame.cotton_close.shift(-h) / frame.cotton_close)
        frame[f'target_date_{h}'] = frame.date.shift(-h)
    return frame


@pytest.fixture(scope='module')
def base_packet(tmp_path_factory, parent):
    tmp_path = tmp_path_factory.mktemp('weak-signal')
    repo, ref, root = [tmp_path / n for n in ('repo', 'reference', 'study')]
    (repo / 'ml').mkdir(parents=True)
    (repo / 'ml/uv.lock').write_text('synthetic lock')
    (repo / 'research/evidence').mkdir(parents=True)
    ref.mkdir()
    parent.to_parquet(ref / 'history.parquet', index=False)
    identity = {'source_id': 'old-source', 'registration_id': 'old-registration',
                'design': {'groups': study.reference_groups()}, 'split': full_year_manifest(parent)}
    freeze_record(ref / 'ready.json', {'identity': identity, 'history_sha256': digest(ref / 'history.parquet')})
    (ref / 'reports').mkdir()
    (ref / 'reports/outer-predictions.csv').write_text('synthetic reference placeholder')
    evidence = repo / 'research/evidence/nass-regional-result-20261009.json'
    freeze_record(evidence, {'ready_sha256': digest(ref / 'ready.json'), 'ml_source_id': 'old-source',
                            'registration_id': 'old-registration', 'predictions_sha256': digest(ref / 'reports/outer-predictions.csv')})
    registry = {'schema': 1, 'records': [], 'local_evidence': [{'path': 'evidence/' + evidence.name, 'sha256': digest(evidence)}]}
    (repo / 'research/registry.json').write_text(json.dumps(registry))
    (repo / 'research/trials.json').write_text(json.dumps({'schema': 1, 'records': []}))
    study.prepare(repo, root, ref)
    proof = repo / 'research/evidence/weak-signal-preregistration-20261009.json'
    reg = study.study(root)
    freeze_record(proof, {'completed_experiment': False, 'synthetic_fits': 0, 'market_fits': 0,
                         'registration_id': content_id(reg), 'registration_sha256': digest(root / study.REGISTRATION),
                         'complete_sha256': digest(root / 'complete.json'), 'source_id': reg['source_id']})
    registry['local_evidence'].append({'path': 'evidence/' + proof.name, 'sha256': digest(proof)})
    (repo / 'research/registry.json').write_text(json.dumps(registry))
    return repo, ref, root


@pytest.fixture
def packet(tmp_path, base_packet):
    for original in base_packet:
        shutil.copytree(original, tmp_path / original.name)
    return tuple(tmp_path / original.name for original in base_packet)


def test_analytic_price_loss_matches_integration():
    from scipy.integrate import quad
    from scipy.stats import norm
    for mu, sigma, p in [(0., .02, 0.), (.008, .015, .008), (-.01, .02, .003)]:
        expected = quad(lambda z, m=mu, s=sigma, prediction=p: 80 * abs(np.exp(m + s * z) - np.exp(prediction)) * norm.pdf(z),
                        -10, 10, points=[(p - mu) / sigma], epsabs=1e-10)[0]
        assert study.expected_absolute_log_price(80, mu, sigma, p) == pytest.approx(expected, abs=1e-9)
    with pytest.raises(ValueError):
        study.expected_absolute_log_price(0, 0, .02, 0)


def test_early_calibration_and_future_isolation(parent):
    c = study.calibrate(parent)
    assert c['expected_calibration_oracle_gain_pct'] == pytest.approx(5, abs=1e-10)
    changed = parent.copy()
    changed.loc[changed.date >= '2015-01-01', [study.SOURCE, 'cotton_volatility_20', 'target_return_1']] = 1000
    assert study.calibrate(changed) == c
    before = study.synthetic_frame(parent, c, 'injected', 1201)
    after = study.synthetic_frame(changed, c, 'injected', 1201)
    pd.testing.assert_frame_equal(before.loc[before.date < '2015-01-01'], after.loc[after.date < '2015-01-01'])
    assert study.SOURCE in study.reference_groups()[study.GROUP]
    assert not {'oracle_return_1', 'target_return_1'} & set(study.reference_groups()[study.GROUP])


def test_null_pair_noise_features_and_origins_unchanged(parent):
    original = parent.copy(deep=True)
    c = study.calibrate(parent)
    null, signal = [study.synthetic_frame(parent, c, condition, 1201) for condition in study.CONDITIONS]
    np.testing.assert_allclose(signal.target_return_1 - signal.oracle_return_1, null.target_return_1, atol=1e-16, rtol=0)
    assert null.oracle_return_1.eq(0).all()
    assert signal.loc[parent[study.SOURCE].isna(), 'oracle_return_1'].eq(0).all()
    for f in (null, signal):
        pd.testing.assert_frame_equal(f.drop(columns=['oracle_return_1', 'target_return_1']), parent.drop(columns='target_return_1'))
        split = full_year_manifest(f)
        assert [b['origins'] for b in split['folds']] == [b['origins'] for b in full_year_manifest(parent)['folds']]
        assert sum(len(b['origins']) for b in split['folds']) == 2006
    pd.testing.assert_frame_equal(parent, original)


def test_prepare_idempotence_registry_additions_and_frozen_inputs(packet):
    repo, ref, root = packet
    files = {p: digest(p) for directory in (ref, root) for p in directory.rglob('*') if p.is_file()}
    assert study.prepare(repo, root, ref) == study.study(root)
    assert all(digest(p) == value for p, value in files.items())
    assert study.verify_registered_evidence(repo, root)
    assert study.compare(repo, root, ref, 20)['required_outputs'] == 160
    assert sum(len(list((root / 'scenarios').glob(n))) for n in ['null-seed*', 'injected-seed*']) == 20


@pytest.mark.parametrize('field', ['source', 'history', 'ready', 'reference', 'proof'])
def test_changed_identity_blocks_execution(packet, field):
    repo, ref, root = packet
    path = {'source': repo / 'ml/uv.lock', 'history': root / 'scenarios/null-seed1201/history.parquet',
            'ready': root / 'scenarios/null-seed1201/ready.json', 'reference': ref / 'history.parquet',
            'proof': repo / 'research/evidence/weak-signal-preregistration-20261009.json'}[field]
    path.write_bytes(path.read_bytes() + b' changed')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        study.run(repo, root, ref, 30)


def test_global_session_deadline_and_checkpoint_resume(monkeypatch, tmp_path):
    registration = {'fit_budget': 3380}
    monkeypatch.setattr(study, 'verify_all', lambda *a: registration)
    monkeypatch.setattr(study, 'verify_registered_evidence', lambda *a: 'proof')
    now = [0.]
    monkeypatch.setattr(study.time, 'monotonic', lambda: now[0])
    from cottonlens_ml.research import engine
    monkeypatch.setattr(engine, 'Experiment', lambda directory, repo: SimpleNamespace(
        identity={'registration_id': content_id(registration)}, root=directory))
    calls = []
    def pilot(experiment, minutes, **kwargs):
        calls.append(minutes)
        assert kwargs['horizons'] == (1,)
        kwargs['before_fit']()
        now[0] += 1000
        return {'status': 'complete', 'saved_outputs': 8}
    monkeypatch.setattr(study.path_pilot, 'run', pilot)
    result = study.run(tmp_path, tmp_path, tmp_path, 30)
    assert result['status'] == 'planned_pause' and result['consumed'] == 2
    assert calls == pytest.approx([30, 800 / 60])
    with pytest.raises(ValueError, match='30 minutes'):
        study.run(tmp_path, tmp_path, tmp_path, 31)


def test_failed_fit_budget_and_unpublished_registration_block(monkeypatch, packet):
    repo, ref, root = packet
    monkeypatch.setattr(study, 'verify_all', lambda *a: {'fit_budget': 3380})
    monkeypatch.setattr(study, 'verify_registered_evidence', lambda *a: 'proof')
    monkeypatch.setattr(study, 'fit_consumption', lambda *a: (169, set()))
    from cottonlens_ml.research import engine
    monkeypatch.setattr(engine, 'Experiment', lambda *a, **k: SimpleNamespace(identity={'registration_id': content_id({'fit_budget': 3380})}))
    def pilot(*a, **k):
        k['before_fit']()
    monkeypatch.setattr(study.path_pilot, 'run', pilot)
    from cottonlens_ml.research.ledger import FitBudgetReached
    with pytest.raises(FitBudgetReached, match='3380-fit'):
        study.run(repo, root, ref, 30)


def test_preregistered_classification_does_not_reinterpret_null_or_shrinkage():
    rules = {'valid_oracle_mean_gain_pct_at_least': 3., 'minimum_positive_seeds': 8,
             'median_fraction_of_oracle_at_least': .5, 'repeated_null_practical_false_positives_at_least': 2}
    scenarios = {study.scenario_name(c, s): {'modes': {m: {'naive_gain_pct': g} for m, g in
                  [('oracle', 4), ('raw', 3), ('selected', 3)]}, 'practical_false_positive': False}
                 for s in study.SEEDS for c in study.CONDITIONS}
    assert study.classify(scenarios, rules)[0] == 'RAW_AND_SELECTED_RECOVER'
    changed = copy.deepcopy(scenarios)
    for name, v in changed.items():
        if name.startswith('injected'):
            v['modes']['selected']['naive_gain_pct'] = 0
    assert study.classify(changed, rules)[0] == 'RAW_ONLY_RECOVERS'
    changed['null-seed1201']['practical_false_positive'] = True
    changed['null-seed1202']['practical_false_positive'] = True
    assert study.classify(changed, rules)[0] == 'NULL_CONTROL_FAILS'


def test_complete_mocked_outputs_metrics_idempotence_and_tamper(packet, monkeypatch):
    repo, ref, root = packet
    from cottonlens_ml.research import weak_signal_review
    monkeypatch.setattr(weak_signal_review, 'replay', lambda *a: {'new_fits': 0, 'synthetic_fixture': True})
    def inner(exp, spec, fold):
        return {'recipe': spec, 'weight': .5, 'iterations': 1, 'inner_score': 1.}
    monkeypatch.setattr(study.path_pilot, 'inner_price', inner)
    monkeypatch.setattr(study.path_pilot, 'predict_chunks', lambda exp, spec, dates, *a:
                        exp.history.set_index('date').loc[pd.to_datetime(dates)].oracle_return_1.to_numpy())
    assert study.run(repo, root, ref, 30)['saved_outputs'] == 160
    result = study.compare(repo, root, ref, 20)
    assert result['prediction_rows'] == 40120 and result['market_skill_demonstrated'] is False
    export = root / 'reports/outer-predictions.csv'
    rows = pd.read_csv(export, float_precision='round_trip', keep_default_na=False)
    for (condition, seed), frame in rows.groupby(['condition', 'seed']):
        saved = result['scenarios'][study.scenario_name(condition, seed)]
        truth = frame.cotton_close.to_numpy() * np.exp(frame.actual_return.to_numpy())
        for mode, field in [('raw', 'raw_predicted_return'), ('selected', 'predicted_return'), ('oracle', 'oracle_return')]:
            forecast = frame.cotton_close.to_numpy() * np.exp(frame[field].to_numpy())
            assert saved['modes'][mode]['price_mae'] == pytest.approx(np.abs(truth - forecast).mean(), abs=1e-12)
    old = digest(export)
    assert study.compare(repo, root, ref, 20) == result and digest(export) == old
    audited = weak_signal_review.verify_report(root)
    assert audited['independent_bootstrap_intervals'] == 120 and audited['scenario_modes_verified'] == 60
    assert study.run(repo, root, ref, 30)['consumed'] == 0
    marker = root / 'scenarios/injected-seed1201/weak-signal-outputs/numeric_D0-t1-year2016.json'
    data = json.loads(marker.read_bytes())
    data['records'][0]['oracle_return'] += .01
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Corrupt completed record'):
        study.run(repo, root, ref, 30)
    data['record_id'] = content_id({k: v for k, v in data.items() if k != 'record_id'})
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Synthetic truth'):
        study.run(repo, root, ref, 30)
    export.write_bytes(export.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Report CSV changed'):
        weak_signal_review.verify_report(root)


def test_no_receipts_cannot_be_called_a_complete_study(packet):
    from cottonlens_ml.research.weak_signal_review import replay
    _, _, root = packet
    with pytest.raises(ValueError, match='169 synthetic fit receipts'):
        replay(root / 'scenarios/injected-seed1201')


def test_failed_uncheckpointed_fit_is_not_automatically_retried(packet):
    repo, ref, root = packet
    freeze_record(root / 'scenarios/null-seed1201/ledger/attempts/failed.json', {'experiment_id': 'failed-key'})
    with pytest.raises(ValueError, match='no automatic refit'):
        study.run(repo, root, ref, 30)


def test_cpu_launcher_profile_default_threads_and_session_limit(tmp_path, monkeypatch):
    launcher = runpy.run_path(str(Path(__file__).parents[1] / 'full_year_cpu.py'))
    python = tmp_path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir()
    python.touch()
    captured = []
    monkeypatch.setattr(launcher['subprocess'], 'run', lambda command, **kwargs: captured.append((command, kwargs)))
    monkeypatch.setattr(sys, 'argv', ['full_year_cpu.py', 'pilot', '--profile', study.PROFILE,
                                    '--cpu-environment', str(tmp_path), '--reference-root', str(tmp_path), '--max-minutes', '60'])
    launcher['main']()
    command, kwargs = captured[0]
    assert command[command.index('--experiment') + 1] == 'research-weak-signal-control-v1-r2'
    assert command[command.index('--max-minutes') + 1] == '30.0'
    assert kwargs['env']['COTTONLENS_ALLOW_LOCAL_CPU_TABULAR'] == '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert kwargs['env'][name] == '2'


def test_registered_revision_proof_is_selected_by_identity(packet):
    repo, _, root = packet
    old = repo / 'research/evidence/weak-signal-preregistration-20261009.json'
    new = old.with_name('weak-signal-preregistration-r2-20261009.json')
    new.write_bytes(old.read_bytes())
    path = repo / 'research/registry.json'
    registry = json.loads(path.read_bytes())
    registry['local_evidence'][-1] = {'path': 'evidence/' + new.name, 'sha256': digest(new)}
    path.write_text(json.dumps(registry))
    assert study.verify_registered_evidence(repo, root) == digest(new)


def test_ambiguous_registered_proofs_fail_closed(packet):
    repo, _, root = packet
    old = repo / 'research/evidence/weak-signal-preregistration-20261009.json'
    new = old.with_name('weak-signal-preregistration-duplicate.json')
    new.write_bytes(old.read_bytes())
    path = repo / 'research/registry.json'
    registry = json.loads(path.read_bytes())
    registry['local_evidence'].append({'path': 'evidence/' + new.name, 'sha256': digest(new)})
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='Exactly one registered zero-fit proof'):
        study.verify_registered_evidence(repo, root)
