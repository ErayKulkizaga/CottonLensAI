import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.paired_price_loss import (
    ARMS,
    decision,
    make_design,
    metric,
    paired_recipe,
    run,
    training_cohort,
    verify_registration,
)
from cottonlens_ml.research.protocol import split_manifest
from cottonlens_ml.research.review import training_rows


@pytest.fixture
def reference():
    rng = np.random.default_rng(81)
    history = pd.DataFrame(rng.normal(size=(3650, 24)), columns=FEATURE_NAMES)
    history['date'] = pd.bdate_range('2010-01-04', periods=len(history))
    history['cotton_close'] = 80 * np.exp(.02 * np.sin(np.arange(len(history)) / 30))
    for h in (1, 5):
        history[f'target_return_{h}'] = np.log(history.cotton_close.shift(-h) / history.cotton_close)
        history[f'target_date_{h}'] = history.date.shift(-h)
    history = history.loc[history.date < '2024-01-01'].copy()
    history.loc[history.index % 17 == 0, FEATURE_NAMES[0]] = np.nan
    split = split_manifest(history)
    identity = {'source_id': 'legacy-source', 'groups': {'base': list(FEATURE_NAMES)}, 'split': split}
    ready, report, receipts = {'identity': identity}, {'source_id': 'legacy-source', 'years': []}, {}
    for fold in split['folds']:
        train = training_rows(history, pd.Timestamp(fold['origins'][0]), {'years': None}, None)
        selected, ids = [], []
        for seed in (17, 42, 101):
            recipe = {'family': 'xgboost', 'horizon': 1, 'task': 'price', 'seed': seed, 'years': None,
                      'window': 1, 'cadence': 126, 'target': 'scaled_log', 'loss': 'reg:squarederror',
                      'features': list(FEATURE_NAMES), 'params': {'eta': .03, 'max_depth': 2}}
            spec = {'recipe': recipe, 'role': f'old-{fold["year"]}-{seed}', 'iterations': 56,
                    'test_dates': fold['origins'], 'validation_dates': [],
                    'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(),
                    'train_identity': frame_identity(train, list(train))}
            key = content_id({'identity': identity, 'specification': spec})
            receipts[key] = {'identity': identity, 'experiment_id': key, 'specification': spec,
                             'result': {'origins': fold['origins']}}
            selected.append({'recipe': recipe, 'iterations': 56}); ids.append(key)
        report['years'].append({'year': fold['year'], 'chosen': {'seeds': selected}, 'outer_fit_ids': ids})
    return ready, history, report, receipts


def test_recipe_arms_differ_only_in_loss_and_preserve_recorded_values():
    original = {'family': 'xgboost', 'horizon': 1, 'seed': 17, 'target': 'scaled_log',
                'loss': 'reg:squarederror', 'features': ['x'], 'params': {'eta': np.nextafter(.03, 1)}}
    before = copy.deepcopy(original)
    a, b = (paired_recipe(original, arm) for arm in ARMS)
    assert {k: v for k, v in a.items() if k != 'loss'} == {k: v for k, v in b.items() if k != 'loss'}
    assert a['target'] == b['target'] == 'price_delta' and a['device'] == b['device'] == 'cpu'
    a['params']['eta'] = 0
    a['features'].append('changed')
    assert original == before and b['params'] == before['params']


def test_design_exact_budget_and_missing_features_retained(reference):
    ready, history, report, receipts = reference
    design = make_design(ready, history, report, receipts)
    assert design['fit_budget'] == 48 and design['tree_budget'] == 2688
    assert design['unique_origins'] == 1008 and design['prediction_rows'] == 2016
    dates = design['periods'][0]['seeds'][0]['train_dates']
    assert history.loc[history.date.isin(pd.to_datetime(dates)), FEATURE_NAMES[0]].isna().any()


@pytest.mark.parametrize('field', ['train_dates', 'train_identity', 'test_dates', 'iterations', 'validation_dates'])
def test_changed_receipt_cannot_enter_design(reference, field):
    ready, history, report, receipts = reference
    key = report['years'][0]['outer_fit_ids'][0]
    bad = copy.deepcopy(receipts)
    if field == 'iterations':
        bad[key]['specification'][field] += 1
    elif field == 'train_identity':
        bad[key]['specification'][field] = 'incorrect'
    else:
        bad[key]['specification'][field] = ['2024-01-01']
    with pytest.raises((ValueError, AssertionError)):
        make_design(ready, history, report, bad)


def test_future_changes_do_not_change_training_or_preprocessing(reference):
    ready, history, report, receipts = reference
    seed = make_design(ready, history, report, receipts)['periods'][0]['seeds'][0]
    cutoff = pd.Timestamp(ready['identity']['split']['folds'][0]['origins'][0])
    train = training_cohort(history, seed['train_dates'], cutoff, seed['train_identity'], None)
    future = history.copy()
    future.loc[future.date >= cutoff, FEATURE_NAMES] = 9999
    future.loc[future.date >= cutoff, ['target_return_1', 'target_return_5']] = .5
    actual = training_cohort(future, seed['train_dates'], cutoff, seed['train_identity'], None)
    pd.testing.assert_frame_equal(train, actual, check_exact=True)
    from cottonlens_ml.research.protocol import Preprocessor, Target
    assert Preprocessor.fit(train, FEATURE_NAMES) == Preprocessor.fit(actual, FEATURE_NAMES)
    assert Target.fit(train, 1, 'price_delta') == Target.fit(actual, 1, 'price_delta')


def test_price_mae_reconstruction_is_independent_of_stored_losses():
    frame = pd.DataFrame({'cotton_close': [80., 100.], 'actual_return': np.log([1.1, .9]),
                          'predicted_return': np.log([1.05, .95])})
    errors, stats = metric(frame)
    np.testing.assert_allclose(errors, [[8., 4.], [10., 5.]], atol=1e-12)
    assert stats['naive_gain_pct'] == pytest.approx(50) and stats['direction_pct'] == 100


def test_decision_distinguishes_tiny_contribution_from_practical_skill():
    arms = {arm: {'naive_gain_pct': 0., 'direction_pct': 50., 'period_wins': 1,
                 'versus_naive': {str(b): {'gain_ci_pct': [-1, 1]} for b in (20, 60)}} for arm in ARMS}
    paired = {str(b): {'difference_ci_95': [-.01, .01]} for b in (20, 60)}
    assert decision(arms, paired) == 'BOTH_FIXED_LOSSES_BELOW_PRACTICAL_GOAL'
    paired = {str(b): {'difference_ci_95': [.001, .002]} for b in (20, 60)}
    assert decision(arms, paired) == 'LIMITED_LOSS_CONTRIBUTION_PRACTICAL_GOAL_UNMET'
    arms['absolute'].update(naive_gain_pct=5, direction_pct=53, period_wins=6)
    assert decision(arms, paired) == 'STRONG_FIXED_LOSS_CANDIDATE_REQUIRES_PROSPECTIVE_VALIDATION'


def test_optimized_python_cannot_disable_prepare_checks(tmp_path):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)}
    code = 'from cottonlens_ml.research.paired_price_loss import prepare; prepare("missing", "output", "missing")'
    r = subprocess.run([sys.executable, '-O', '-c', code], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert r.returncode != 0 and 'Scientific integrity assertions must remain enabled' in r.stderr
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('field', ['ready_sha256', 'preregistered_sha256', 'identity_id', 'fit_budget', 'tree_budget', 'market_fits'])
def test_changed_public_registration_stops_training(tmp_path, field):
    folder, repo = tmp_path / 'prepared', tmp_path / 'repo'
    folder.mkdir(); (repo / 'research/evidence').mkdir(parents=True)
    (folder / 'ready.json').write_text('ready', encoding='utf-8')
    (folder / 'preregistered.json').write_text('registration', encoding='utf-8')
    ready = {'identity': {'test': True}}
    proof = {'ready_sha256': digest(folder / 'ready.json'), 'preregistered_sha256': digest(folder / 'preregistered.json'),
             'identity_id': content_id(ready['identity']), 'market_fits': 0, 'fit_budget': 48, 'tree_budget': 2688}
    path = repo / 'research/evidence/paired-price-loss-preregistration-20261010.json'
    path.write_text(json.dumps(proof), encoding='utf-8')
    verify_registration(repo, folder, ready)
    proof[field] = 'changed' if isinstance(proof[field], str) else proof[field] + 1
    path.write_text(json.dumps(proof), encoding='utf-8')
    with pytest.raises(ValueError, match='Published zero-fit'):
        verify_registration(repo, folder, ready)


def test_cpu_launcher_routes_fixed_profile_and_caps_session(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'full_year_cpu.py'
    spec = importlib.util.spec_from_file_location('paired_loss_launcher_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    environment = tmp_path / 'env'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    python.parent.mkdir(parents=True); python.write_bytes(b'fixture')
    seen = []
    monkeypatch.setattr(sys, 'argv', ['launcher', 'pilot', '--profile', 'paired-price-loss-control-v1',
                                    '--cpu-environment', str(environment), '--max-minutes', '999'])
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: seen.append((command, kwargs)))
    module.main()
    command, kwargs = seen[0]
    assert command[command.index('--experiment') + 1] == 'research-paired-price-loss-control-v1'
    assert command[command.index('--max-minutes') + 1] == '30.0'
    assert kwargs['env']['COTTONLENS_ALLOW_LOCAL_CPU_TABULAR'] == '1'
    assert all(kwargs['env'][name] == '2' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))


def test_runner_resumes_without_refitting_cached_market_records(reference, tmp_path, monkeypatch):
    from cottonlens_ml.research import engine, paired_price_loss
    from cottonlens_ml.research.ledger import Ledger
    ready, history, report, receipts = reference
    design = make_design(ready, history, report, receipts)
    experiment = object.__new__(engine.Experiment)
    experiment.root, experiment.history = tmp_path, history
    experiment.identity = {'profile': paired_price_loss.PROFILE, 'design': design, 'split': ready['identity']['split']}
    experiment.ledger = Ledger(tmp_path / 'ledger', experiment.identity)
    experiment.repo_for_registration = tmp_path
    monkeypatch.setattr(paired_price_loss, 'verify_execution', lambda p: (ready, history))
    monkeypatch.setattr(paired_price_loss, 'verify_registration', lambda *a: None)
    monkeypatch.setattr(paired_price_loss, 'learning_controls', lambda *a: None)
    operations = []

    def fake_fit(history, train, validation, test, spec, workspace, *, iterations):
        operations.append(spec['loss'])
        (workspace / 'mock-payload.txt').write_text('synthetic test fixture; no fitted model', encoding='utf-8')
        return {'predictions': (test.target_return_1.to_numpy() * .5).tolist(), 'iterations': iterations,
                'origins': test.date.dt.strftime('%Y-%m-%d').tolist()}

    monkeypatch.setattr(engine, 'fit_predict', fake_fit)
    times = iter([0, .1, .2, .3, .7])
    monkeypatch.setattr(paired_price_loss, 'time', SimpleNamespace(monotonic=lambda: next(times)))
    assert run(experiment, .01)['status'] == 'planned_pause'
    assert len(operations) == 3
    monkeypatch.setattr(paired_price_loss, 'time', SimpleNamespace(monotonic=lambda: 0))
    assert run(experiment, 1)['market_fits'] == 48
    assert len(operations) == 48 and len(experiment.ledger.results()) == 48
    assert len(list((tmp_path / 'paired-outputs').glob('*.json'))) == 16
    assert run(experiment, 1)['market_fits'] == 48 and len(operations) == 48
