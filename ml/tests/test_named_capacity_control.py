from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import named_capacity_control as capacity
from cottonlens_ml.research import named_label_control as parent


def test_fixed_recipe_changes_family_only_and_no_early_stop():
    assert capacity.RECIPE['features'] == parent.RECIPE['features']
    for name in ('horizon', 'seed', 'task', 'target', 'loss', 'window', 'cadence', 'years', 'device'):
        assert capacity.RECIPE[name] == parent.RECIPE[name]
    assert capacity.RECIPE['params'] == {'max_depth': 2, 'eta': .03, 'min_child_weight': 20, 'alpha': 0, 'lambda': 1}
    assert capacity.TREES == capacity.RECIPE['max_iterations'] == 100
    assert 'patience' not in capacity.RECIPE
    assert not any('target' in name or 'named' in name for name in capacity.RECIPE['features'])


def test_design_uses_parent_jobs_and_origins_without_choosing_years():
    fold = {'year': 2023, 'origins': ['2023-01-03'], 'inner': [['2022-01-03']],
            'jobs': [{'role': 'outer', 'bucket': 0, 'cutoff': '2023-01-03', 'train_dates': []}]}
    ready = {'identity': {'design': {'profile': parent.PROFILE, 'recipe': parent.RECIPE,
             'folds': [fold], 'selection': 'past only', 'clock': 'assumed'}}}
    design = capacity.design(ready)
    assert design['folds'] == [fold]
    assert design['fit_budget']['market'] == 1 and design['fit_budget']['synthetic'] == 4
    assert design['iterations'] == 100 and design['gate_evaluation_allowed'] is False
    ready['identity']['design']['recipe'] = {**parent.RECIPE, 'window': 20}
    with pytest.raises(ValueError, match='completed matched'):
        capacity.design(ready)


def test_nonlinear_control_is_zero_linear_coefficient_quadratic_source():
    frame, train, test, recipe, trees = capacity.synthetic('known_nonlinear')
    expected = .02 * (frame[recipe['features'][0]] ** 2 - 1)
    np.testing.assert_array_equal(frame.target_return_5, expected)
    np.testing.assert_array_equal(train.target_return_5, expected.loc[train.index])
    np.testing.assert_array_equal(test.target_return_5, expected.loc[test.index])
    assert train.date.max() < test.date.min() and trees == 100
    assert len(recipe['features']) == 28


def test_negative_shift_labels_not_lost_when_adding_features():
    _, linear, _, _, _ = capacity.synthetic('known_signal')
    _, shifted, test, _, _ = capacity.synthetic('negative_block_shift')
    np.testing.assert_array_equal(shifted.target_return_5, np.roll(linear.target_return_5, 137))
    assert not np.array_equal(shifted.target_return_5, linear.target_return_5)
    assert shifted.date.max() < test.date.min()


def test_family_memorization_is_separate_from_fixed_market_recipe():
    _, train, test, recipe, trees = capacity.synthetic('small_subset_overfit')
    assert len(train) == len(test) == 128
    pd.testing.assert_frame_equal(train, test)
    assert recipe['params']['max_depth'] == 8 and trees == 400
    assert capacity.RECIPE['params']['max_depth'] == 2 and capacity.TREES == 100


@pytest.fixture
def mock_learning(tmp_path, monkeypatch):
    class FakeLedger:
        def __init__(self, *_):
            pass
        def run(self, spec, operation, before_compute):
            before_compute()
            kind = spec['role']
            _, _, test, _, _ = capacity.synthetic(kind)
            pred = np.zeros(len(test)) if kind == 'negative_block_shift' else .99 * test.target_return_5.to_numpy()
            if kind == settings['failed_kind']:
                pred = np.zeros(len(test))
            if settings.get('leaking_negative') and kind == 'negative_block_shift':
                pred = .99 * test.target_return_5.to_numpy()
            calls.append((kind, spec['iterations']))
            return {'experiment_id': kind, 'result': {'predictions': pred.tolist()}}
    calls, settings = [], {'failed_kind': None}
    monkeypatch.setattr(capacity, 'Ledger', FakeLedger)
    experiment = SimpleNamespace(root=tmp_path, identity={'source_id': 'test', 'versions': {}, 'python': 'test'})
    return experiment, settings, calls


def test_all_four_controls_and_nonlinear_measurement_are_required(mock_learning):
    ex, _, calls = mock_learning
    capacity.learning(ex)
    record = capacity.read_record(ex.root / 'capacity-learning.json')
    assert record['status'] == 'passed' and record['market_evidence'] is False
    assert [k for k, _ in calls] == list(capacity.KINDS)
    assert record['weak_signal_power_claimed'] is False


@pytest.mark.parametrize('kind', ['known_signal', 'known_nonlinear', 'small_subset_overfit'])
def test_failed_family_control_cannot_pass(mock_learning, kind):
    ex, settings, _ = mock_learning
    settings['failed_kind'] = kind
    with pytest.raises(ValueError, match='Learning control failed'):
        capacity.learning(ex)
    assert capacity.read_record(ex.root / 'capacity-learning.json')['status'] == 'failed'


def test_cached_passed_is_remeasured(mock_learning):
    ex, settings, _ = mock_learning
    capacity.learning(ex)
    settings['failed_kind'] = 'known_nonlinear'
    with pytest.raises(ValueError):
        capacity.learning(ex)


def test_expired_session_stops_before_any_control_fit(mock_learning):
    ex, _, calls = mock_learning
    with pytest.raises(capacity.FitBudgetReached):
        capacity.learning(ex, deadline=-1)
    assert calls == []


def test_exhausted_attempt_budget_cannot_restart_controls(mock_learning, monkeypatch):
    ex, _, calls = mock_learning
    monkeypatch.setattr(capacity, 'fit_consumption', lambda _: (4, set()))
    with pytest.raises(capacity.FitBudgetReached):
        capacity.learning(ex)
    assert calls == []


def test_negative_control_improvement_cannot_pass(mock_learning):
    ex, settings, _ = mock_learning
    settings['leaking_negative'] = True
    with pytest.raises(ValueError, match='Learning control failed'):
        capacity.learning(ex)
    assert any('negative_block_shift' in f for f in
               capacity.read_record(ex.root / 'capacity-learning.json')['failures'])


def test_changed_completed_parent_payload_is_rejected_before_replay(tmp_path):
    p = tmp_path / 'parent/ready.json'
    p.parent.mkdir()
    p.write_text('changed', encoding='utf-8')
    capacity.freeze_record(tmp_path / 'parent-files.json', {'files': {'ready.json': '0' * 64}})
    with pytest.raises(ValueError, match='completed parent changed'):
        capacity.verify_parent(tmp_path, tmp_path)


def test_unsafe_parent_member_is_rejected(tmp_path):
    capacity.freeze_record(tmp_path / 'parent-files.json', {'files': {'../escape': '0' * 64}})
    with pytest.raises(ValueError, match='completed parent changed'):
        capacity.verify_parent(tmp_path, tmp_path)


def test_source_change_rejects_new_cache_before_parent_access(tmp_path, monkeypatch):
    capacity.freeze_record(tmp_path / 'ready.json', {'identity': {
        'profile': capacity.PROFILE, 'source_id': 'old'}})
    monkeypatch.setattr(capacity, 'research_source_identity', lambda _: {'source_id': 'new'})
    with pytest.raises(ValueError, match='new namespace required'):
        capacity.verify(tmp_path, tmp_path)


def test_source_change_during_learning_stops_before_fit(mock_learning, monkeypatch):
    ex, _, calls = mock_learning
    monkeypatch.setattr(capacity, 'research_source_identity', lambda _: {'source_id': 'changed'})
    with pytest.raises(ValueError, match='during learning controls'):
        capacity.learning(ex, repo=ex.root)
    assert calls == []
