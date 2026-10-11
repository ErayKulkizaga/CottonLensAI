from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import named_capacity_control as capacity
from cottonlens_ml.research import named_history as source
from cottonlens_ml.research import named_history_control as pilot
from cottonlens_ml.research.protocol import Preprocessor, inputs


def test_only_source_information_changes_recipe_and_jobs_stay_fixed():
    old = {'profile': capacity.PROFILE, 'recipe': capacity.RECIPE,
           'folds': [{'jobs': [{}, {}]}], 'fit_budget': {'market': 2, 'synthetic': 4, 'threads': 2, 'processes': 1, 'session_minutes': 30},
           'selection': 'past only', 'clock': 'assumed', 'bootstrap': {'blocks': [20, 60]}}
    design = pilot.design({'identity': {'design': old}})
    assert {k: v for k, v in pilot.RECIPE.items() if k != 'features'} == {
        k: v for k, v in capacity.RECIPE.items() if k != 'features'}
    assert pilot.RECIPE['features'] == [*capacity.RECIPE['features'], *source.FEATURES]
    assert len(pilot.RECIPE['features']) == 32
    assert design['folds'] == old['folds'] and design['fit_budget']['market'] == 4
    assert design['fit_budget']['synthetic'] == 4
    assert design['gate_evaluation_allowed'] is False and design['release_allowed'] is False
    old['recipe'] = {**capacity.RECIPE, 'max_iterations': 200}
    with pytest.raises(ValueError, match='frozen named-capacity'):
        pilot.design({'identity': {'design': old}})


def example():
    h = pd.DataFrame({'date': pd.bdate_range('2023-01-02', periods=5),
                      'cotton_session_index': np.arange(5), 'cotton_ret_1': [.1, .2, .3, .4, .5]})
    for i, name in enumerate(source.FEATURES):
        h[name] = [np.nan, .1+i, .2+i, .3+i, .4+i]
    return h


def test_mask_width_dates_and_missing_bits_match_and_numeric_is_preserved():
    h = example()
    num, mask = pilot.arm_history(h, 'numeric'), pilot.arm_history(h, 'mask')
    pd.testing.assert_frame_equal(num, h, check_exact=True)
    pd.testing.assert_frame_equal(num.drop(columns=list(source.FEATURES)), mask.drop(columns=list(source.FEATURES)))
    np.testing.assert_array_equal(num.isna(), mask.isna())
    assert (mask[list(source.FEATURES)].fillna(0).to_numpy() == 0).all()
    names = ['cotton_ret_1', *source.FEATURES]
    pn, pm = Preprocessor.fit(num, names), Preprocessor.fit(mask, names)
    xn, xm = inputs(num, num, pn), inputs(mask, mask, pm)
    np.testing.assert_array_equal(xn[:, len(names):], xm[:, len(names):])
    assert np.any(xn[:, 1:len(names)] != xm[:, 1:len(names)])
    pd.testing.assert_frame_equal(h, example())


@pytest.mark.parametrize('arm', ['wrong', 'ct'])
def test_unknown_arm_fails(arm):
    with pytest.raises(ValueError, match='Exact source arm'):
        pilot.arm_history(example(), arm)


def test_missing_source_fields_fail():
    with pytest.raises(ValueError, match='attached fields'):
        pilot.arm_history(example().drop(columns=[source.FEATURES[0]]), 'numeric')


def test_learning_controls_test_the_added_information_not_just_old_CT_fields():
    linear = pilot.synthetic('known_signal')
    nonlinear = pilot.synthetic('known_nonlinear')
    shifted = pilot.synthetic('negative_block_shift')
    for frame, train, test, recipe, trees in (linear, nonlinear, shifted):
        assert recipe['features'] == pilot.RECIPE['features'] and trees == 100
        assert train.date.max() < test.date.min()
        for f in source.FEATURES:
            np.testing.assert_array_equal(train[f], frame.loc[train.index, f])
            np.testing.assert_array_equal(test[f], frame.loc[test.index, f])
    np.testing.assert_array_equal(linear[0].target_return_5, .02*linear[0][source.FEATURES[0]])
    np.testing.assert_array_equal(nonlinear[0].target_return_5, .02*(nonlinear[0][source.FEATURES[0]]**2-1))
    np.testing.assert_array_equal(shifted[1].target_return_5, np.roll(linear[1].target_return_5, 137))
    _, train, test, recipe, trees = pilot.synthetic('small_subset_overfit')
    pd.testing.assert_frame_equal(train, test)
    assert len(train) == 128 and recipe['max_iterations'] == trees == 400


def test_source_change_rejects_cache_before_parent_access(tmp_path, monkeypatch):
    pilot.freeze_record(tmp_path/'ready.json', {'identity': {'profile': pilot.PROFILE, 'source_id': 'old'}})
    monkeypatch.setattr(pilot, 'research_source_identity', lambda _: {'source_id': 'new'})
    with pytest.raises(ValueError, match='new namespace'):
        pilot.verify(tmp_path, tmp_path)


def test_incomplete_or_unsafe_parent_is_rejected(tmp_path):
    pilot.freeze_record(tmp_path/'parent-files.json', {'files': {'../escape': '0'*64}})
    with pytest.raises(ValueError, match='completed parent changed'):
        pilot.verify_parent(tmp_path, tmp_path)


@pytest.mark.parametrize('minutes', [0, 31])
def test_invalid_time_budget_stops_before_verification(tmp_path, minutes):
    with pytest.raises(ValueError, match='<=30'):
        pilot.run(SimpleNamespace(root=tmp_path), tmp_path, minutes)


@pytest.mark.parametrize('interrupt', [False, True])
def test_run_masks_full_engine_history_and_restores_it_on_pause(tmp_path, monkeypatch, interrupt):
    h = example()
    dates = h.date.dt.strftime('%Y-%m-%d').tolist()
    fold = {'year': 2023, 'jobs': [{'role': 'outer', 'bucket': 0, 'train_dates': dates[:3], 'origins': dates[3:]}]}
    design = {'folds': [fold], 'fit_budget': {'market': 2}}
    ex = SimpleNamespace(root=tmp_path, identity={'source_id': 'test'}, history=h.copy())
    original = ex.history
    calls = []
    ex.ledger = SimpleNamespace(results=lambda: calls)
    monkeypatch.setattr(pilot, 'verify', lambda *_: ({'identity': {'design': design}}, h))
    monkeypatch.setattr(capacity, 'learning', lambda *a, **kw: None)
    monkeypatch.setattr(pilot, 'fit_consumption', lambda _: (0, set()))
    monkeypatch.setattr(pilot, 'research_source_identity', lambda _: {'source_id': 'test'})
    monkeypatch.setattr(pilot.labels, 'projection', lambda history, ds, *_a, **_kw: history.loc[history.date.dt.strftime('%Y-%m-%d').isin(ds)].copy())
    monkeypatch.setattr(pilot.labels, 'make_output', lambda *_: {'synthetic_test': True})
    monkeypatch.setattr(pilot, 'verify_output', lambda *_: None)
    def fit(recipe, train, validation, test, role, iterations):
        ex.before_compute()
        expected = pilot.arm_history(h, role.split('-')[0])
        pd.testing.assert_frame_equal(ex.history, expected, check_exact=True)
        processor = Preprocessor.fit(train, list(source.FEATURES))
        actual = inputs(ex.history, test, processor)
        wanted = inputs(expected, test, processor)
        np.testing.assert_array_equal(actual, wanted)
        if role.startswith('mask'):
            assert not actual[:, :4].any()
            if interrupt:
                raise pilot.FitBudgetReached('test pause')
        calls.append(role)
        return {'synthetic_test': True}
    ex.fit = fit
    result = pilot.run(ex, tmp_path, 30)
    assert result['status'] == ('planned_pause' if interrupt else 'complete')
    assert ex.history is original
    assert len(calls) == (1 if interrupt else 2)


@pytest.mark.parametrize('positive,gain,direction,upper,expected', [
    (True, 5, 55, 8, 'NAMED_HISTORY_CANDIDATE_RESEARCH_ONLY'),
    (True, 5, 54.999, 8, 'NAMED_HISTORY_CONTRIBUTION_BELOW_POINT_GOAL'),
    (True, 4.999, 55, 8, 'NAMED_HISTORY_CONTRIBUTION_BELOW_POINT_GOAL'),
    (True, 1, 60, 4.9, 'NAMED_HISTORY_CONTRIBUTION_BELOW_POINT_GOAL'),
    (False, 1, 60, 4.9, 'FIXED_NAMED_HISTORY_BELOW_PRACTICAL_GOAL'),
    (False, 1, 60, 5, 'INCONCLUSIVE_NAMED_HISTORY_CONTROL'),
])
def test_source_contribution_and_practical_decisions_stay_separate(positive, gain, direction, upper, expected):
    intervals = {}
    for block in (20, 60):
        intervals[f'mask_minus_numeric_raw_{block}'] = {'difference_ci_95': [1 if positive else 0, 2]}
        intervals[f'numeric_raw_vs_naive_{block}'] = {'gain_ci_95_pct': [-1, upper]}
    result = pilot.classify({'numeric_raw': {'gain_pct': gain, 'direction_pct': direction}}, intervals)
    assert result['decision'] == expected
    assert result['source_contribution_supported'] is positive
    assert result['numeric_practical_effect_excluded'] is (upper < 5)


def test_injected_learning_controls_are_remeasured_and_profile_bound(tmp_path, monkeypatch):
    settings, calls = {'failed': False}, []
    class Ledger:
        def __init__(self, root, identity):
            assert identity['profile'] == pilot.PROFILE
        def run(self, spec, operation, before_compute):
            before_compute()
            kind = spec['role']
            _, _, test, recipe, trees = pilot.synthetic(kind)
            assert spec['recipe'] == recipe and spec['iterations'] == trees
            pred = np.zeros(len(test)) if kind == 'negative_block_shift' or settings['failed'] else .99*test.target_return_5.to_numpy()
            calls.append(kind)
            return {'experiment_id': kind, 'result': {'predictions': pred.tolist()}}
    monkeypatch.setattr(capacity, 'Ledger', Ledger)
    ex = SimpleNamespace(root=tmp_path, identity={'source_id': 'test', 'versions': {}, 'python': 'test'})
    capacity.learning(ex, synthetic_fn=pilot.synthetic, profile=pilot.PROFILE)
    assert calls == list(pilot.KINDS)
    original = (tmp_path/'capacity-learning.json').read_bytes()
    settings['failed'] = True
    with pytest.raises(ValueError, match='Frozen record changed'):
        capacity.learning(ex, synthetic_fn=pilot.synthetic, profile=pilot.PROFILE)
    assert calls == [*pilot.KINDS, *pilot.KINDS]
    assert (tmp_path/'capacity-learning.json').read_bytes() == original
    ex.root = tmp_path/'failed'
    with pytest.raises(ValueError, match='Learning control failed'):
        capacity.learning(ex, synthetic_fn=pilot.synthetic, profile=pilot.PROFILE)
    assert pilot.read_record(ex.root/'capacity-learning.json')['status'] == 'failed'


def test_learning_failure_stops_before_market_fit(tmp_path, monkeypatch):
    h = example()
    ex = SimpleNamespace(root=tmp_path, history=h)
    monkeypatch.setattr(pilot, 'verify', lambda *_: ({}, h))
    def fail(*_a, **_kw):
        raise ValueError('Learning control failed')
    monkeypatch.setattr(capacity, 'learning', fail)
    with pytest.raises(ValueError, match='Learning control failed'):
        pilot.run(ex, tmp_path, 30)
    assert ex.history is h and not (tmp_path/'ledger').exists()
