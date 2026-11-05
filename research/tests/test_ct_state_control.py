from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research import ct_shock_control as study
from research.verify_ct_shock_control import independent_dxy, profile_definition

PROFILE = study.STATE_PROFILE


def history():
    dates = pd.bdate_range('2000-01-03', periods=100)
    close = 100*np.exp(np.sin(np.arange(len(dates)))/100)
    h = pd.DataFrame({'date': dates, 'cotton_close': close})
    h['cotton_ret_1'] = np.log(h.cotton_close/h.cotton_close.shift())
    h['ct_ewma'] = h.cotton_ret_1.pow(2).ewm(alpha=.06, adjust=False).mean().clip(lower=1e-12)
    h['variance_target_1'] = h.cotton_ret_1.shift(-1).pow(2)
    for k in (1, 5, 22):
        h[f'har_{k}'] = np.log(h.cotton_ret_1.pow(2).rolling(k).mean().clip(lower=1e-12))
    market = pd.DataFrame({'date': dates, 'series': 'dxy', 'close': 80+np.arange(len(dates))**1.1})
    return h, market


def test_state_capacity_raw_calendar_and_independent_alignment():
    h, market = history()
    # Cotton has a holiday which is not a DXY holiday; source windows stay raw.
    h = h.drop(index=40).reset_index(drop=True)
    h['cotton_ret_1'] = np.log(h.cotton_close/h.cotton_close.shift())
    h['ct_ewma'] = h.cotton_ret_1.pow(2).ewm(alpha=.06, adjust=False).mean().clip(lower=1e-12)
    attached = study.attach(h, PROFILE, market)
    checked = independent_dxy(market, h.date)
    np.testing.assert_allclose(attached.dxy_rv5, checked.dxy_rv5, rtol=1e-10, atol=1e-14, equal_nan=True)
    for name in ('dxy_state_source_date', 'dxy_state_available_at', 'dxy_state_window_start', 'dxy_state_age', 'decision_at'):
        pd.testing.assert_series_equal(attached[name], checked[name])
    dates = h.date.iloc[45:50].dt.strftime('%Y-%m-%d').tolist()
    own, external = (study.relative(study.project(attached, dates, a, PROFILE), False, PROFILE)
                     for a in ('own_level', 'dxy_state'))
    np.testing.assert_array_equal(own[study.parent.FEATURES], external[study.parent.FEATURES])
    np.testing.assert_allclose(own.origin_state, np.log(own.ct_ewma))
    expected = attached.set_index(attached.date.dt.strftime('%Y-%m-%d')).loc[dates]
    np.testing.assert_allclose(external.origin_state, np.log(expected.dxy_rv5))
    assert external.date.dt.strftime('%Y-%m-%d').tolist() == dates
    for arm in ('own_level', 'dxy_state'):
        assert study.recipe(arm, PROFILE)['parameters'] == 5
        assert study.recipe(arm, PROFILE)['variance_features'][-1] == 'origin_state'
    assert profile_definition(PROFILE) == (('own_level', 'dxy_state'), 'dxy_state', 'own_level', 'origin_state', False)


@pytest.mark.parametrize('arm', ['own_level', 'dxy_state'])
def test_declared_state_is_fitted_without_future_targets(tmp_path, arm):
    rng = np.random.default_rng(42); x = rng.uniform(-2, 2, (600, 4))
    b = np.exp(rng.uniform(-12, -4, len(x)))
    f = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=len(x)), 'ct_ewma': b, 'origin_state': x[:, 3]})
    for i, name in enumerate(study.parent.FEATURES):
        f[name] = np.log(b)+x[:, i]
    f['variance_target_1'] = b*np.exp(.1+x@np.array([.2, .1, -.1, -.15]))
    first, second = tmp_path/'a', tmp_path/'b'; first.mkdir(); second.mkdir()
    with patch.object(study, 'require_training'), patch('cottonlens_ml.research.volatility.require_training'):
        before = study.fit(f.iloc[:500], f.iloc[500:], first, arm, PROFILE)
        np.testing.assert_allclose(before['variance_predictions'], f.variance_target_1.iloc[500:], rtol=1e-5)
        f.loc[500:, 'variance_target_1'] = np.nan
        after = study.fit(f.iloc[:500], f.iloc[500:], second, arm, PROFILE)
    assert before == after
    assert (first/'variance-model.json').read_bytes() == (second/'variance-model.json').read_bytes()


def test_future_raw_prices_do_not_change_past_source_state():
    h, market = history(); before = study.attach(h, PROFILE, market)
    market.loc[70:, 'close'] *= 10
    after = study.attach(h, PROFILE, market)
    pd.testing.assert_frame_equal(before.iloc[:70], after.iloc[:70])


@pytest.mark.parametrize('invalid', [0., -1., np.nan])
def test_invalid_latest_window_is_not_replaced_with_older_state(invalid):
    h, market = history(); market.loc[50, 'close'] = invalid
    attached = study.attach(h, PROFILE, market)
    with pytest.raises(ValueError, match='Do not drop or impute'):
        study.project(attached, [str(h.date.iloc[50].date())], 'dxy_state', PROFILE)
    checked = independent_dxy(market, h.date)
    assert np.isnan(checked.dxy_rv5.iloc[50])


def test_trim_is_sensitivity_for_original_three_risk_conditions():
    intervals = {'20': [.01, .04], '60': [.01, .05]}
    years = {str(y): .01 if y < 2022 else -.01 for y in range(2016, 2024)}
    assert study.risk_gate(.01, intervals, years, -.02)
    assert not study.risk_gate(.01, intervals, years, -.02, require_positive_trim=True)
    assert not study.risk_gate(.01, {'20': [-.01, .04], '60': [.01, .05]}, years, .02)


def test_profile_or_arm_cannot_be_silently_relabelled():
    with pytest.raises(ValueError, match='registered'):
        study.definition('another-profile')
    with pytest.raises(ValueError, match='two frozen'):
        study.recipe('signed', PROFILE)
    with pytest.raises(ValueError, match='Unknown frozen'):
        profile_definition('another-profile')


def test_completed_profile_cannot_be_fitted_again(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'validate', lambda *a: None)
    monkeypatch.setattr(study, 'load_registry', lambda *a: ({}, {}))
    monkeypatch.setattr(study, 'check', lambda *a, **k: {'experiments': [{'completed_market_fits': 169}]})
    with pytest.raises(ValueError, match='Completed profile'):
        study.prepare(tmp_path, tmp_path/'new-run', tmp_path/'reference', PROFILE, tmp_path/'market.parquet')
