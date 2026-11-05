from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import volatility as v


def data():
    x = np.random.default_rng(42).uniform(-10, -4, (600, 3))
    a = np.exp(-1 + x @ np.array([.2, .4, .1]))
    f = pd.DataFrame(x, columns=['har_1', 'har_5', 'har_22'])
    f['variance_target_1'] = a
    f['date'] = pd.bdate_range('2000-01-03', periods=len(f))
    return f


def test_analytic_gradient_matches_finite_difference_with_zero_target():
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(20), rng.normal(size=(20, 3))])
    a = rng.uniform(0, 1, 20); a[0] = 0
    theta = np.array([.2, .3, -.1, .1])
    _, gradient = v.qlike_value_gradient(theta, x, a)
    numeric = []
    for i in range(4):
        delta = np.eye(4)[i] * 1e-6
        numeric.append((v.qlike_value_gradient(theta+delta, x, a)[0]-v.qlike_value_gradient(theta-delta, x, a)[0])/2e-6)
    np.testing.assert_allclose(gradient, numeric, atol=1e-9)


def test_noiseless_known_mean_is_recovered():
    f = data()
    coefficient, info = v.qlike_har(f.iloc[:500], 1)
    prediction = np.exp(np.column_stack([np.ones(100), f.iloc[500:][['har_1', 'har_5', 'har_22']]]) @ coefficient)
    np.testing.assert_allclose(prediction, f.variance_target_1.iloc[500:], rtol=1e-6)
    assert info['success']
    assert len(coefficient) == 4


def test_objective_not_worse_than_initial_and_gradient_certified():
    f = data()
    f['variance_target_1'] *= np.random.default_rng(12).lognormal(0, .8, len(f))
    f.loc[0, 'variance_target_1'] = 0.
    _, info = v.qlike_har(f, 1)
    assert info['final_training_qlike'] <= info['initial_training_qlike'] + 1e-10
    assert info['gradient_inf_norm'] <= 1e-6


@pytest.mark.parametrize('value', [np.nan, np.inf, -1])
def test_invalid_targets_rejected(value):
    f = data(); f.loc[0, 'variance_target_1'] = value
    with pytest.raises(v.InvalidVarianceFit):
        v.qlike_har(f, 1)


def test_all_zero_targets_have_no_finite_unconstrained_optimum():
    f = data(); f['variance_target_1'] = 0.
    with pytest.raises(v.InvalidVarianceFit, match='positive mass'):
        v.qlike_har(f, 1)


def test_numerical_failure_cannot_fallback_to_ols():
    f = data()
    fake = SimpleNamespace(x=np.zeros(4), success=False, nit=0)
    with patch('scipy.optimize.minimize', return_value=fake), pytest.raises(v.InvalidVarianceFit, match='no OLS fallback'):
        v.qlike_har(f, 1)


def test_future_test_values_cannot_change_fitted_parameters(tmp_path):
    f = data();train=f.iloc[:500];test=f.iloc[500:].copy()
    spec = {'family': 'har', 'horizon': 1, 'device': 'cpu', 'training_loss': 'qlike'}
    with patch.object(v, 'require_training'):
        v.fit_predict(f, train, test, spec, tmp_path)
        first = (tmp_path/'variance-model.json').read_bytes()
        test['variance_target_1'] *= 100
        v.fit_predict(f, train, test, spec, tmp_path)
        assert first == (tmp_path/'variance-model.json').read_bytes()


def test_unknown_loss_rejected(tmp_path):
    f = data()
    with patch.object(v, 'require_training'), pytest.raises(ValueError, match='supported HAR'):
        v.fit_predict(f, f.iloc[:500], f.iloc[500:], {'family':'har','horizon':1,'training_loss':'unknown'}, tmp_path)


def test_declared_fourth_feature_reaches_fit_and_heldout_prediction(tmp_path):
    f = data(); f['origin_shock'] = np.random.default_rng(7).uniform(-2, 2, len(f))
    f['variance_target_1'] *= np.exp(.35*f.origin_shock)
    names = ['har_1', 'har_5', 'har_22', 'origin_shock']
    spec = {'family': 'har', 'horizon': 1, 'training_loss': 'qlike', 'variance_features': names}
    with patch.object(v, 'require_training'):
        result = v.fit_predict(f, f.iloc[:500], f.iloc[500:], spec, tmp_path)
    import json
    model = json.loads((tmp_path/'variance-model.json').read_bytes())
    assert model['features'] == names
    assert len(model['coef']) == 5
    assert model['coef'][-1] == pytest.approx(.35, abs=1e-6)
    np.testing.assert_allclose(result['variance_predictions'], f.variance_target_1.iloc[500:], rtol=1e-6)


def test_legacy_default_matches_explicit_three_features_exactly():
    f = data()
    default, info = v.qlike_har(f, 1)
    explicit, second_info = v.qlike_har(f, 1, feature_names=['har_1', 'har_5', 'har_22'])
    np.testing.assert_array_equal(default, explicit)
    assert info == second_info


@pytest.mark.parametrize('names', [[], ['har_1', 'har_1'], ['variance_target_1'], 'har_1'])
def test_invalid_declared_feature_list_is_not_silently_ignored(names):
    with pytest.raises(ValueError, match='Unique declared'):
        v.qlike_har(data(), 1, feature_names=names)


def test_declared_features_cannot_silently_enter_another_loss(tmp_path):
    with patch.object(v, 'require_training'), pytest.raises(ValueError, match='QLIKE HAR path'):
        v.fit_predict(data(), data(), data(), {'family': 'har', 'horizon': 1,
                      'variance_features': ['har_1'], 'training_loss': 'log_mse'}, tmp_path)
