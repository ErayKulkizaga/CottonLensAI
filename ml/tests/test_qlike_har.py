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
