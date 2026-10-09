import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.price_mae_calibration import (
    errors,
    factor,
    past_calibration,
)
from scipy.optimize import linprog


def test_weighted_price_median_not_unweighted_return_median():
    assert factor([1., 1., 1000.], [1., 1., 100.]) == 10.
    assert factor([2., 6.], [1., 1.]) == 2.
    assert factor([3., 6.], [3., 6.]) == 1.


def test_analytic_solution_matches_independent_linear_program():
    rng = np.random.default_rng(7)
    p = rng.uniform(1, 100, 30)
    a = p * rng.uniform(.8, 1.2, 30)
    n = len(a)
    # min sum(u), -u <= A-fP <= u. No shared weighted-median algorithm.
    constraints = np.vstack([np.column_stack([p, -np.eye(n)]), np.column_stack([-p, -np.eye(n)])])
    result = linprog(np.r_[0., np.ones(n)], A_ub=constraints, b_ub=np.r_[a, -a], bounds=[(0, None)] * (n + 1), method='highs')
    assert result.success
    assert np.abs(a - factor(a, p) * p).sum() == pytest.approx(result.fun)


@pytest.mark.parametrize('actual,predicted', [([], []), ([0.], [1.]), ([1.], [-1.]), ([np.inf], [1.]), ([1., 2.], [1.])])
def test_invalid_prices_fail(actual, predicted):
    with pytest.raises(ValueError):
        factor(actual, predicted)


def training():
    return pd.DataFrame({'date': pd.to_datetime(['2015-01-01', '2015-01-02']),
                         'target_date_5': pd.to_datetime(['2015-01-08', '2015-01-09']),
                         'cotton_close': [50., 100.], 'target_return_1': np.log([.95, .95])})


def test_known_bias_removed_using_only_matured_training_rows():
    result = past_calibration(training(), [0., 0.], '2015-01-10')
    assert result['factor'] == pytest.approx(.95)
    assert result['training_mae_after'] == pytest.approx(0.)
    with pytest.raises(ValueError, match='matured'):
        past_calibration(training(), [0., 0.], '2015-01-09')
    late = training()
    late.loc[0, 'target_date_5'] = pd.Timestamp('2030-01-01')
    with pytest.raises(ValueError, match='matured'):
        past_calibration(late, [0., 0.], '2015-01-10')


def test_zero_return_is_exactly_the_same_price_error_as_naive():
    frame = pd.DataFrame({'cotton_close': [50., 100.], 'actual_return': [.001, -.002], 'prediction': [0., 0.]})
    naive, model = errors(frame, 'prediction')
    np.testing.assert_array_equal(naive, model)
