import numpy as np
import pytest

from research.ct_signed_risk_score import basis, gate, score


def fixture():
    rng = np.random.default_rng(42)
    return rng.normal(size=(500, 3)), rng.normal(size=500)


def test_past_projection_controls_unsigned_size_without_erasing_sign():
    x, shock = fixture()
    phi, metadata = basis(x, x, shock, shock)
    nuisance = np.column_stack([np.ones(len(x)), x, np.abs(shock)])
    np.testing.assert_allclose(nuisance.T@phi/len(x), 0, atol=1e-12)
    assert np.mean(phi**2) == pytest.approx(1)
    assert np.mean(phi*shock) > 0
    assert metadata['feature_only']


def test_changing_future_features_cannot_change_earlier_basis_or_past_projection():
    x, shock = fixture()
    first, meta = basis(x[:400], x[400:], shock[:400], shock[400:])
    changed_x, changed_s = x[400:].copy(), shock[400:].copy()
    changed_x[50:] += 100; changed_s[50:] -= 100
    second, new_meta = basis(x[:400], changed_x, shock[:400], changed_s)
    np.testing.assert_array_equal(first[:50], second[:50])
    assert meta == new_meta


def test_one_sided_shocks_are_not_new_signed_information():
    x, shock = fixture()
    with pytest.raises(ValueError, match='not identifiable'):
        basis(x, x, np.abs(shock), np.abs(shock))


def test_no_silent_missing_row_intersection_or_nonfinite_imputation():
    x, shock = fixture()
    with pytest.raises(ValueError, match='Aligned'):
        basis(x, x, shock[:-1], shock)
    shock[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        basis(x, x, shock, shock)


def test_qlike_score_is_negative_loss_derivative_and_zero_target_is_valid():
    y, p, phi = np.array([0., .3, 2.]), np.array([.1, .2, 1.]), np.array([-1., 2., .3])
    epsilon = 1e-6
    def loss(gamma):
        predicted = p*np.exp(gamma*phi)
        return np.log(predicted)+y/predicted
    np.testing.assert_allclose(score(y, p, phi), -(loss(epsilon)-loss(-epsilon))/(2*epsilon), atol=1e-9)
    with pytest.raises(ValueError, match='positive forecasts'):
        score(y, np.array([0., 1., 1.]), phi)
    with pytest.raises(ValueError, match='nonnegative targets'):
        score(-np.ones(3), p, phi)


def test_both_signs_are_allowed_but_year_and_tail_gates_remain():
    intervals = {'20': [-3., -1.], '60': [-2., -.5]}
    years = {str(i): -1. for i in range(8)}
    assert gate(intervals, years, -.1) == (True, -1, 8)
    assert not gate(intervals, years, .1)[0]
    years.update({'0': 1., '1': 1., '2': 1.})
    assert gate(intervals, years, -.1) == (False, -1, 5)
    assert gate({'20': [-1., 1.], '60': [-2., -1.]}, years, -.1) == (False, 0, 0)
