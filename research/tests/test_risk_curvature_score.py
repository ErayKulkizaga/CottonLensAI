import numpy as np
import pytest

from research.risk_curvature_score import curvature_basis, improvement_score, supported


def test_basis_is_past_only_and_orthogonal_to_original_affine_features():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(100, 3)); test = rng.normal(size=(20, 3))
    mean, scale = train.mean(axis=0), train.std(axis=0)
    phi, info = curvature_basis(train, test, mean, scale)
    changed, same_info = curvature_basis(train, np.vstack([test, test*100]), mean, scale)
    assert info == same_info
    np.testing.assert_allclose(phi, changed[:len(test)], atol=1e-12)
    past_phi, _ = curvature_basis(train, train, mean, scale)
    z = np.column_stack([np.ones(len(train)), (train-mean)/scale])
    np.testing.assert_allclose(z.T@past_phi, 0, atol=1e-10)
    assert np.mean(past_phi**2) == pytest.approx(1.)


def test_curvature_score_is_negative_qlike_derivative_including_zero_actual():
    actual = np.array([0., .2, .6, .1]); predicted = np.array([.1, .3, .2, .5])
    phi = np.array([-1., .1, .6, 2.]); epsilon = 1e-6
    score = improvement_score(actual, predicted, phi).mean()
    up, down = predicted*np.exp(epsilon*phi), predicted*np.exp(-epsilon*phi)
    difference = ((np.log(down)+actual/down).mean()-(np.log(up)+actual/up).mean())/(2*epsilon)
    assert score == pytest.approx(difference, abs=1e-9)
    np.testing.assert_array_equal(improvement_score(predicted, predicted, phi), np.zeros(4))


def test_no_degenerate_or_future_normalization_or_invalid_predictions():
    rng = np.random.default_rng(1); x = rng.normal(size=(30, 3))
    with pytest.raises(AssertionError):
        curvature_basis(x, x, x.mean(axis=0)+1, x.std(axis=0))
    with pytest.raises(ValueError, match='identifiable'):
        curvature_basis(np.zeros((30, 3)), np.zeros((1, 3)), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        improvement_score([1.], [0.], [1.])


def test_two_sided_rule_requires_consistent_intervals_years_and_trim():
    negative = {str(i): -1. for i in range(8)}
    assert supported(-1., {'20': [-2., -.1], '60': [-1.5, -.2]}, negative, -.3) == (True, -1, 8)
    assert supported(-1., {'20': [-2., .1], '60': [-1.5, -.2]}, negative, -.3)[0] is False
    assert supported(-1., {'20': [-2., -.1], '60': [-1.5, -.2]}, negative, .3)[0] is False
