import numpy as np
import pytest

from research.asymmetric_risk_moment import intervals, partial_covariance


def test_unsigned_risk_relationship_does_not_manufacture_asymmetry():
    z = np.array([-1., 2, -3, 4, -5, 6])
    w = np.abs(z)
    assert partial_covariance(np.column_stack([z, 5*w, w, np.ones(6)])) == pytest.approx(0, abs=1e-12)


def test_known_asymmetry_survives_size_control_and_ordinal_missingness():
    z = np.tile([-1., 2, -3, 4, -5, 6], 20); w = np.abs(z)
    part = np.column_stack([z, 3*z+5*w, w, np.ones(len(z))])
    assert partial_covariance(part) > 0
    assert intervals([part], 20, 256)['ci_95'][0] > 0
    missing = np.array([[999, -999, 999, 0]], float)
    assert partial_covariance(np.vstack([part, missing])) == partial_covariance(part)


def test_degenerate_control_is_unavailable_not_negative():
    with pytest.raises(ValueError, match='no usable variation'):
        partial_covariance(np.array([[1, 2, 1, 1], [2, 3, 1, 1], [3, 4, 1, 1]], float))
