"""Missing source days retain calendar slots without shifting the moment."""

import numpy as np

from research.ams_curve_risk_moment import interval, weighted_covariance


def test_missing_slot_has_zero_weight_and_moment_is_signed():
    observed = np.array([[0., 0., 1.], [1., 2., 1.], [2., 4., 1.]])
    with_gap = np.insert(observed, 1, [100., -100., 0.], axis=0)
    assert weighted_covariance(observed) == weighted_covariance(with_gap)
    assert weighted_covariance(with_gap) > 0
    assert weighted_covariance(np.column_stack([with_gap[:, 0], -with_gap[:, 1],
                                               with_gap[:, 2]])) < 0


def test_block_interval_is_deterministic_for_constant_relation():
    part = np.column_stack([np.arange(12.), 2*np.arange(12.), np.ones(12)])
    first = interval([part], block=3, repetitions=20)
    second = interval([part], block=3, repetitions=20)
    assert first == second
    assert first['covariance'] > 0
