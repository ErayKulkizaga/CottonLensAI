import numpy as np
import pytest

from research.seasonal_risk_moment import centered_covariance, intervals


def test_fixed_season_covariance_and_interval_are_positive_for_known_relation():
    part = np.column_stack([np.tile([0., 1.], 120), np.tile([0., 2.], 120)])
    assert centered_covariance(part) == 0.5
    summary = intervals([part]*8, 20, repetitions=256)
    assert summary['covariance'] == 0.5
    assert summary['ci_95'][0] > 0


def test_one_season_cannot_produce_a_score():
    with pytest.raises(ValueError, match='Both seasons'):
        centered_covariance(np.column_stack([np.ones(10), np.arange(10)]))
