"""Independent sufficient statistics preserve absent calendar slots."""

import numpy as np
import pytest

from research.ams_curve_risk_moment import interval
from research.verify_ams_curve_intervals import bootstrap, covariance


def test_independent_intervals_match_with_calendar_gaps():
    rng = np.random.default_rng(7)
    parts = []
    for count in (71, 83):
        x = rng.normal(size=count)
        part = np.column_stack([x, .2*x+rng.normal(size=count), np.ones(count)])
        part[::9, 2] = 0
        parts.append(part)
    for block in (3, 20, 60):
        expected = interval(parts, block, repetitions=260)
        point, limits = bootstrap(parts, block, repetitions=260)
        assert point == pytest.approx(expected['covariance'], abs=1e-12)
        assert limits == pytest.approx(expected['ci_95'], abs=1e-12)


def test_independent_covariance_rejects_bad_weights_and_coverage():
    with pytest.raises(ValueError, match='calendar'):
        covariance(np.array([[1., 2., 1.], [2., 3., 0.]]))
    with pytest.raises(ValueError, match='calendar'):
        covariance(np.array([[1., 2., 1.], [2., 3., 1.], [3., 4., 2.]]))
