"""Independent finite-cohort and arithmetic checks; no fitting."""
import numpy as np
import pytest
from cottonlens_ml.research.full_year import interval_metrics


@pytest.mark.parametrize('actual,lo,hi,coverage', [
    ([0., 1.], [-1.], [2.], .9),
    ([0.], [-1., -2.], [2., 3.], .9),
    ([[0.]], [[-1.]], [[1.]], .9),
    ([], [], [], .9),
    ([np.nan], [-1.], [1.], .9),
    ([np.inf], [-1.], [1.], .9),
    ([0.], [-np.inf], [1.], .9),
    ([0.], [-1.], [np.nan], .9),
    ([0.], [1.], [-1.], .9),
    ([0.], [-1.], [1.], 0.),
    ([0.], [-1.], [1.], 1.),
    ([0.], [-1.], [1.], -1.),
    ([0.], [-1.], [1.], 1.1),
    ([0.], [-1.], [1.], np.nan),
    ([0.], [-1.], [1.], np.inf),
])
def test_interval_metrics_reject_invalid_cohort_before_scoring(actual, lo, hi, coverage):
    with pytest.raises(ValueError):
        interval_metrics(actual, lo, hi, coverage)


def test_interval_metrics_match_independent_manual_scores_and_boundary_hits():
    metrics, scores, hits = interval_metrics([-2., 0., 4.], [-1.]*3, [1.]*3, .8)
    # Width2, alpha0.2: penalties10 and30 outside; one point inside.
    np.testing.assert_allclose(scores, [12., 2., 32.], rtol=0, atol=1e-13)
    np.testing.assert_array_equal(hits, [0., 1., 0.])
    assert metrics == pytest.approx({'coverage': 1/3, 'mean_width': 2.,
        'interval_score': 46/3, 'pinball_lower': .5, 'pinball_upper': 3.1/3})
    metrics, scores, hits = interval_metrics([-1., 1., 0.], [-1., -1., 0.], [1., 1., 0.], .9)
    np.testing.assert_allclose(scores, [2., 2., 0.])
    np.testing.assert_array_equal(hits, [1., 1., 1.])
    assert metrics['coverage'] == 1.
