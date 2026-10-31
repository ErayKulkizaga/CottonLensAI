import itertools
from unittest.mock import patch

import numpy as np
import pytest

from research import ct_inference_calibration as study


def test_fast_intervals_match_the_existing_frozen_helper_and_short_stratum_is_fixed():
    sizes = (43, 209)
    values = np.random.default_rng(51).normal(size=(3, sum(sizes)))
    for block in (20, 60):
        weights = study.bootstrap_weights(sizes, block)
        assert study.check_native(weights, values, sizes, block) == 3
        np.testing.assert_allclose(weights.sum(axis=1), 1., rtol=0., atol=1e-14)
        if block == 60:
            np.testing.assert_array_equal(weights[:, :43], np.full((10000, 43), 1/252))


def test_analytic_variance_matches_exhaustive_small_resampler():
    # n=5, width3: starts0,1,2 for each of two blocks, final block truncated2.
    rows = []
    for a, b in itertools.product(range(3), repeat=2):
        rows.append(np.bincount([a, a+1, a+2, b, b+1], minlength=5)/5)
    weights = np.array(rows)
    ratio = 5*float(np.sum((weights-weights.mean(axis=0))**2)/len(weights))
    assert study.iid_variance_ratio((5,), 3)['ratio'] == pytest.approx(ratio)
    assert study.iid_variance_ratio((43,), 60)['ratio'] == pytest.approx(0.)


def test_known_null_uses_valid_qlike_pairs_and_only_declared_scenarios():
    for phi in (0., .8):
        null = study.synthetic_differences(phi, 0.)
        assert null.shape == (1024, 252)
        assert np.min(null) >= -study.LOG_RATIO
        np.testing.assert_array_equal(null, study.synthetic_differences(phi, 0.))
    with pytest.raises(ValueError, match='preregistered'):
        study.synthetic_differences(.5, 0.)


def test_positive_gain_is_not_an_automatic_gate_pass():
    weights = study.bootstrap_weights((43, 209), 60)
    values = np.ones((2, 252))*.001
    values[1, -1] = -1.
    mean, ci = study.intervals(weights, values)
    assert mean[0] > 0 and ci[0, 0] > 0
    assert mean[1] < 0


def test_no_market_reader_or_fit_and_existing_output_is_never_overwritten(tmp_path):
    existing = tmp_path/'existing'
    existing.mkdir()
    with (patch.object(study, 'synthetic_differences', side_effect=AssertionError('No run')),
          pytest.raises(ValueError, match='Preserve completed')):
        study.audit(tmp_path, existing)
    assert study.wilson(0, 1024)[1] < .004
    assert study.wilson(1024, 1024)[0] > .996


@pytest.mark.parametrize('sizes,block', [([], 20), ([0], 20), ([4.5], 2), ([43], 0)])
def test_invalid_resampling_settings_fail(sizes, block):
    with pytest.raises(ValueError, match='Positive integer'):
        study.bootstrap_weights(sizes, block)
