from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.named_ewma_har import fit_offset


def rows():
    rng = np.random.default_rng(42)
    baseline = np.exp(rng.uniform(-12, -4, 600))
    x = rng.uniform(-3, 3, (600, 3))
    f = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=600), 'named_ewma': baseline})
    for i, k in enumerate((1, 5, 22)):
        f[f'har_{k}'] = np.log(baseline)+x[:, i]
    f['variance_target_1'] = baseline*np.exp(.1+x@np.array([.2, .1, -.1]))
    return f


def test_four_parameter_known_relative_mean_recovers_absolute_variance(admit_generated_backends, tmp_path):
    f = rows()
    with patch('cottonlens_ml.research.named_ewma_har.require_training'), patch('cottonlens_ml.research.volatility.require_training'):
        r = fit_offset(f.iloc[:500], f.iloc[500:], tmp_path)
    np.testing.assert_allclose(r['variance_predictions'], f.variance_target_1.iloc[500:], rtol=1e-5)


def test_future_labels_cannot_change_native_model_or_forecast(admit_generated_backends, tmp_path):
    f = rows(); first_path = tmp_path/'first'; second_path = tmp_path/'second'
    first_path.mkdir(); second_path.mkdir()
    with patch('cottonlens_ml.research.named_ewma_har.require_training'), patch('cottonlens_ml.research.volatility.require_training'):
        before = fit_offset(f.iloc[:500], f.iloc[500:], first_path)
        f.loc[500:, 'variance_target_1'] = np.nan
        after = fit_offset(f.iloc[:500], f.iloc[500:], second_path)
    assert before == after
    assert (first_path/'variance-model.json').read_bytes() == (second_path/'variance-model.json').read_bytes()


@pytest.mark.parametrize('value', [0, np.nan, -1])
def test_invalid_baseline_is_not_imputed(value, tmp_path):
    f = rows(); f.loc[0, 'named_ewma'] = value
    with patch('cottonlens_ml.research.named_ewma_har.require_training'), pytest.raises(ValueError, match='EWMA'):
        fit_offset(f.iloc[:500], f.iloc[500:], tmp_path)
    assert not (tmp_path/'variance-model.json').exists()
