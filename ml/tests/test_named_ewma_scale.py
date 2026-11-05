from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.named_ewma_scale import fit_scale


def rows():
    b = np.exp(np.random.default_rng(42).uniform(-12, -4, 600))
    return pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=600),
                         'named_ewma': b, 'variance_target_1': 1.2*b})


def test_known_scale_recovered_without_using_test_labels(tmp_path):
    f = rows()
    with patch('cottonlens_ml.research.named_ewma_scale.require_training'):
        first = fit_scale(f.iloc[:500], f.iloc[500:], tmp_path)
        native = (tmp_path/'variance-model.json').read_bytes()
        f.loc[500:, 'variance_target_1'] = np.nan
        second = fit_scale(f.iloc[:500], f.iloc[500:], tmp_path)
    assert first == second
    assert native == (tmp_path/'variance-model.json').read_bytes()
    np.testing.assert_allclose(first['variance_predictions'], 1.2*f.named_ewma.iloc[500:], rtol=1e-12)


def test_zero_actuals_are_not_log_floor_imputed(tmp_path):
    f = rows(); f.loc[0, 'variance_target_1'] = 0
    with patch('cottonlens_ml.research.named_ewma_scale.require_training'):
        r = fit_scale(f.iloc[:500], f.iloc[500:], tmp_path)
    np.testing.assert_allclose(r['variance_predictions'], 1.2*499/500*f.named_ewma.iloc[500:])


@pytest.mark.parametrize(('column', 'value'), [('named_ewma', 0), ('named_ewma', np.nan),
                                              ('variance_target_1', -1), ('variance_target_1', np.inf)])
def test_invalid_training_inputs_fail_closed(tmp_path, column, value):
    f = rows(); f.loc[0, column] = value
    with patch('cottonlens_ml.research.named_ewma_scale.require_training'), pytest.raises(ValueError):
        fit_scale(f.iloc[:500], f.iloc[500:], tmp_path)
    assert not (tmp_path/'variance-model.json').exists()


def test_all_zero_labels_do_not_fallback(tmp_path):
    f = rows(); f['variance_target_1'] = 0
    with patch('cottonlens_ml.research.named_ewma_scale.require_training'), pytest.raises(ValueError, match='optimum'):
        fit_scale(f.iloc[:500], f.iloc[500:], tmp_path)


def test_training_guard_precedes_fit(tmp_path):
    f = rows()
    with patch('cottonlens_ml.research.named_ewma_scale.require_training', side_effect=RuntimeError('blocked')), pytest.raises(RuntimeError, match='blocked'):
        fit_scale(f.iloc[:500], f.iloc[500:], tmp_path)
    assert not (tmp_path/'variance-model.json').exists()
