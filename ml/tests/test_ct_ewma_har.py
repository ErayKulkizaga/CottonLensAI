from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import ct_ewma_har as study
from cottonlens_ml.research.ct_ewma_har import fit_offset, transform


def fixture():
    rng = np.random.default_rng(42); baseline = np.exp(rng.uniform(-12, -4, 600)); x = rng.uniform(-3, 3, (600, 3))
    rows = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=600), 'ct_ewma': baseline})
    for i, k in enumerate((1, 5, 22)):
        rows[f'har_{k}'] = np.log(baseline)+x[:, i]
    rows['variance_target_1'] = baseline*np.exp(.1+x@np.array([.2, .1, -.1]))
    return rows


def test_known_relative_variance_recovers_absolute_scale(admit_generated_backends, tmp_path):
    rows = fixture()
    with patch('cottonlens_ml.research.ct_ewma_har.require_training'), patch('cottonlens_ml.research.volatility.require_training'):
        result = fit_offset(rows.iloc[:500], rows.iloc[500:], tmp_path)
    np.testing.assert_allclose(result['variance_predictions'], rows.variance_target_1.iloc[500:], rtol=1e-5)


def test_future_targets_cannot_change_native_model(admit_generated_backends, tmp_path):
    rows = fixture(); first, second = tmp_path/'first', tmp_path/'second'; first.mkdir(); second.mkdir()
    with patch('cottonlens_ml.research.ct_ewma_har.require_training'), patch('cottonlens_ml.research.volatility.require_training'):
        before = fit_offset(rows.iloc[:500], rows.iloc[500:], first)
        rows.loc[500:, 'variance_target_1'] = np.nan
        after = fit_offset(rows.iloc[:500], rows.iloc[500:], second)
    assert before == after
    assert (first/'variance-model.json').read_bytes() == (second/'variance-model.json').read_bytes()


@pytest.mark.parametrize('value', [0, -1, np.nan])
def test_invalid_reference_is_not_imputed(value):
    rows = fixture(); rows.loc[0, 'ct_ewma'] = value
    with pytest.raises(ValueError, match='EWMA'):
        transform(rows, True)


def test_expired_session_reaches_ledger_guard_before_market_compute(tmp_path, monkeypatch):
    budget = {'market': 1, 'synthetic': 1, 'processes': 1, 'threads': 2, 'session_minutes': 30}
    job = {'year': 2016, 'role': 'outer', 'bucket': 0, 'train_dates': ['2000-01-03'], 'origins': ['2000-01-04']}
    identity = {'source_id': 'source', 'design_id': 'design', 'design': {'fit_budget': budget, 'jobs': [job]}}
    authorization = {'ready_sha256': 'digest', 'source_id': 'source', 'design_id': 'design',
                     'fit_budget': budget, 'fresh_restore_verified': True}
    computed = []

    class FakeLedger:
        def __init__(self, root, identity):
            self.root = root

        def run(self, specification, operation, *, before_compute=None):
            if specification.get('kind') == 'known_signal':
                rng = np.random.default_rng(42)
                baseline = np.exp(rng.uniform(-12, -4, 1200)); x = rng.uniform(-3, 3, (1200, 3))
                actual = baseline*np.exp(.1+x@np.array([.2, .1, -.1]))
                return {'experiment_id': 'synthetic-fixture', 'result': {'variance_predictions': actual[1000:].tolist()}}
            if before_compute is not None:
                before_compute()
            computed.append(specification)
            return {}

    moments = iter([0., 2000.])
    monkeypatch.setattr(study, 'time', SimpleNamespace(monotonic=lambda: next(moments)))
    monkeypatch.setattr(study, 'require_training', lambda *args: None)
    monkeypatch.setattr(study, 'verify', lambda *args: ({'identity': identity}, fixture()))
    monkeypatch.setattr(study, 'read_record', lambda *args: authorization)
    monkeypatch.setattr(study, 'digest', lambda *args: 'digest')
    monkeypatch.setattr(study, 'fit_consumption', lambda *args: (0, set()))
    monkeypatch.setattr(study, 'Ledger', FakeLedger)
    monkeypatch.setattr(study, 'projection', lambda history, dates: history.iloc[:1])
    result = study.run(tmp_path/'study', tmp_path, tmp_path/'authorization.json')
    assert result['status'] == 'planned_pause'
    assert result['market_fits'] == 0
    assert computed == []
