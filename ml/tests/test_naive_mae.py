"""Synthetic-only regressions for the fixed median experiment."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import models
from cottonlens_ml.research import naive_mae as m
from cottonlens_ml.research.admission import model_inputs_id
from cottonlens_ml.research.ledger import Ledger
from cottonlens_ml.research.protocol import Target
from cottonlens_ml.runtime_guard import require_training


def test_price_features_are_prefix_invariant_and_use_only_price():
    rng = np.random.default_rng(42)
    data = pd.DataFrame({'date': pd.bdate_range('2010-01-01', periods=180),
                         'cotton_close': 80*np.exp(np.cumsum(rng.normal(0, .01, 180))), 'dxy_ret_1': 0.})
    a = m.features(data)
    data.loc[120:, 'cotton_close'] *= 5
    data['dxy_ret_1'] = 100
    pd.testing.assert_frame_equal(a.loc[:119, m.FEATURES], m.features(data).loc[:119, m.FEATURES])
    np.testing.assert_allclose(a.mae_ret5.iloc[30], np.log(data.cotton_close.iloc[30]/data.cotton_close.iloc[25]))
    flat = m.features(data.assign(cotton_close=80.))
    assert flat.mae_z20.isna().all()


def test_zero_correction_is_exact_naive_and_price_delta_roundtrip():
    data = pd.DataFrame({'date': pd.bdate_range('2010-01-01', periods=10),
                         'cotton_close': 80., 'target_return_1': np.linspace(-.02, .02, 10),
                         'predicted_return': 0.})
    naive, model = m.loss_arrays(data)
    np.testing.assert_array_equal(naive, model)
    target = Target.fit(data, 1, 'price_delta')
    np.testing.assert_allclose(target.inverse(target.forward(data, 1), data.cotton_close),
                               data.target_return_1, atol=1e-8)
    with pytest.raises(ValueError, match='positive domain'):
        target.inverse(np.array([-1e12]), np.array([80.]))
    with pytest.raises(ValueError, match='unique'):
        m.loss_arrays(pd.concat([data, data]))


def test_quantile_permission_is_profile_only_and_ci_closed(monkeypatch):
    monkeypatch.setenv('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR', '1')
    monkeypatch.delenv('CI', raising=False)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    with pytest.raises(RuntimeError, match='restricted'):
        require_training('quantile', 'cpu')
    require_training('quantile', 'cpu', profile=m.PROFILE)
    monkeypatch.setenv('CI', 'true')
    with pytest.raises(RuntimeError, match='CI'):
        require_training('quantile', 'cpu', profile=m.PROFILE)
    spec = m.recipe({}, 'quantile', 1)
    m.validate_quantile(spec)
    altered = deepcopy(spec)
    altered['params']['alpha'] = .1
    with pytest.raises(ValueError, match='frozen'):
        m.validate_quantile(altered)


@pytest.mark.parametrize('family', m.GROUPS)
def test_native_replay_resume_and_corrupt_payload_rejected(tmp_path, monkeypatch, family):
    from admission_fixture import synthetic_admitter
    # Explicit synthetic fixture, never an environment-based real-data bypass.
    monkeypatch.setattr(models, 'require_training', lambda *a, **kw: None)
    kind, frame = next(m.control_frames())
    train, test = frame.iloc[:600], frame.iloc[610:790]
    spec = m.recipe({}, family, 1)
    ledger = Ledger(tmp_path/'ledger', {'profile': m.PROFILE})
    admission = synthetic_admitter(tmp_path/'synthetic-admission', ledger.identity)
    specification = {'kind': kind, 'recipe': spec, 'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(),
                     'test_dates': test.date.dt.strftime('%Y-%m-%d').tolist(),
                     'model_inputs_id': model_inputs_id(frame, train, None, test, spec, 1)}
    record = ledger.run(specification, lambda workspace: models.fit_predict(
        frame, train, None, test, spec, workspace, iterations=1), admission=admission)
    assert m.verify_linear(record, ledger, frame) < 1e-7
    def fail(_):
        pytest.fail('Completed fit must not retrain')
    assert ledger.run(specification, fail)['experiment_id'] == record['experiment_id']
    path = ledger.root/record['payload_root']/'model.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        m.verify_linear(record, ledger, frame)


def test_quantile_solver_failure_is_not_a_fallback(tmp_path, monkeypatch):
    from admission_fixture import synthetic_admitter
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import QuantileRegressor
    monkeypatch.setattr(models, 'require_training', lambda *a, **kw: None)
    def failed(*a, **kw):
        import warnings
        warnings.warn('synthetic timeout', ConvergenceWarning)
    monkeypatch.setattr(QuantileRegressor, 'fit', failed)
    _, frame = next(m.control_frames())
    recipe = m.recipe({}, 'quantile', 1)
    train, test = frame.iloc[:600], frame.iloc[610:790]
    ledger_identity = {'profile': 'synthetic-solver-failure'}
    specification = {'recipe': recipe, 'model_inputs_id': model_inputs_id(frame, train, None, test, recipe, 1)}
    admission = synthetic_admitter(tmp_path/'synthetic-admission', ledger_identity)
    with admission.training_scope(ledger_identity, specification), pytest.raises(ConvergenceWarning, match='timeout'):
        models.fit_predict(frame, frame.iloc[:600], None, frame.iloc[610:790],
                           recipe, tmp_path, iterations=1)


def test_cached_control_status_requires_native_evidence(tmp_path):
    from cottonlens_ml.research.ledger import freeze_record
    exp = SimpleNamespace(root=tmp_path, ledger=Ledger(tmp_path/'ledger', {}))
    freeze_record(tmp_path/'learning-controls.json', {'status': 'passed', 'checks': []})
    with pytest.raises(ValueError, match='incomplete'):
        m.verify_controls(exp)


def test_audit_target_date_mismatch_fails_before_analysis(tmp_path, monkeypatch):
    import hashlib
    _, frame = next(m.control_frames())
    frame = frame.iloc[:100].copy()
    raw = frame[['date', 'cotton_close']].rename(columns={'cotton_close': 'close'}).assign(series='cotton')
    path = tmp_path/'market.parquet'
    raw.to_parquet(path)
    ready = {'identity': {'raw_audit': {'market_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    monkeypatch.setattr(m, 'MARKET_SHA', ready['identity']['raw_audit']['market_sha256'])
    monkeypatch.setattr(m, 'verify_history', lambda r: (ready, frame))
    with pytest.raises(ValueError, match='target'):
        m.load_reference(tmp_path, path)


def test_annotations_cover_all_five_steps_and_keep_unknowns():
    history = pd.DataFrame({'date': pd.bdate_range('2010-01-01', periods=7)})
    ann = history.assign(date=history.date.dt.strftime('%Y-%m-%d'),
                         wasde_event=['no', 'no', 'yes', 'no', 'no', 'no', 'no'],
                         contract_transition=['unknown', 'unknown', 'no', 'no', 'no', 'no', 'no'])
    out = m.annotate_windows(history.head(1).copy(), history, ann, 5)
    assert out.wasde_event.iloc[0] == 'yes'
    assert out.contract_transition.iloc[0] == 'unknown'
