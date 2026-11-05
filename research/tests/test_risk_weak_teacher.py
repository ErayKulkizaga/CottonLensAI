"""Known risk means must not become learner inputs or evaluation calibration."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research import risk_weak_teacher as control


def history():
    x = np.linspace(-1., 1., 50)
    return pd.DataFrame({'date': pd.bdate_range('2010-01-01', periods=50),
                         'own_har_5': x-6., 'named_ewma': np.full(50, np.exp(-6.)),
                         'named_variance': np.exp(-6.)*(x+1.1),
                         'own_feature_ready': True, 'named_ewma_eligible': True,
                         'named_eligible': True})


def calibration(h):
    with patch.object(control.parent, 'projection', side_effect=lambda h, dates, **kw: h.loc[h.date.isin(dates)]):
        return control.calibrate(h, h.date.iloc[:30].tolist())


def test_calibration_and_past_teacher_do_not_use_future_labels():
    h = history(); c = calibration(h)
    changed = h.copy(); changed.loc[30:, 'named_variance'] *= 1e8
    assert calibration(changed) == c
    a = control.simulate(h, c, 'weak', 42)
    b = control.simulate(changed, c, 'weak', 42)
    pd.testing.assert_frame_equal(a, b)
    assert np.mean(c['innovation_pool']) == pytest.approx(1.)


def test_mean_is_representable_and_null_is_baseline():
    h = history(); c = calibration(h)
    weak = control.simulate(h, c, 'weak', 42)
    expected = np.exp(.15*((h.own_har_5-np.log(h.named_ewma))-c['feature_mean'])
                      / c['feature_scale']-c['weak_log_normalizer'])
    np.testing.assert_allclose(weak.teacher_mean/h.named_ewma, expected)
    null = control.simulate(h, c, 'null', 42)
    np.testing.assert_array_equal(null.teacher_mean, null.named_ewma)
    np.testing.assert_array_equal(null.teacher_innovation, weak.teacher_innovation)


def test_oracle_upper_bound_and_null_expected_gain():
    base = np.array([.2, .4, .8]); mean = np.array([.3, .2, 1.])
    oracle = control.expected_gain(mean, base, mean)
    assert (oracle >= 0).all()
    assert (control.expected_gain(mean, base, np.array([.1, .8, .3])) <= oracle).all()
    assert (control.expected_gain(base, base, mean) <= 0).all()
    with pytest.raises(ValueError):
        control.expected_gain(mean, base, np.array([0., 1., 2.]))


def test_latent_teacher_is_removed_from_model_frames():
    frame = pd.DataFrame({'har_1': [1.], 'teacher_mean': [100.], 'teacher_innovation': [999.]})
    with patch.object(control.parent, 'projection', return_value=frame):
        result = control.model_rows(frame, [], True)
    assert list(result) == ['har_1']


def test_unregistered_condition_or_seed_and_duplicate_origins_rejected():
    h = history(); c = calibration(h)
    with pytest.raises(ValueError):
        control.simulate(h, c, 'stronger', 42)
    with pytest.raises(ValueError):
        control.simulate(h, c, 'weak', 99)
    h.loc[1, 'date'] = h.loc[0, 'date']
    with pytest.raises(ValueError, match='Unique'):
        calibration(h)


def test_frozen_inputs_exclude_transient_writer_and_outputs(tmp_path):
    for name in ['inputs/h.json', 'cases/weak.parquet', '.writer-lock/owner.json',
                 'reports/result.json', 'calibration.json', 'decision-contract.json',
                 'history-check.json', 'source-manifest.json']:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fixture')
    files = control.frozen_inputs(tmp_path)
    assert '.writer-lock/owner.json' not in files
    assert 'reports/result.json' not in files
    assert len(files) == 6


def test_native_role_is_classified_as_synthetic_by_existing_indexer():
    frame = pd.DataFrame({'date': pd.bdate_range('2010-01-01', periods=2), 'x': [1., 2.]})
    job = {'role': 'outer', 'bucket': 0, 'train_dates': ['2010-01-01'], 'origins': ['2010-01-04']}
    record = control.spec(job, frame, 'weak', 42)
    assert 'synthetic' in record['role']
    assert record['kind'] == 'synthetic_risk_weak_teacher'
    assert record['recipe']['synthetic_seed'] == 42
