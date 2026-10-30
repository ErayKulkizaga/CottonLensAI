from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research.ct_shock_control import NAMES, attach, fit, project, recipe


def fixture():
    rng = np.random.default_rng(42); x = rng.uniform(-2, 2, (600, 4))
    baseline = np.exp(rng.uniform(-12, -4, len(x)))
    f = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=len(x)),
                      'ct_ewma': baseline, 'origin_shock': x[:, 3]})
    for i, name in enumerate(NAMES[:3]):
        f[name] = np.log(baseline)+x[:, i]
    f['variance_target_1'] = baseline*np.exp(.1+x@np.array([.2, .1, -.1, -.15]))
    return f


def test_known_fourth_effect_recovers_scale_and_ignores_future_targets(tmp_path):
    f = fixture(); first, second = tmp_path/'a', tmp_path/'b'; first.mkdir(); second.mkdir()
    with patch('research.ct_shock_control.require_training'), patch('cottonlens_ml.research.volatility.require_training'):
        before = fit(f.iloc[:500], f.iloc[500:], first, 'signed')
        np.testing.assert_allclose(before['variance_predictions'], f.variance_target_1.iloc[500:], rtol=1e-5)
        f.loc[500:, 'variance_target_1'] = np.nan
        after = fit(f.iloc[:500], f.iloc[500:], second, 'signed')
    assert before == after
    assert (first/'variance-model.json').read_bytes() == (second/'variance-model.json').read_bytes()


def test_origin_shock_does_not_use_future_prices_or_temporally_compress_dates():
    dates = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-08', '2020-01-09'])
    h = pd.DataFrame({'date': dates, 'cotton_close': [100., 99., 101., 100., 110.]})
    h['cotton_ret_1'] = np.log(h.cotton_close/h.cotton_close.shift())
    h['ct_ewma'] = np.maximum(h.cotton_ret_1.pow(2).ewm(alpha=.06, adjust=False).mean(), 1e-12)
    h['variance_target_1'] = h.cotton_ret_1.shift(-1).pow(2)
    for name in NAMES[:3]:
        h[name] = np.log(h.ct_ewma)
    original = attach(h)
    changed = h.copy(); changed.loc[4, 'cotton_close'] = 150.
    changed['cotton_ret_1'] = np.log(changed.cotton_close/changed.cotton_close.shift())
    changed['ct_ewma'] = changed.cotton_ret_1.pow(2).ewm(alpha=.06, adjust=False).mean()
    altered = attach(changed)
    np.testing.assert_array_equal(original.signed_shock.iloc[:4], altered.signed_shock.iloc[:4])
    np.testing.assert_array_equal(original.unsigned_shock.iloc[:4], altered.unsigned_shock.iloc[:4])
    days = ['2020-01-03', '2020-01-08']
    result = project(original, days, 'signed')
    assert result.date.dt.strftime('%Y-%m-%d').tolist() == days
    assert np.sign(result.origin_shock).tolist() == [-1., -1.]


def test_unsigned_signed_have_equal_declared_capacity_and_magnitude():
    assert recipe('unsigned')['parameters'] == recipe('signed')['parameters'] == 5
    assert recipe('unsigned')['variance_features'] == recipe('signed')['variance_features'] == NAMES
    with pytest.raises(ValueError, match='two frozen'):
        recipe('best_oos_arm')


def test_missing_source_value_is_not_dropped_or_imputed():
    f = fixture()
    f['signed_shock'] = f.origin_shock
    f.loc[2, 'signed_shock'] = np.nan
    with pytest.raises(ValueError, match='Do not drop'):
        project(f, f.date.iloc[:5].dt.strftime('%Y-%m-%d').tolist(), 'signed')


def test_expired_session_stops_before_first_market_fit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from research import ct_shock_control as study
    budget = {'market': 338, 'synthetic': 1, 'processes': 1, 'threads': 2, 'session_minutes': 30}
    identity = {'source_id': 'source', 'design_id': 'design'}
    job = {'year': 2016, 'role': 'outer', 'bucket': 0,
           'train_dates': ['2000-01-03'], 'origins': ['2000-01-04']}
    old = {'identity': {'design': {'jobs': [job]}}}
    authorization = {'ready_sha256': 'digest', 'source_id': 'source', 'design_id': 'design',
                     'fit_budget': budget, 'fresh_restore_verified': True}
    moments = iter([0., 2000.])
    monkeypatch.setattr(study, 'time', SimpleNamespace(monotonic=lambda: next(moments)))
    monkeypatch.setattr(study, 'verify', lambda *a: ({'identity': identity}, {'fit_budget': budget}, old, fixture()))
    monkeypatch.setattr(study, 'read_record', lambda *a: authorization)
    monkeypatch.setattr(study, 'digest', lambda *a: 'digest')
    monkeypatch.setattr(study, 'project', lambda h, d, a, profile: fixture().iloc[:1])
    with patch.object(study, 'require_training'), patch('cottonlens_ml.research.volatility.require_training'):
        result = study.run(tmp_path/'run', tmp_path, tmp_path/'authorization.json')
    assert result == {'status': 'planned_pause', 'market_fits': 0, 'synthetic_fits': 1}
    assert not list((tmp_path/'run/ledger/completed').glob('*.json'))


@pytest.mark.parametrize('tamper', ['source', 'input'])
def test_frozen_cache_rejects_changed_source_or_data(tmp_path, monkeypatch, tamper):
    from cottonlens_ml.cohort import content_id
    from cottonlens_ml.research.ledger import freeze_record

    from research import ct_shock_control as study
    (tmp_path/'ml').mkdir(); (tmp_path/'ml/uv.lock').write_text('frozen')
    root = tmp_path/'run'; root.mkdir(); (root/'input.txt').write_text('original')
    design = {'fixed': True}; manifest = {'source_id': 'source', 'files': {}}
    identity = {'profile': study.PROFILE, 'source_id': 'source', 'design_id': content_id(design),
                'python': study.sys.version.split()[0],
                'versions': {p: study.importlib.metadata.version(p) for p in study.VERSIONS},
                'lock_sha256': study.digest(tmp_path/'ml/uv.lock')}
    freeze_record(root/'ready.json', {'identity': identity, 'frozen_hashes': {'input.txt': study.digest(root/'input.txt')}})
    freeze_record(root/'decision-contract.json', design); freeze_record(root/'source-manifest.json', manifest)
    monkeypatch.setattr(study, 'source', lambda repo: {**manifest, 'source_id': 'changed'} if tamper == 'source' else manifest)
    if tamper == 'input':
        (root/'input.txt').write_text('changed')
    with pytest.raises(ValueError, match='Frozen'):
        study.verify(root, tmp_path)
