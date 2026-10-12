"""Synthetic contracts for report-only risk evidence; no market training."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.full_year import volatility_recipe
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.protocol import mature
from cottonlens_ml.research.risk_audit import (
    align_output,
    audit,
    decision,
    loss_summary,
    predictions,
    reconstruct,
    verify_fit,
)
from cottonlens_ml.research.volatility import features


def history():
    dates = pd.bdate_range('2010-01-01', periods=700)
    prices = 80 * np.exp(np.cumsum(.004 * np.sin(np.arange(700) / 9)))
    result = pd.DataFrame({'date': dates, 'cotton_close': prices,
                           'cotton_session_index': np.arange(700)})
    result['cotton_ret_1'] = np.log(result.cotton_close / result.cotton_close.shift())
    for h in (1, 5):
        result[f'target_date_{h}'] = result.date.shift(-h)
        result[f'target_return_{h}'] = np.log(result.cotton_close.shift(-h) / result.cotton_close)
    return features(result)


def test_reconstruct_targets_and_causal_ewma():
    rows = history()
    verified = reconstruct(rows)
    expected = rows.daily_variance_proxy.ewm(alpha=.06, adjust=False).mean()
    np.testing.assert_allclose(verified.audit_ewma_1, expected.clip(lower=1e-12), equal_nan=True)
    future = rows.copy()
    future.loc[600:, 'cotton_close'] *= 3
    future = features(future)
    for h in (1, 5):
        future[f'target_return_{h}'] = np.log(future.cotton_close.shift(-h) / future.cotton_close)
    changed = reconstruct(future)
    np.testing.assert_array_equal(verified.audit_ewma_1.iloc[:600], changed.audit_ewma_1.iloc[:600])


@pytest.mark.parametrize('field', ['variance_target_1', 'target_return_5', 'har_22'])
def test_reconstructed_values_reject_corruption(field):
    rows = history()
    rows.loc[500, field] += .01
    with pytest.raises(ValueError, match='mismatch'):
        reconstruct(rows)


def test_target_dates_cannot_be_changed():
    rows = history()
    rows.loc[500, 'target_date_1'] = rows.loc[500, 'date']
    with pytest.raises(ValueError, match='Target date'):
        reconstruct(rows)


@pytest.mark.parametrize('change', ['origin', 'target', 'horizon', 'prediction'])
def test_comparisons_reject_unpaired_origins_and_values(change):
    rows = reconstruct(history())
    selected = rows.iloc[500:505]
    records = [{'date': r.date.strftime('%Y-%m-%d'), 'cotton_close': r.cotton_close,
                'target_return_1': r.target_return_1, 'variance_target_1': r.variance_target_1,
                'predicted_return': 0., 'variance': .001, 'ewma_variance': r.audit_ewma_1}
               for r in selected.itertuples()]
    original = {'year': 2011, 'horizon': 1, 'records': records}
    derived = copy.deepcopy(original)
    fold = {'year': 2011, 'origins': [r['date'] for r in records]}
    align_output(rows, original, derived, fold, 1)
    if change == 'origin':
        derived['records'][0]['date'] = '2011-01-01'
    elif change == 'target':
        derived['records'][0]['variance_target_1'] += .1
    elif change == 'horizon':
        derived['horizon'] = 5
    else:
        derived['records'][0]['variance'] *= 2
    with pytest.raises(ValueError):
        align_output(rows, original, derived, fold, 1)


def sample_receipt(tmp_path):
    rows = history()
    test = rows.iloc[600:605]
    train = mature(rows, test.date.min())
    train = train.loc[train.date >= rows.date.iloc[119]]
    recipe = volatility_recipe('har', 1)
    spec = {'recipe': recipe, 'role': 'vol-outer-2012-0', 'iterations': 1,
            'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(), 'validation_dates': [],
            'test_dates': test.date.dt.strftime('%Y-%m-%d').tolist(),
            'train_identity': frame_identity(train, list(train.columns)), 'repeat_reason': None}
    identity = {'split': {}}
    model = {'kind': 'log_har', 'features': recipe['features'],
             'coef': [np.log(.001), 0., 0., 0.], 'smearing': 1.}
    path = tmp_path / 'ledger/payloads/test/variance-model.json'
    path.parent.mkdir(parents=True)
    path.write_text(__import__('json').dumps(model))
    from cottonlens_ml.code_identity import digest
    receipt = {'specification': spec, 'identity': identity,
               'experiment_id': content_id({'identity': identity, 'specification': spec}),
               'files': {'payloads/test/variance-model.json': digest(path)},
               'result': {'origins': spec['test_dates'], 'variance_predictions': [.001] * len(test)}}
    return rows, identity, receipt, path


def test_native_inference_and_maturity_without_fit(tmp_path, monkeypatch):
    from cottonlens_ml.research import volatility
    monkeypatch.setattr(volatility, 'fit_predict', lambda *a, **k: pytest.fail('Training called'))
    rows, identity, receipt, _ = sample_receipt(tmp_path)
    verify_fit(tmp_path, receipt, rows, identity)
    receipt['specification']['train_dates'].append(receipt['specification']['test_dates'][0])
    receipt['experiment_id'] = content_id({'identity': identity, 'specification': receipt['specification']})
    with pytest.raises(ValueError, match='maturity'):
        verify_fit(tmp_path, receipt, rows, identity)


def test_truncated_history_may_retain_unscored_future_endpoint_metadata():
    rows = history()
    for h in (1, 5):
        rows.loc[rows.index[-h:], f'target_date_{h}'] = pd.bdate_range(rows.date.max(), periods=h + 1)[1:]
        rows.loc[rows.index[-h:], f'target_return_{h}'] = .1
    reconstruct(rows)
    rows.loc[rows.index[-1], 'target_date_1'] = rows.date.iloc[-2]
    with pytest.raises(ValueError, match='Out-of-snapshot'):
        reconstruct(rows)


def test_native_payload_hash_and_missing_payload_rejected(tmp_path):
    rows, identity, receipt, path = sample_receipt(tmp_path)
    path.write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        verify_fit(tmp_path, receipt, rows, identity)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_fit(tmp_path, receipt, rows, identity)


@pytest.mark.parametrize('change', ['training_frame', 'origins', 'prediction'])
def test_coherently_rehashed_fit_still_rejects_wrong_semantics(tmp_path, change):
    rows, identity, receipt, _ = sample_receipt(tmp_path)
    if change == 'training_frame':
        receipt['specification']['train_identity'] = '0' * 64
    elif change == 'origins':
        receipt['result']['origins'] = list(reversed(receipt['result']['origins']))
    else:
        receipt['result']['variance_predictions'][0] = .01
    receipt['experiment_id'] = content_id({'identity': identity, 'specification': receipt['specification']})
    with pytest.raises(ValueError, match='mismatch'):
        verify_fit(tmp_path, receipt, rows, identity)


def test_missing_or_ambiguous_receipt_never_counts_completed(tmp_path):
    rows, _, receipt, _ = sample_receipt(tmp_path)
    spec = receipt['specification']
    for receipts in ([], [receipt, receipt]):
        with pytest.raises(ValueError, match='Missing/ambiguous'):
            predictions(receipts, rows, spec['recipe'], spec['test_dates'], 'vol-outer-2012')


def losses():
    dates = pd.bdate_range('2010-01-01', periods=240)
    return pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'year': np.repeat([2010, 2011], 120),
                         'actual_variance': np.linspace(0, .003, 240), 'variance': .0011,
                         'ewma_variance': .001})


def test_zero_actual_valid_negative_variance_rejected_and_no_percent():
    frame = losses()
    summary, scored = loss_summary(frame, repetitions=100)
    assert np.isfinite(scored.difference).all()
    assert 'gain_pct' not in summary
    assert summary['top1pct_sensitivity']['removed_count'] == 3
    for field in ('actual_variance', 'variance', 'ewma_variance'):
        bad = frame.copy()
        bad.loc[0, field] = -.01
        with pytest.raises(ValueError):
            loss_summary(bad, repetitions=100)


def test_decision_requires_both_blocks_and_six_years():
    positive = {'mean_difference': .01, 'year_wins': 6,
                'bootstrap': {str(b): {'difference_ci_95': [.001, .02]} for b in (20, 60)}}
    assert decision(positive) == 'HISTORICAL_FOLLOWUP_CANDIDATE'
    positive['bootstrap']['60']['difference_ci_95'][0] = -.01
    assert decision(positive) == 'INCONCLUSIVE'
    positive['bootstrap'] = {str(b): {'difference_ci_95': [-.02, 0]} for b in (20, 60)}
    assert decision(positive) == 'FIXED_RECIPE_SUPERIORITY_NOT_SUPPORTED'


def test_new_namespace_and_hash_failure_publish_no_score(tmp_path):
    source, corrected = tmp_path / 'input', tmp_path / 'corrected'
    source.mkdir()
    corrected.mkdir()
    freeze_record(source / 'ready.json', {'history_sha256': '0' * 64})
    freeze_record(corrected / 'ready.json', {})
    (source / 'history.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        audit(source, corrected, tmp_path / 'new-output')
    assert not (tmp_path / 'new-output').exists()
    with pytest.raises(ValueError, match='namespace'):
        audit(source, corrected, source / 'overwriting-input')


def test_audit_cli_has_no_training_stage():
    import cottonlens_ml.research.risk_audit as risk
    source = Path(risk.__file__).read_text()
    assert 'fit_predict(' not in source and '.fit(' not in source
