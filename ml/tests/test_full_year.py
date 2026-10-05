import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.full_year import (
    interval_metrics,
    interval_rows,
    price_recipes,
    report,
)
from cottonlens_ml.research.ledger import Ledger, read_record
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.research.protocol import full_year_manifest, mature
from cottonlens_ml.research.statistics import bh_adjust, oos_r2, paired_bootstrap
from cottonlens_ml.research.volatility import ewma, features, qlike
from cottonlens_ml.runtime_guard import require_training


def test_optional_ohlc_proxy_gaps_are_null_not_invalid_json():
    from cottonlens_ml.research.full_year import json_predictions
    rows = pd.DataFrame({'predicted_return':[.01], 'gk_target_5':[np.nan]})
    assert json_predictions(rows) == [{'predicted_return':.01,'gk_target_5':None}]
    rows['predicted_return'] = np.nan
    with pytest.raises(ValueError,match='Nonfinite primary'):
        json_predictions(rows)


def history():
    dates = pd.bdate_range('2010-01-01', '2024-02-01')
    close = 80 * np.exp(np.cumsum(np.sin(np.arange(len(dates)) / 8) * .003))
    rows = pd.DataFrame({'date': dates, 'cotton_close': close, 'cotton_session_index': np.arange(len(dates))})
    rows['cotton_ret_1'] = np.log(rows.cotton_close / rows.cotton_close.shift())
    for h in (1, 5):
        rows[f'target_date_{h}'] = rows.date.shift(-h)
        rows[f'target_return_{h}'] = np.log(rows.cotton_close.shift(-h) / rows.cotton_close)
    return rows


def test_full_year_cohort_keeps_missing_features_and_purges():
    rows = history()
    rows['feature'] = np.nan
    frozen = full_year_manifest(rows)
    assert len(frozen['folds']) == 8 and frozen['refit_cadence'] == 21
    assert len(frozen['folds'][0]['origins']) > 240
    assert frozen['folds'][0]['origins'][0].startswith('2016-01')
    for fold in frozen['folds']:
        assert len(fold['inner']) == 3
        for block in fold['inner']:
            assert len(block['origins']) == 63
            train = mature(rows, block['cutoff'])
            assert train.target_date_5.max() < pd.Timestamp(block['cutoff'])
        assert max(fold['inner'][-1]['origins']) < min(fold['origins'])
    changed = rows.copy()
    changed.loc[changed.date >= '2024-01-01', 'cotton_close'] = 900
    assert full_year_manifest(changed) == frozen


def test_past_variance_features_are_future_invariant():
    rows = history()
    changed = rows.copy()
    changed.loc[changed.date > '2018-01-01', 'cotton_close'] *= 10
    a, b = features(rows), features(changed)
    past = a.date < '2018-01-01'
    np.testing.assert_allclose(a.loc[past, ['har_1', 'har_5', 'har_22']], b.loc[past, ['har_1', 'har_5', 'har_22']], equal_nan=True)
    np.testing.assert_allclose(ewma(a, a.loc[past, 'date'], 5), ewma(b, b.loc[past, 'date'], 5))
    assert np.isfinite(qlike([0, .01], [.01, .01])).all()
    with pytest.raises(ValueError):
        qlike([0], [0])


def test_cpu_permission_is_scoped(monkeypatch):
    monkeypatch.setenv('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR', '1')
    monkeypatch.delenv('CI', raising=False)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    require_training('ridge', 'cpu')
    for family, threads in [('lstm', 2), ('catboost', 2), ('ridge', 3), ('ridge', True)]:
        with pytest.raises(RuntimeError):
            require_training(family, 'cpu', threads)
    monkeypatch.setenv('CI', 'true')
    with pytest.raises(RuntimeError):
        require_training('ridge', 'cpu')


def test_paired_blocks_and_actual_r2():
    folds = [np.column_stack([np.arange(50.) + 10, np.arange(50.) + 9]), np.array([[1000., 999.]]*30)]
    result = paired_bootstrap(folds, repetitions=100, block=20)
    np.testing.assert_allclose(result['difference_ci_95'], [1, 1])
    assert oos_r2([1, 2], [4, 4], [0, 0]) < 0
    assert bh_adjust([.01, .04, .02]) == pytest.approx([.03, .04, .03])


def test_identical_or_constant_paired_losses_do_not_claim_perfect_power():
    losses = np.linspace(1., 4., 60)
    for offset in (0., 1.):
        result = paired_bootstrap([np.column_stack([losses, losses - offset])], repetitions=100)
        assert not result['power_identifiable']
        assert result['power_at_relative_effect'] is None
        assert result['normal_approx_mde_80pct_relative'] is None
        assert result['standard_error'] == 0
        np.testing.assert_allclose(result['difference_ci_95'], [offset, offset])


def test_intervals_use_only_mature_prequential_scores():
    rows = features(history()).iloc[500:800].copy()
    rows['variance'] = .001
    test = rows.tail(1).copy()
    test['ewma_variance'] = .001
    before = interval_rows(rows, test, [.001], rows, 5)
    future = rows.copy()
    future.loc[future.target_date_5 >= test.date.iloc[0], 'target_return_5'] = 1000
    pd.testing.assert_frame_equal(before, interval_rows(rows, test, [.001], future, 5))
    assert before.status.iloc[0] == 'available'
    assert interval_rows(rows, test, [.001], rows.tail(200), 5).status.iloc[0] == 'insufficient_mature_calibration'
    assert interval_metrics([0], [-1], [1], .9)[0]['coverage'] == 1


def test_resume_and_mirror_never_retrain_completed_fit(tmp_path):
    ledger = Ledger(tmp_path/'local/ledger', {'profile': 'full-year-v1'})
    calls = []
    def operation(work):
        calls.append(1)
        (work/'model.json').write_text('synthetic')
        return {'predictions': [0.]}
    record = ledger.run({'synthetic_test': True}, operation)
    ledger.run({'synthetic_test': True}, operation)
    assert len(calls) == 1
    mirror = Mirror(tmp_path/'local', tmp_path/'drive')
    mirror.fit(record)
    mirror.fit(record)
    receipts = list((tmp_path/'local/transfer-receipts').glob('*.json'))
    assert len(receipts) == 1
    assert read_record(receipts[0])['cloud_synchronization_confirmed'] is False
    restored = Mirror(tmp_path/'restored', tmp_path/'drive')
    assert restored.hydrate()['restored_packages'] == 1
    assert restored.hydrate()['restored_packages'] == 0
    assert read_record(tmp_path/'restored/ledger/completed'/f'{record["experiment_id"]}.json') == record
    (ledger.root / next(iter(record['files']))).write_text('corrupt')
    with pytest.raises(ValueError, match='Corrupt'):
        ledger.run({'synthetic_test': True}, operation)
    assert len(calls) == 1


def test_small_fixed_budget_and_pending_is_not_failure(tmp_path):
    assert len(price_recipes(1)) == 6
    assert price_recipes(1)[-1]['max_iterations'] == 600
    assert report(tmp_path)['decision'] == 'pending_not_negative'


def test_causal_interval_replay_ignores_future_prices_and_annual_model_selection():
    from cottonlens_ml.research.full_year import causal_intervals
    raw = history()
    rows = features(raw)
    test = rows.iloc[800:804].copy()
    expected = causal_intervals(rows, test, 5)
    changed = raw.copy()
    changed.loc[changed.date > test.date.max(), 'cotton_close'] *= 3
    # Recompute all targets; still-unmatured outcomes must never enter calibration.
    for h in (1, 5):
        changed[f'target_return_{h}'] = np.log(changed.cotton_close.shift(-h) / changed.cotton_close)
    actual = causal_intervals(features(changed), test.assign(variance=999.), 5)
    pd.testing.assert_frame_equal(expected, actual)


def test_interval_correction_preserves_prices_and_old_evidence_without_fit(tmp_path):
    from cottonlens_ml.code_identity import digest
    from cottonlens_ml.research.full_year import correct_intervals, json_predictions
    from cottonlens_ml.research.ledger import freeze_record
    rows = features(history())
    rows.to_parquet(tmp_path/'history.parquet', index=False)
    folds = []
    original_sha = {}
    for year in range(2016, 2024):
        origin = rows.loc[rows.date.dt.year == year].iloc[[0]].copy()
        date = origin.date.dt.strftime('%Y-%m-%d').iloc[0]
        folds.append({'year':year, 'origins':[date]})
        for h in (1, 5):
            frame = origin[['date','cotton_close',f'target_return_{h}']].copy()
            frame['date'] = date
            frame['predicted_return'] = .001
            frame['variance'] = .0003
            path = tmp_path/'pilot-full-year'/f't{h}-year{year}.json'
            freeze_record(path, {'year':year,'horizon':h,'records':json_predictions(frame)})
            original_sha[path.name] = digest(path)
    freeze_record(tmp_path/'ready.json', {'history_sha256':digest(tmp_path/'history.parquet'),
        'identity':{'split':{'folds':folds}}})
    derived = correct_intervals(tmp_path)
    assert derived != tmp_path
    assert correct_intervals(tmp_path) == derived
    assert read_record(derived/'ready.json')['derived_analysis']['model_training'] is False
    for p in (tmp_path/'pilot-full-year').glob('*.json'):
        assert digest(p) == original_sha[p.name]
        before = read_record(p)['records'][0]
        after = read_record(derived/'pilot-full-year'/p.name)['records'][0]
        for field in ('date','cotton_close','predicted_return','variance'):
            assert before[field] == after[field]
        assert after['status'] == 'available'
    rows.loc[0,'cotton_close'] *= 2
    rows.to_parquet(tmp_path/'history.parquet', index=False)
    with pytest.raises(ValueError,match='verified frozen history'):
        correct_intervals(tmp_path)


def test_mirror_receipt_cache_detects_changed_receipt(tmp_path, monkeypatch):
    import cottonlens_ml.research.mirror as module
    from cottonlens_ml.research.ledger import freeze_record
    local = tmp_path/'local'
    freeze_record(local/'evidence.json', {'synthetic':True})
    mirror = Mirror(local, tmp_path/'drive')
    mirror.enqueue(['evidence.json'])
    assert mirror.flush()['new_verified_copy_batches'] == 1
    original = module.read_record
    calls = []
    def observed(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(module, 'read_record', observed)
    mirror.flush()  # Cache established after the first transfer.
    assert not calls
    receipt = next((local/'transfer-receipts').glob('*.json'))
    receipt.write_text('corrupt')
    with pytest.raises((ValueError, __import__('json').JSONDecodeError)):
        mirror.flush()


def test_status_metadata_restore_excludes_payloads_but_restores_timing(tmp_path):
    ledger = Ledger(tmp_path/'local/ledger', {'profile': 'full-year-v1'})
    def operation(work):
        (work/'model.json').write_text('synthetic')
        return {'predictions': [0.]}
    record = ledger.run({'synthetic': True}, operation)
    Mirror(tmp_path/'local', tmp_path/'drive').fit(record)
    mirror = Mirror(tmp_path/'restored', tmp_path/'drive')
    assert mirror.hydrate(metadata_only=True)['metadata_only']
    assert not (tmp_path/'restored/ledger/payloads').exists()
    summary = Ledger(tmp_path/'restored/ledger', {}).summary()
    assert summary['durably_saved'] == summary['timing_receipts'] == 1
    assert summary['timing']['compute_seconds'] == record['compute_seconds']
    assert summary['payloads_verified_by_status'] is False
    assert mirror.hydrate()['restored_packages'] == 1
    assert (tmp_path/'restored/ledger/payloads').exists()

