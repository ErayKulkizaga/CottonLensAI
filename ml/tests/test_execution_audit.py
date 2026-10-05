import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research import execution_audit as audit
from cottonlens_ml.research.ledger import freeze_record, read_record


def fixture():
    market = pd.DataFrame({'date': pd.to_datetime(['2020-06-12', '2020-06-15', '2020-06-16']),
        'series': 'cotton', 'open': [60., 61., 62.], 'close': [60., 62., 61.],
        'high': [61., 61.5, 63.], 'low': [59., 60., 61.], 'volume': [10, 0, 10]})
    rows = pd.DataFrame({'date': ['2020-06-12', '2020-06-15'], 'target_date': ['2020-06-15', '2020-06-16'],
        'cotton_close': [60., 62.], 'actual_return': np.log([62 / 60, 61 / 62]),
        'decision_time': ['2020-06-13T00:15:00Z', '2020-06-16T00:15:00Z'],
        'control_predicted_return': [0.01, -0.01], 'example_position': [1., -1.],
        'example_gross_usd_equivalent': [500., 500.]})
    return rows, market


def test_flags_do_not_repair_prices_or_change_strategy_sample():
    rows, market = fixture()
    original_rows, original_market = rows.copy(deep=True), market.copy(deep=True)
    result, flags = audit.audit(rows, market)
    pd.testing.assert_frame_equal(rows, original_rows)
    pd.testing.assert_frame_equal(market, original_market)
    assert len(flags) == 2 and flags.close_outside_range.tolist() == [True, False]
    assert result['flag_counts']['zero_volume'] == 1
    assert result['strategies']['example']['original_gross_usd_equivalent'] == 1000.
    assert result['strategies']['example']['range_projection_diagnostic_gross_usd_equivalent'] == 750.
    assert result['strategies']['example']['buckets']['any_flag']['unflagged_gross_usd_equivalent'] == 500.
    assert not result['prices_repaired'] and not result['execution_verified'] and result['origins_dropped'] == 0
    indexed = rows.copy()
    indexed.index = [100, 200]
    again, again_flags = audit.audit(indexed, market)
    assert again == result
    pd.testing.assert_frame_equal(again_flags, flags)


@pytest.mark.parametrize('field,value', [('high', 59.), ('volume', -1), ('low', np.nan), ('open', 0.)])
def test_invalid_prices_or_volume_cannot_produce_completed_audit(field, value):
    rows, market = fixture()
    market.loc[1, field] = value
    with pytest.raises(ValueError):
        audit.audit(rows, market)


def test_broken_target_or_saved_pnl_rejected():
    rows, market = fixture()
    rows.loc[0, 'example_gross_usd_equivalent'] += 1
    with pytest.raises(ValueError, match='PnL'):
        audit.audit(rows, market)
    rows, market = fixture()
    rows.loc[0, 'target_date'] = '2020-06-16'
    with pytest.raises(ValueError, match='target'):
        audit.audit(rows, market)


def test_signed_inputs_completion_and_no_overwrite(tmp_path):
    rows, market = fixture()
    path = tmp_path / 'market.parquet'
    market.to_parquet(path, index=False)
    folder = tmp_path / 'diagnostic'
    folder.mkdir()
    rows.to_csv(folder / 'decisions.csv', index=False)
    freeze_record(folder / 'report.json', {'horizon': 1, 'inputs': {'market.parquet': digest(path)}})
    freeze_record(folder / 'complete.json', {'completed': True, 'fits': 0,
        'version': 'clock-t1-open-close-proxy-v1', 'files': {name: digest(folder / name) for name in ['report.json', 'decisions.csv']}})
    output = tmp_path / 'audit'
    before = {p.name: digest(p) for p in folder.iterdir()}
    audit.analyze(path, folder, output)
    assert read_record(output / 'complete.json')['completed']
    assert {p.name: digest(p) for p in folder.iterdir()} == before
    with pytest.raises(ValueError, match='overwrite'):
        audit.analyze(path, folder, output)
    (folder / 'decisions.csv').write_text('corrupted')
    with pytest.raises(ValueError, match='checksum'):
        audit.analyze(path, folder, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()
