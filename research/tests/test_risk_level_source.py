from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research import risk_level_source as pilot


def market():
    return pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=30),
                         'series': 'dxy', 'close': 80+np.arange(30.)**1.1})


def test_asof_boundary_is_inclusive_and_later_bar_is_excluded():
    raw = market(); dates = raw.date
    clocks = pd.to_datetime(dates, utc=True)+pd.Timedelta(days=1)
    equal = pilot.dxy_state(raw, dates, clocks)
    before = pilot.dxy_state(raw, dates, clocks-pd.Timedelta(microseconds=1))
    assert equal.dxy_state_source_date.iloc[10] == dates.iloc[10]
    assert before.dxy_state_source_date.iloc[10] == dates.iloc[9]
    squared = np.log(raw.close/raw.close.shift()).pow(2)
    assert equal.dxy_rv5.iloc[10] == pytest.approx(squared.iloc[6:11].mean())
    assert equal.dxy_state_window_start.iloc[10] == dates.iloc[5]


def test_future_source_mutation_does_not_change_past_features():
    raw = market(); base = pilot.dxy_state(raw, raw.date)
    changed = raw.copy(); changed.loc[15:, 'close'] *= 10
    pd.testing.assert_frame_equal(base.iloc[:15], pilot.dxy_state(changed, raw.date).iloc[:15])


def test_source_calendar_not_cotton_compressed_and_invalid_window_preserved():
    raw = market(); dates = raw.date.iloc[::2]
    state = pilot.dxy_state(raw, dates)
    expected = np.log(raw.close/raw.close.shift()).pow(2).iloc[16:21].mean()
    assert state.dxy_rv5.iloc[10] == pytest.approx(expected)
    changed = raw.copy(); changed.loc[18, 'close'] = 0
    assert np.isnan(pilot.dxy_state(changed, dates).dxy_rv5.iloc[10])
    sparse = raw.drop(index=range(10, 25))
    assert np.isnan(pilot.dxy_state(sparse, raw.date).dxy_rv5.iloc[20])


def test_duplicate_source_or_reversed_origin_calendar_is_rejected():
    raw = market()
    with pytest.raises(ValueError, match='Unique'):
        pilot.dxy_state(pd.concat([raw, raw.iloc[:1]]), raw.date)
    with pytest.raises(ValueError, match='chronological'):
        pilot.dxy_state(raw, raw.date.iloc[::-1])


def fixture():
    x = np.random.default_rng(42).uniform(-2, 2, (1200, 4))
    b = np.exp(-8+x[:, 3])
    rows = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1200),
                          'named_ewma': b, 'dxy_rv5': np.exp(x[:, 3]-12)})
    for j, k in enumerate((1, 5, 22)):
        rows[f'har_{k}'] = np.log(b)+x[:, j]
    rows['variance_target_1'] = b*np.exp(.1+pilot.features(rows, 'own_level')@np.array([.2, .1, -.1, .08]))
    return rows


def test_capacity_matched_states_differ_without_relative_dxy_baseline_subtraction():
    rows = fixture()
    assert pilot.recipe('own_level')['parameters'] == pilot.recipe('dxy_state')['parameters'] == 5
    a, b = (pilot.features(rows, arm) for arm in pilot.ARMS)
    np.testing.assert_array_equal(a[:, :3], b[:, :3])
    np.testing.assert_allclose(a[:, 3], np.log(rows.named_ewma))
    np.testing.assert_allclose(b[:, 3], np.log(rows.dxy_rv5))


def test_kernel_recovers_fifth_coefficient_and_never_uses_test_labels(tmp_path):
    rows = fixture(); train, test = rows.iloc[:1000], rows.iloc[1000:].copy()
    first = tmp_path/'first'; second = tmp_path/'second'; first.mkdir(); second.mkdir()
    with patch.object(pilot, 'require_training'):
        result = pilot.fit(train, test, first, 'own_level')
        test['variance_target_1'] = np.nan
        changed = pilot.fit(train, test, second, 'own_level')
    assert result == changed
    np.testing.assert_allclose(result['variance_predictions'], rows.variance_target_1.iloc[1000:], rtol=1e-5)
    # Models are plain JSON protected by the enclosing ledger hash, not freeze_record JSON.
    job = {'role': 'outer', 'bucket': 0, 'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(),
           'origins': test.date.dt.strftime('%Y-%m-%d').tolist()}
    ledger = pilot.Ledger(tmp_path, {'fixture': True})
    specification = pilot.spec(job, train, 'own_level'); key = ledger.key(specification)
    pilot.freeze_record(tmp_path/'completed'/f'{key}.json', {
        'identity': ledger.identity, 'specification': specification, 'experiment_id': key,
        'files': {'first/variance-model.json': pilot.digest(first/'variance-model.json')},
        'payload_root': 'first', 'result': result})
    native = pilot.native(ledger, job, train, test, 'own_level')
    assert native['result'] == result


def test_invalid_training_labels_fail_without_fallback(tmp_path):
    rows = fixture(); rows.loc[1, 'variance_target_1'] = -1
    with patch.object(pilot, 'require_training'), pytest.raises(ValueError, match='Nonnegative'):
        pilot.fit(rows.iloc[:1000], rows.iloc[1000:], tmp_path, 'own_level')


def test_no_silent_dropped_origins_when_source_unready():
    rows = fixture(); states = pd.DataFrame({'date': rows.date, 'dxy_rv5': np.nan})
    with patch.object(pilot.parent, 'projection', return_value=rows.iloc[:2].copy()), pytest.raises(ValueError, match='drop/impute'):
        pilot.project(rows, states, rows.date.iloc[:2], False)


def test_expired_contract_month_is_rejected_without_guessing_true_identity():
    h = pd.DataFrame({'date': pd.to_datetime(['2020-02-26']), 'named_contract': ['2019-07-01']})
    with pytest.raises(ValueError, match='past contract month'):
        pilot.require_contract_lifecycle(h)
    h['named_contract'] = '2020-03-01'
    pilot.require_contract_lifecycle(h)
    # Same-month validity needs the actual expiry calendar; this necessary gate makes no claim about it.
    h['named_contract'] = '2020-02-01'
    pilot.require_contract_lifecycle(h)
