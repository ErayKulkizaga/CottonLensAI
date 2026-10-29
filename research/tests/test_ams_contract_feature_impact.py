import math

import pandas as pd

from research.ams_contract_feature_impact import reconstruct


def fixture():
    dates = pd.bdate_range('2020-01-01', periods=30)
    history = pd.DataFrame({'date': dates, 'own_har_1': -12., 'own_har_5': -12., 'own_har_22': -12.})
    quotes = {d.strftime('%Y-%m-%d'): {'2020-03-01': str(60+i*.1), '2020-05-01': str(65+i*.2)} for i, d in enumerate(dates)}
    return history, quotes


def test_target_contract_history_is_contiguous_and_reaches_no_future():
    h, quotes = fixture(); result = reconstruct(h, quotes)
    row = result.iloc[22]
    assert row.verified_contiguous_window and row.changed
    assert row.true_contract == '2020-05-01'
    assert abs(row.true_har_1-math.log(math.log((65+22*.2)/(65+21*.2))**2)) < 1e-12
    changed = {k: dict(v) for k, v in quotes.items()}
    changed[h.date.iloc[23].strftime('%Y-%m-%d')]['2020-05-01'] = '999'
    pd.testing.assert_frame_equal(result.iloc[:23], reconstruct(h, changed).iloc[:23])


def test_missing_bar_does_not_compress_a_23_cotton_observation_window():
    h, quotes = fixture(); quotes.pop(h.date.iloc[10].strftime('%Y-%m-%d'))
    result = reconstruct(h, quotes)
    assert not result.iloc[22].verified_contiguous_window
    assert not result.iloc[29].verified_contiguous_window
    assert len(result) == len(h)
