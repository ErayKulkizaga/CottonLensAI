import numpy as np
import pandas as pd
import pytest

from research.dxy_risk_source_audit import reconstruct


def market():
    return pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=16),
                         'series': 'dxy', 'close': np.arange(16.)+80})


def test_reconstruction_never_uses_same_origin_or_future_close():
    source = market(); dates = source.date
    baseline = reconstruct(source, dates)
    changed = source.copy(); changed.loc[8:, 'close'] *= 100
    actual = reconstruct(changed, dates)
    pd.testing.assert_frame_equal(actual.iloc[:9], baseline.iloc[:9])
    assert baseline.source_date.iloc[8] == dates.iloc[7]
    assert baseline.ret_1.iloc[8] == pytest.approx(np.log(source.close.iloc[7]/source.close.iloc[6]))


def test_missing_source_preserves_calendar_and_stale_prices_stay_missing():
    source = market(); dates = source.date
    source = source.drop(index=list(range(4, 10)))
    actual = reconstruct(source, dates)
    assert len(actual) == len(dates)
    assert actual.age.iloc[8] == 5
    assert np.isnan(actual.ret_1.iloc[8])
    assert np.isnan(actual.variance_5.iloc[8])


def test_duplicate_quotes_or_reversed_calendar_fail_closed():
    source = market()
    with pytest.raises(ValueError, match='Unique'):
        reconstruct(pd.concat([source, source.iloc[:1]]), source.date)
    with pytest.raises(ValueError, match='chronological'):
        reconstruct(source, source.date.iloc[::-1])
