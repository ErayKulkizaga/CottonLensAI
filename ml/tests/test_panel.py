import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.panel import FEATURES, asset_rows, mature_auxiliary


def fixture_bars():
    dates = pd.bdate_range('2020-01-01', periods=60).delete([10, 20])
    return pd.DataFrame({'date':dates, 'close':100*np.exp(np.arange(len(dates))*.01)})


def test_features_use_preceding_recorded_bar_and_never_future_data():
    bars = fixture_bars()
    original = asset_rows(bars, 'cotton')
    changed = bars.copy()
    changed.loc[35:, 'close'] *= 3
    future = asset_rows(changed, 'cotton')
    pd.testing.assert_frame_equal(original.loc[:35, FEATURES], future.loc[:35, FEATURES])
    assert original.own_return_5.iloc[30] == pytest.approx(.05)
    assert original.target_return_5.iloc[30] == pytest.approx(.05)
    assert original.target_date_5.iloc[30] == bars.date.iloc[35]
    assert original.feature_asof.iloc[30] == bars.date.iloc[29]
    assert original.source_observation_index.tolist() == list(range(len(bars)))


def test_missing_feature_rows_stay_and_missing_labels_are_not_filled():
    bars = fixture_bars()
    bars.loc[25,'close'] = np.nan
    frame = asset_rows(bars,'corn')
    assert len(frame)==len(bars)
    assert frame.own_return_1.iloc[26:28].isna().all()
    assert pd.isna(frame.target_return_5.iloc[20])
    assert frame.iloc[-5:].target_return_5.isna().all()
    assert frame.own_volatility_20.iloc[26:46].isna().all()


def test_auxiliary_maturity_respects_recorded_purge_and_label_access():
    bars = fixture_bars()
    frame = asset_rows(bars,'soybean')
    decision = bars.date.iloc[40].tz_localize('UTC')+pd.Timedelta(days=1,minutes=15)
    purge = bars.date.iloc[35]
    eligible = mature_auxiliary(frame,decision,purge)
    assert eligible.source_observation_index.tolist()==list(range(35))
    assert eligible.target_date_5.max()==bars.date.iloc[39]
    access = bars.date.iloc[15].tz_localize('UTC')+pd.Timedelta(days=1,minutes=15)
    at_boundary = mature_auxiliary(frame,access,bars.date.iloc[40])
    assert 10 not in at_boundary.source_observation_index.tolist()
    assert 10 in mature_auxiliary(frame,access+pd.Timedelta(seconds=1),bars.date.iloc[40]).source_observation_index.tolist()


@pytest.mark.parametrize('kind',['duplicate','unsorted','nonpositive','audit','unknown_asset'])
def test_unverified_or_invalid_inputs_fail_closed(kind):
    bars = fixture_bars()
    if kind=='duplicate': bars.loc[1,'date']=bars.date.iloc[0]
    if kind=='unsorted': bars=bars.iloc[::-1]
    if kind=='nonpositive': bars.loc[1,'close']=0
    if kind=='audit': bars.loc[len(bars)-1,'date']=pd.Timestamp('2024-01-01')
    with pytest.raises(ValueError):
        asset_rows(bars,'unknown' if kind=='unknown_asset' else 'cotton')


def test_auxiliary_maturity_rejects_mixed_assets_and_ambiguous_clock():
    bars = fixture_bars()
    frame = asset_rows(bars,'soybean')
    cutoff = bars.date.iloc[40].tz_localize('UTC')+pd.Timedelta(days=1,minutes=15)
    boundary = bars.date.iloc[35]
    with pytest.raises(ValueError, match='UTC'):
        mature_auxiliary(frame,cutoff.tz_localize(None),boundary)
    with pytest.raises(ValueError, match='UTC'):
        mature_auxiliary(frame,cutoff,boundary.tz_localize('UTC'))
    frame.loc[0,'asset']='corn'
    with pytest.raises(ValueError, match='one asset'):
        mature_auxiliary(frame,cutoff,boundary)
