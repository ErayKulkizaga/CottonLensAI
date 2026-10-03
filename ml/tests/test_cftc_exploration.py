"""Synthetic CFTC contracts and causal alignment; never fit a model."""
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.cftc_exploration import (
    add_positions,
    derive,
    dispatch,
    group_names,
    positions,
    read_positions,
)
from cottonlens_ml.research.ledger import freeze_record


def reports(days=('2020-01-07','2020-01-14','2020-01-28')):
    return derive(pd.DataFrame({'report_date':pd.to_datetime(days),'market_code':'033661',
        'report_type':'FutOnly','open_interest':1000,'managed_long':[200,250,300],
        'managed_short':100,'producer_long':100,'producer_short':300}))


def zipped(kind='FutOnly',code='033661',interest=1000):
    cells={'Market_and_Exchange_Names':['COTTON NO. 2 - ICE FUTURES U.S.'],
        'Report_Date_as_MM_DD_YYYY':['01/07/2020'],'CFTC_Contract_Market_Code':[code],
        'Open_Interest_All':[interest],'M_Money_Positions_Long_All':[200],
        'M_Money_Positions_Short_All':[100],'Prod_Merc_Positions_Long_All':[100],
        'Prod_Merc_Positions_Short_All':[300],'FutOnly_or_Combined':[kind]}
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:z.writestr('f_year.txt',pd.DataFrame(cells).to_csv(index=False))
    return out.getvalue()


def test_schema_market_type_and_counts():
    frame=positions(zipped(),2020)
    assert len(frame)==1 and frame.market_code.iloc[0]=='033661'
    assert derive(frame).managed_net_share.iloc[0]==.1
    for raw in (zipped(kind='Combined'),zipped(code='005602'),zipped(interest=0),zipped(interest=50)):
        with pytest.raises(ValueError):positions(raw,2020)
    with pytest.raises(ValueError):positions(zipped(),2019)


def test_ratio_change_requires_adjacent_unmasked_report():
    rows=reports()
    assert np.isnan(rows.managed_share_change.iloc[0])
    assert rows.managed_share_change.iloc[1]==pytest.approx(.05)
    assert np.isnan(rows.managed_share_change.iloc[2])
    rows=reports(('2019-03-19','2019-03-26','2019-04-02'))
    assert rows.input_mask_reason.iloc[1]=='cotton_position_revision'
    assert rows.managed_share_change.isna().all()


def test_no_future_alignment_lag_stress_or_time_compression():
    rows=reports();history=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-03-10')})
    result=add_positions(history,rows)
    assert result.loc[result.date<='2020-01-14','cftc_managed_net_share_L1'].isna().all()
    assert result.loc[result.date=='2020-01-15','cftc_managed_net_share_L1'].iloc[0]==.1
    assert result.loc[result.date=='2020-01-15','cftc_managed_net_share_L2'].isna().all()
    assert result.loc[result.date=='2020-01-22','cftc_managed_net_share_L6'].iloc[0]==.1
    assert result.loc[result.date=='2020-03-10','cftc_managed_net_share_L1'].isna().all()
    changed=rows.copy();changed.loc[2,'managed_net_share']=.9
    other=add_positions(history,changed)
    pd.testing.assert_frame_equal(result.loc[result.date<='2020-02-04'],other.loc[other.date<='2020-02-04'])
    assert len(result)==len(history)
    with pytest.raises(ValueError):add_positions(history.iloc[::-1],rows)


def test_blackout_cannot_refresh_age_or_expose_masked_values():
    rows=reports(('2023-01-24','2023-01-31','2023-02-07'))
    history=pd.DataFrame({'date':pd.bdate_range('2023-01-20','2023-03-20')})
    result=add_positions(history,rows)
    assert result.loc[result.date=='2023-02-08','cftc_managed_net_share_L1'].iloc[0]==.1
    assert result.loc[result.date=='2023-02-08','cftc_age_L1'].iloc[0]==11
    assert result.loc[result.date>='2023-02-17','cftc_missing_L1'].eq(1).all()


def test_pinned_source_rejects_changed_formula_and_bytes(tmp_path):
    table=tmp_path/'positions.csv';rows=reports();rows.to_csv(table,index=False)
    def pin():
        freeze_record(table.with_suffix('.manifest.json'),{'table_sha256':digest(table),'rows':3,
            'model_eligible':False,'release_allowed':False,'publication_timestamp_verified':False,
            'first_version_verified':False})
    pin();loaded,_=read_positions(table);assert len(loaded)==3
    bad=tmp_path/'bad.csv';rows.loc[0,'managed_net_share']=.8;rows.to_csv(bad,index=False)
    freeze_record(bad.with_suffix('.manifest.json'),{'table_sha256':digest(bad),'rows':3,
        'model_eligible':False,'release_allowed':False,'publication_timestamp_verified':False,'first_version_verified':False})
    with pytest.raises(ValueError,match='ratios/change'):read_positions(bad)
    table.write_text('corrupt')
    with pytest.raises(ValueError,match='Pinned Tier-A'):read_positions(table)


def test_same_origin_controls_and_release_closed(tmp_path):
    groups=group_names()
    for lag in (1,2,6):assert groups[f'position_L{lag}'][:-3]==groups[f'missing_L{lag}']
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())
