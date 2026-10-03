import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.fas_exploration import (
    FIELDS,
    add_weekly,
    compile_weekly,
    dispatch,
    group_names,
)
from cottonlens_ml.research.ledger import freeze_record


def inputs(tmp_path, *, duplicate=False, unit=2):
    raw=tmp_path/'fas';catalog=tmp_path/'catalog.json';acquisition=tmp_path/'audit.json';recovery=tmp_path/'review.json'
    catalog.write_text(json.dumps([{'commodityCode':1404,'commodityName':'All Upland Cotton','unitId':2}]))
    archives=[]
    for year in range(2010,2025):
        # Boundary week appears in consecutive files with different values.
        dates=[f'{year-1}-08-06T00:00:00',f'{year}-07-30T00:00:00',f'{year}-08-06T00:00:00']
        rows=[{'commodityCode':1404,'unitId':unit,'countryCode':country,'weekEndingDate':date,
               'currentMYNetSales': -year if country==1 else year+10,
               'weeklyExports':year} for date in dates for country in (1,2)]
        if duplicate:rows.append(rows[0])
        payload=json.dumps(rows).encode()
        sha=hashlib.sha256(payload).hexdigest();p=raw/str(year)/sha/'source.json';p.parent.mkdir(parents=True);p.write_bytes(payload)
        archives.append({'provider':'fas','year':year,'rows':len(rows),'source_sha256':sha})
    acquisition.write_text(json.dumps({'archives':archives}))
    freeze_record(recovery,{'responses':[{'cotton_content_comparison':{'content_matches_at_printed_precision':True}}]*2,
        'numeric_inputs':{'commodity_catalog_sha256':digest(catalog),'api_snapshot_sha256':next(r['source_sha256'] for r in archives if r['year']==2020)}})
    return raw,acquisition,catalog,recovery


def test_compile_boundary_does_not_add_two_marketing_years_and_preserves_negative_sales(tmp_path):
    args=inputs(tmp_path);table=tmp_path/'weekly.csv'
    manifest=compile_weekly(*args,table)
    frame=pd.read_csv(table)
    assert not frame.week_date.duplicated().any()
    r=frame.loc[frame.week_date=='2020-08-06'].iloc[0]
    assert r.market_year==2021 and r.weeklyExports==4042 and r.currentMYNetSales==10
    assert pd.isna(r.weeklyExports_change)
    assert frame.week_date.max()<'2024-01-01'
    assert manifest['model_eligible'] is False and manifest['release_allowed'] is False
    assert compile_weekly(*args,table)==manifest


@pytest.mark.parametrize('kwargs,match',[({'duplicate':True},'Duplicate'),({'unit':1},'Mixed commodity')])
def test_duplicate_or_mixed_units_rejected(tmp_path,kwargs,match):
    with pytest.raises(ValueError,match=match):compile_weekly(*inputs(tmp_path,**kwargs),tmp_path/'weekly.csv')


def test_corrupt_pinned_annual_bytes_rejected(tmp_path):
    args=inputs(tmp_path);next(args[0].glob('*/*/source.json')).write_text('changed')
    with pytest.raises(ValueError,match='missing/corrupt'):compile_weekly(*args,tmp_path/'weekly.csv')


def test_lags_future_prefix_gaps_and_expiration():
    dates=pd.bdate_range('2020-01-01','2020-03-01');history=pd.DataFrame({'date':dates,'cotton_close':80.})
    weekly=pd.DataFrame({'week_date':pd.to_datetime(['2020-01-02','2020-02-20']),
        'assumed_day':pd.to_datetime(['2020-01-09','2020-02-27']),
        'currentMYNetSales':[-5.,10.],'weeklyExports':[20.,30.],
        'currentMYNetSales_change':[np.nan,15.],'weeklyExports_change':[np.nan,10.]})
    expected=add_weekly(history,weekly);a=expected.set_index('date')
    assert pd.isna(a.loc['2020-01-09','fas_net_sales_L1'])
    assert a.loc['2020-01-10','fas_net_sales_L1']==-5/1e6
    assert pd.isna(a.loc['2020-01-10','fas_net_sales_L2'])
    assert pd.isna(a.loc['2020-01-27','fas_exports_L1'])
    assert a.loc['2020-01-10','fas_missing_L1']==1 # Missing change not zero-filled.
    changed=weekly.copy();changed.loc[1,list(FIELDS)]=900
    actual=add_weekly(history,changed)
    pd.testing.assert_frame_equal(expected.loc[expected.date<'2020-02-20'],actual.loc[actual.date<'2020-02-20'])
    assert len(expected)==len(history)


def test_equal_origin_groups_and_export_prohibition(tmp_path):
    groups=group_names();assert len(groups)==7
    for lag in (1,2,6):assert groups[f'sales_L{lag}'][:-4]==groups[f'missing_L{lag}']
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())
