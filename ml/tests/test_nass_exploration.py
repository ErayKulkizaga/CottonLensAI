from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.nass_exploration import (
    add_condition,
    dispatch,
    group_names,
    read_condition,
)


def inputs(tmp_path):
    table=tmp_path/'condition.csv'
    pd.DataFrame({'week_ending':['2020-05-31','2020-06-07','2021-05-30'],
        'release_page_date':['2020-06-01','2020-06-08','2021-06-01'],
        'very_poor_pct':[0,1,10],'poor_pct':[10,10,20],'fair_pct':[30,20,30],
        'good_pct':[40,49,30],'excellent_pct':[20,20,10],
        'quickstats_check':['reference_missing_zero','numeric_match','numeric_match'],
        'release_page_url':['https://example.org/2020-06-01','https://example.org/2020-06-08','https://example.org/2021-06-01'],
        'report_sha256':['a'*64,'b'*64,'c'*64]}).to_csv(table,index=False)
    years={}
    frame=pd.read_csv(table)
    for year in (2020,2021):
        rows=frame.loc[frame.week_ending.str.startswith(str(year))]
        p=tmp_path/f'year-audit-{year}.json'
        freeze_record(p,{'year':year,'results':[{'week_ending':r.week_ending,'status':r.quickstats_check,
            'report_sha256':r.report_sha256,'release_page':r.release_page_url} for r in rows.itertuples()]})
        years[str(year)]=digest(p)
    freeze_record(table.with_suffix('.manifest.json'),{'table_sha256':digest(table),'model_eligible':False,
        'publication_clock_verified':False,'row_count':3,'source_year_audit_sha256':years,
        'content_check_counts':{'numeric_match':2,'reference_missing_zero':1}})
    return table


def test_content_bound_zero_is_real_and_changes_do_not_cross_seasons(tmp_path):
    rows,source=read_condition(inputs(tmp_path),tmp_path)
    assert rows.very_poor_pct.iloc[0]==0 and rows.good_excellent.iloc[0]==.6
    assert rows.good_excellent_change.iloc[1]==pytest.approx(.09)
    assert np.isnan(rows.good_excellent_change.iloc[2])
    assert not source['model_eligible'] and not source['release_allowed']


def test_changed_audit_or_table_rejected(tmp_path):
    table=inputs(tmp_path);(tmp_path/'year-audit-2020.json').write_text('changed')
    with pytest.raises(ValueError,match='audit changed'):read_condition(table,tmp_path)
    table.write_text('changed')
    with pytest.raises(ValueError,match='Pinned diagnostic'):read_condition(table,tmp_path)


def test_wrong_category_totals_rejected_even_with_updated_manifest(tmp_path):
    table=inputs(tmp_path);f=pd.read_csv(table);f.loc[0,'good_pct']=90;f.to_csv(table,index=False)
    manifest=table.with_suffix('.manifest.json')
    # Synthetic fixture only: reseal a new valid envelope to test value validation.
    other=tmp_path/'bad.csv';other.write_bytes(table.read_bytes())
    from cottonlens_ml.research.ledger import read_record
    body=read_record(manifest);body['table_sha256']=digest(other)
    freeze_record(other.with_suffix('.manifest.json'),body)
    with pytest.raises(ValueError,match='sum to100'):read_condition(other,tmp_path)


def test_no_backfill_winter_carry_future_dependency_or_origin_drop(tmp_path):
    condition,_=read_condition(inputs(tmp_path),tmp_path)
    dates=pd.bdate_range('2020-01-01','2021-06-30');history=pd.DataFrame({'date':dates,'cotton_close':80.})
    expected=add_condition(history,condition);out=expected.set_index('date')
    assert np.isnan(out.loc['2020-06-01','nass_good_excellent_L1'])
    assert out.loc['2020-06-02','nass_good_excellent_L1']==.6
    assert np.isnan(out.loc['2020-06-02','nass_good_excellent_L2'])
    assert np.isnan(out.loc['2020-07-01','nass_good_excellent_L1'])
    assert out.loc['2021-01-04','nass_missing_L1']==1
    assert np.isnan(out.loc['2021-01-04','nass_good_excellent_L6'])
    changed=condition.copy();changed.loc[2,'good_excellent']=.99
    actual=add_condition(history,changed)
    pd.testing.assert_frame_equal(expected.loc[expected.date<'2021-01-01'],actual.loc[actual.date<'2021-01-01'])
    assert len(expected)==len(history)


def test_groups_have_exact_matching_controls_and_export_is_forbidden(tmp_path):
    groups=group_names();assert len(groups)==7
    for lag in (1,2,6):assert groups[f'condition_L{lag}'][:-3]==groups[f'missing_L{lag}']
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())


def test_late_older_report_cannot_replace_newer_and_invalid_dates_rejected(tmp_path):
    condition,_=read_condition(inputs(tmp_path),tmp_path)
    condition.loc[0,'assumed_day']=pd.Timestamp('2020-06-10')
    dates=pd.bdate_range('2020-05-25','2020-06-30')
    history=pd.DataFrame({'date':dates})
    out=add_condition(history,condition).set_index('date')
    assert out.loc['2020-06-11','nass_good_excellent_L1']==pytest.approx(.69)
    assert out.loc['2020-06-22','nass_good_excellent_L1']==pytest.approx(.69)
    assert out.loc['2020-06-23','nass_good_excellent_L1']==pytest.approx(.69)
    assert np.isnan(out.loc['2020-06-24','nass_good_excellent_L1'])
    condition.loc[0,'assumed_day']=pd.NaT
    with pytest.raises(ValueError,match='Unique sorted'):add_condition(history,condition)
