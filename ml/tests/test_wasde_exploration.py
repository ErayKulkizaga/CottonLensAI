from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.wasde_exploration import (
    add_balance,
    balance,
    compile_balance,
    dispatch,
    group_names,
    read_balance,
)


def cells():
    result=[]
    for season,month in [('2019/20 Est.',''),('2020/21 Proj.','Apr'),('2020/21 Proj.','May')]:
        for field,value in [('Production','100'),('Domestic Use','110'),('Ending Stocks','80')]:
            result.append({'region':'World','report_month':'May 2020','marketing_year':season,
                'forecast_month':month,'attribute':field,'value':value,
                'units':'(Million 480-Pound Bales)'})
    return result


def test_latest_crop_current_month_and_ratio_units():
    rows=cells();rows[0]['value']='9999';rows[5]['value']='1234'
    result=balance(rows,'2020-05-12')
    assert result['crop_year']==2020 and result['stock_use']==pytest.approx(80/110)
    assert result['production_use']==pytest.approx(100/110)
    with pytest.raises(ValueError,match='Report month'):balance(rows,'2020-06-12')


def test_ambiguous_current_forecast_or_invalid_consumption_rejected():
    rows=cells();rows[-2]['value']='0'
    with pytest.raises(ValueError,match='Positive consumption'):balance(rows,'2020-05-12')
    rows=cells();rows.append(rows[-1])
    with pytest.raises(ValueError,match='One finite'):balance(rows,'2020-05-12')


def reports():
    return pd.DataFrame({'assumed_day':pd.to_datetime(['2020-03-10','2020-04-09','2020-05-12']),
        'crop_year':[2019,2019,2020],'stock_use':[.8,.85,.7],
        'production_use':[.9,1.,1.1],'stock_use_change':[np.nan,.05,np.nan]})


def test_no_early_use_future_dependence_or_feature_based_origin_drop():
    history=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-10-01')})
    source=reports();a=add_balance(history,source);out=a.set_index('date')
    assert np.isnan(out.loc['2020-03-10','wasde_stock_use_L1'])
    assert out.loc['2020-03-11','wasde_stock_use_L1']==.8
    assert np.isnan(out.loc['2020-03-11','wasde_stock_use_L2'])
    assert np.isnan(out.loc['2020-09-01','wasde_stock_use_L1'])
    changed=source.copy();changed.loc[2,'stock_use']=99
    b=add_balance(history,changed)
    pd.testing.assert_frame_equal(a.loc[a.date<'2020-05-12'],b.loc[b.date<'2020-05-12'])
    assert len(a)==len(history)
    with pytest.raises(ValueError,match='chronological'):add_balance(history,source.iloc[::-1])


def test_pinned_table_rejects_bad_same_year_revisions_and_crop_resets(tmp_path):
    source=reports();table=tmp_path/'balance.csv';source.to_csv(table,index=False)
    freeze_record(table.with_suffix('.manifest.json'),{'table_sha256':digest(table),'rows':3,
        'model_eligible':False,'release_allowed':False,'publication_timestamp_verified':False})
    data,_=read_balance(table);assert np.isnan(data.stock_use_change.iloc[2])
    source.loc[2,'stock_use_change']=-.15
    other=tmp_path/'bad.csv';source.to_csv(other,index=False)
    freeze_record(other.with_suffix('.manifest.json'),{'table_sha256':digest(other),'rows':3,
        'model_eligible':False,'release_allowed':False,'publication_timestamp_verified':False})
    with pytest.raises(ValueError,match='causal same-crop'):read_balance(other)
    table.write_text('changed')
    with pytest.raises(ValueError,match='Pinned Tier-A'):read_balance(table)


def test_matching_controls_and_release_forbidden(tmp_path):
    groups=group_names()
    for lag in (1,2,6):assert groups[f'balance_L{lag}'][:-3]==groups[f'missing_L{lag}']
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-example'))
    assert not list(tmp_path.iterdir())


def test_unchanged_later_same_month_page_does_not_reset_revision_or_age(tmp_path,monkeypatch):
    from cottonlens_ml.research import wasde_exploration as module
    root=tmp_path/'raw';items=[]
    def archive(url,raw):
        import hashlib
        sha=hashlib.sha256(raw).hexdigest();folder=root/'wasde'/sha;folder.mkdir(parents=True,exist_ok=True)
        (folder/'source.bin').write_bytes(raw)
        freeze_record(folder/'retrieval.json',{'kind':'wasde','source_url':url,'sha256':sha})
    for day in ('2020-05-12','2020-05-15'):
        url='https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/'+day
        xml_url='https://esmis.nal.usda.gov/'+day+'.xml'
        archive(url,f'<a href="{xml_url}">xml</a>'.encode());archive(xml_url,day.encode())
        items.append({'url':url,'numeric_status':'passed'})
    for i,name in enumerate(module.EVIDENCE):
        freeze_record(root/name,{'releases':items if i==0 else [],'model_eligible':False,'publication_timing_verified':False})
    monkeypatch.setattr(module,'cotton_rows',lambda _:cells())
    table=tmp_path/'balance.csv';manifest=compile_balance(root,table)
    assert manifest['rows']==1 and len(manifest['source_versions'])==2
    rows,_=read_balance(table);assert rows.assumed_day.iloc[0]==pd.Timestamp('2020-05-12')
    assert np.isnan(rows.stock_use_change.iloc[0])


def test_short_cohort_priority_is_preregistered_and_cannot_be_release(tmp_path):
    from cottonlens_ml.config import FEATURE_NAMES
    from cottonlens_ml.research.ams_exploration import prepare_information
    dates=pd.bdate_range('2010-01-01','2023-12-31')
    history=pd.DataFrame({'date':dates,'cotton_close':80.,'cotton_session_index':np.arange(len(dates))})
    for h in (1,5):
        history[f'target_date_{h}']=history.date.shift(-h)
        history[f'target_return_{h}']=.001
    for feature in FEATURE_NAMES:
        if feature not in history:history[feature]=0.
    parent=tmp_path/'parent';parent.mkdir();history.to_parquet(parent/'history.parquet',index=False)
    freeze_record(parent/'ready.json',{'history_sha256':digest(parent/'history.parquet')})
    target=tmp_path/'experiment'
    args={'profile':'wasde-exploration-v1','source':{'tier':'A'},'add_features':lambda h:h,
        'first_source_day':pd.Timestamp('2016-01-12'),'groups':{'base':FEATURE_NAMES},'controls':{},'prefix':'fixture',
        'hypothesis':'fixture','policy':{},'priority_fraction':5/8}
    prepare_information(Path(__file__).parents[2],target,parent,**args)
    ready=read_record(target/'ready.json');prereg=read_record(target/'preregistered.json')
    assert len(ready['identity']['split']['folds'])==5
    assert prereg['priority_rule']['outer_fold_wins']==4
    assert prereg['priority_rule']==ready['identity']['research_priority_rule']
    assert not ready['identity']['gate_evaluation_allowed'] and not prereg['release_allowed']
    frozen=digest(target/'ready.json');prepare_information(Path(__file__).parents[2],target,parent,**args)
    assert digest(target/'ready.json')==frozen
