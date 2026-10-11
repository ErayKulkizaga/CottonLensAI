import copy

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import named_history as nh
from cottonlens_ml.research.protocol import Preprocessor


def quote(day, values, *, displayed=None):
    contracts=list(values)
    return {'report_date':day, 'publication_local_naive':displayed or day+'T01:00:00',
        'published_at':None, 'timezone_verified':False, 'first_vintage_verified':False, 'model_eligible':False,
        'document_sha256':'a'*64, 'quotes':values, 'first_contract':contracts[0], 'second_contract':contracts[1],
        'front_second_gap_months':2}


def example():
    dates=pd.bdate_range('2023-01-02',periods=24)
    records=[quote(d.strftime('%Y-%m-%d'), {'Mar-23':70+i,'May-23':80+2*i,'Jul-23':90+3*i}) for i,d in enumerate(dates)]
    return pd.DataFrame({'date':dates}), records


def test_current_selected_contract_used_at_both_endpoints_across_rank_change():
    h,q=example()
    for i in range(10,len(q)):
        q[i]=quote(q[i]['report_date'], {'May-23':80+2*i,'Jul-23':90+3*i,'Oct-23':100+4*i})
        q[i]['front_second_gap_months']=2
    features,traces=nh.build(h,q)
    assert features.iloc[10].named_contract_ret_1==pytest.approx(np.log(120/117))
    assert features.iloc[10].named_contract_ret_5==pytest.approx(np.log(120/105))
    assert features.iloc[10].named_contract_ret_1!=pytest.approx(np.log(120/98))
    assert next(t for t in traces if t['date']==q[10]['report_date'] and t['window']==5)['contract']=='Jul-23'


def test_assumed_availability_by_decision_and_late_report_exclusion():
    h,q=example()
    q[3]['publication_local_naive']=q[4]['report_date']+'T23:59:59'
    f,traces=nh.build(h,q)
    at_day3=next(t for t in traces if t['date']==q[3]['report_date'] and t['window']==1)
    assert at_day3['current_report_date']==q[2]['report_date']
    at_day4=next(t for t in traces if t['date']==q[4]['report_date'] and t['window']==1)
    assert at_day4['current_report_date']==q[4]['report_date']
    assert f.iloc[3].named_contract_ret_1==0
    assert pd.Timestamp(at_day3['current_assumed_available_at'])==pd.Timestamp(q[2]['report_date'],tz='UTC')+pd.Timedelta(days=1)
    assert pd.Timestamp(at_day4['current_assumed_available_at'])<=pd.Timestamp(h.date.iloc[4],tz='UTC')+pd.Timedelta(days=1,minutes=15)
    _,without_newer=nh.build(h,[r for i,r in enumerate(q) if i!=4])
    late_available=next(t for t in without_newer if t['date']==q[4]['report_date'] and t['window']==1)
    assert late_available['current_report_date']==q[3]['report_date']


def test_future_quotes_and_future_calendar_extension_cannot_change_past_features():
    h,q=example()
    before,_=nh.build(h.iloc[:12],q)
    future=copy.deepcopy(q)
    for r in future[12:]:
        r['quotes']={k:v*10 for k,v in r['quotes'].items()}
        r['document_sha256']='b'*64
    after,traces=nh.build(h,future)
    pd.testing.assert_frame_equal(before,after.iloc[:12])
    assert all(pd.Timestamp(t['current_report_date'])<=pd.Timestamp(t['date']) for t in traces if t['current_report_date'])


def test_late_past_bar_available_now_used_but_future_release_rejected():
    h,q=example()
    q[3]['publication_local_naive']=q[4]['report_date']+'T23:59:59'
    f,traces=nh.build(h,q)
    t=next(t for t in traces if t['date']==q[4]['report_date'] and t['window']==1)
    assert t['past_report_date']==q[3]['report_date']
    assert f.iloc[4].named_contract_ret_1==pytest.approx(np.log(88/86))
    assert pd.Timestamp(t['past_assumed_available_at'])>pd.Timestamp(h.date.iloc[3],tz='UTC')+pd.Timedelta(days=1,minutes=15)
    q[3]['publication_local_naive']=q[5]['report_date']+'T01:00:00'
    f,traces=nh.build(h,q)
    t=next(t for t in traces if t['date']==q[4]['report_date'] and t['window']==1)
    assert t['past_report_date']==q[2]['report_date']
    assert f.iloc[4].named_contract_ret_1==pytest.approx(np.log(88/84))


def test_changed_old_reference_quote_with_future_publication_does_not_leak():
    h,q=example()
    q[3]['publication_local_naive']=q[15]['report_date']+'T01:00:00'
    before,_=nh.build(h,q)
    later=copy.deepcopy(q)
    later[3]['quotes']={k:v*10 for k,v in later[3]['quotes'].items()}
    later[3]['document_sha256']='b'*64
    after,_=nh.build(h,later)
    pd.testing.assert_frame_equal(before.iloc[:15],after.iloc[:15])


def test_targets_prices_and_named_label_metadata_are_not_feature_inputs():
    h,q=example()
    f,_=nh.build(h,q)
    changed=h.assign(cotton_close=1e6,target_return_5=-99,named_price=-1,named_return=99,named_contract='Dec-99')
    other,_=nh.build(changed,q)
    pd.testing.assert_frame_equal(f,other)


def test_missing_sources_keep_cotton_time_and_stale_limit():
    h,q=example()
    f,t=nh.build(h,[q[0],*q[6:]])
    assert len(f)==len(h) and len(t)==4*len(h)
    assert f.iloc[4].isna().all() and f.iloc[5].isna().all()
    assert f.iloc[6].named_contract_ret_5==pytest.approx(np.log(92/80))
    assert nh.mask(f).isna().equals(f.isna())
    assert nh.mask(f).fillna(0).eq(0).all().all()


def test_contract_not_yet_listed_is_missing_not_different_rank_price():
    h,q=example()
    for i in range(10,len(q)):
        q[i]=quote(q[i]['report_date'], {'Jul-23':90+3*i,'Oct-23':100+4*i,'Dec-23':110+5*i})
        q[i]['front_second_gap_months']=3
    f,t=nh.build(h,q)
    assert np.isnan(f.iloc[10].named_contract_ret_1)
    assert next(r for r in t if r['date']==q[10]['report_date'] and r['window']==1)['reason']=='contract_not_listed_at_past_source'


@pytest.mark.parametrize('bad', [0.,-1.,np.nan,np.inf])
def test_invalid_quote_domain_rejected(bad):
    h,q=example()
    q[4]['quotes']['May-23']=bad
    with pytest.raises(ValueError,match='positive quotes'):
        nh.build(h,q)


def test_duplicate_report_date_rejected():
    h,q=example()
    with pytest.raises(ValueError,match='Unique report'):
        nh.build(h,[*q,q[0]])


def test_non_monotone_calendar_rejected():
    h,q=example()
    with pytest.raises(ValueError,match='chronological'):
        nh.build(h.iloc[::-1],q)


def test_preprocessing_only_uses_training_features_and_matches_masks():
    h,q=example()
    f,_=nh.build(h,q)
    train=f.iloc[:12]
    processor=Preprocessor.fit(train.copy(),list(nh.FEATURES))
    original=processor.as_dict()
    altered=f.copy()
    altered.iloc[12:]=1e6
    assert Preprocessor.fit(altered.iloc[:12].copy(),list(nh.FEATURES)).as_dict()==original
    np.testing.assert_array_equal(processor.transform(train)[:,4:], Preprocessor.fit(nh.mask(train),list(nh.FEATURES)).transform(nh.mask(train))[:,4:])


def test_mask_rejects_changed_schema_and_preserves_indices():
    h,q=example()
    f,_=nh.build(h,q)
    f.index=pd.RangeIndex(100,124)
    assert nh.mask(f).index.equals(f.index)
    with pytest.raises(ValueError,match='schema'):
        nh.mask(f.iloc[:,:3])
