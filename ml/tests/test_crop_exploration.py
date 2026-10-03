"""Synthetic provider identity, causal crop windows, immutable packets; no fits."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.crop_exploration import (
    add_prices,
    dispatch,
    group_names,
    read_prices,
)
from cottonlens_ml.sources.crop_prices import (
    QUERY,
    archive,
    compile_prices,
    parse_chart,
)


def payload(symbol='ZC=F', closes=(400., 410., 420.)):
    timestamps=[int(t.timestamp()) for t in pd.date_range('2020-01-01 01:00',periods=3,tz='UTC')]
    return json.dumps({'chart': {'error': None, 'result': [{'meta': {'symbol': symbol,
        'instrumentType': 'FUTURE', 'exchangeTimezoneName': 'America/New_York', 'currency': 'USX'},
        'timestamp': timestamps, 'indicators': {'quote': [{'open': list(closes), 'high': list(closes),
        'low': list(closes), 'close': list(closes), 'volume': [0, 1, 2]}]}}]}}).encode()


class Session:
    def __init__(self, raw, status=200):self.raw,self.status=raw,status
    def get(self, url, **kwargs):
        self.url,self.kwargs=url,kwargs
        raw,status=self.raw,self.status
        class Response:
            status_code=status
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def iter_content(self,size):yield raw
        return Response()


def prices():
    dates=pd.bdate_range('2020-01-01','2020-03-01')
    return pd.concat([pd.DataFrame({'date':dates,'series':s,'close':400.+np.arange(len(dates))})
        for s in ('corn','soybean')],ignore_index=True).sort_values(['date','series'])


def test_provider_timezone_identity_and_raw_quote_domain():
    f,a=parse_chart(payload(closes=(400., None, -1.)), 'corn')
    assert f.date.iloc[0]==pd.Timestamp('2019-12-31')  # UTC date is not source-local day.
    assert a['missing_or_nonpositive_closes']==2 and a['zero_volume_rows']==1
    with pytest.raises(ValueError,match='identity'):parse_chart(payload(symbol='ZS=F'),'corn')
    with pytest.raises(ValueError):parse_chart(b'not json','corn')
    with pytest.raises(ValueError):parse_chart(payload(),'wheat')


def test_chart_rejects_duplicate_future_and_misaligned_dates():
    body=json.loads(payload());r=body['chart']['result'][0]
    r['timestamp'][1]=r['timestamp'][0]
    with pytest.raises(ValueError,match='Unique'):parse_chart(json.dumps(body).encode(),'corn')
    body=json.loads(payload());body['chart']['result'][0]['timestamp'][0]=1735689600
    with pytest.raises(ValueError,match='chronological'):parse_chart(json.dumps(body).encode(),'corn')
    body=json.loads(payload());body['chart']['result'][0]['indicators']['quote'][0]['close'].pop()
    with pytest.raises(ValueError):parse_chart(json.dumps(body).encode(),'corn')


def test_archive_bounded_no_credentials_and_no_false_vintage(tmp_path):
    session=Session(payload());folder=archive('corn',tmp_path,session=session)
    receipt=json.loads((folder/'retrieval.json').read_text())
    assert session.kwargs['params']==QUERY and session.kwargs['allow_redirects'] is False
    assert not receipt['model_eligible'] and not receipt['first_version_verified']
    before=(folder/'retrieval.json').read_bytes()
    assert archive('corn',tmp_path,session=Session(payload()))==folder
    assert (folder/'retrieval.json').read_bytes()==before
    with pytest.raises(RuntimeError,match='429'):archive('soybean',tmp_path,session=Session(b'',429))
    assert not (tmp_path/'soybean').exists()


def test_two_pinned_sources_compile_and_corruption_rejected(tmp_path):
    folders={s:archive(s,tmp_path/'raw',session=Session(payload(t))) for s,t in
             (('corn','ZC=F'),('soybean','ZS=F'))}
    table=tmp_path/'inputs.csv';m=compile_prices(folders,table)
    assert m['rows']==6 and not m['release_allowed']
    assert len(read_prices(table)[0])==6
    assert compile_prices(folders,table)==m
    (folders['corn']/'chart.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Pinned'):compile_prices(folders,tmp_path/'bad.csv')
    table.write_text('changed')
    with pytest.raises(ValueError,match='Pinned'):read_prices(table)


def test_nonpositive_missing_quotes_never_refresh_age_and_no_origin_drop():
    p=prices();p.loc[p.date>'2020-01-01','close']=np.nan
    p.loc[p.date=='2020-01-03','close']=-1
    h=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-01-13')})
    out=add_prices(h,p)
    assert len(out)==len(h)
    assert out.crop_corn_age_L1.iloc[4]==3
    assert out.crop_corn_return_1_L1.iloc[2]==0
    assert out.crop_corn_return_1_L1.iloc[5]!=out.crop_corn_return_1_L1.iloc[5]
    assert out.crop_corn_age_L1.iloc[5]==4


def test_causal_lagged_retained_calendar_windows():
    p=prices();h=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-03-10')})
    out=add_prices(h,p)
    expected=np.log(401/400)
    for lag,day in ((1,'2020-01-03'),(2,'2020-01-06'),(6,'2020-01-10')):
        assert out.loc[out.date==day,f'crop_corn_return_1_L{lag}'].iloc[0]==pytest.approx(expected)
    assert out.crop_corn_volatility_21_L1.iloc[:22].isna().all()
    changed=p.copy();changed.loc[changed.date>='2020-02-03','close']*=2
    other=add_prices(h,changed)
    pd.testing.assert_frame_equal(out.loc[out.date<='2020-02-03'],other.loc[other.date<='2020-02-03'])
    with pytest.raises(ValueError):add_prices(h.iloc[::-1],p)
    with pytest.raises(ValueError):add_prices(h,p.iloc[::-1])


def test_matching_control_and_export_blocked(tmp_path):
    groups=group_names()
    for lag in (1,2,6):
        assert groups[f'prices_L{lag}'][:-8]==groups[f'missing_L{lag}']
        assert len(groups[f'prices_L{lag}'])==36
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())
