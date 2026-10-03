"""Synthetic FX identity, missingness and causal windows; no training."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.fx_exploration import (
    add_rates,
    annual_rates,
    crosses,
    dispatch,
    group_names,
    read_rates,
)
from cottonlens_ml.research.ledger import freeze_record


def rates():
    dates=pd.bdate_range('2020-01-01','2020-02-28')
    return crosses(pd.DataFrame({'date':dates,'eur_USD':2.,'eur_BRL':10.+np.arange(len(dates))*.02,
        'eur_CNY':14.,'eur_INR':160.}))


def raw(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode()


def cells():
    return [{'KEY':f'EXR.D.{c}.EUR.SP00.A','FREQ':'D','CURRENCY':c,'CURRENCY_DENOM':'EUR',
        'EXR_TYPE':'SP00','EXR_SUFFIX':'A','TIME_PERIOD':'2020-01-02','OBS_VALUE':v,'OBS_STATUS':'A'}
        for c,v in zip(('BRL','CNY','INR','USD'),(10,14,160,2),strict=True)]


def test_same_date_cross_and_exact_source_type():
    f,_=annual_rates(raw(cells()),2020)
    assert f.brl_per_usd.iloc[0]==5 and f.inr_per_usd.iloc[0]==80
    with pytest.raises(ValueError,match='missing a requested currency'):annual_rates(raw(cells()[:-1]),2020)
    later=[{**r,'TIME_PERIOD':'2020-01-03'} for r in cells()[:-1]]
    with pytest.raises(ValueError,match='four-currency'):annual_rates(raw(cells()+later),2020)
    with pytest.raises(ValueError,match='revisions'):annual_rates(raw(cells()+[cells()[0]]),2020)
    with pytest.raises(ValueError,match='year'):annual_rates(raw(cells()),2021)


def test_positive_domain_and_missing_denominator():
    f=rates();f.loc[0,'eur_USD']=np.nan
    out=crosses(f);assert out[['brl_per_usd','cny_per_usd','inr_per_usd']].iloc[0].isna().all()
    for value in (0.,-1.,np.inf):
        f.loc[0,'eur_USD']=value
        with pytest.raises(ValueError):crosses(f)
    f.loc[0,'eur_USD']=1e-320
    with pytest.raises(ValueError,match='cross-rate'):crosses(f)


def test_holiday_missing_cannot_refresh_age_or_quote():
    f=rates().iloc[:5].copy();f.loc[f.index[1]:,'eur_USD']=np.nan
    h=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-01-13')})
    out=add_rates(h,f)
    assert np.isnan(out.fx_brl_age_L1.iloc[0])
    assert out.fx_brl_age_L1.iloc[1]==0
    assert out.fx_brl_age_L1.iloc[4]==3
    assert out.fx_brl_return_1_L1.iloc[2]==0
    assert np.isnan(out.fx_brl_return_1_L1.iloc[5])
    assert out.fx_brl_age_L1.iloc[5]==4
    assert len(out)==len(h)


def test_no_same_day_input_and_stresses_share_cotton_calendar():
    f=rates();h=pd.DataFrame({'date':pd.bdate_range('2020-01-01','2020-03-10')})
    out=add_rates(h,f)
    assert out.loc[out.date=='2020-01-02','fx_brl_return_1_L1'].isna().all()
    expected=np.log(f.brl_per_usd.iloc[1]/f.brl_per_usd.iloc[0])
    assert out.loc[out.date=='2020-01-03','fx_brl_return_1_L1'].iloc[0]==pytest.approx(expected)
    assert out.loc[out.date=='2020-01-06','fx_brl_return_1_L2'].iloc[0]==pytest.approx(expected)
    assert out.loc[out.date=='2020-01-10','fx_brl_return_1_L6'].iloc[0]==pytest.approx(expected)
    assert out.fx_brl_volatility_21_L1.iloc[:22].isna().all()
    assert np.isfinite(out.fx_brl_volatility_21_L1.iloc[22])
    changed=f.copy();changed.loc[changed.date>='2020-02-03','eur_BRL']*=2
    other=add_rates(h,changed)
    pd.testing.assert_frame_equal(out.loc[out.date<='2020-02-03'],other.loc[other.date<='2020-02-03'])
    with pytest.raises(ValueError):add_rates(h.iloc[::-1],f)


def test_pinned_rates_reject_wrong_cross_formula_and_bytes(tmp_path):
    f=rates();table=tmp_path/'fx.csv';f.to_csv(table,index=False)
    def pin(p):
        freeze_record(p.with_suffix('.manifest.json'),{'table_sha256':digest(p),'rows':len(f),
            'model_eligible':False,'release_allowed':False,'publication_timestamp_verified':False,'first_version_verified':False})
    pin(table);loaded,_=read_rates(table);assert len(loaded)==len(f)
    f.brl_per_usd*=2;bad=tmp_path/'bad.csv';f.to_csv(bad,index=False);pin(bad)
    with pytest.raises(ValueError,match='identity'):read_rates(bad)
    table.write_text('corrupt')
    with pytest.raises(ValueError,match='Pinned Tier-A'):read_rates(table)


def test_exact_matching_controls_and_no_export(tmp_path):
    groups=group_names()
    for lag in (1,2,6):
        assert groups[f'currency_L{lag}'][:-12]==groups[f'missing_L{lag}']
        assert len(groups[f'currency_L{lag}'])==42
    with pytest.raises(ValueError,match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export',drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())
