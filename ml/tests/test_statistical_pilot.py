"""Synthetic CPU API contracts and mocked ledger work; no market-data fitting."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_transfer
from cottonlens_ml import runtime_guard
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import path_pilot, statistical_models, statistical_pilot
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.full_year import chunks
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record


@pytest.fixture
def synthetic_series():
    rng=np.random.default_rng(42)
    returns=np.zeros(190)
    for i in range(1,len(returns)): returns[i]=.6*returns[i-1]+rng.normal(0,.01)
    rows=pd.DataFrame({'date':pd.bdate_range('2010-01-01',periods=len(returns)),
        'cotton_close':100*np.exp(np.cumsum(returns)),'cotton_session_index':np.arange(len(returns))})
    for h in (1,5):
        rows[f'target_date_{h}']=rows.date.shift(-h)
        rows[f'target_return_{h}']=np.log(rows.cotton_close.shift(-h)/rows.cotton_close)
    return rows


def test_statistical_guard_does_not_extend_local_or_ci_permission(monkeypatch):
    monkeypatch.setenv('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR','1')
    monkeypatch.delenv('CI',raising=False); monkeypatch.delenv('GITHUB_ACTIONS',raising=False)
    def deny(): raise RuntimeError('Colab required')
    monkeypatch.setattr(runtime_guard,'require_colab_training',deny)
    for family in ('naive','drift','arima'):
        with pytest.raises(RuntimeError,match='Colab'): runtime_guard.require_training(family,'cpu')
    monkeypatch.setattr(runtime_guard,'require_colab_training',lambda:None)
    runtime_guard.require_training('arima','cpu')
    for device,threads in (('cuda',2),('cpu',3),('cpu',True)):
        with pytest.raises(RuntimeError): runtime_guard.require_training('arima',device,threads)
    monkeypatch.setenv('CI','true')
    with pytest.raises(RuntimeError,match='CI'): runtime_guard.require_training('arima','cpu')


def test_actual_statsmodels_synthetic_forecast_is_causal(synthetic_series,tmp_path,monkeypatch):
    pytest.importorskip('statsmodels')
    monkeypatch.setattr(statistical_models,'require_training',lambda *args:None)
    rows=synthetic_series; train=rows.iloc[:150].copy(); test=rows.iloc[160:166].copy()
    spec=statistical_pilot.recipes({},statistical_pilot.GROUPS[1],5)[2]
    first=statistical_models.fit_predict(rows,train,test,spec,tmp_path)
    assert len(first['predictions'])==6 and np.isfinite(first['predictions']).all()
    changed=rows.copy(); boundary=test.date.iloc[3]
    changed.loc[changed.date>boundary,'cotton_close']*=2
    other=statistical_models.fit_predict(changed,train,changed.iloc[160:166],spec,tmp_path)
    np.testing.assert_array_equal(first['predictions'][:4],other['predictions'][:4])
    assert first['details']['state_updates_refit_parameters'] is False
    bad=train.copy(); bad.loc[bad.index[-1],'target_date_5']=test.date.min()
    with pytest.raises(ValueError,match='maturity'):
        statistical_models.fit_predict(rows,bad,test,spec,tmp_path)
    for family in ('naive','drift'):
        recipe=next(s for s in statistical_pilot.recipes({},statistical_pilot.GROUPS[1],1) if s['family']==family)
        result=statistical_models.fit_predict(rows,train,test,recipe,tmp_path)
        expected=0 if family=='naive' else float(np.diff(np.log(train.cotton_close)).mean())
        np.testing.assert_allclose(result['predictions'],expected)


def test_failed_inner_candidate_stays_excluded_and_does_not_retry(tmp_path,monkeypatch):
    exp=SimpleNamespace(root=tmp_path,identity={'synthetic':True})
    options=statistical_pilot.recipes({},statistical_pilot.GROUPS[1],1)
    calls=[]
    def inner(experiment,spec,fold,**kwargs):
        calls.append(spec['family'])
        if spec['family']=='arima': raise statistical_models.InvalidStatisticalFit('Synthetic nonconvergence')
        return {'recipe':spec,'weight':0.,'iterations':1,'inner_score':1.}
    monkeypatch.setattr(statistical_pilot,'inner_price',inner)
    first=statistical_pilot.select(exp,options,{'year':2016})
    assert first['selected']['recipe']['family']=='naive'
    assert [c['status'] for c in first['candidates']]==['complete','complete','numerical_failure','numerical_failure']
    second=statistical_pilot.select(exp,options,{'year':2016})
    assert second==first and calls.count('arima')==2
    file=tmp_path/'choice.json'; freeze_record(file,{**first,'selection_used_outer':False})
    assert path_pilot.selected_decision(file,options)==first['selected']
    bad=copy.deepcopy(first); bad['selected']=bad['candidates'][1]['selected']
    freeze_record(tmp_path/'changed.json',{**bad,'selection_used_outer':False})
    with pytest.raises(ValueError,match='choice'): path_pilot.selected_decision(tmp_path/'changed.json',options)


def test_actual_adapter_rejects_nonconvergence_without_writing_model(synthetic_series,tmp_path,monkeypatch):
    from statsmodels.tsa.arima import model
    monkeypatch.setattr(statistical_models,'require_training',lambda *args:None)
    class Nonconverged:
        def __init__(self,*args,**kwargs): pass
        def fit(self,**kwargs):
            return SimpleNamespace(params=np.zeros(2),mle_retvals={'converged':False})
    monkeypatch.setattr(model,'ARIMA',Nonconverged)
    spec=statistical_pilot.recipes({},statistical_pilot.GROUPS[1],1)[2]
    with pytest.raises(statistical_models.InvalidStatisticalFit,match='converge'):
        statistical_models.fit_predict(synthetic_series,synthetic_series.iloc[:150],synthetic_series.iloc[160:166],spec,tmp_path)
    assert not (tmp_path/'statistical-model.json').exists()


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    _,old_packet=test_transfer.experiment.__wrapped__(tmp_path)
    reference=old_packet/'reference'; old=read_record(reference/'ready.json')
    # Synthetic old fixture deliberately contains terminal2024 targets; preparation must use a masked source.
    rows=pd.read_parquet(reference/'history.parquet')
    for h in (1,5):
        bad=rows[f'target_date_{h}'].ge('2024-01-01')
        rows.loc[bad,f'target_date_{h}']=pd.NaT; rows.loc[bad,f'target_return_{h}']=np.nan
    packet=tmp_path/'statistical-packet'; (packet/'reference').mkdir(parents=True)
    rows.to_parquet(packet/'reference/history.parquet',index=False)
    freeze_record(packet/'reference/ready.json',{'identity':old['identity'],
        'history_sha256':digest(packet/'reference/history.parquet')})
    inner=sum(len(chunks(rows,b['origins'])) for f in old['identity']['split']['folds'] for b in f['inner'])
    outer=sum(len(chunks(rows,f['origins'])) for f in old['identity']['split']['folds'])
    design={'profile':statistical_pilot.PROFILE,'policy':statistical_pilot.POLICY,
        'groups':list(statistical_pilot.GROUPS),'shrinkage_weights':[0,.25,.5,.75,1],
        'split':old['identity']['split'],'price_gate':{'gain_pct':5,'direction_pct':{'1':53,'5':55},'year_wins':6},
        'automatic_release':False,'audit_2024_used':False,'ready_sha256':digest(packet/'reference/ready.json'),
        'history_sha256':digest(packet/'reference/history.parquet'),
        'helper_sha256':digest(Path(__file__).parents[1]/'src/cottonlens_ml/research/statistical_models.py'),
        'fit_budget':{'maximum_ledger_jobs':8*inner+4*outer,'maximum_arima_estimations':4*inner+2*outer,'annual_outputs':32}}
    freeze_record(packet/'preregistered.json',{'design_id':content_id(design),'design':design})
    monkeypatch.setattr(statistical_pilot,'require_colab_training',lambda:None)
    root=tmp_path/'statistical-experiment'; statistical_pilot.prepare(Path(__file__).parents[2],root,packet)
    return Experiment(root)


def test_existing_runner_handles_multiple_candidates_resume_and_no_new_fit(prepared,monkeypatch):
    exp=prepared; calls=[]; pause=[True]
    def inner(experiment,spec,fold,*,weights):
        if pause[0] and len(calls)==1: raise FitBudgetReached('Synthetic pause')
        def operation(work):
            calls.append((fold['year'],spec['horizon'],spec['family']))
            (work/'model.txt').write_text('synthetic')
            return {'iterations':1}
        experiment.ledger.run({'year':fold['year'],'spec':spec},operation)
        return {'recipe':spec,'iterations':1,'weight':weights[0],'inner_score':1.}
    def predict(experiment,spec,origins,role,iterations): return np.zeros(len(origins))
    monkeypatch.setattr(statistical_pilot,'inner_price',inner)
    monkeypatch.setattr(path_pilot,'predict_chunks',predict)
    assert path_pilot.run(exp,60,**statistical_pilot.options(),selection_fn=statistical_pilot.select)['status']=='planned_pause'
    pause[0]=False
    assert path_pilot.run(exp,60,**statistical_pilot.options(),selection_fn=statistical_pilot.select)['saved_outputs']==32
    count=len(calls)
    assert path_pilot.run(exp,.000001,**statistical_pilot.options(),selection_fn=statistical_pilot.select)['status']=='complete'
    assert len(calls)==count
    report=path_pilot.compare(exp.root,repetitions=100,**statistical_pilot.options(),hypothesis_prefix='S')
    assert report['release_allowed'] is False and set(report['hypotheses'])=={'S1','S5'}
