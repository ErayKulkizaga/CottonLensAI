"""Synthetic temporal/weighted-fit/resume contracts; no estimator is trained."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import models, path_pilot, transfer
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.research.panel import FEATURES, asset_rows
from cottonlens_ml.research.protocol import full_year_manifest


@pytest.fixture
def experiment(tmp_path):
    packet = tmp_path/'packet'
    reference = packet/'reference'
    reference.mkdir(parents=True)
    dates = pd.bdate_range('2010-01-01','2023-12-29')
    frames = [asset_rows(pd.DataFrame({'date':dates,'close':100*np.exp(np.sin(np.arange(len(dates))/30)/5+i)}),asset)
        for i,asset in enumerate(('cotton','corn','soybean'))]
    panel = pd.concat(frames,ignore_index=True)
    history = frames[0].rename(columns={'close':'cotton_close'})
    history['cotton_session_index'] = np.arange(len(history))
    split = full_year_manifest(history)
    for fold in split['folds']:
        fold['origins'] = fold['origins'][:240]
    # Real reference snapshots retain a few 2024 target labels at their tail.
    # They must never enter the new pre-2024 training/evaluation table.
    for h in (1,5):
        tail = history.index[-h:]
        history.loc[tail,f'target_date_{h}'] = pd.bdate_range('2024-01-01',periods=h)
        history.loc[tail,f'target_return_{h}'] = 99.
    history.to_parquet(reference/'history.parquet',index=False)
    freeze_record(reference/'ready.json',{'identity':{'split':split},'history_sha256':digest(reference/'history.parquet')})
    panel.to_parquet(packet/'panel-history.parquet',index=False)
    design = {'profile':transfer.PROFILE,'series':['cotton','corn','soybean'],'features':FEATURES,
        'arms':list(transfer.GROUPS),'feature_lag_own_observations':1,
        'model':{'family':'ridge','alpha':10,'target':'scaled_log','seed':42},'cadence':21,
        'purge_cotton_observations':5,'split':split,'shrinkage_weights':[0,.25,.5,.75,1],
        'fit_budget':{'total_maximum':676,'annual_outputs':32},
        'price_gate_unchanged':{'gain_pct':5,'direction_pct':{'1':53,'5':55},'year_wins':6},
        'audit_2024_used':False,'automatic_release':False,'data_hashes':{
            'Cotton_ready':digest(reference/'ready.json'),'Cotton_history':digest(reference/'history.parquet'),
            'panel_table':digest(packet/'panel-history.parquet'),
            'feature_helper':digest(Path(__file__).parents[1]/'src/cottonlens_ml/research/panel.py')}}
    freeze_record(packet/'preregistered.json',{'design_id':content_id(design),'design':design})
    folder = tmp_path/'experiment'
    transfer.prepare(Path(__file__).parents[2],folder,packet)
    return transfer.TransferExperiment(folder),packet


def train_test(exp,group):
    spec = transfer.recipe(exp.identity['design'],group,1)
    test = exp.history.loc[exp.history.date.isin(pd.to_datetime(exp.identity['split']['folds'][0]['origins'][:21]))].copy()
    return spec,exp.train_rows(test.date.min(),spec),test


def test_transforms_weights_and_maturity_are_matched_to_cotton_only(experiment):
    exp,_ = experiment
    spec,train,test = train_test(exp,transfer.GROUPS[1])
    processor,target,weights,info = transfer.training_state(exp.history,train,test,spec)
    control_spec,control,_ = train_test(exp,transfer.GROUPS[0])
    cp,ct,cw,_ = transfer.training_state(exp.history,control,test,control_spec)
    assert processor.as_dict()==cp.as_dict() and target.as_dict()==ct.as_dict()
    assert cw.sum()==pytest.approx(len(control)) and weights.sum()==pytest.approx(len(control))
    assert info['asset_loss_weight_totals']==pytest.approx({'cotton':len(control)*.5,'corn':len(control)*.25,'soybean':len(control)*.25})
    altered = train.copy()
    altered.loc[altered.asset.ne('cotton'),FEATURES] *= 1000
    ap,at,_,_ = transfer.training_state(exp.history,altered,test,spec)
    assert ap.as_dict()==cp.as_dict() and at.as_dict()==ct.as_dict()
    exp.panel.loc[exp.panel.date.ge(test.date.min()),['close','target_return_1','target_return_5']] *= 100
    pd.testing.assert_frame_equal(exp.train_rows(test.date.min(),spec),train)
    bad = train.copy()
    bad.loc[bad.index[-1],'target_date_5'] = test.date.min()
    with pytest.raises(ValueError,match='Immature'):
        transfer.training_state(exp.history,bad,test,spec)


def test_existing_model_fit_receives_frozen_weights_and_saves_adapter(experiment,tmp_path,monkeypatch):
    exp,_ = experiment
    spec,train,test = train_test(exp,transfer.GROUPS[1])
    expected = transfer.training_state(exp.history,train,test,spec)
    captured = {}

    class FakeRidge:
        def __init__(self,alpha):
            assert alpha==10

        def fit(self,x,y,sample_weight=None):
            captured.update(x=x,y=y,weights=sample_weight)
            self.coef_,self.intercept_ = np.zeros(x.shape[1]),0.
            return self

        def predict(self,x):
            return np.zeros(len(x))

    import sklearn.linear_model
    monkeypatch.setattr(sklearn.linear_model,'Ridge',FakeRidge)
    monkeypatch.setattr(models,'require_training',lambda family,device: None)
    result = models.fit_predict(exp.history,train,None,test,spec,tmp_path,iterations=1)
    np.testing.assert_array_equal(captured['weights'],expected[2])
    np.testing.assert_array_equal(captured['x'],expected[0].transform(train))
    np.testing.assert_array_equal(captured['y'],expected[1].forward(train,1))
    adapter = __import__('json').loads((tmp_path/'adapter.json').read_text())
    assert adapter['processor']==expected[0].as_dict() and adapter['target']==expected[1].as_dict()
    assert result['details']['asset_rows']==train.asset.value_counts().to_dict()
    assert result['training_metrics']['sample_count']==len(train.loc[train.asset.eq('cotton')])
    assert len(result['predictions'])==len(test)


def test_frozen_design_and_auxiliary_checksum_are_not_silently_replaced(experiment):
    exp,packet = experiment
    assert exp.history.target_date_5.max()<pd.Timestamp('2024-01-01')
    assert exp.history.target_return_5.tail(5).isna().all()
    before = digest(exp.root/'ready.json')
    transfer.prepare(Path(__file__).parents[2],exp.root,packet)
    assert digest(exp.root/'ready.json')==before
    changed = copy.deepcopy(read_record(packet/'preregistered.json'))
    changed['design']['model']['alpha']=1
    changed['design_id']=content_id(changed['design'])
    with pytest.raises(ValueError,match='design'):
        transfer.validate_design(changed)
    path = exp.root/'panel.parquet'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='snapshot'):
        transfer.TransferExperiment(exp.root)


def test_shared_runner_interruption_resume_baselines_and_complete_report(experiment,monkeypatch):
    exp,_ = experiment
    computed,pause = [],[True]

    def inner(experiment,spec,fold):
        if pause[0] and len(computed)==1:
            raise FitBudgetReached('Synthetic pause')

        def operation(work):
            computed.append((fold['year'],spec['horizon'],spec['arm']))
            (work/'model.txt').write_text('synthetic placeholder')
            return {'iterations':1}

        experiment.ledger.run({'recipe':spec,'year':fold['year'],'synthetic':True},operation)
        return {'recipe':spec,'weight':0.,'inner_score':1.,'iterations':1}

    monkeypatch.setattr(path_pilot,'inner_price',inner)
    monkeypatch.setattr(path_pilot,'predict_chunks',lambda experiment,spec,origins,role,iterations: np.zeros(len(origins)))
    assert path_pilot.run(exp,60,**transfer.options())['status']=='planned_pause'
    pause[0]=False
    assert path_pilot.run(exp,60,**transfer.options())['saved_outputs']==32
    assert len(computed)==32
    assert path_pilot.run(exp,.00001,**transfer.options())['status']=='complete'
    assert len(computed)==32
    result = path_pilot.compare(exp.root,repetitions=100,**transfer.options(),hypothesis_prefix='P',scope='Tier A exploration')
    assert result['release_allowed'] is False and set(result['hypotheses'])=={'P1','P5'}
    assert not any(h['research_priority_signal'] for h in result['horizons'].values())
    for h in (1,5):
        for year in range(2016,2024):
            a=read_record(exp.root/'transfer-outputs'/f'cotton_only-t{h}-year{year}.json')['records']
            b=read_record(exp.root/'transfer-outputs'/f'cotton_corn_soybean-t{h}-year{year}.json')['records']
            assert a==b  # Baselines remain Cotton-only, even when fitting auxiliaries.
    marker=next((exp.root/'transfer-outputs').glob('*.json'))
    marker.write_bytes(marker.read_bytes()+b'bad')
    with pytest.raises(ValueError):
        path_pilot.run(exp,60,**transfer.options())
    assert len(computed)==32
