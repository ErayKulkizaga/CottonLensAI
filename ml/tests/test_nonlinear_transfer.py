"""Mocked GPU/weighted-tree/temporal contracts; never train an estimator."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_transfer
from cottonlens_ml.cohort import content_id
from cottonlens_ml.evaluation import evaluate
from cottonlens_ml.research import full_year, models, path_pilot, transfer
from cottonlens_ml.research import nonlinear_transfer as nonlinear
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record


@pytest.fixture
def linear_case(tmp_path):
    return test_transfer.experiment.__wrapped__(tmp_path)


@pytest.fixture
def nonlinear_case(linear_case,monkeypatch,tmp_path):
    _,packet = linear_case
    registration = copy.deepcopy(read_record(packet/'preregistered.json'))
    registration['design'].update(profile=nonlinear.PROFILE,model=nonlinear.MODEL,fit_budget=nonlinear.BUDGET)
    registration['design_id'] = content_id(registration['design'])
    freeze_record(packet/'nonlinear-registration.json',registration)
    replacement = tmp_path/'nonlinear-packet'
    replacement.mkdir()
    import shutil
    shutil.copytree(packet/'reference',replacement/'reference')
    shutil.copyfile(packet/'panel-history.parquet',replacement/'panel-history.parquet')
    shutil.copyfile(packet/'nonlinear-registration.json',replacement/'preregistered.json')
    monkeypatch.setattr(nonlinear,'require_colab_training',lambda:None)
    folder = tmp_path/'nonlinear-experiment'
    nonlinear.prepare(Path(__file__).parents[2],folder,replacement)
    exp = nonlinear.NonlinearExperiment(folder)
    assert exp.identity['execution']=={'device':'cuda','threads':2}
    assert exp.identity['exact_fit_budget']<=772
    assert not exp.history.target_return_5.tail(5).notna().any()
    return exp


def test_early_validation_is_cotton_only_and_tree_counts_are_past_locked(nonlinear_case):
    exp = nonlinear_case
    captured = []
    spec = nonlinear.recipe(exp.identity['design'],transfer.GROUPS[1],1)
    fold = exp.identity['split']['folds'][0]

    def fake_fit(recipe,train,validation,test,role,*,iterations=None,repeat=None):
        assert test.asset.eq('cotton').all()
        transfer.training_state(exp.history,train,test,recipe)
        if validation is not None:
            assert len(validation)==63 and validation.asset.eq('cotton').all()
            assert validation.date.max()<min(np.array(fold['inner'][int(role.rsplit('-',1)[1])]['origins'],dtype='datetime64'))
            count = (7,11,19)[int(role.rsplit('-',1)[1])]
        else:
            count = iterations
            captured.append(count)
        return {'result':{'iterations':count,'predictions':np.zeros(len(test)).tolist()}}

    exp.fit = fake_fit
    chosen = full_year.inner_price(exp,spec,fold)
    assert chosen['iterations']==11 and set(captured)=={7,11,19}
    assert chosen['weight']==0  # Stable simple tie break on synthetic zero predictions.
    assert chosen['inner_score']==pytest.approx(1.)


def test_gpu_fit_passes_asset_weights_and_stops_on_cotton_price_mae(nonlinear_case,tmp_path,monkeypatch):
    exp = nonlinear_case
    spec = nonlinear.recipe(exp.identity['design'],transfer.GROUPS[1],1)
    test = exp.history.loc[exp.history.date.isin(pd.to_datetime(exp.identity['split']['folds'][0]['origins'][:21]))].copy()
    train = exp.train_rows(test.date.min(),spec)
    expected = transfer.training_state(exp.history,train,test,spec)
    seen, device = [],['cuda:0']

    class Matrix:
        def __init__(self,data,label=None,weight=None):
            self.data,self.label,self.weight = data,label,weight
        def get_label(self): return self.label

    class Booster:
        best_iteration = 3
        def save_config(self): return json.dumps({'learner':{'generic_param':{'device':device[0]}}})
        def __getitem__(self,count):
            seen.append(('sliced',count.stop)); return self
        def predict(self,data): return np.zeros(len(data.data))
        def save_model(self,path): path.write_text('synthetic model')
        def get_dump(self,**kwargs): return ['{}']*4
        def get_score(self): return {}

    def fake_train(config,dtrain,**kwargs):
        assert config['device']=='cuda' and config['tree_method']=='hist'
        assert kwargs['num_boost_round']==600 and kwargs['early_stopping_rounds']==50
        assert np.array_equal(dtrain.weight,expected[2])
        assert kwargs['evals'][-1][1]=='validation'
        curves = kwargs['evals_result']
        for data,name in kwargs['evals']:
            results = kwargs['custom_metric'](np.zeros(len(data.data)),data)
            assert [key for key,_ in results]==['weighted_target_mse','price_mae']
            curves[name] = {key:[value]*4 for key,value in results}
        cotton = train.asset.eq('cotton').to_numpy()
        truth = expected[1].inverse(dtrain.label,train.cotton_close.to_numpy())
        predictions = expected[1].inverse(np.zeros(len(train)),train.cotton_close.to_numpy())
        assert curves['train']['price_mae'][0]==pytest.approx(
            evaluate(train.cotton_close.to_numpy()[cotton],truth[cotton],predictions[cotton])['mae'])
        seen.append(('trained',1)); return Booster()

    import xgboost
    monkeypatch.setattr(xgboost,'DMatrix',Matrix)
    monkeypatch.setattr(xgboost,'train',fake_train)
    monkeypatch.setattr(models,'require_training',lambda *args:None)
    monkeypatch.setattr(models,'Telemetry',models.CPUTelemetry)
    result = models.fit_predict(exp.history,train,test,test,spec,tmp_path)
    assert result['iterations']==4 and ('sliced',4) in seen
    assert result['details']['asset_loss_weight_totals']==pytest.approx(expected[3]['asset_loss_weight_totals'])
    assert result['training_metrics']['sample_count']==expected[3]['cotton_transform_rows']
    device[0]='cpu'
    with pytest.raises(RuntimeError,match='fell back'):
        models.fit_predict(exp.history,train,test,test,spec,tmp_path)


def test_locked_outer_tree_count_resume_and_no_duplicate_fit(nonlinear_case,monkeypatch):
    exp = nonlinear_case
    computed, counts, pause = [],[],[True]

    def inner(experiment,spec,fold):
        if pause[0] and len(computed)==1: raise FitBudgetReached('Synthetic pause')
        def operation(work):
            computed.append((fold['year'],spec['horizon'],spec['arm']))
            (work/'model.txt').write_text('mock')
            return {'iterations':11}
        experiment.ledger.run({'recipe':spec,'year':fold['year'],'synthetic':True},operation)
        return {'recipe':spec,'weight':0.,'inner_score':1.,'iterations':11}

    def predictions(experiment,spec,origins,role,iterations):
        counts.append(iterations)
        assert iterations==11
        return np.zeros(len(origins))
    monkeypatch.setattr(path_pilot,'inner_price',inner)
    monkeypatch.setattr(path_pilot,'predict_chunks',predictions)
    assert path_pilot.run(exp,60,**nonlinear.options())['status']=='planned_pause'
    pause[0]=False
    assert path_pilot.run(exp,60,**nonlinear.options())['saved_outputs']==32
    assert len(computed)==len(counts)==32
    assert path_pilot.run(exp,.00001,**nonlinear.options())['status']=='complete'
    assert len(computed)==len(counts)==32
    report = path_pilot.compare(exp.root,repetitions=100,**nonlinear.options(),hypothesis_prefix='N',scope='Tier A')
    assert report['release_allowed'] is False and set(report['hypotheses'])=={'N1','N5'}
    bad = copy.deepcopy(exp.identity['design'])
    bad['model']['params']['max_depth']=3
    with pytest.raises(ValueError,match='design'):
        nonlinear.validate_design({'design':bad,'design_id':content_id(bad)})


def test_gpu_profile_is_not_authorized_for_local_cpu_prepare(linear_case,tmp_path,monkeypatch):
    _,packet = linear_case
    def deny(): raise RuntimeError('Colab required')
    monkeypatch.setattr(nonlinear,'require_colab_training',deny)
    with pytest.raises(RuntimeError,match='Colab'):
        nonlinear.prepare(Path(__file__).parents[2],tmp_path/'denied',packet)
    assert not (tmp_path/'denied').exists()
