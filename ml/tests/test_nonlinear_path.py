"""Synthetic/mocked nonlinear-path contracts; no real CPU or CUDA market fit."""
import copy
from pathlib import Path

import numpy as np
import pytest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import nonlinear_path, path_pilot
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.full_year import chunks, inner_price
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record
from cottonlens_ml.research.protocol import rows_at
from test_path_pilot import packet as build_ridge_packet


@pytest.fixture
def packet(tmp_path):
    _, reference, registration = build_ridge_packet.__wrapped__(tmp_path)
    d = copy.deepcopy(registration['design'])
    d.update({'profile':nonlinear_path.PROFILE,'recipes':nonlinear_path.MODEL,
              'fit_budget':nonlinear_path.BUDGET})
    extra = sum(len(f['origins']) for f in d['split']['folds']) - 2006
    assert extra >= 0
    if extra:
        d['split']['folds'][-1]['origins'] = d['split']['folds'][-1]['origins'][:-extra]
    # Synthetic packet needs its own exact budget, so test preparation against the
    # original actual frozen topology separately; the mocked runner verifies 32 outputs.
    registration = {'design':d,'design_id':content_id(d)}
    freeze_record(reference.parent/'preregistered.json',registration)
    nonlinear_path.validate_design(registration)
    folder = reference.parent.parent/'nonlinear'
    old = read_record(reference.parent.parent/'experiment/ready.json')
    old['identity'].update({'profile':nonlinear_path.PROFILE,'design':d,'design_id':registration['design_id'],'split':d['split']})
    old.pop('record_id',None)
    folder.mkdir()
    import shutil
    shutil.copyfile(reference.parent.parent/'experiment/history.parquet',folder/'history.parquet')
    freeze_record(folder/'ready.json',old)
    return folder, registration


def test_fixed_recipe_and_design_cannot_expand(packet):
    _, registration = packet
    for group in nonlinear_path.GROUPS:
        r = nonlinear_path.recipe(registration['design'],group,5)
        assert r['device']=='cuda' and r['max_iterations']==600
        assert len(r['features']) == (24 if group=='base' else 83)
        r['params']['eta'] = 99
        assert nonlinear_path.MODEL['params']['eta']==.03
    for key,value in [('recipes',{**nonlinear_path.MODEL,'device':'cpu'}),
                       ('fit_budget',{**nonlinear_path.BUDGET,'total':900})]:
        changed = copy.deepcopy(registration)
        changed['design'][key]=value
        changed['design_id']=content_id(changed['design'])
        with pytest.raises(ValueError,match='design changed'): nonlinear_path.validate_design(changed)


def test_prepare_is_colab_only_before_reading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(nonlinear_path,'require_colab_training',lambda: (_ for _ in ()).throw(RuntimeError('Colab only')))
    with pytest.raises(RuntimeError,match='Colab only'):
        nonlinear_path.prepare(tmp_path,tmp_path/'new',tmp_path/'packet')
    assert not (tmp_path/'new').exists()


def test_inner_stopping_uses_past_blocks_and_locks_median_count(packet):
    folder, registration = packet
    exp = Experiment(folder)
    spec = nonlinear_path.recipe(registration['design'],'base_plus_return_path',5)
    calls = []
    def fit(recipe, train, validation, test, role, iterations=1, repeat=None):
        assert recipe['device']=='cuda' and recipe['features']==spec['features']
        assert train.target_date_5.max() < test.date.min()
        if validation is not None:
            assert len(validation)==63 and validation.date.max()<rows_at(exp.history,fold['inner'][len(calls)]['origins']).date.min()
            counts = (5,7,9)
            calls.append(role)
            return {'result':{'iterations':counts[len(calls)-1]}}
        return {'result':{'predictions':[0.]*len(test)}}
    exp.fit = fit
    fold = registration['design']['split']['folds'][0]
    chosen = inner_price(exp,spec,fold)
    assert len(calls)==3 and chosen['iterations']==7 and chosen['weight']==0.


def test_interruption_resume_and_outside_iteration_lock(packet, monkeypatch):
    folder, _registration = packet
    exp = Experiment(folder)
    calls, pause = [], [True]
    def inner(experiment, spec, fold):
        if pause[0] and len(calls)==1: raise FitBudgetReached('Synthetic pause')
        calls.append((fold['year'],spec['horizon'],len(spec['features'])))
        return {'recipe':spec,'weight':.5,'inner_score':.999,'iterations':7}
    def predict(experiment,spec,origins,role,iterations):
        assert iterations==7 and spec['device']=='cuda'
        return np.full(len(origins),.0001)
    monkeypatch.setattr(path_pilot,'inner_price',inner)
    monkeypatch.setattr(path_pilot,'predict_chunks',predict)
    assert path_pilot.run(exp,15,**nonlinear_path.options())['status']=='planned_pause'
    pause[0]=False
    assert path_pilot.run(exp,15,**nonlinear_path.options())['saved_outputs']==32
    before=len(calls)
    assert path_pilot.run(exp,.00001,**nonlinear_path.options())['status']=='complete'
    assert len(calls)==before==32
    result=path_pilot.compare(folder,repetitions=50,**nonlinear_path.options())
    assert not result['release_allowed'] and result['status']=='complete'
    marker=next((folder/'nonlinear-path-outputs').glob('*.json'))
    marker.write_bytes(marker.read_bytes()+b'changed')
    with pytest.raises(ValueError): path_pilot.run(exp,15,**nonlinear_path.options())


def test_actual_frozen_budget_uses_early_stop_and_refit_jobs():
    # No model fit: use the pinned pre-2024 input's origin calendar.
    import pandas as pd
    root=Path(__file__).parents[2]/'output/reports/statistical-implementation-20261003/inputs-v1/reference'
    if not root.exists(): pytest.skip('Local frozen packet unavailable in CI')
    ready=read_record(root/'ready.json'); history=pd.read_parquet(root/'history.parquet')
    split=ready['identity']['split']
    refit=4*(sum(len(chunks(history,b['origins'])) for f in split['folds'] for b in f['inner'])
             +sum(len(chunks(history,f['origins'])) for f in split['folds']))
    early=4*sum(len(f['inner']) for f in split['folds'])
    assert refit==676 and early==96 and refit+early==772


def test_frozen_packet_prepares_same_origin_without_fitting(tmp_path,monkeypatch):
    repo=Path(__file__).parents[2]
    packet=repo/'output/reports/nonlinear-path-implementation-20261003/inputs-v2'
    if not packet.exists(): pytest.skip('Frozen research packet is local-only')
    monkeypatch.setattr(nonlinear_path,'require_colab_training',lambda:None)
    folder=tmp_path/'prepared'
    first=nonlinear_path.prepare(repo,folder,packet)
    second=nonlinear_path.prepare(repo,folder,packet)
    assert first==second
    assert first['identity']['execution']['device']=='cuda'
    assert first['identity']['design']['fit_budget']['total']==772
    assert not (folder/'ledger').exists()
    exp=Experiment(folder,repo=repo)
    assert set(first['identity']['design']['groups']['base_plus_return_path']).issubset(exp.history.columns)
    assert exp.history.date.max().year==2023 and not exp.history.target_date_5.ge('2024-01-01').any()
    other=read_record(packet/'reference/ready.json')
    assert first['identity']['split']==other['identity']['split']
