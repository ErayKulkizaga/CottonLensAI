"""Metadata, synthetic orchestration and Tier-A contracts; no market training."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as runner
from cottonlens_ml.research import wasde_narrative as pilot
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import mature, rows_at


@pytest.fixture
def frozen(tmp_path):
    repo, reference, folder = (tmp_path/name for name in ('repo','reference','experiment'))
    (repo/'ml').mkdir(parents=True)
    (repo/'ml/uv.lock').write_text('synthetic-lock')
    dates = pd.bdate_range('2010-01-04','2023-12-29')
    history = pd.DataFrame({'date':dates,'cotton_close':60.,'cotton_session_index':np.arange(len(dates))})
    for h in (1,5):
        history[f'target_date_{h}'] = history.date.shift(-h)
        history[f'target_return_{h}'] = .005
    for name in FEATURE_NAMES:
        if name not in history:
            history[name] = .1
    groups = {'base':list(FEATURE_NAMES)}
    for lag in pilot.LAGS:
        numeric = [f'wasde_age_L{lag}',f'wasde_missing_L{lag}',f'stock_L{lag}',f'production_L{lag}',f'use_L{lag}']
        groups[f'balance_L{lag}'] = list(FEATURE_NAMES)+numeric
        for name in numeric:
            history[name] = .2
    folds = []
    for year,count in zip(range(2019,2024),(252,253,252,251,246),strict=True):
        outer = history.loc[history.date.dt.year.eq(year) & history.target_date_5.notna()].tail(count)
        preceding = mature(history,outer.date.min()).tail(189)
        inner = []
        for i in range(3):
            block = preceding.iloc[i*63:(i+1)*63]
            inner.append({'origins':block.date.dt.strftime('%Y-%m-%d').tolist(),'cutoff':block.date.min().isoformat()})
        folds.append({'year':year,'fold':year-2018,'origins':outer.date.dt.strftime('%Y-%m-%d').tolist(),'inner':inner})
    reference.mkdir()
    history.to_parquet(reference/'history.parquet',index=False)
    parent = {'identity':{'profile':'wasde-exploration-v1','groups':groups,
                         'split':{'folds':folds,'coverage_start':'2016-01-04','purge_observations':5,'refit_cadence':21}},
              'history_sha256':digest(reference/'history.parquet')}
    freeze_record(reference/'ready.json',parent)
    corpus = tmp_path/'corpus.json'
    records = [{'report_day':day.strftime('%Y-%m-%d'),'narrative':f'Cotton supply demand drought texas india china brazil pakistan rain outlook stocks consumption exports production {("flood", "tariff", "weather", "planting", "harvest", "textiles")[i%6]} report {i}',
                'narrative_id':content_id(f'Cotton supply demand drought texas india china brazil pakistan rain outlook stocks consumption exports production {("flood", "tariff", "weather", "planting", "harvest", "textiles")[i%6]} report {i}')}
               for i,day in enumerate(pd.date_range('2016-01-01','2023-12-01',freq='MS'))]
    freeze_record(corpus,{'kind':'wasde-cotton-narrative-corpus-v1','source_tier':'A_exploration_only',
        'model_eligible':False,'release_allowed':False,'publication_timestamp_verified':False,'records':records})
    ready = pilot.prepare(repo,folder,reference,corpus)
    return repo,folder,reference,corpus,ready


def replace(path,body):
    path.unlink()
    freeze_record(path,body)


def fill_outputs(folder,ready,*,bad_lag=False):
    history = pd.read_parquet(folder/'history.parquet')
    for group in ready['identity']['groups']:
        recipe = ready['identity']['fixed_recipes'][group][0]
        for fold in ready['identity']['split']['folds']:
            score = .98 if group.startswith('text') else .99 if group.startswith('numeric') else 1.
            if bad_lag and group=='text_L6':
                score = 1.01
            choice = {'selected':{'recipe':recipe,'weight':1.,'iterations':1,'inner_score':score},'selection_used_outer':False}
            name = f'{group}-{fold["year"]}.json'
            decision = folder/'information-decisions'/name
            freeze_record(decision,choice)
            frame = rows_at(history,fold['origins'])[['date','cotton_close','target_return_5']].copy()
            frame['date'] = fold['origins']
            frame['raw_predicted_return'] = .0025 if group.startswith('text') else 0.
            frame['predicted_return'] = frame.raw_predicted_return
            frame['median_return'],frame['past_majority_sign'] = .005,1.
            freeze_record(folder/'information-outputs'/name,{'group':group,'year':fold['year'],'recipe':recipe,
                'decision_sha256':digest(decision),'weight':1.,'inner_score':score,'gate_evaluated':False,
                'records':frame.to_dict(orient='records')})


def test_prepare_preserves_cohort_and_fits_no_transform(frozen,monkeypatch):
    from cottonlens_ml.research.text_adapter import TextPreprocessor
    monkeypatch.setattr(TextPreprocessor,'fit',lambda *a:pytest.fail('prepare must not fit'))
    repo,folder,reference,corpus,ready = frozen
    identity = ready['identity']
    assert pilot.prepare(repo,folder,reference,corpus)==ready
    assert identity['split']==read_record(reference/'ready.json')['identity']['split']
    assert identity['fit_budget']=={'annual_outputs':45,'inner_refits':405,'outer_refits':549,'total_refits':954}
    assert identity['text_preflight']['minimum_unique_mature_documents']>=12
    assert identity['text_preflight']['learned_transform_fitted'] is False
    assert [len(identity['groups'][name]) for name in ('base_L1','numeric_L1','text_L1')]==[26,31,39]
    assert not identity['release_allowed'] and not identity['gate_evaluation_allowed']
    assert not list((folder/'ledger').glob('completed/*'))
    history = pd.read_parquet(folder/'history.parquet')
    lag6 = history.wasde_narrative_id_L6.isna()
    assert lag6.any() and len(history)==len(pd.read_parquet(reference/'history.parquet'))
    assert all(r[0]['params']=={'alpha':1.} and r[0]['device']=='cpu' for r in identity['fixed_recipes'].values())


def test_changed_corpus_cannot_resume_or_overwrite(frozen):
    repo,folder,reference,corpus,ready = frozen
    before = digest(folder/'history.parquet')
    body = read_record(corpus)
    body['records'][-1]['narrative'] = 'Changed future report'
    body['records'][-1]['narrative_id'] = content_id('Changed future report')
    replace(corpus,body)
    with pytest.raises(ValueError,match='new namespace'):
        pilot.prepare(repo,folder,reference,corpus)
    assert digest(folder/'history.parquet')==before==ready['history_sha256']


@pytest.mark.parametrize('damage',['origin','recipe','decision_missing','snapshot','weight'])
def test_completed_evidence_corruption_rejected(frozen,damage):
    _,folder,_,_,ready = frozen
    fill_outputs(folder,ready)
    pilot.verify_cached(folder)
    path = folder/'information-outputs/base_L1-2019.json'
    row = read_record(path)
    if damage=='snapshot':
        (folder/'history.parquet').write_bytes(b'corrupt')
    elif damage=='decision_missing':
        (folder/'information-decisions/base_L1-2019.json').unlink()
    else:
        if damage=='origin':row['records'][0]['date']='2099-01-01'
        if damage=='recipe':row['recipe']['params']['alpha']=10
        if damage=='weight':row['records'][0]['predicted_return']=1.
        replace(path,row)
    with pytest.raises(ValueError):
        pilot.verify_cached(folder)


@pytest.mark.parametrize('bad_lag',[False,True])
def test_compare_both_controls_and_all_lag_stresses_required(frozen,monkeypatch,bad_lag):
    _,folder,_,_,ready = frozen
    assert pilot.compare(folder)['status']=='pending'
    fill_outputs(folder,ready,bad_lag=bad_lag)
    monkeypatch.setattr(runner,'paired_bootstrap',lambda *a,**k:{'synthetic_test':True})
    result = pilot.compare(folder)
    assert result['status']=='complete' and result['origin_count']==1254
    assert result['research_priority_passed'] is (not bad_lag)
    assert set(result['priority_checks'])=={'1','2','6'}
    assert set(result['priority_checks']['1'])=={'base_L1','numeric_L1'}
    assert 'vs_numeric_L1' in result['groups']['text_L1']
    assert result['release_allowed'] is False and result['gate_evaluated'] is False


def test_shared_runner_uses_only_registered_recipe_and_cached_result(tmp_path,monkeypatch):
    calls=[]
    history=pd.DataFrame({'date':pd.to_datetime(['2020-01-03']),'cotton_close':[60.],'target_return_5':[.005]})
    recipe={'family':'ridge','params':{'alpha':1},'task':'price'}
    exp=SimpleNamespace(root=tmp_path,history=history,identity={'groups':{'text_L1':['x']},
        'fixed_recipes':{'text_L1':[recipe]},'split':{'folds':[{'year':2020,'origins':['2020-01-03']}]}})
    monkeypatch.setattr(runner,'specs',lambda *a:pytest.fail('legacy two-model search forbidden'))
    def inner(*args):
        calls.append('inner')
        return {'recipe':args[1],'weight':0.,'inner_score':1.,'iterations':1}
    def outer(*args):
        calls.append('outer')
        return np.zeros(1)
    monkeypatch.setattr(runner,'inner_price',inner)
    monkeypatch.setattr(runner,'predict_chunks',outer)
    assert runner.run(exp,1)['status']=='complete'
    assert runner.run(exp,1)['status']=='complete'
    assert calls==['inner','outer']


def test_narrative_cannot_export_or_launch_wide_search(tmp_path):
    for stage in ('export','lock','search'):
        with pytest.raises(ValueError,match='cannot lock/export/search'):
            pilot.dispatch(SimpleNamespace(stage=stage,drive_root=tmp_path,experiment='research-fixture'))
    assert not list(tmp_path.iterdir())


def test_text_fit_resumes_from_same_ledger_and_corrupt_payload_is_rejected(frozen,monkeypatch):
    from cottonlens_ml.research import models
    from cottonlens_ml.research.engine import Experiment
    _,folder,_,_,ready = frozen
    experiment = Experiment(folder)
    spec = ready['identity']['fixed_recipes']['text_L1'][0]
    fold = ready['identity']['split']['folds'][0]
    test = rows_at(experiment.history,fold['inner'][0]['origins']).iloc[:2]
    train = experiment.train_rows(test.date.min(),spec)
    assert train.target_date_5.max()<test.date.min()
    monkeypatch.setattr(models,'require_training',lambda *a:None)  # Synthetic fixture only.
    first = experiment.fit(spec,train,None,test,'synthetic-text-resume',iterations=1)
    from cottonlens_ml.research import engine
    monkeypatch.setattr(engine,'fit_predict',lambda *a,**k:pytest.fail('completed fit must not retrain'))
    second = experiment.fit(spec,train,None,test,'synthetic-text-resume',iterations=1)
    assert first['experiment_id']==second['experiment_id']
    assert first['result']['predictions']==second['result']['predictions']
    model_path = next(folder/'ledger'/name for name in first['files'] if name.endswith('model.json'))
    model_path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt completed payload'):
        experiment.fit(spec,train,None,test,'synthetic-text-resume',iterations=1)


def test_new_runtime_compares_metadata_without_restoring_model_payloads(frozen,tmp_path,monkeypatch):
    from cottonlens_ml.research.mirror import Mirror
    _,folder,_,_,ready = frozen
    fill_outputs(folder,ready)
    remote = tmp_path/'remote'
    target = remote/'experiments/research-text-fixture'
    members = ['ready.json','history.parquet','preregistered.json']
    members += [p.relative_to(folder).as_posix() for directory in ('information-decisions','information-outputs')
                for p in (folder/directory).glob('*.json')]
    Mirror(folder,target).publish_metadata(members)
    monkeypatch.setattr(runner,'paired_bootstrap',lambda *a,**k:{'synthetic_test':True})
    local = tmp_path/'fresh-local'
    pilot.dispatch(SimpleNamespace(stage='compare',drive_root=local,mirror_root=remote,
                                  experiment='research-text-fixture'))
    restored = local/'experiments/research-text-fixture'
    assert read_record(restored/'reports/wasde-text-decision.json')['status']=='complete'
    assert len(list((restored/'information-outputs').glob('*.json')))==45
    assert not (restored/'ledger/payloads').exists()
