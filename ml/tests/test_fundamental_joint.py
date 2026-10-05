"""Synthetic contracts for joint information; no market fitting."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as runner
from cottonlens_ml.research import engine
from cottonlens_ml.research import fundamental_joint as joint
from cottonlens_ml.research.fixed_information import dispatch_fixed
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.research.protocol import mature, rows_at


@pytest.fixture
def packet(tmp_path):
    root=tmp_path/'packet'; parents,histories={},{}
    dates=pd.bdate_range('2010-01-04','2023-12-29')
    base=pd.DataFrame({'date':dates,'cotton_close':60.,'cotton_session_index':np.arange(len(dates))})
    for h in (1,5):
        base[f'target_date_{h}']=base.date.shift(-h)
        base[f'target_return_{h}']=.005
    for name in FEATURE_NAMES:
        base[name]=.1
    folds=[]
    for year,count in zip(range(2016,2024),(250,251,251,252,253,252,251,246),strict=True):
        outer=base.loc[base.date.dt.year.eq(year) & base.target_date_5.notna()].tail(count)
        before=mature(base,outer.date.min()).tail(189)
        inner=[{'origins':before.iloc[i*63:(i+1)*63].date.dt.strftime('%Y-%m-%d').tolist(),
                'cutoff':before.iloc[i*63].date.isoformat()} for i in range(3)]
        folds.append({'year':year,'fold':year-2015,'origins':outer.date.dt.strftime('%Y-%m-%d').tolist(),'inner':inner})
    files={}
    for kind,arm in joint.KINDS.items():
        folder=root/kind;folder.mkdir(parents=True)
        history=base.copy(); groups={'base':list(FEATURE_NAMES)}
        for lag in joint.LAGS:
            control=[*FEATURE_NAMES,*[f'{kind}_{v}_L{lag}' for v in joint.CONTROLS[kind]]]
            full=[*control,*[f'{kind}_{v}_L{lag}' for v in joint.VALUES[kind]]]
            groups[f'missing_L{lag}'],groups[f'{arm}_L{lag}']=control,full
            for name in full[24:]:history[name]=.2
        history.to_parquet(folder/'history.parquet',index=False)
        parent={'identity':{'profile':f'{kind}-exploration-v1','groups':groups,
            'split':{'folds':folds,'coverage_start':'2010-06-02','refit_cadence':21,'purge_observations':5},
            'release_allowed':False,'gate_evaluation_allowed':False},'history_sha256':digest(folder/'history.parquet')}
        freeze_record(folder/'ready.json',parent)
        parents[kind],histories[kind]=parent,history
        for name in ('ready.json','history.parquet'):files[f'{kind}/{name}']=digest(folder/name)
    freeze_record(root/'input-manifest.json',{'profile':joint.PROFILE,'files':files})
    return root,parents,histories


def test_same_origins_missing_inputs_and_explicit_interactions(packet):
    _,parents,histories=packet
    histories['nass'].loc[0,'nass_poor_very_poor_L1']=np.nan
    frame,split=joint.combine(parents,histories)
    assert len(frame)==len(histories['fas'])
    assert frame.loc[0,'joint_sales_under_poor_condition_L1'] is not None
    assert np.isnan(frame.loc[0,'joint_sales_under_poor_condition_L1'])
    assert frame.loc[1,'joint_sales_under_poor_condition_L1']==pytest.approx(.04)
    assert sum(len(f['origins']) for f in split['folds'])==2006
    groups,recipes=joint.groups_and_recipes(parents)
    assert [len(groups[g]) for g in ('timing_L1','joint_L1','interaction_L1')]==[31,46,49]
    assert all(len(r)==1 and r[0]['family']=='ridge' and r[0]['params']=={'alpha':1.} for r in recipes.values())


def test_future_source_values_cannot_change_past_joint_features(packet):
    _,parents,histories=packet
    expected,_=joint.combine(parents,histories)
    histories['fas'].loc[histories['fas'].date>='2020-01-01','fas_net_sales_change_L1']=999.
    actual,_=joint.combine(parents,histories)
    pd.testing.assert_frame_equal(expected.loc[expected.date<'2020-01-01'],actual.loc[actual.date<'2020-01-01'])


@pytest.mark.parametrize('damage',['calendar','label','feature_order','vintage','fraction','origin'])
def test_misaligned_or_unregistered_sources_rejected(packet,damage):
    _,parents,histories=packet
    if damage=='calendar':histories['nass']=histories['nass'].iloc[1:].reset_index(drop=True)
    if damage=='label':histories['weather'].loc[0,'target_return_5']=.8
    if damage=='feature_order':parents['fas']['identity']['groups']['sales_L1'].reverse()
    if damage=='vintage':parents['fas']['identity']['release_allowed']=True
    if damage=='fraction':histories['nass'].loc[0,'nass_poor_very_poor_L1']=2.
    if damage=='origin':parents['nass']['identity']['split']['folds']=[]
    with pytest.raises(ValueError):
        joint.combine(parents,histories)
        joint.groups_and_recipes(parents)


def prepare_fixture(tmp_path,packet):
    root,_,_=packet
    repo=tmp_path/'repo';(repo/'ml').mkdir(parents=True)
    (repo/'ml/uv.lock').write_text('synthetic-lock')
    folder=tmp_path/'experiment'
    ready=joint.prepare(repo,folder,root)
    return repo,folder,root,ready


def test_metadata_prepare_is_fixed_resumable_and_tier_a(tmp_path,packet):
    repo,folder,root,ready=prepare_fixture(tmp_path,packet)
    assert joint.prepare(repo,folder,root)==ready
    assert joint.validate(folder)==ready
    assert ready['identity']['fit_budget']['annual_outputs']==72
    assert ready['identity']['fit_budget']['total_refits']==1521
    assert not ready['identity']['release_allowed'] and not ready['identity']['gate_evaluation_allowed']
    assert not (folder/'ledger').exists()
    assert joint.compare(folder)['status']=='pending'


def outputs(folder,ready,bad_lag=False):
    history=pd.read_parquet(folder/'history.parquet')
    for group,recipes in ready['identity']['fixed_recipes'].items():
        for fold in ready['identity']['split']['folds']:
            score=.98 if group.startswith('interaction') else 1.
            if bad_lag and group=='interaction_L6':score=1.01
            choice={'selected':{'recipe':recipes[0],'weight':1.,'inner_score':score,'iterations':1},'selection_used_outer':False}
            name=f'{group}-{fold["year"]}.json';decision=folder/'information-decisions'/name
            freeze_record(decision,choice)
            frame=rows_at(history,fold['origins'])[['date','cotton_close','target_return_5']].copy()
            frame['date']=fold['origins'];frame['raw_predicted_return']=.0025 if group.startswith('interaction') else 0.
            frame['predicted_return']=frame.raw_predicted_return;frame['median_return'],frame['past_majority_sign']=.005,1.
            freeze_record(folder/'information-outputs'/name,{'group':group,'year':fold['year'],'recipe':recipes[0],
                'decision_sha256':digest(decision),'inner_score':score,'weight':1.,'gate_evaluated':False,
                'records':frame.to_dict(orient='records')})


@pytest.mark.parametrize('bad_lag',[False,True])
def test_both_controls_and_all_lag_stresses_required(tmp_path,packet,monkeypatch,bad_lag):
    _,folder,_,ready=prepare_fixture(tmp_path,packet)
    outputs(folder,ready,bad_lag)
    monkeypatch.setattr(runner,'paired_bootstrap',lambda *a,**k:{'synthetic_test':True})
    result=joint.compare(folder)
    assert result['research_priority_passed'] is (not bad_lag)
    assert result['release_allowed'] is False and result['gate_evaluated'] is False
    assert set(result['priority_checks']['1'])=={'timing_L1','joint_L1'}


@pytest.mark.parametrize('damage',['price','selection','recipe','extra'])
def test_completed_output_corruption_blocks_resume(tmp_path,packet,damage):
    _,folder,_,ready=prepare_fixture(tmp_path,packet)
    outputs(folder,ready)
    path=folder/'information-outputs/joint_L1-2016.json'; row=read_record(path)
    if damage=='price':row['records'][0]['cotton_close']=100.
    if damage=='selection':row['weight']=.5
    if damage=='recipe':row['recipe']['params']['alpha']=10.
    if damage=='extra':freeze_record(folder/'information-outputs/duplicate.json',row)
    else:
        path.unlink();freeze_record(path,row)
    with pytest.raises(ValueError):joint.validate(folder)


def test_tier_a_export_is_denied_without_reading_or_fitting(tmp_path):
    with pytest.raises(ValueError,match='cannot lock/export/search'):
        dispatch_fixed(SimpleNamespace(stage='export'),prepare=None,validate=None,compare=None)
    assert not list(tmp_path.iterdir())


def test_fresh_session_reuses_completed_metadata_without_payload_scan_or_fit(tmp_path,packet,monkeypatch,capsys):
    repo,folder,_,ready=prepare_fixture(tmp_path,packet)
    outputs(folder,ready)
    name='research-joint-test'; target=tmp_path/'drive/experiments'/name
    members=['ready.json','history.parquet']+[p.relative_to(folder).as_posix()
        for d in ('information-decisions','information-outputs') for p in (folder/d).glob('*.json')]
    Mirror(folder,target).publish_metadata(members)
    real_hydrate=Mirror.hydrate
    def metadata_only(self,**kwargs):
        assert kwargs.get('metadata_only') is True, 'Completed results must not rescan training packages'
        return real_hydrate(self,**kwargs)
    monkeypatch.setattr(Mirror,'hydrate',metadata_only)
    monkeypatch.setattr(engine,'Experiment',lambda root,**kwargs:SimpleNamespace(root=root,identity=ready['identity']))
    monkeypatch.setattr(runner,'inner_price',lambda *a,**k:pytest.fail('Completed decision refitted'))
    monkeypatch.setattr(runner,'predict_chunks',lambda *a,**k:pytest.fail('Completed prediction refitted'))
    args=SimpleNamespace(stage='pilot',drive_root=tmp_path/'fresh',mirror_root=tmp_path/'drive',
        experiment=name,repo=repo,max_minutes=1.)
    joint.dispatch(args)
    restored=tmp_path/'fresh/experiments'/name
    assert len(list((restored/'information-outputs').glob('*.json')))==72
    assert not (restored/'ledger/payloads').exists()
    assert 'CACHE timing_L1 year=2016' in capsys.readouterr().out


def test_full_restore_is_revalidated_before_any_training(tmp_path,packet,monkeypatch):
    repo,folder,_,ready=prepare_fixture(tmp_path,packet)
    local=tmp_path/'local'; name='research-joint-test'
    target=tmp_path/'drive/experiments'/name
    Mirror(folder,target).publish_metadata(['ready.json','history.parquet'])
    real_hydrate=Mirror.hydrate
    def restore_corrupt(self,**kwargs):
        if kwargs.get('metadata_only'):
            return real_hydrate(self,**kwargs)
        outputs(self.local,ready)
        path=self.local/'information-outputs/joint_L1-2016.json'
        row=read_record(path);row['records'][0]['cotton_close']=999.
        path.unlink();freeze_record(path,row)
    monkeypatch.setattr(Mirror,'hydrate',restore_corrupt)
    monkeypatch.setattr(engine,'Experiment',lambda *a,**k:pytest.fail('Corrupt restoration reached training'))
    with pytest.raises(ValueError,match='output/origins changed'):
        joint.dispatch(SimpleNamespace(stage='pilot',drive_root=local,mirror_root=tmp_path/'drive',
            experiment=name,repo=repo,max_minutes=1.))
