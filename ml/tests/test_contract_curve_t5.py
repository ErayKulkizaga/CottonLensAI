"""T5 is a new frozen horizon, not a renamed T1 checkpoint or an oracle feature."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_contract_curve as t1_tests
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import contract_curve as curve
from cottonlens_ml.research import contract_curve_diagnostic as diagnostic
from cottonlens_ml.research import contract_curve_execution as execute
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research import wasde_regional_execution as shared
from cottonlens_ml.research.ledger import FitBudgetReached, freeze_record, read_record

design = t1_tests.design  # pytest discovers the shared synthetic fixture


@pytest.fixture
def t5(design):
    expanded, old = design
    parent = expanded[[c for c in expanded if not c.startswith(('curve_', 'mask_', 'numeric_'))]].copy()
    quotes = [t1_tests.report(str(d.date()), price=71. + np.sin(j))
              for j, d in enumerate(parent.date[parent.date >= '2020-01-01'])]
    new, spec, _ = curve.make_design(parent, quotes, profile=curve.T5_PROFILE)
    pd.testing.assert_frame_equal(new, expanded, check_exact=True)
    assert spec['split'] == old['split'] and spec['feature_data_id'] == old['feature_data_id']
    assert spec['groups'] == old['groups'] and spec['recipe'] == {**old['recipe'], 'horizon': 5}
    assert spec['rules']['practical_goal'] == {'naive_gain_pct': 5, 'direction_pct': 55, 'year_wins': '6/8'}
    assert spec['fit_budget']['total'] == 252 and spec['fit_budget']['prediction_rows'] == 2996
    return new, spec


def test_exact_horizon_recipe_and_no_oracle_feature(t5, design):
    _, new = t5
    _, old = design
    assert execute.recipe(new, 'numeric_D0', 5)['horizon'] == 5
    assert execute.profile_settings(curve.T5_PROFILE)[1] != execute.profile_settings(curve.PROFILE)[1]
    for spec, wrong in ((old, 5), (new, 1)):
        with pytest.raises(ValueError, match='exact preregistered'):
            execute.recipe(spec, 'numeric_D0', wrong)
    with pytest.raises(ValueError):
        curve.definition('unregistered')
    contaminated = copy.deepcopy(new)
    contaminated['groups']['numeric_D0'].append('future_first_contract')
    with pytest.raises(ValueError):
        execute.recipe(contaminated, 'numeric_D0', 5)


def test_t5_output_labels_not_t1(t5):
    history, _spec = t5
    test = history.iloc[1000:1003].copy()
    test['target_return_1'] = 99.
    test['target_date_1'] = pd.Timestamp('2099-01-01')
    payload = pd.DataFrame({'date':test.date.dt.strftime('%Y-%m-%d')}, index=test.index)
    saved = execute.record_frame(test, payload, 'numeric_D0', 5, {'weight': .25})
    np.testing.assert_array_equal(saved.actual_return, test.target_return_5)
    assert saved.target_date.tolist() == test.target_date_5.dt.strftime('%Y-%m-%d').tolist()
    assert saved.horizon.eq(5).all() and 'future_first_contract' not in saved


def test_t5_pause_resume_keeps_full_h5_origins(t5, tmp_path, monkeypatch):
    history, spec = t5
    calls = []
    def select(exp, recipe, fold):
        calls.append(recipe['horizon'])
        if len(calls) == 2:
            raise FitBudgetReached('Synthetic planned pause')
        return {'recipe':recipe,'weight':.25,'iterations':1,'inner_score':1.}
    monkeypatch.setattr(execute, 'verify_execution', lambda root: (None,history))
    monkeypatch.setattr(execute, 'learning_control', lambda exp: None)
    monkeypatch.setattr(path_pilot, 'inner_price', select)
    monkeypatch.setattr(path_pilot, 'predict_chunks', lambda *a: np.full(len(a[2]),.001))
    exp = SimpleNamespace(root=tmp_path,history=history,
        identity={'profile':curve.T5_PROFILE,'design':spec,'design_id':content_id(spec),'split':spec['split']},
        train_rows=lambda cutoff, recipe: history.loc[history.target_date_5 < cutoff])
    assert execute.run(exp,30)['status'] == 'planned_pause'
    assert execute.run(exp,30)['saved_outputs'] == 12
    execute.run(exp,30)
    assert len(calls) == 13 and set(calls) == {5}
    paths = list((tmp_path/'contract-curve-t5-outputs').glob('*.json'))
    assert len(paths) == 12 and not (tmp_path/'contract-curve-outputs').exists()
    assert sum(len(read_record(p)['records']) for p in paths) == 2996
    for p in paths:
        assert '-t5-' in p.name and read_record(p)['horizon'] == 5


def test_shared_comparison_cannot_default_t5_to_t1(t5, tmp_path):
    history, spec = t5
    ready = {'identity':{'design':spec}}
    with pytest.raises(ValueError,match='Comparison horizon'):
        shared.compare(tmp_path,verify_fn=lambda root:(ready,history))


def test_cli_profile_cannot_resume_t1_root(tmp_path):
    folder = tmp_path/'experiments/research-run'
    freeze_record(folder/'ready.json', {'identity':{'profile':curve.PROFILE}})
    args = SimpleNamespace(drive_root=tmp_path,experiment='research-run',profile=curve.T5_PROFILE)
    with pytest.raises(ValueError,match='CLI profile'):
        execute.dispatch(args)


def test_parent_identity_cannot_change(t5, design):
    _, spec = t5
    _, old = design
    proof = {'profile':curve.PROFILE,'status':'completed_assumption_sensitivity','registration_id':'frozen',
             'source_id':'old-source'}
    ready = {'history_sha256':'history','identity':{'profile':curve.PROFILE,'registration_id':'frozen',
        'source_id':'old-source','split':old['split'],'research_data_id':old['feature_data_id'],'design':old}}
    audit = {'parent_registration_id':'frozen','inputs_sha256':{'history':'history'},'new_fits':0,
             'market_skill_claimed':False,'historical_source_admitted':False}
    good = curve.bind_t5_parent(copy.deepcopy(spec),ready,proof,audit)
    assert good['parent_t1_registration_id'] == 'frozen'
    bad = copy.deepcopy(ready)
    bad['identity']['split']['folds'][0]['origins'].pop()
    with pytest.raises(ValueError,match='preserve completed T1'):
        curve.bind_t5_parent(copy.deepcopy(spec),bad,proof,audit)
    with pytest.raises(ValueError):
        curve.bind_t5_parent(copy.deepcopy(spec),ready,proof,{**audit,'market_skill_claimed':True})


def test_ex_post_strata_never_change_primary_or_predictions():
    anchor = pd.DataFrame({'date':['2023-01-05','2023-01-06','2023-01-09'],
        'target_date':['2023-01-12','2023-01-13','2023-01-16'],'horizon':5,
        'cotton_close':70.,'actual_return':[.01,-.02,.03],
        'predicted_return':0.,'raw_predicted_return':0.})
    frames = {g:anchor.copy() for g in curve.groups()}
    for g in ('numeric_D0','numeric_D1'):
        frames[g]['predicted_return'] = [.005,-.003,0.]
        frames[g]['raw_predicted_return'] = [.01,-.006,0.]
    before = {g:f.copy(deep=True) for g,f in frames.items()}
    panel = {'rows':[{'report_date':d,'first_contract':n} for d,n in
        [('2023-01-05','Mar-23'),('2023-01-06','Mar-23'),('2023-01-12','May-23'),('2023-01-13','Mar-23')]]}
    saved = diagnostic.summarize(frames,panel)
    assert saved['counts'] == {'transition':1,'unchanged':1,'unknown':1}
    assert saved['new_fits'] == 0 and not saved['selection_used'] and not saved['feature_used']
    assert saved['confidence_intervals'] is None
    for g, frame in frames.items():
        pd.testing.assert_frame_equal(before[g],frame,check_exact=True)
    changed = copy.deepcopy(panel)
    changed['rows'][2]['first_contract'] = 'Mar-23'
    alternate = diagnostic.summarize(frames,changed)
    assert alternate['counts'] == {'transition':0,'unchanged':2,'unknown':1}
    for k in saved['paired']:
        assert saved['paired'][k]['full_paired_error_sum'] == alternate['paired'][k]['full_paired_error_sum']
    frames['numeric_D0'].loc[0,'target_date'] = '2023-01-11'
    with pytest.raises(ValueError):
        diagnostic.summarize(frames,panel)


def test_t1_synthetic_control_cannot_prove_t5(tmp_path, monkeypatch):
    record = {'identity':{'source_id':'same-source'},'files':{'model.txt':'hash'},
              'specification':{'recipe':{'horizon':1}}}
    control = {'fit_record':'control','policy':shared.VALIDATION_POLICY,'market_evidence':False,
               'status':'passed','controls':{'ridge/known_signal':{'relative_mae_gain':.99}}}
    monkeypatch.setattr(shared,'read_record',lambda p:control if p.name=='learning-control.json' else record)
    with pytest.raises(ValueError,match='Learning evidence'):
        shared.verify_learning_evidence(tmp_path,{'source_id':'same-source'},horizon=5)
