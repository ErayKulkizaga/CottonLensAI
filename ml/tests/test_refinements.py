"""Generated fold-local choices/OOF predictions; no estimator or market fits."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from cottonlens_ml.research import refinements
from cottonlens_ml.research.decision_cache import execution_binding, read_selection
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.release import component_seeds
from cottonlens_ml.research.uncertainty import inner_predictions


def experiment(root, changed=False):
    dates = pd.bdate_range('2000-01-03', periods=800)
    frame = pd.DataFrame({'date': dates, 'cotton_close': 80., 'x': 1.,
                          'target_return_1': .01, 'target_return_5': .05,
                          'target_date_1': pd.Series(dates).shift(-1),
                          'target_date_5': pd.Series(dates).shift(-5)})
    if changed:
        frame.loc[600:, ['cotton_close', 'x']] = [999., 999.]
    folds = [{'fold': n+1, 'origins': dates[start:start+30].strftime('%Y-%m-%d').tolist(),
              'inner': [{'origins': dates[i:i+63].strftime('%Y-%m-%d').tolist()}
                        for i in starts]} for n, (start, starts) in enumerate(
                            ((300, (20, 83, 146)), (600, (320, 383, 446))))]
    exp = SimpleNamespace(root=root, history=frame,
                          identity={'groups': {'expanded': ['x'], 'base': ['x']},
                                    'split': {'folds': folds}})
    seen = []
    def decision(spec, fold, score):
        origins = [d for block in fold['inner'] for d in block['origins']]
        return {'recipe': spec, 'score': score, 'seeds': [
            {'recipe': spec, 'iterations': 1, 'origins': origins, 'predictions': [.005]*189}]}
    for fold in folds:
        for h in (1, 5):
            for family in ('xgboost', 'catboost', 'lstm'):
                score = {'xgboost': .9, 'catboost': .95, 'lstm': .92}[family]
                if changed and fold['fold'] == 2:
                    score = {'xgboost': 2., 'catboost': .1, 'lstm': .4}[family]
                namespace = 'C' if family == 'lstm' else 'A'
                spec = {'family': family, 'horizon': h, 'features': ['x'],
                        'params': {}, 'seed': 42, 'task': 'price'}
                request = {'operation': 'tune', 'family': family, 'horizon': h, 'namespace': namespace}
                freeze_record(root/'decisions'/namespace/f'{family}-t{h}-fold{fold["fold"]}.json',
                    {'chosen': decision(spec, fold, score), 'execution_request': request,
                     'selection_used_outer': False,
                     'execution_binding': execution_binding(exp, fold, request, selection=True)})
    def tune(family, h, fold, candidates, namespace):
        seen.append((h, fold['fold'], family, namespace))
        return {'chosen': decision(candidates[0], fold, .8)}
    def outer(chosen, fold, namespace, **kwargs):
        return {'origins': fold['origins'], 'predictions': [.005]*len(fold['origins'])}
    exp.tune, exp.outer = tune, outer
    return exp, seen


@pytest.mark.parametrize('round_name', ['D', 'E'])
def test_later_fold_scores_and_future_rows_cannot_change_earlier_choice(tmp_path, round_name):
    first, calls = experiment(tmp_path/'original')
    changed, changed_calls = experiment(tmp_path/'changed', changed=True)
    refinements.run_refinement(first, round_name)
    refinements.run_refinement(changed, round_name)
    assert [c for c in calls if c[1] == 1] == [c for c in changed_calls if c[1] == 1]
    assert [c for c in calls if c[1] == 2] != [c for c in changed_calls if c[1] == 2]
    original_records = sorted((first.root/'decisions').glob('*policy-v2/*fold1.json'))
    assert original_records
    for p in original_records:
        assert read_record(p) == read_record(changed.root/p.relative_to(first.root))


def test_policy_wrapper_retains_native_leaf_and_oof_contract(tmp_path):
    exp, _ = experiment(tmp_path)
    refinements.run_refinement(exp, 'D')
    path = next((tmp_path/'decisions').glob('*tabular-policy-v2/*t1-fold1.json'))
    chosen = read_selection(exp, path, exp.identity['split']['folds'][0],
                            path.parent.name, 'ensemble', 1)['chosen']
    assert chosen['weights'] == [1.]
    assert refinements.leaf_recipe(chosen)['family'] == 'xgboost'
    native = component_seeds(chosen)
    assert len(native) == 1 and native[0][1] == 1.
    assert len(inner_predictions(chosen)[0]) == 189


def test_invalid_temporal_fold_stops_before_any_refinement_work(tmp_path):
    exp, _ = experiment(tmp_path)
    exp.identity = deepcopy(exp.identity)
    exp.identity['split']['folds'][0]['inner'][0]['origins'][0] = '2099-01-01'
    exp.tune = lambda *a, **k: pytest.fail('Malformed selection reached computation')
    with pytest.raises(ValueError):
        refinements.run_refinement(exp, 'D')


def test_direct_refinement_ensemble_rejects_rehashed_outer_label_substitution(tmp_path, monkeypatch):
    import json

    from cottonlens_ml.cohort import content_id
    exp, _ = experiment(tmp_path)
    path = tmp_path/'decisions/A/xgboost-t1-fold1.json'
    body = read_record(path)
    body['chosen']['seeds'][0]['origins'] = exp.identity['split']['folds'][0]['origins']
    path.write_text(json.dumps({**body, 'record_id': content_id(body)}))
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid OOF origins reached predictions or weight optimization')
    exp.outer = forbidden
    monkeypatch.setattr('cottonlens_ml.research.free_rounds.simplex_weights', forbidden)
    with pytest.raises(ValueError, match='OOF cohort'):
        refinements.ensemble(exp, 1)


def test_nonpositive_fold_stops_both_native_refinement_and_ensemble(tmp_path):
    import json

    from cottonlens_ml.cohort import content_id
    exp, seen = experiment(tmp_path)
    for path in (tmp_path/'decisions').glob('*/*fold1.json'):
        body = read_record(path)
        body['chosen']['score'] = 1.1
        path.write_text(json.dumps({**body, 'record_id': content_id(body)}))
    original = exp.outer
    def outer(chosen, fold, namespace, **kwargs):
        if fold['fold'] == 1:
            pytest.fail('Nonpositive fold reached ensemble computation')
        return original(chosen, fold, namespace, **kwargs)
    exp.outer = outer
    refinements.run_refinement(exp, 'E')
    assert all(c[1] != 1 for c in seen)
    assert not (tmp_path/'outer/E-ensemble-v2/ensemble-t1-fold1.json').exists()
    assert (tmp_path/'refinement-v2/skipped-t1-fold1.json').exists()
