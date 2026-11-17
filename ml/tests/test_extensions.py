"""Fold-local extension routing with generated immutable callbacks only."""
import json

import pytest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import engine, extensions
from cottonlens_ml.research.decision_cache import execution_binding
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.search import BUDGETS, SEEDS, recipes

from ml.tests.test_refinements import experiment as base_experiment


def fixture(root, changed=False, stop=False, zero=False, round_name='A'):
    exp, _ = base_experiment(root, changed)
    calls = []
    for fold in exp.identity['split']['folds']:
        for h in (1, 5):
            for family in BUDGETS[round_name]:
                score = {'xgboost': .9, 'catboost': .95, 'ridge': .97, 'elasticnet': .98,
                         'mlp': .9, 'lstm': .95, 'tcn': .98}[family]
                if changed and fold['fold'] == 2:
                    score = {'xgboost': 2., 'catboost': .1, 'ridge': .05, 'elasticnet': .02,
                             'mlp': 2., 'lstm': .02, 'tcn': .01}[family]
                if stop and fold['fold'] == 1:
                    score = 0. if zero else 1.1
                candidates = recipes(family, h, ['x'], BUDGETS[round_name][family])
                save_decision(exp, family, h, fold, candidates, round_name, score, replace=True)
                save_outer(exp, read_record(root/'decisions'/round_name/f'{family}-t{h}-fold{fold["fold"]}.json')['chosen'],
                           fold, round_name)
    def tune(family, h, fold, candidates, name):
        calls.append((fold['fold'], h, family, len(candidates)))
        return save_decision(exp, family, h, fold, candidates, name, .8)
    exp.tune = tune
    exp.outer = lambda chosen, fold, name: save_outer(exp, chosen, fold, name)
    freeze_record(root/'diagnosis.json', {'status': 'passed'})
    return exp, calls


def save_decision(exp, family, h, fold, candidates, name, score, replace=False):
    spec = candidates[0]
    dates = [d for b in fold['inner'] for d in b['origins']]
    chosen = {'recipe': spec, 'score': score, 'seeds': [
        {'recipe': spec, 'iterations': 1, 'origins': dates, 'predictions': [.005]*189}]}
    request = {'operation': 'tune', 'family': family, 'horizon': h,
               'namespace': name, 'candidates': candidates, 'seeds': list(SEEDS)}
    result = {'chosen': chosen, 'execution_request': request, 'selection_used_outer': False,
              'execution_binding': execution_binding(exp, fold, request, selection=True)}
    path = exp.root/'decisions'/name/f'{family}-t{h}-fold{fold["fold"]}.json'
    if replace and path.exists():
        path.unlink()  # Generated fixture only.
    freeze_record(path, result)
    return result


def save_outer(exp, chosen, fold, name):
    spec = chosen['recipe']
    request = {'operation': 'outer', 'chosen': chosen, 'namespace': name, 'repeat': None}
    freeze_record(exp.root/'outer'/name/f'{spec["family"]}-t{spec["horizon"]}-fold{fold["fold"]}.json',
                  {'recipe': spec, 'origins': fold['origins'], 'fold': fold['fold'],
                   'predictions': [.005]*len(fold['origins']), 'metrics': {'mae': 1.},
                   'inner_score': chosen['score'],
                   'execution_binding': execution_binding(exp, fold, request, selection=False)})


def forbidden(*args, **kwargs):
    pytest.fail('Completed or stopped work reached computation')


@pytest.mark.parametrize('round_name', ['A', 'C'])
def test_later_fold_data_and_scores_cannot_change_first_extension(tmp_path, round_name):
    first, calls = fixture(tmp_path/'first', round_name=round_name)
    changed, later_calls = fixture(tmp_path/'changed', changed=True, round_name=round_name)
    extensions.extend(first, round_name, 1)
    extensions.extend(changed, round_name, 1)
    assert [c for c in calls if c[0] == 1] == [c for c in later_calls if c[0] == 1]
    assert [c for c in calls if c[0] == 2] != [c for c in later_calls if c[0] == 2]
    for path in (first.root/'decisions'/f'{round_name}-extension-1-policy-v2').glob('*fold1.json'):
        assert read_record(path) == read_record(changed.root/path.relative_to(first.root))


@pytest.mark.parametrize('zero', [False, True])
def test_stopped_fold_carries_cached_baseline_without_computation(tmp_path, zero):
    exp, calls = fixture(tmp_path, stop=True, zero=zero)
    extensions.extend(exp, 'A', 1)
    assert all(c[0] != 1 for c in calls)
    record = read_record(tmp_path/'extensions/fold-local-v2/A-1-fold1.json')
    assert record['action'] == 'STOP'
    output = read_record(tmp_path/'outer/A-extension-1-policy-v2/ensemble-t1-fold1.json')
    assert output['predictions'] == [.005]*30
    exp.tune = exp.outer = forbidden
    extensions.extend(exp, 'A', 1)


def test_completed_receipts_never_refit_and_ignore_legacy_global_receipt(tmp_path):
    exp, _ = fixture(tmp_path)
    freeze_record(tmp_path/'extensions/A-1.json', {'legacy_global_gain': 999})
    original = (tmp_path/'extensions/A-1.json').read_bytes()
    extensions.extend(exp, 'A', 1)
    exp.tune = exp.outer = forbidden
    extensions.extend(exp, 'A', 1)
    assert (tmp_path/'extensions/A-1.json').read_bytes() == original


@pytest.mark.parametrize('fault', ['receipt', 'source', 'candidate'])
def test_changed_context_or_rehashed_receipt_rejected_before_fit(tmp_path, fault):
    exp, _ = fixture(tmp_path)
    extensions.extend(exp, 'A', 1)
    if fault == 'source':
        exp.history.loc[0, 'x'] += 1
    else:
        path = (tmp_path/'extensions/fold-local-v2/A-1-fold1.json' if fault == 'receipt'
                else tmp_path/'decisions/A/xgboost-t1-fold1.json')
        body = read_record(path)
        if fault == 'receipt':
            body['relative_improvement'] = 999
        else:
            body['execution_request']['candidates'] = []
        path.write_text(json.dumps({**body, 'record_id': content_id(body)}))
    exp.tune = exp.outer = forbidden
    with pytest.raises(ValueError):
        extensions.extend(exp, 'A', 1)


def test_skipped_extension_prefix_is_not_silently_computed(tmp_path):
    exp, _ = fixture(tmp_path)
    exp.tune = exp.outer = forbidden
    with pytest.raises(ValueError, match='Previous fold extension incomplete'):
        extensions.extend(exp, 'A', 2)


def test_engine_routes_extension_without_global_compare(tmp_path, monkeypatch):
    exp, calls = fixture(tmp_path)
    monkeypatch.setattr(engine, 'require_colab_training', lambda: None)
    monkeypatch.setattr(engine, 'compare', forbidden)
    engine.search(exp, 'A', extension=1)
    assert calls


def test_two_zero_gains_stop_further_extensions_without_computation(tmp_path):
    exp, calls = fixture(tmp_path)
    for index in (1, 2, 3):
        extensions.extend(exp, 'A', index)
    count = len(calls)
    exp.tune = exp.outer = forbidden
    for index in (4, 5):
        extensions.extend(exp, 'A', index)
        for fold in (1, 2):
            assert read_record(tmp_path/'extensions/fold-local-v2'/f'A-{index}-fold{fold}.json')['action'] == 'STOP'
    assert len(calls) == count


def test_missing_derived_policy_is_restored_from_native_cache_without_fit(tmp_path):
    exp, _ = fixture(tmp_path)
    extensions.extend(exp, 'A', 1)
    path = tmp_path/'outer/A-extension-1-policy-v2/ensemble-t1-fold1.json'
    original = read_record(path)
    path.unlink()  # Generated derived fixture only.
    exp.tune = exp.outer = forbidden
    extensions.extend(exp, 'A', 1)
    assert read_record(path) == original
