"""Generated record contracts only; no market or estimator fits."""
from types import SimpleNamespace

import pytest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import ablation
from cottonlens_ml.research.decision_cache import execution_binding
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.search import SEEDS, recipes

from ml.tests.test_selection_boundary import fixture


def prepared(root):
    history, fold = fixture()
    fold['fold'] = 1
    history['x'] = 1.
    experiment = SimpleNamespace(root=root, history=history,
        identity={'groups': {'base': ['x']}, 'split': {'folds': [fold]}})
    hashes = []
    for family in ('xgboost', 'catboost'):
        for h in (1, 5):
            request = {'operation': 'tune', 'family': family, 'horizon': h,
                       'namespace': 'ablate-base', 'seeds': list(SEEDS),
                       'candidates': recipes(family, h, ['x'], 16)}
            result = {'chosen': {'score': 1.}, 'execution_request': request,
                      'execution_binding': execution_binding(experiment, fold, request, selection=True)}
            freeze_record(root/'decisions/ablate-base'/f'{family}-t{h}-fold1.json', result)
            hashes.append(content_id(result))
    plan = {'fixture': 'generated summary contract, not eight-year research admission'}
    freeze_record(root/'ablation-plan.json', plan)
    summary = {'plan': plan,
               'inner_scores': {f'{family}-t{h}': {'base': {'1': 1.}}
                                for family in ('xgboost', 'catboost') for h in (1, 5)},
               'comparisons': {f'{family}-t{h}': {} for family in ('xgboost', 'catboost') for h in (1, 5)},
               'outer_used_for_admission': False,
               'execution_binding': ablation.summary_binding(experiment, plan, hashes)}
    freeze_record(root/'ablation-summary.json', summary)
    return experiment, summary


@pytest.mark.parametrize('change', [None, 'history', 'groups', 'plan', 'aggregate'])
def test_summary_rechecks_bound_decisions_and_current_context(tmp_path, change):
    experiment, summary = prepared(tmp_path)
    if change == 'history':
        experiment.history.loc[0, 'x'] = 999
    elif change == 'groups':
        experiment.identity['groups']['base'] = ['other']
    elif change == 'plan':
        # New self-hash cannot make an aggregate match a changed plan.
        import json
        body = {'fixture': 'changed'}
        (tmp_path/'ablation-plan.json').write_text(json.dumps({**body, 'record_id': content_id(body)}))
    elif change == 'aggregate':
        import json
        body = {**summary, 'comparisons': {'forged': {'admitted': True}}}
        (tmp_path/'ablation-summary.json').write_text(json.dumps({**body, 'record_id': content_id(body)}))
    if change:
        with pytest.raises(ValueError, match='binding|contract|plan|aggregate'):
            ablation.read_summary(experiment)
    else:
        assert ablation.read_summary(experiment) == summary
