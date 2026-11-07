"""Past-only family selection and immutable ensemble with fake predictions."""
from types import SimpleNamespace

import numpy as np
import pytest
from cottonlens_ml.research.free_rounds import ensemble_fold, prior_choices
from cottonlens_ml.research.ledger import freeze_record, read_record
from test_sprint import history


def choice(family, score, dates):
    recipe = {'family': family, 'horizon': 1, 'task': 'price', 'features': ['cotton_ret_1'], 'params': {}}
    return {'recipe': recipe, 'score': score, 'seeds': [
        {'recipe': recipe, 'origins': dates, 'predictions': [0.] * len(dates), 'iterations': 1}]}


def test_family_selection_reads_only_requested_fold(tmp_path):
    first = choice('xgboost', .99, ['2020-01-01'])
    future = choice('catboost', .01, ['2023-01-01'])
    freeze_record(tmp_path / 'decisions/adaptive-A/xgboost-t1-fold1.json', {'chosen': first})
    freeze_record(tmp_path / 'decisions/adaptive-A/catboost-t1-fold8.json', {'chosen': future})
    found = prior_choices(SimpleNamespace(root=tmp_path), 1, {'fold': 1}, ['adaptive-A'])
    assert [c['family'] for c in found] == ['xgboost']


@pytest.mark.parametrize('fault', [None, 'outer_labels', 'missing_origin', 'immature'])
def test_ensemble_family_uniqueness_and_common_origin_contract(tmp_path, fault, monkeypatch):
    frame = history().head(500)
    blocks = [{'origins': frame.date.iloc[i:i+63].dt.strftime('%Y-%m-%d').tolist()}
              for i in (126, 189, 252)]
    inner_dates = [date for block in blocks for date in block['origins']]
    fold = {'fold': 1, 'inner': blocks,
            'origins': frame.date.iloc[400:450].dt.strftime('%Y-%m-%d').tolist()}
    if fault == 'outer_labels':
        inner_dates = fold['origins']
    elif fault == 'missing_origin':
        inner_dates = inner_dates[:-1]
    elif fault == 'immature':
        frame.loc[126, 'target_date_5'] = frame.date.iloc[400]
    selected = [{'family': family, 'namespace': namespace, 'decision': choice(family, .99, inner_dates)}
                for family, namespace in [('xgboost', 'adaptive-A'), ('xgboost', 'free-E'), ('catboost', 'adaptive-A')]]
    def outer(chosen, supplied_fold, namespace):
        return {'origins': supplied_fold['origins'], 'predictions': np.zeros(len(supplied_fold['origins'])).tolist()}
    experiment = SimpleNamespace(root=tmp_path, history=frame, outer=outer)
    if fault:
        def forbidden(*args, **kwargs):
            pytest.fail('Malformed OOF cohort reached prediction or weight optimization')
        experiment.outer = forbidden
        monkeypatch.setattr('cottonlens_ml.research.free_rounds.simplex_weights', forbidden)
        with pytest.raises(ValueError, match='OOF cohort|mature strictly'):
            ensemble_fold(experiment, 1, fold, selected)
        return
    ensemble_fold(experiment, 1, fold, selected)
    decision = read_record(tmp_path / 'decisions/free-E-ensemble/ensemble-t1-fold1.json')
    assert len(decision['chosen']['members']) == 2
    assert abs(sum(decision['chosen']['weights']) - 1) < 1e-8
    assert decision['family_selection'] == 'this_fold_inner_only'
