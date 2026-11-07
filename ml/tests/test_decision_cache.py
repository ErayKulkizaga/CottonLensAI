"""Generated orchestration only; never train a market estimator."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.ledger import freeze_record

from ml.tests.test_selection_boundary import fixture as split_fixture


def test_unbound_historical_decision_is_preserved_but_not_reused(tmp_path):
    history, fold = split_fixture()
    fold['fold'] = 1
    marker = tmp_path/'decisions/test/ridge-t1-fold1.json'
    freeze_record(marker, {'chosen': {'old': True}})
    original = marker.read_bytes()
    experiment = SimpleNamespace(history=history, root=tmp_path)
    with pytest.raises(ValueError, match='Cache execution binding'):
        Experiment.tune(experiment, 'ridge', 1, fold, [], 'test')
    assert marker.read_bytes() == original


def tune_case(tmp_path):
    history, fold = split_fixture()
    history['value'] = np.arange(len(history), dtype=float)
    fold['fold'] = 1
    recipe = {'family': 'ridge', 'horizon': 1, 'seed': 42, 'features': ['value'], 'params': {'alpha': 1}}
    calls = []
    def inner(spec, fold, **kwargs):
        calls.append(spec)
        return {'recipe': spec, 'score': 1., 'records': [], 'iterations': 1}
    experiment = SimpleNamespace(history=history, root=tmp_path, inner=inner)
    return experiment, fold, [recipe], calls


@pytest.mark.parametrize('change', ['candidate', 'history', 'fold', 'panel', None])
def test_tune_cache_binds_request_and_actual_past_inputs(tmp_path, change):
    experiment, fold, candidates, calls = tune_case(tmp_path)
    experiment.panel = experiment.history.copy()
    first = Experiment.tune(experiment, 'ridge', 1, fold, candidates, 'test')
    count = len(calls)
    if change == 'candidate':
        candidates[0]['params']['alpha'] = 2
    elif change == 'history':
        experiment.history.loc[0, 'value'] += 1
    elif change == 'fold':
        fold['year'] = 2099
    elif change == 'panel':
        experiment.panel.loc[0, 'value'] += 1
    if change:
        with pytest.raises(ValueError, match='Cache execution binding'):
            Experiment.tune(experiment, 'ridge', 1, fold, candidates, 'test')
    else:
        experiment.history.loc[300:, 'value'] = 99999  # Outer values cannot guide selection.
        assert Experiment.tune(experiment, 'ridge', 1, fold, candidates, 'test') == first
    assert len(calls) == count


@pytest.mark.parametrize('change', ['weights', 'member', 'history', 'origins', None])
def test_ensemble_outer_cache_binds_decision_and_evaluation(tmp_path, change):
    dates = pd.bdate_range('2001-01-01', periods=10)
    history = pd.DataFrame({'date': dates, 'cotton_close': 100., 'target_return_1': .01})
    fold = {'fold': 1, 'origins': dates[5:8].strftime('%Y-%m-%d').tolist()}
    calls = []
    def outer(*args, **kwargs):
        calls.append(args)
        return {'predictions': [0., 0., 0.]}
    experiment = SimpleNamespace(history=history, root=tmp_path, identity={}, outer=outer)
    chosen = {'recipe': {'family': 'ensemble', 'horizon': 1}, 'score': 1., 'weights': [1.],
              'members': [{'decision': {'recipe': {'family': 'ridge'}}, 'namespace': 'member'}]}
    first = Experiment.outer(experiment, chosen, fold, 'test')
    if change == 'weights':
        chosen['weights'] = [.5]
    elif change == 'member':
        chosen['members'][0]['namespace'] = 'other'
    elif change == 'history':
        history.loc[6, 'cotton_close'] = 200.
    elif change == 'origins':
        fold['origins'][-1] = dates[8].strftime('%Y-%m-%d')
    if change:
        with pytest.raises(ValueError, match='Cache execution binding'):
            Experiment.outer(experiment, chosen, fold, 'test')
    else:
        assert Experiment.outer(experiment, chosen, fold, 'test') == first
    assert len(calls) == 1
