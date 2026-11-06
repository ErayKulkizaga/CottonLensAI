"""Injected selection defects must stop before any training call."""
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
from cottonlens_ml.research.engine import Experiment
from cottonlens_ml.research.full_year import inner_price
from cottonlens_ml.research.protocol import validate_inner_fold


def fixture():
    dates = pd.bdate_range('2000-01-03', periods=400)
    history = pd.DataFrame({'date': dates, 'target_date_1': pd.Series(dates).shift(-1),
                            'target_date_5': pd.Series(dates).shift(-5)})
    inner = [{'origins': dates[i:i+63].strftime('%Y-%m-%d').tolist(),
              'cutoff': dates[i].isoformat()} for i in (20, 83, 146)]
    return history, {'origins': dates[300:363].strftime('%Y-%m-%d').tolist(), 'inner': inner}


def test_valid_selection_and_future_outer_label_perturbation():
    history, fold = fixture()
    validate_inner_fold(history, fold)
    changed = history.copy()
    changed.loc[300:, 'target_date_5'] = pd.Timestamp('2099-01-01')
    validate_inner_fold(changed, fold)


@pytest.mark.parametrize('kind,reason', [
    ('outer_origin', 'overlaps outer'), ('unmatured_t1', 'mature strictly'),
    ('unmatured_t5', 'mature strictly'), ('unknown_maturity', 'mature strictly'),
    ('duplicate', 'unique and chronological'), ('reverse', 'unique and chronological'),
    ('overlap', 'overlap or are reordered'), ('cutoff', 'cutoff differs'),
    ('missing_origin', 'absent from history'), ('missing_target', 'maturity is unknown'),
    ('wrong_horizon_order', r'T\+5 labels must follow'),
    ('dropped_origin', 'Exactly 63 origins'), ('dropped_block', 'Exactly three'),
    ('compressed', 'compressed source'),
])
def test_each_injected_defect_stops_both_selection_entrypoints(kind, reason):
    history, valid = fixture()
    fold = deepcopy(valid)
    if kind == 'outer_origin':
        fold['inner'][0]['origins'] = fold['origins']
    elif kind.startswith('unmatured'):
        h = 1 if kind.endswith('t1') else 5
        history.loc[20, f'target_date_{h}'] = history.date.iloc[300]
    elif kind == 'unknown_maturity':
        history.loc[20, 'target_date_5'] = pd.NaT
    elif kind == 'duplicate':
        fold['inner'][0]['origins'][1] = fold['inner'][0]['origins'][0]
    elif kind == 'reverse':
        fold['inner'][0]['origins'].reverse()
    elif kind == 'overlap':
        fold['inner'][1] = deepcopy(fold['inner'][0])
    elif kind == 'cutoff':
        fold['inner'][0]['cutoff'] = history.date.iloc[21].isoformat()
    elif kind == 'missing_origin':
        fold['inner'][0]['origins'][0] = '1999-12-31'
    elif kind == 'missing_target':
        history = history.drop(columns='target_date_5')
    elif kind == 'dropped_origin':
        fold['inner'][0]['origins'].pop()
    elif kind == 'dropped_block':
        fold['inner'].pop()
    elif kind == 'compressed':
        fold['inner'][0]['origins'][-1] = history.date.iloc[83].strftime('%Y-%m-%d')
    else:
        history.loc[20, 'target_date_5'] = history.target_date_1.iloc[20]

    def forbidden(*args, **kwargs):
        pytest.fail('Malformed selection reached training')

    experiment = SimpleNamespace(history=history, fit=forbidden, train_rows=forbidden)
    for call in (lambda: Experiment.inner(experiment, {'family': 'ridge', 'horizon': 1}, fold),
                 lambda: inner_price(experiment, {'family': 'ridge', 'horizon': 1}, fold),
                 lambda: Experiment.tune(experiment, 'ridge', 1, fold, [], 'test')):
        with pytest.raises(ValueError, match=reason):
            call()
    # The same valid control must still pass after the injected defect.
    clean, original = fixture()
    validate_inner_fold(clean, original)


@pytest.mark.parametrize('entrypoint', ['engine', 'full_year', 'tune'])
def test_valid_selection_executes_and_ignores_future_outer_values(tmp_path, entrypoint):
    """Actual selection loops, generated callback only; no backend/model fit proof."""
    import numpy as np
    from cottonlens_ml.research.protocol import mature

    dates = pd.bdate_range('2000-01-03', periods=720)
    history = pd.DataFrame({'date': dates, 'cotton_session_index': np.arange(720),
                            'cotton_close': 80.+np.arange(720)/100,
                            'target_return_1': .01+np.sin(np.arange(720))*.001,
                            'target_return_5': .05+np.cos(np.arange(720))*.001,
                            'target_date_1': pd.Series(dates).shift(-1),
                            'target_date_5': pd.Series(dates).shift(-5)})
    fold = {'fold': 0, 'year': 2002, 'origins': dates[620:683].strftime('%Y-%m-%d').tolist(),
            'inner': [{'origins': dates[i:i+63].strftime('%Y-%m-%d').tolist(),
                       'cutoff': dates[i].isoformat()} for i in (300, 363, 426)]}
    spec = {'family': 'ridge', 'horizon': 1, 'seed': 42, 'task': 'price',
            'cadence': 21, 'features': ['cotton_close'], 'params': {'alpha': 1.}}
    runs = []

    def execute(frame, root):
        seen = []
        experiment = SimpleNamespace(history=frame, root=root)

        def train_rows(cutoff, recipe):
            return mature(frame, cutoff).loc[lambda x: x.date >= dates[119]].copy()

        def generated_fit(recipe, train, validation, test, role, **kwargs):
            # This callback substitutes all model computation. Check the frames
            # delivered by the real orchestration, not inferred fold metadata.
            cutoff = test.date.min() if validation is None else validation.date.min()
            assert len(train) > 0 and (train.target_date_5 < cutoff).all()
            assert test.date.max() < dates[620]
            if validation is not None:
                assert (validation.target_date_5 < dates[620]).all()
            value = float(train.target_return_1.median()) / recipe['params']['alpha']
            seen.append((role, train.date.dt.strftime('%Y-%m-%d').tolist(),
                         test.date.dt.strftime('%Y-%m-%d').tolist(), value))
            return {'experiment_id': f'generated-{len(seen)}',
                    'result': {'iterations': 2, 'predictions': [value]*len(test)}}

        experiment.train_rows = train_rows
        experiment.fit = generated_fit
        experiment.inner = lambda recipe, block, **kw: Experiment.inner(experiment, recipe, block, **kw)
        if entrypoint == 'engine':
            result = Experiment.inner(experiment, spec, fold)
        elif entrypoint == 'full_year':
            result = inner_price(experiment, spec, fold)
        else:
            candidates = [spec, {**spec, 'params': {'alpha': 10.}}]
            result = Experiment.tune(experiment, 'ridge', 1, fold, candidates, 'generated')
        assert seen  # A positive control must execute, not merely reject early.
        runs.append(seen)
        return result

    original = execute(history, tmp_path/'original')
    changed = history.copy(deep=True)
    changed.loc[620:, ['cotton_close', 'target_return_1']] = [999999., -50.]
    perturbed = execute(changed, tmp_path/'changed')
    assert original == perturbed
    assert runs[0] == runs[1]
