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
