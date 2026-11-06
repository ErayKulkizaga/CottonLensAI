import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.temporal_audit import (
    TemporalAuditFailure,
    audit_features,
    audit_predictions,
)


def data():
    return pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=30),
                         'price': np.arange(30, dtype=float) + 80})


def test_past_rolling_builder_passes():
    def builder(f):
        return f.assign(value=f.price.rolling(5).mean())
    result = audit_features(data(), builder, ['value'], [10, 20])
    assert result['new_fits'] == 0 and not result['historical_source_availability_proven']


@pytest.mark.parametrize('kind', ['future_join', 'centered', 'global_scaling', 'backfill'])
def test_leak_injections_are_rejected(kind):
    def builder(f):
        if kind == 'future_join':
            value = f.price.shift(-1)
        elif kind == 'centered':
            value = f.price.rolling(5, center=True).mean()
        elif kind == 'global_scaling':
            value = (f.price-f.price.mean())/f.price.std()
        else:
            value = f.price.where(f.index % 5 == 0).bfill()
        return f.assign(value=value)
    with pytest.raises(TemporalAuditFailure, match='Future'):
        audit_features(data(), builder, ['value'], [9, 19])


def test_mutating_builder_and_dropped_origins_rejected():
    def mutate(f):
        f['value'] = f.price
        return f
    with pytest.raises(TemporalAuditFailure, match='mutated'):
        audit_features(data(), mutate, ['value'], [10])
    with pytest.raises(TemporalAuditFailure, match='dropped/reordered'):
        audit_features(data(), lambda f: f.iloc[1:].assign(value=0.), ['value'], [10])


def test_fixed_prediction_state_passes_without_mutating_caller():
    state = {'recipe': {'kind': 'generated'}, 'selection': {'weight': .5}, 'native': {'coef': 2.}}
    def predict(f, context):
        value = context['selection']['weight']*context['native']['coef']*f.price.rolling(3).mean()
        return f.assign(prediction=value)
    result = audit_predictions(data(), predict, state, ['prediction'], [10, 20])
    assert result['new_fits'] == 0 and not result['selection_procedure_validity_proven']
    assert state['selection']['weight'] == .5


@pytest.mark.parametrize('kind', ['future_scaler', 'future_selection', 'state_mutation'])
def test_prediction_pipeline_leak_or_state_mutation_rejected(kind):
    state = {'recipe': {}, 'selection': {'weight': .5}, 'native': {'coef': 2.}}
    def predict(f, context):
        if kind == 'state_mutation':
            context['selection']['weight'] = 1.
            value = f.price
        elif kind == 'future_scaler':
            value = (f.price-f.price.mean())/f.price.std()
        else:
            value = f.price*(1. if f.price.iloc[-1] > 100 else 0.)
        return f.assign(prediction=value)
    with pytest.raises(TemporalAuditFailure, match='mutated|Future'):
        audit_predictions(data(), predict, state, ['prediction'], [10, 20])
    assert state['selection']['weight'] == .5
