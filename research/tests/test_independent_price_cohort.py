from copy import deepcopy
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from research.verify_naive_mae import decimal_losses, validate_cohort


def fixture():
    dates = pd.bdate_range('2016-01-04', periods=6)
    history = pd.DataFrame({'date': dates, 'cotton_close': 80.+np.arange(6),
                            'target_return_1': [.01, 0., -.02, .03, -.01, .04],
                            'target_date_1': pd.Series(dates).shift(-1)})
    folds = [{'year': 2016, 'origins': dates[:4].strftime('%Y-%m-%d').tolist()}]
    base = history.head(4).copy().rename(columns={'target_date_1': 'target_date'})
    base['date'] = base.date.dt.strftime('%Y-%m-%d')
    base['target_date'] = base.target_date.dt.strftime('%Y-%m-%d')
    base = base.assign(year=2016, horizon=1, predicted_return=0., raw_predicted_return=.005)
    frame = pd.concat([base.assign(group=g) for g in ('ridge', 'quantile')], ignore_index=True)
    return frame, history, folds


def test_exact_common_frozen_cohort_and_valid_control():
    frame, history, folds = fixture()
    assert validate_cohort(frame, history, folds)['origins_per_arm'] == 4


@pytest.mark.parametrize('kind,reason', [
    ('target_date', 'target dates'), ('price', 'price values'), ('return', 'price values'),
    ('replaced_origin', 'exact frozen cohort'), ('dropped', 'exact frozen cohort'),
    ('duplicate', 'exact frozen cohort'), ('reordered', 'exact frozen cohort'),
    ('year', 'year/horizon'), ('horizon', 'year/horizon'), ('extra_arm', 'arm'),
    ('nonfinite_prediction', 'Finite predictions'),
])
def test_shared_errors_and_incomplete_predictions_are_rejected(kind, reason):
    frame, history, folds = fixture()
    changed = deepcopy(frame)
    # Change BOTH arms so cross-arm agreement alone cannot detect the defect.
    idx = changed.index[changed.date.eq(frame.date.iloc[0])]
    if kind == 'target_date':
        changed.loc[idx, 'target_date'] = '2016-01-08'
    elif kind == 'price':
        changed.loc[idx, 'cotton_close'] *= 100
    elif kind == 'return':
        changed.loc[idx, 'target_return_1'] += .01
    elif kind == 'replaced_origin':
        changed.loc[idx, 'date'] = '2015-12-31'
    elif kind == 'dropped':
        changed = changed.drop(index=idx)
    elif kind == 'duplicate':
        changed.loc[changed.date.eq(frame.date.iloc[1]), 'date'] = frame.date.iloc[0]
    elif kind == 'reordered':
        changed = changed.iloc[::-1]
    elif kind in ('year', 'horizon'):
        changed.loc[idx, kind] += 1
    elif kind == 'extra_arm':
        changed = pd.concat([changed, changed.iloc[:1].assign(group='unregistered')])
    else:
        changed.loc[idx, 'predicted_return'] = np.nan
    with pytest.raises(ValueError, match=reason):
        validate_cohort(changed, history, folds)
    assert validate_cohort(frame, history, folds)['exact_frozen_cohort_verified']


def test_decimal_naive_identity_zero_movement_and_price_unit_scaling():
    prices, actual = [80., 90., 100.], [.01, 0., -.02]
    naive, zero = decimal_losses(prices, actual, [0., 0., 0.])
    assert naive == zero and naive[1] == 0
    _, perfect = decimal_losses(prices, actual, actual)
    assert perfect == [0, 0, 0]
    scaled, _ = decimal_losses([p*100 for p in prices], actual, [0., 0., 0.])
    with localcontext() as ctx:
        ctx.prec = 50
        assert max(abs(x*100-y) for x, y in zip(naive, scaled, strict=True)) < Decimal('1e-43')


@pytest.mark.parametrize('price,actual,prediction', [(0., 0., 0.), (80., np.nan, 0.), (80., 0., np.inf)])
def test_invalid_loss_inputs_rejected(price, actual, prediction):
    with pytest.raises(ValueError, match='Finite inputs'):
        decimal_losses([price], [actual], [prediction])
