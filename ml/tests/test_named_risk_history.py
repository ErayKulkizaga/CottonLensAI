from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import named_risk_history as control


def available(q):
    return datetime.fromisoformat(q['report_date']).replace(tzinfo=UTC)+timedelta(days=1)


def panel(dates):
    return {d: {'report_date': d, 'second_contract': 'May',
                'quotes': {'May': 80*np.exp(i*.01), 'Jul': 90*np.exp(i*.02)}} for i, d in enumerate(dates)}


def test_features_select_current_contract_history():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    quotes = panel(dates)
    quotes[dates[-1]]['second_contract'] = 'Jul'
    result = control.own_history(dates, quotes, available)
    for k in (1, 5, 22):
        assert result.iloc[-2][f'own_har_{k}'] == pytest.approx(np.log(.01**2))
        assert result.iloc[-1][f'own_har_{k}'] == pytest.approx(np.log(.02**2))
    assert result.iloc[-1].own_22_session_span == 22


def test_future_quotes_leave_past_features_exact():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    quotes = panel(dates)
    before = control.own_history(dates, quotes, available)
    quotes[dates[-1]]['quotes']['May'] = 130.
    after = control.own_history(dates, quotes, available)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1], check_exact=True)


def test_missing_calendar_pairs_do_not_become_daily_returns():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    quotes = panel(dates)
    del quotes[dates[10]]
    result = control.own_history(dates, quotes, available)
    assert result.iloc[-1].own_observed_pairs == 22
    assert result.iloc[-1].own_22_session_span == 24
    assert result.iloc[-1].own_har_22 == pytest.approx(np.log(.01**2))
    assert len(result) == 25
    assert not result.iloc[10].own_feature_ready


def test_clock_equality_and_later_exclusion():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    quotes = panel(dates)
    assert control.own_history(dates, quotes, lambda q: control.base.decision(q['report_date'])).iloc[-1].own_feature_ready
    assert not control.own_history(dates, quotes, lambda q: control.base.decision(q['report_date'])+timedelta(seconds=1)).iloc[-1].own_feature_ready


def test_zero_returns_keep_positive_log_floor():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    quotes = panel(dates)
    for q in quotes.values():
        q['quotes']['May'] = 80.
    result = control.own_history(dates, quotes, available)
    assert result.iloc[-1].own_har_22 == np.log(control.FLOOR)


@pytest.mark.parametrize('value', [0, -1, np.inf, np.nan])
def test_bad_prices_are_errors_not_missingness(value):
    dates = ['2020-01-01', '2020-01-02']
    quotes = panel(dates)
    quotes[dates[-1]]['quotes']['May'] = value
    with pytest.raises(ValueError, match='Positive finite'):
        control.own_history(dates, quotes, available)


@pytest.fixture
def history():
    dates = pd.bdate_range('2019-01-01', periods=1100)
    h = pd.DataFrame({'date': dates, 'cotton_session_index': np.arange(len(dates)), 'cotton_close': 80.,
                      'target_date_1': pd.Series(dates).shift(-1), 'target_date_5': pd.Series(dates).shift(-5),
                      'variance_target_1': .01, 'named_variance': .02, 'named_eligible': True,
                      'named_ewma_eligible': True, 'named_ewma': .03, 'named_ewma_observations': 50,
                      'named_contract': 'May', 'named_reason': 'conditional',
                      'named_label_available_at': pd.Series(dates).shift(-1).dt.tz_localize('UTC'),
                      'own_feature_ready': True, 'own_22_session_span': 22, 'own_latest_pair_age': 0})
    for k in (1, 5, 22):
        h[f'har_{k}'] = -4.
        h[f'own_har_{k}'] = -5.
    return h


def test_projection_changes_only_har_features_between_arms(history):
    dates = control.mature_dates(history, history.date.iloc[950])
    ct, own = [control.projection(history, dates, a, training=True) for a in control.ARMS]
    pd.testing.assert_frame_equal(ct.drop(columns=control.RECIPE['features']), own.drop(columns=control.RECIPE['features']))
    assert ct.variance_target_1.eq(.02).all()
    assert own.variance_target_1.eq(.02).all()


def test_mature_common_population_requires_own_history_and_label_clock(history):
    cut = history.date.iloc[950].strftime('%Y-%m-%d')
    history.loc[500, 'own_feature_ready'] = False
    assert history.date.iloc[500].strftime('%Y-%m-%d') not in control.mature_dates(history, cut)
    history.loc[501, 'named_label_available_at'] = control.base.decision(cut)+pd.Timedelta(seconds=1)
    assert history.date.iloc[501].strftime('%Y-%m-%d') not in control.mature_dates(history, cut)


def test_plan_retains_full_calendar_cutoffs_and_missing_output_slots(history):
    origins = history.date.iloc[950:1076].dt.strftime('%Y-%m-%d').tolist()
    history.loc[950, 'own_feature_ready'] = False
    history.loc[950, 'named_eligible'] = False
    history.loc[950, 'named_ewma_eligible'] = False
    with patch('cottonlens_ml.research.volatility.fit_predict', side_effect=AssertionError('no fit in plan')):
        design = control.plan(history, {'folds': [{'year': 2023, 'origins': origins}]})
    fold = design['folds'][0]
    assert fold['origins'] == origins
    job = next(j for j in fold['jobs'] if j['role']=='outer' and j['bucket']==0)
    assert job['cutoff'] == origins[0]
    assert job['origin_slots'][0] == origins[0]
    assert job['origins'][0] == origins[1]
    records = [{'experiment_id': str(i), 'result': {'origins': j['origins'],
                'variance_predictions': [.02]*len(j['origins'])}} for i, j in enumerate(fold['jobs'])]
    result = control.output(history, fold, 'named', records)
    assert len(result['records']) == 126
    assert not result['records'][0]['forecast_available']
    assert result['records'][0]['raw'] is None
    assert result['records'][1]['forecast_available']


def test_future_outer_targets_cannot_choose_weight(history):
    origins = history.date.iloc[950:1076].dt.strftime('%Y-%m-%d').tolist()
    fold = control.plan(history, {'folds': [{'year': 2023, 'origins': origins}]})['folds'][0]
    records = [{'experiment_id': str(i), 'result': {'origins': j['origins'],
                'variance_predictions': [.02]*len(j['origins'])}} for i, j in enumerate(fold['jobs'])]
    before = control.output(history, fold, 'named', records)
    history.loc[950:, 'named_variance'] = 10.
    after = control.output(history, fold, 'named', records)
    assert before['selection']==after['selection']


def test_source_unready_test_is_not_imputed(history):
    history.loc[950, 'own_feature_ready'] = False
    with pytest.raises(ValueError, match='imputation'):
        control.projection(history, [history.date.iloc[950].strftime('%Y-%m-%d')], 'named', training=False)
