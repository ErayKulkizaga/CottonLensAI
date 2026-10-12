from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import named_risk_control as control


def available(quote):
    return datetime.fromisoformat(quote['report_date']).replace(tzinfo=UTC) + timedelta(days=1)


def quotes(days):
    return {day: {'report_date': day, 'second_contract': 'May',
                  'quotes': {'May': 80 * np.exp(i * .01), 'Jul': 90 * np.exp(i * .02)}}
            for i, day in enumerate(days)}


def test_named_ewma_uses_origin_contract_history_not_pooled_second():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    panel = quotes(dates)
    panel[dates[-1]]['second_contract'] = 'Jul'
    result = control.named_ewma(dates, panel, available)
    assert result.iloc[-2].named_ewma == pytest.approx(.01 ** 2)
    assert result.iloc[-1].named_ewma == pytest.approx(.02 ** 2)
    assert result.iloc[-1].named_ewma_observations == 24
    assert result.iloc[-1].named_ewma_eligible


def test_future_prices_do_not_change_past_named_ewma():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    panel = quotes(dates)
    before = control.named_ewma(dates, panel, available)
    panel[dates[-1]]['quotes']['May'] = 120
    after = control.named_ewma(dates, panel, available)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1], check_exact=True)
    assert before.iloc[-1].named_ewma != after.iloc[-1].named_ewma


def test_missing_pair_is_not_a_multiday_return_or_compressed_calendar():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    panel = quotes(dates)
    del panel[dates[10]]
    result = control.named_ewma(dates, panel, available)
    assert len(result) == 25
    assert result.iloc[-1].named_ewma_observations == 22
    assert result.iloc[-1].named_ewma == pytest.approx(.01 ** 2)
    assert np.isnan(result.iloc[10].named_ewma)


def test_delayed_quote_is_not_used_before_decision():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    panel = quotes(dates)

    def delayed(q):
        value = available(q)
        return value + timedelta(days=1) if q['report_date'] == dates[-1] else value

    result = control.named_ewma(dates, panel, delayed)
    assert not result.iloc[-1].named_ewma_eligible
    assert np.isnan(result.iloc[-1].named_ewma)


def test_clock_equality_is_allowed_but_later_quote_rejected():
    dates = pd.bdate_range('2020-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    panel = quotes(dates)
    exact = lambda q: control.decision(q['report_date'])
    assert control.named_ewma(dates, panel, exact).iloc[-1].named_ewma_eligible
    late = lambda q: exact(q) + timedelta(seconds=1)
    assert not control.named_ewma(dates, panel, late).iloc[-1].named_ewma_eligible


def test_nonpositive_named_price_is_failure_not_missingness():
    dates = ['2020-01-02', '2020-01-03']
    panel = quotes(dates)
    panel[dates[0]]['quotes']['May'] = 0
    with pytest.raises(ValueError, match='Positive finite'):
        control.named_ewma(dates, panel, available)


@pytest.fixture
def history():
    dates = pd.bdate_range('2019-01-01', periods=1100)
    h = pd.DataFrame({'date': dates, 'cotton_session_index': np.arange(len(dates)),
                      'cotton_close': 80., 'target_date_5': pd.Series(dates).shift(-5),
                      'target_date_1': pd.Series(dates).shift(-1),
                      'variance_target_1': .01, 'named_variance': .02, 'named_eligible': True,
                      'named_contract': 'May', 'named_reason': 'conditional', 'named_ewma': .03,
                      'named_ewma_observations': 50,
                      'named_ewma_eligible': True,
                      'named_label_available_at': pd.Series(dates).shift(-1).dt.tz_localize('UTC')})
    for name in control.RECIPE['features']:
        h[name] = -3.
    return h


def test_training_uses_same_population_and_only_target_changes(history):
    dates = control.mature_dates(history, history.date.iloc[950])
    ct, named = [control.projection(history, dates, arm) for arm in control.ARMS]
    pd.testing.assert_frame_equal(ct.drop(columns='variance_target_1'), named.drop(columns='variance_target_1'))
    assert ct.variance_target_1.eq(.01).all()
    assert named.variance_target_1.eq(.02).all()
    assert len(dates) == 950 - 5 - 119


def test_maturity_checks_source_time_and_five_observation_purge(history):
    cutoff = history.date.iloc[950].strftime('%Y-%m-%d')
    history.loc[900, 'named_label_available_at'] = control.decision(cutoff)
    assert history.date.iloc[900].strftime('%Y-%m-%d') in control.mature_dates(history, cutoff)
    history.loc[900, 'named_label_available_at'] += pd.Timedelta(seconds=1)
    assert history.date.iloc[900].strftime('%Y-%m-%d') not in control.mature_dates(history, cutoff)
    assert history.date.iloc[945].strftime('%Y-%m-%d') not in control.mature_dates(history, cutoff)


def test_incomplete_common_label_cannot_enter_training(history):
    history.loc[900, 'named_eligible'] = False
    with pytest.raises(ValueError, match='common finite'):
        control.projection(history, [history.date.iloc[900].strftime('%Y-%m-%d')], 'ct')


def test_plan_preserves_origins_and_counts_fits_without_training(history):
    origins = history.date.iloc[950:1076].dt.strftime('%Y-%m-%d').tolist()
    parent = {'folds': [{'year': 2023, 'origins': origins}]}
    with patch('cottonlens_ml.research.volatility.fit_predict', side_effect=AssertionError('No fit in plan')):
        design = control.plan(history, parent)
    assert design['folds'][0]['origins'] == origins
    assert len(design['folds'][0]['inner']) == 3
    assert design['fit_budget']['market'] == 2 * len(design['folds'][0]['jobs'])
    assert all(len(j['train_dates']) >= 500 for j in design['folds'][0]['jobs'])
    assert not design['gate_evaluation_allowed']
    assert not design['release_allowed']


def test_incomplete_named_baseline_cannot_support_selection(history):
    history.loc[:950, 'named_ewma_eligible'] = False
    origins = history.date.iloc[950:1076].dt.strftime('%Y-%m-%d').tolist()
    design = control.plan(history, {'folds': [{'year': 2023, 'origins': origins}]})
    assert not design['folds']
    assert not design['readiness'][0]['named_baseline_ready']


def test_outer_targets_cannot_select_weight(history):
    origins = history.date.iloc[950:1076].dt.strftime('%Y-%m-%d').tolist()
    fold = control.plan(history, {'folds': [{'year': 2023, 'origins': origins}]})['folds'][0]
    records = [{'experiment_id': str(i), 'result': {'origins': j['origins'],
                'variance_predictions': [.02] * len(j['origins'])}} for i, j in enumerate(fold['jobs'])]
    first = control.output(history, fold, 'named', records)
    history.loc[950:, 'named_variance'] = 10.
    second = control.output(history, fold, 'named', records)
    assert first['selection']['weight'] == 1
    assert first['selection'] == second['selection']
    assert first['all_inner_weights'] == second['all_inner_weights']


def test_bad_fit_origins_and_variance_cannot_be_completed(history):
    origins = history.date.iloc[950:1076].dt.strftime('%Y-%m-%d').tolist()
    fold = control.plan(history, {'folds': [{'year': 2023, 'origins': origins}]})['folds'][0]
    records = [{'experiment_id': str(i), 'result': {'origins': j['origins'],
                'variance_predictions': [.02] * len(j['origins'])}} for i, j in enumerate(fold['jobs'])]
    records[0]['result']['variance_predictions'][0] = -1
    with pytest.raises(ValueError, match='positive disjoint'):
        control.output(history, fold, 'named', records)
