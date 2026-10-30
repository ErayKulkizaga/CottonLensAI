import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from research import ct_prospective_readiness as study
from research.verify_ct_prospective_readiness import decimal_difference


def prices(n=100):
    return pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=n),
                         'cotton_close': 100*np.exp(np.sin(np.arange(n))/100)})


def test_append_preserves_every_seed_session_and_its_values():
    seed = prices(50); before = seed.copy(deep=True)
    new = prices().iloc[30:].reset_index(drop=True)
    combined = study.append_snapshot(seed, new)
    pd.testing.assert_frame_equal(combined, prices())
    pd.testing.assert_frame_equal(seed, before)
    pd.testing.assert_frame_equal(study.append_snapshot(seed, new.iloc[:20]), seed)


@pytest.mark.parametrize('change', ['gap', 'omit', 'insert', 'revise'])
def test_append_rejects_unproved_bridges_and_overlapping_source_changes(change):
    seed = prices(50); new = prices().iloc[30:].reset_index(drop=True)
    if change == 'gap':
        new = prices().iloc[60:]
    elif change == 'omit':
        new = new.drop(index=4)
    elif change == 'insert':
        extra = pd.DataFrame({'date': [new.date.iloc[0]+pd.Timedelta(days=3)], 'cotton_close': [101.]})
        # The selected date is an unobserved weekend session.
        new = pd.concat([new, extra]).sort_values('date').reset_index(drop=True)
    else:
        new.loc[4, 'cotton_close'] += .00001
    with pytest.raises(ValueError):
        study.append_snapshot(seed, new)


@pytest.mark.parametrize('value', [0., -1., np.nan])
def test_invalid_price_cannot_be_filled_or_removed(value):
    p = prices(); p.loc[3, 'cotton_close'] = value
    with pytest.raises(ValueError, match='Positive finite'):
        study.candidate_history(p)


def test_past_features_do_not_see_future_prices_and_builder_never_fits():
    p = prices(); changed = p.copy(); changed.loc[70:, 'cotton_close'] *= 1.5
    with patch('cottonlens_ml.research.volatility.fit_predict', side_effect=AssertionError('No fits')):
        before, after = study.candidate_history(p), study.candidate_history(changed)
    columns = ['date', 'cotton_close', 'daily_variance_proxy', 'har_1', 'har_5', 'har_22', 'ct_ewma', 'origin_state']
    pd.testing.assert_frame_equal(before[columns].iloc[:70], after[columns].iloc[:70])
    assert len(before) == len(p)


def test_hac_equals_independent_squared_block_sum_and_preserves_year_edges():
    x = np.random.default_rng(42).normal(.02, .2, 120)
    f = pd.DataFrame({'year': np.repeat([2020, 2021], 60), 'delta': x})
    for block in (20, 60):
        expected = sum(float(np.convolve(g.delta-x.mean(), np.ones(block), mode='full')@np.convolve(
            g.delta-x.mean(), np.ones(block), mode='full')) for _, g in f.groupby('year'))/(len(x)*block)
        assert study.long_run_variance(f, block) == pytest.approx(expected, rel=1e-12, abs=1e-14)
    f['year'] = 2020
    assert study.long_run_variance(f, 60) != pytest.approx(expected)


def test_normal_planning_is_conditional_and_requires_positive_effect():
    small = study.normal_scenario(.02, .09, 126)
    larger = study.normal_scenario(.02, .09, 504)
    assert small['single_interval_positive_lci_probability'] < larger['single_interval_positive_lci_probability']
    assert small['single_interval_positive_lci_probability'] < .2
    needed = small['origins_for_80pct_single_interval_probability']
    assert study.normal_scenario(.02, .09, needed)['single_interval_positive_lci_probability'] >= .8
    with pytest.raises(ValueError, match='positive'):
        study.normal_scenario(0., .09, 126)


def test_equal_forecasts_have_exact_zero_paired_loss_and_window_sign():
    equal = [decimal_difference(.0001378212, .000571382134, .000571382134) for _ in range(126)]
    assert equal == [0.]*126
    assert not any(sum(equal[i:]) > 0 for i in range(len(equal)))
    assert decimal_difference(0., .000571382134, .000571382134) == 0.


def test_hac_rejects_unregistered_bandwidth_and_nonfinite_losses():
    f = pd.DataFrame({'year': 2020, 'delta': np.arange(100.)})
    with pytest.raises(ValueError, match='declared'):
        study.long_run_variance(f, 10)
    f.loc[2, 'delta'] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        study.long_run_variance(f, 20)


def test_unregistered_parent_proof_fails_before_any_output_or_fit(tmp_path):
    (tmp_path/'research').mkdir()
    (tmp_path/'research/registry.json').write_text(json.dumps({'local_evidence': []}))
    p = tmp_path/'research/parent.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='registered immutable'):
        study.audit(tmp_path, tmp_path/'missing', tmp_path/'missing-market', p, tmp_path/'new-output')
    assert not (tmp_path/'new-output').exists()
