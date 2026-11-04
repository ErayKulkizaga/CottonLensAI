from decimal import Decimal

import pandas as pd
import pytest

from research import ct_pre2010_risk as risk
from research import ct_pre2010_risk_source as source


def test_returns_hold_the_same_contract_across_a_calendar_month():
    dates = ['2008-11-28', '2008-12-01', '2008-12-02']
    rows = {
        dates[0]: {'quotes': {'2008-12-01': '60.00'}},
        dates[1]: {'quotes': {'2008-12-01': '61.00', '2009-03-01': '80.00'}},
        dates[2]: {'quotes': {'2009-03-01': '81.00'}},
    }
    assert risk.required_quotes(dates)[dates[1]] == ['2008-12-01', '2009-03-01']
    frame, target, _ = risk.history(dates, rows)
    assert target.iloc[0] == pytest.approx((61/60-1)**2, rel=.02)
    rows[dates[2]]['quotes']['2009-03-01'] = '84.00'
    changed, changed_target, _ = risk.history(dates, rows)
    pd.testing.assert_frame_equal(frame[risk.forward.NAMES].iloc[:2],
                                  changed[risk.forward.NAMES].iloc[:2])
    assert target.iloc[1] != changed_target.iloc[1]
    with pytest.raises(ValueError, match='coverage'):
        risk.history(dates, {k: v for k, v in rows.items() if k != dates[1]})


def test_folds_respect_maturity_and_late_repair():
    dates = pd.bdate_range('2005-01-03', periods=1050).strftime('%Y-%m-%d').tolist()
    releases = {dates[520]: dates[525]}
    design = risk.job_plan(dates, dates[850:1000], releases)
    for job in design['jobs']:
        assert all(dates[dates.index(d)+5] < job['cutoff'] for d in job['train_dates'])
        assert not set(job['train_dates']) & set(dates[520:526])
        assert not set(job['origins']) & set(dates[520:526])


def test_invalid_far_quote_does_not_substitute_a_missing_near_quote():
    rows = [('2009-03-01', Decimal('61.0'), Decimal('60.0'), Decimal(1)),
            ('2009-05-01', Decimal('62.0'), Decimal(0), Decimal(0))]
    assert source.validated_quotes(rows, '2009-01-02') == {'2009-03-01': '61.0'}
    with pytest.raises(ValueError, match='Duplicate'):
        source.validated_quotes(rows+rows, '2009-01-02')


def test_stale_origins_use_last_complete_state_until_weekly_release():
    dates = ['2009-05-21', '2009-05-22', '2009-05-26', '2009-05-29', '2009-06-01']
    late = {'2009-05-22': {'published': '2009-05-29'}}
    assert risk.stale_origins(dates, dates[1:], late) == {
        '2009-05-22': '2009-05-21', '2009-05-26': '2009-05-21', '2009-05-29': '2009-05-21',
    }
