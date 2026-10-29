import numpy as np
import pandas as pd
import pytest

from research.ct_risk_extreme_attribution import attribute


def fixture():
    dates = pd.bdate_range('2016-01-04', periods=2007)
    close = np.where(dates < '2022-06-24', 136.32, 103.76)
    history = pd.DataFrame({'date': dates, 'cotton_close': close, 'cotton_volume': 174.,
                            'target_date_1': pd.Series(dates).shift(-1)})
    actual = np.log(close[1:]/close[:-1])**2
    frame = pd.DataFrame({'date': dates[:-1].strftime('%Y-%m-%d'),
                          'target_date': dates[1:].strftime('%Y-%m-%d'),
                          'year': dates[:-1].year, 'actual_variance': actual,
                          'ewma_variance': .01, 'raw_variance': .02,
                          'baseline_loss': np.log(.01)+actual/.01,
                          'raw_loss': np.log(.02)+actual/.02, 'score': 1.})
    contracts = ['Jul-22', 'Oct-22', 'Dec-22', 'Mar-23', 'May-23',
                 'Jul-23', 'Oct-23', 'Dec-23', 'Mar-24', 'May-24']
    def report(day, price):
        lines = [f'MEMPHIS {day}', 'FUTURES TODAY', f'Jul-22 {price}']
        lines += [f'{c} 100.00' for c in contracts[1:]]
        return ('\n'.join([*lines, 'COTTON QUOTATIONS'])).encode()
    return frame, history, report('6/23/2022', '136.32'), report('6/24/2022', '103.76')


def test_same_printed_contract_is_not_automatically_called_a_continuous_roll():
    frame, history, first, second = fixture()
    result = attribute(frame, history, first, second)
    assert result['front_contract_on_both_prints'] == '2022-07-01'
    assert result['focus_raw_QLIKE_gain'] > 0
    assert not result['continuous_contract_switch_established']
    assert result['decimal_target_rows_checked'] == 2006
    assert result['decimal_loss_scalars_checked'] == 4012
    assert result['original_origins'] == len(frame)


def test_mismatched_source_day_or_price_is_rejected_without_source_repair():
    frame, history, first, second = fixture()
    with pytest.raises(ValueError, match='header date'):
        attribute(frame, history, first, first)
    with pytest.raises(ValueError, match='does not match'):
        attribute(frame, history, first, second.replace(b'103.76', b'104.76'))


def test_wrong_target_and_dropped_origins_are_rejected():
    frame, history, first, second = fixture()
    with pytest.raises(ValueError, match='original2006'):
        attribute(frame.iloc[1:], history, first, second)
    frame.loc[0, 'target_date'] = frame.date.iloc[0]
    with pytest.raises(ValueError, match='endpoints'):
        attribute(frame, history, first, second)
