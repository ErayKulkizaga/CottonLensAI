import math

import pytest

from research import ams_contract_reconciliation as audit


def report():
    names = ['Mar-20', 'May-20', 'Jul-20', 'Oct-20', 'Dec-20', 'Mar-21', 'May-21', 'Jul-21', 'Oct-21', 'Dec-21']
    return ('MP_CN001 2/26/2020\nFUTURES TODAY\n'+
            '\n'.join(f'SPOT 999 888 10 {name} {65+i/10:.2f}' for i, name in enumerate(names))+
            '\nCOTTON QUOTATIONS 999 1000').encode()


def test_header_date_and_future_column_are_required():
    quotes = audit.parse_daily(report(), '2020-02-26')
    assert quotes['2020-05-01'] == '65.10'
    assert len(quotes) == 10
    with pytest.raises(ValueError, match='date differs'):
        audit.parse_daily(report(), '2020-02-27')


def test_duplicate_and_expired_contracts_fail_closed():
    with pytest.raises(ValueError, match='unique'):
        audit.parse_daily(report().replace(b'May-20', b'Mar-20'), '2020-02-26')
    with pytest.raises(ValueError, match='Past delivery'):
        audit.parse_daily(report().replace(b'Mar-20', b'Jul-19'), '2020-02-26')


def test_equal_prices_allow_witnessed_relabel_but_not_guessed_year_shift():
    daily = audit.parse_daily(report(), '2020-02-26')
    old = {name.replace('2020', '2019'): {'price': value} for name, value in daily.items()}
    status, mapping = audit.reconcile_day(old, daily)
    assert status == 'verified_relabel'
    assert mapping[1]['true_contract'] == '2020-05-01'
    old[next(iter(old))]['price'] = '65.02'
    assert audit.reconcile_day(old, daily) == ('price_conflict_no_repair', [])


def test_roll_seam_uses_origin_contract_at_next_date_and_keeps_unknown_slots():
    origin = {'2020-03-01': '65', '2020-05-01': '66'}
    target = {'2020-05-01': '65', '2020-07-01': '68'}
    dates = {'2020-02-26': '2020-02-27', '2020-02-27': '2020-02-28'}
    old = [{'date': day, 'target_date': end, 'contract': '2019-07-01', 'named_actual': 0.} for day, end in dates.items()]
    result = audit.labels(dates, old, {'2020-02-26': origin, '2020-02-27': target})
    assert result[0]['true_contract'] == '2020-05-01'
    assert result[0]['ordinal_roll_seam'] and result[0]['label_changed']
    assert result[0]['true_risk'] == pytest.approx(math.log(65/66)**2)
    assert len(result) == 2 and not result[1]['source_pair_verified']


def test_wrong_target_date_or_subcent_value_cannot_be_silently_fixed():
    with pytest.raises(ValueError, match='next Cotton'):
        audit.labels({'2020-02-26': '2020-02-27'}, [{'date': '2020-02-26', 'target_date': '2020-02-28',
                     'contract': None, 'named_actual': None}], {})
    with pytest.raises(ValueError, match='two-decimal'):
        audit.cents('65.123')


def test_unrelated_cp1252_prose_does_not_change_ascii_futures_values():
    raw = report()+b'\n'+b'x'*600+b' cotton\x92s report prose'
    assert audit.parse_daily(raw, '2020-02-26') == audit.parse_daily(report(), '2020-02-26')
    with pytest.raises(UnicodeDecodeError):
        audit.parse_daily(report().replace(b'65.10', b'65\x9210'), '2020-02-26')
