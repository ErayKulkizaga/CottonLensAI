from datetime import date

import pytest

from research.weekly_named_source import parse_table, printed_date

HEADER = 'ICE futures contract settlement\nM ar-13 May-13 Jul-13 Oct-13 Dec-13 M ar-14 Average\n'
ROWS = '\n'.join(f'Feb {day} 82.98 83.55 83.88 82.29 81.46 82.40 77.45' for day in (4, 5, 6, 7))


def test_six_contracts_ignore_spot_price_and_separate_publication():
    result = parse_table(HEADER + ROWS, date(2013, 2, 8))
    assert len(result) == 4
    assert result[-1]['settlements']['2013-03'] == '82.98'
    assert len(result[-1]['settlements']) == 6
    assert result[0]['publication_lag_calendar_days'] == 4
    assert result[-1]['observation_date'] == '2013-02-07'


def test_year_boundary_and_printed_clock():
    header = HEADER.replace('-13', '-16').replace('-14', '-17')
    rows = ROWS.replace('Feb 4', 'Dec 29').replace('Feb 5', 'Dec 30').replace('Feb 6', 'Dec 31').replace('Feb 7', 'Jan 1')
    result = parse_table(header + rows, date(2016, 1, 2))
    assert result[0]['observation_date'] == '2015-12-29'
    assert result[-1]['observation_date'] == '2016-01-01'
    assert printed_date('Vol 94 No 27 February 8, 2013') == date(2013, 2, 8)


@pytest.mark.parametrize('text', [
    HEADER + ROWS.replace('Feb 7', 'Feb 8'),
    HEADER + ROWS.replace('82.98', '-'),
    HEADER + ROWS.replace('82.98', '0.00'),
    HEADER + ROWS.replace('Feb 7', 'Feb 6'),
    HEADER.replace('May-13', 'M ar-13') + ROWS,
    HEADER + ROWS + '\n' + HEADER + ROWS,
    (HEADER + ROWS).replace('ICE futures contract settlement', 'spot quotations'),
])
def test_ambiguities_and_missing_values_fail_closed(text):
    with pytest.raises(ValueError):
        parse_table(text, date(2013, 2, 8))
