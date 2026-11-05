from datetime import date

import pytest

from research.weekly_named_reconcile import parse_reconciled_table

HEAD = 'ICE futures contract settlement\nJul-10 Oct-10 Dec-10 Mar-11 May-11 Jul-11 Average\n'
ROWS = '\n'.join(f'June {d} 77.06 74.69 75.28 76.78 77.53 77.88 71.65 89.60 *' for d in (7, 8, 9, 10))


def test_full_month_auxiliary_markers_and_same_publication_day_are_separate():
    result = parse_reconciled_table(HEAD + ROWS, date(2010, 6, 10))
    assert len(result) == 4
    assert result[-1]['publication_lag_calendar_days'] == 0
    assert result[-1]['settlements']['2010-07'] == '77.06'
    assert result[-1]['ignored_non_futures_cells'] == ['71.65', '89.60', '*']


def test_closed_row_retained_without_filled_prices():
    text = (HEAD + ROWS).replace('June 9 77.06 74.69 75.28 76.78 77.53 77.88 71.65 89.60 *', 'June 9 No Trading Holiday')
    result = parse_reconciled_table(text, date(2010, 6, 11))
    assert len(result) == 4
    assert result[2]['available'] is False
    assert result[2]['settlements'] == {}


@pytest.mark.parametrize('text', [
    (HEAD + ROWS).replace('77.06', '*'),
    (HEAD + ROWS).replace('Oct-10', 'Jul-10'),
    (HEAD + ROWS).replace('89.60', 'bad'),
    (HEAD + ROWS).replace('June 10', 'June 12'),
])
def test_no_contract_repair_price_imputation_or_unknown_format(text):
    with pytest.raises(ValueError):
        parse_reconciled_table(text, date(2010, 6, 11))
