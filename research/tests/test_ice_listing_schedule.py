from datetime import date

import pytest

from research.ice_listing_schedule import classify, default_listing_month


@pytest.mark.parametrize('contract,month', [('2026-12-01', date(2024, 1, 1)),
                                          ('2027-03-01', date(2024, 4, 1)),
                                          ('2012-12-01', date(2010, 1, 1)),
                                          ('2013-03-01', date(2010, 4, 1))])
def test_default_month_arithmetic_is_not_a_listing_date(contract, month):
    assert default_listing_month(contract) == month


@pytest.mark.parametrize('contract', ['2012-01-01', '2012-12-02', '2012-2-1'])
def test_invalid_delivery_month_rejected(contract):
    with pytest.raises(ValueError):
        default_listing_month(contract)


def test_conditional_partition_preserves_every_unknown_without_admission():
    cells = [{'source_date': day, 'contract': '2013-03-01'}
             for day in ('2010-03-31', '2010-04-01', '2010-04-30', '2010-05-03')]
    row = classify(cells)[0]
    assert (row['before_default_month'], row['within_default_month'], row['after_default_month']) == (1, 2, 1)
    assert row['total_unknown_cells'] == 4 and row['source_cells_admitted'] == 0
    assert not row['actual_listing_verified'] and not row['board_exceptions_verified']
    with pytest.raises(ValueError, match='Duplicate'):
        classify([*cells, cells[0]])
