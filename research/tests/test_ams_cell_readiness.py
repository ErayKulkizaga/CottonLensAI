import csv

import pytest

from research.ams_cell_readiness import (
    ABSENT,
    PRESENT,
    UNKNOWN,
    quote_cell,
    source_tables,
    trace,
)
from research.verify_ams_cell_readiness import source_cell_status


def test_partial_far_quote_is_unknown_not_absent():
    table = {'quotes': {'2019-03-01': '75.00', '2019-05-01': '76.00'},
             'ranked': ['2019-03-01', '2019-05-01'], 'complete': False}
    assert quote_cell(table, '2019-10-01', '2018-10-15') == (UNKNOWN, None)
    assert quote_cell(table, '2019-03-01', '2018-10-15') == (PRESENT, 75.)
    table['complete'] = True
    assert quote_cell(table, '2019-10-01', '2018-10-15') == (ABSENT, None)


@pytest.mark.parametrize('contract', ['2018-03-01', '2019-04-01'])
def test_partial_table_cannot_prove_absence_from_month_or_rank(contract):
    table = {'quotes': {'2019-03-01': '75.00', '2019-05-01': '76.00'},
             'ranked': ['2019-03-01', '2019-05-01'], 'complete': False}
    assert quote_cell(table, contract, '2018-10-15') == (UNKNOWN, None)
    assert source_cell_status((table['quotes'], table['ranked'], False), contract) == UNKNOWN


def test_missing_table_is_not_a_complete_empty_listing():
    table = {'quotes': {}, 'ranked': [], 'complete': False}
    assert quote_cell(table, '2018-03-01', '2020-01-01') == (UNKNOWN, None)
    assert source_cell_status(({}, [], False), '2018-03-01') == UNKNOWN


def test_unknown_ewma_state_does_not_reset_when_har_recovers():
    cells = [(PRESENT, 75.), (UNKNOWN, None), *[(PRESENT, 76.+i*.01) for i in range(24)]]
    rows = [dict(row) for row, _ in trace(cells)]
    assert rows[-1]['har_ready'] is True
    assert rows[-1]['ewma_ready'] is False
    assert rows[-1]['ewma_unknown_cells'] == 1
    assert rows[-1]['known_pair_updates_after_last_unknown'] == 23
    assert rows[-1]['unknown_state_multiplier_after_known_updates'] == pytest.approx(.94**23)


def test_absent_bar_does_not_create_a_compressed_return():
    rows = [dict(row) for row, _ in trace([(PRESENT, 75.), (ABSENT, None), (PRESENT, 80.), (PRESENT, 81.)])]
    assert rows[-1]['known_pair_count'] == 1
    assert rows[-1]['ewma_unknown_cells'] == 0


def test_unknown_adjacent_to_only_absent_bars_cannot_update_state():
    rows = [dict(row) for row, _ in trace([(ABSENT, None), (UNKNOWN, None), (ABSENT, None)])]
    assert rows[-1]['ewma_unknown_cells'] == 0


def test_future_change_does_not_affect_past_readiness():
    cells = [(PRESENT, 75.+i*.01) for i in range(40)]
    old = [dict(row) for row, _ in trace(cells)]
    cells[-1] = (UNKNOWN, None)
    new = [dict(row) for row, _ in trace(cells)]
    assert new[:-1] == old[:-1]
    assert old[-1]['ewma_ready'] is True and new[-1]['ewma_ready'] is False


@pytest.mark.parametrize('suspect', [False, True])
def test_prefix_does_not_discard_unchanged_annual_far_quotes(tmp_path, suspect):
    path = tmp_path/'annual.csv'
    names = ['2018-10-01', '2018-12-01', '2019-03-01', '2019-05-01', '2019-07-01',
             '2019-10-01', '2019-12-01', '2020-03-01', '2020-05-01', '2020-07-01']
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'contract', 'settlement_cents_raw', 'source_date_matches', 'price_precision_2dp', 'container'])
        writer.writeheader()
        writer.writerows({'date': '2018-08-01', 'contract': c, 'settlement_cents_raw': '75.00',
                          'source_date_matches': 'True', 'price_precision_2dp': 'True',
                          'container': '2018-2019 Price file.xlsx' if suspect else '2017-2018-price-file.xlsx'} for c in names)
    merged = {'quotes': {'2018-08-01': dict.fromkeys(names[:6], '75.00')}}
    tables, _ = source_tables(path, {'quotes': {}}, merged, set())
    t = tables['2018-08-01']
    assert len(t['quotes']) == (6 if suspect else 10)
    assert quote_cell(t, names[-1], '2018-08-01')[0] == (UNKNOWN if suspect else PRESENT)
    full = {'2018-08-01': dict.fromkeys(names, '75.00')}
    tables, _ = source_tables(path, {'quotes': {}}, merged, {'2018-08-01'}, full)
    assert tables['2018-08-01']['complete'] is True
    assert tables['2018-08-01']['kind'] == 'verified_full_daily_settlement'
