from copy import deepcopy

import pytest

from research.source_request_matrix import summarize


def fixture():
    cells = [{'source_date': '2010-01-04', 'contract': '2012-12-01',
              'first_using_origin': '2012-08-31', 'last_using_origin': '2012-09-04',
              'using_origin_count': '2', 'kind': 'no_accepted_original_source_table'}]
    return cells, [{'ewma_unknown_cells': '1'}, {'ewma_unknown_cells': '1'}]


def test_request_does_not_infer_contract_was_listed():
    cells, origins = fixture()
    rows, edges = summarize(cells, origins)
    assert edges == 2 and rows[0]['unknown_cells_upper_bound'] == 1
    assert 'official_listed_on' in rows[0]['required_evidence']
    assert 'listed_on' not in rows[0]


@pytest.mark.parametrize('kind', ['dated_daily', 'verified_full_daily_settlement',
                                 'unchanged_annual_outside_known_bad_blocks',
                                 'unchanged_annual_outside_known_bad_blocks_with_prefix_witness'])
def test_complete_report_without_listing_coverage_is_a_request_not_absence(kind):
    cells, origins = fixture()
    cells[0]['kind'] = kind
    rows, _ = summarize(cells, origins)
    assert rows[0]['report_without_listing_coverage'] == 1
    assert rows[0]['unknown_cells_upper_bound'] == 1


@pytest.mark.parametrize('kind,reason', [('duplicate', 'Duplicate'), ('count', 'totals differ'),
                                       ('future', 'interval'), ('unknown_kind', 'Unrecognized')])
def test_bad_request_inputs_rejected_and_clean_control_survives(kind, reason):
    cells, origins = fixture()
    changed = deepcopy(cells)
    if kind == 'duplicate':
        changed.append(deepcopy(changed[0]))
    elif kind == 'count':
        changed[0]['using_origin_count'] = '3'
    elif kind == 'future':
        changed[0]['source_date'] = '2013-01-01'
    else:
        changed[0]['kind'] = 'guessed_absence'
    with pytest.raises(ValueError, match=reason):
        summarize(changed, origins)
    assert summarize(cells, origins)[1] == 2
