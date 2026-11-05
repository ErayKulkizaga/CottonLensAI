"""Rounding compatibility is not source repair or historical-vintage proof."""
import sys
from types import SimpleNamespace

import pytest
from cottonlens_ml.sources.fas_country_review import SUBTYPES, subtype_precision
from test_fas_country_review import CATALOG, sample


def inputs():
    rows, all_upland, reference = sample()
    # Fixed synthetic stock totals: printed grade sums equal the All Upland
    # value, but exact aggregate is 55 bales lower for China. National stock
    # totals still match because the counter-country absorbs the difference.
    rows[0]['accumulatedExports'] -= 60
    rows[0]['currentMYTotalCommitment'] -= 60
    rows[1]['accumulatedExports'] += 60
    rows[1]['currentMYTotalCommitment'] += 60
    sections = []
    for code, heading in SUBTYPES.items():
        value = {1401: '19.8', 1402: '0.2', 1403: '0.0'}[code]
        section = (heading + ' MARKETING YEAR 08/01 - 07/31\n'
                   '1000 RUNNING BALES AS OF JUNE 04, 2020\n'
                   'DESTINATION :THIS WEEK: YR AGO:THIS WEEK: YR AGO :SECOND YR: THIRD YR\n')
        section += ''.join(label + f' : 0.0 99.0 {value} 999.0 0.0 888.0\n'
                           for label in ['CHINA', 'PAKISTN'])
        sections.append(section)
    catalog = CATALOG + [{'commodityCode': code, 'unitId': 2} for code in SUBTYPES]
    return rows, catalog, '\n'.join(sections) + '\n' + all_upland, reference


def test_compatible_rounding_preserves_failed_direct_gate_and_all_admission_blocks():
    result = subtype_precision(*inputs())
    china = result['countries'][0]
    assert china['direct_difference_bales'] == -55
    assert not china['direct_within_50_bales']
    assert china['printed_subtype_sum_bales'] == china['printed_all_upland_bales']
    assert china['api_compatible_with_subtype_rounding']
    assert china['assumed_sum_min_bales'] == 19900  # Zero stock interval starts at zero.
    assert china['assumed_sum_max_bales'] == 20150
    for key in ['publisher_rounding_method_verified', 'exact_api_grade_values_present',
                'rounding_only_cause_verified', 'vintage_verified', 'direct_stock_gate_changed',
                'publication_timestamp_verified', 'first_version_verified', 'model_eligible', 'release_allowed']:
        assert result[key] is False


def test_aggregate_outside_grade_intervals_is_not_explained_away():
    rows, catalog, text, reference = inputs()
    rows[0]['accumulatedExports'] -= 100
    rows[0]['currentMYTotalCommitment'] -= 100
    rows[1]['accumulatedExports'] += 100
    rows[1]['currentMYTotalCommitment'] += 100
    assert not subtype_precision(rows, catalog, text, reference)['countries'][0]['api_compatible_with_subtype_rounding']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unit', 'date', 'suppressed'])
def test_subtype_identity_missingness_and_units_fail_closed(mutation):
    rows, catalog, text, reference = inputs()
    if mutation == 'missing':
        text = text.replace('PAKISTN :', 'OTHER :', 1)
    elif mutation == 'duplicate':
        text = SUBTYPES[1401] + ' MARKETING YEAR 08/01 - 07/31\n' + text
    elif mutation == 'unit':
        text = text.replace('1000 RUNNING BALES', '1000 METRIC TONS', 1)
    elif mutation == 'date':
        text = text.replace('AS OF JUNE 04', 'AS OF JUNE 05', 1)
    else:
        text = text.replace('19.8', '*', 1)
    with pytest.raises(ValueError):
        subtype_precision(rows, catalog, text, reference)


def test_wrong_subtype_catalog_never_accepted():
    rows, catalog, text, reference = inputs()
    catalog[-1]['unitId'] = 1
    with pytest.raises(ValueError, match='catalog'):
        subtype_precision(rows, catalog, text, reference)


def test_cli_requires_reference_before_any_report_or_result(tmp_path, monkeypatch):
    from review_fas_archive import review
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace())
    with pytest.raises(ValueError, match='pinned country reference'):
        review(tmp_path, [], 2020, tmp_path / 'new.json', subtype_review=True)
    assert not (tmp_path / 'new.json').exists()
