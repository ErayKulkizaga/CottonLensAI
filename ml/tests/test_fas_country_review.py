import json
import sys
from types import SimpleNamespace

import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.sources.fas_country_review import (
    country_reference,
    reconcile_countries,
)

NAMES = [('China', 'CHINA', 5700, 'CN'), ('Vietnam', 'VIETNAM', 5520, 'VN'),
         ('Turkey', 'TURKEY', 4890, 'TR'), ('Pakistan', 'PAKISTN', 5350, 'PK')]
CATALOG = [{'commodityCode': 1404, 'commodityName': 'All Upland Cotton', 'unitId': 2}]


def sample():
    html = '<table>' + ''.join(f'<tr><td>{name}</td><td>{code}</td><td>{iso}</td></tr>'
                              for name, _, code, iso in NAMES) + '</table>'
    text = ('This summary is based on reports from exporters for the period May 29-June 4, 2020.\n'
            'ALL UPLAND COTTON MARKETING YEAR 08/01 - 07/31\n'
            '1000 RUNNING BALES AS OF JUNE 04, 2020\n'
            'DESTINATION :THIS WEEK: YR AGO:THIS WEEK: YR AGO :SECOND YR: THIRD YR\n')
    text += ''.join(label + ' : 10.0 99.0 20.0 999.0 5.0 888.0\n' for _, label, _, _ in NAMES)
    text += 'TOTAL KNOWN & UNKNOWN : 40.0 1.0 80.0 1.0 20.0 1.0\n'
    rows = [{'commodityCode': 1404, 'countryCode': code, 'unitId': 2,
             'weekEndingDate': '2020-06-04T00:00:00', 'outstandingSales': 10005,
             'accumulatedExports': 20005, 'nextMYOutstandingSales': 5005,
             'currentMYTotalCommitment': 30010} for _, _, code, _ in NAMES]
    return rows, text, html


def test_country_current_year_columns_and_derived_commitment_are_not_publication_proof():
    rows, text, html = sample()
    result = reconcile_countries(rows, CATALOG, text, html)
    assert result['direct_stock_checks'] == 12 and result['all_stock_values_match']
    assert result['countries'][0]['comparisons']['outstandingSales']['difference_bales'] == 5
    assert result['countries'][0]['commitment_is_derived_not_separately_printed']
    for field in ['weekly_exports_verified', 'weekly_net_sales_verified', 'historical_code_validity_verified',
                  'publication_timestamp_verified', 'first_version_verified', 'model_eligible', 'release_allowed']:
        assert result[field] is False


def test_country_revisions_can_cancel_in_national_total_but_fail_country_check():
    rows, text, html = sample()
    rows[0]['outstandingSales'] += 60
    rows[0]['currentMYTotalCommitment'] += 60
    rows[1]['outstandingSales'] -= 60
    rows[1]['currentMYTotalCommitment'] -= 60
    result = reconcile_countries(rows, CATALOG, text, html)
    assert result['national_stock_values_match']
    assert not result['all_stock_values_match']


@pytest.mark.parametrize('difference,passes', [(50, True), (51, False)])
def test_rounding_bound_not_relaxed_when_country_disagrees(difference, passes):
    rows, text, html = sample()
    delta = difference - 5
    rows[0]['outstandingSales'] += delta
    rows[0]['currentMYTotalCommitment'] += delta
    rows[1]['outstandingSales'] -= delta
    rows[1]['currentMYTotalCommitment'] -= delta
    assert reconcile_countries(rows, CATALOG, text, html)['all_stock_values_match'] is passes


@pytest.mark.parametrize('replace,match', [
    (('PAKISTN :', 'OTHER ASIA :'), 'absent/ambiguous'),
    (('PAKISTN : 10.0', 'PAKISTN : *'), 'suppression'),
    (('1000 RUNNING BALES', '1000 METRIC TONS'), 'running-bale'),
    (('YR AGO:THIS WEEK', 'YR AGO:UNKNOWN'), 'stock header'),
    (('AS OF JUNE 04', 'AS OF JUNE 05'), 'periods differ'),
])
def test_missing_suppressed_wrong_units_or_columns_never_replaced(replace, match):
    rows, text, html = sample()
    with pytest.raises(ValueError, match=match):
        reconcile_countries(rows, CATALOG, text.replace(*replace), html)


def test_missing_api_country_not_zero_filled():
    rows, text, html = sample()
    rows[-1]['countryCode'] = 2000
    with pytest.raises(ValueError, match='API row absent'):
        reconcile_countries(rows, CATALOG, text, html)


def test_cover_date_spelling_and_other_commodity_rows_are_scoped():
    rows, text, html = sample()
    pima = ('AMERICAN PIMA COTTON MARKETING YEAR 08/01 - 07/31\n'
            'CHINA : 999.0 0.0 0.0 0.0 0.0 0.0\n')
    assert reconcile_countries(rows, CATALOG, pima + text + pima, html)['all_stock_values_match']
    assert reconcile_countries(rows, CATALOG, text.replace('JUNE 04,', 'June 4'), html)['all_stock_values_match']


@pytest.mark.parametrize('mutation', ['duplicate', 'iso', 'missing', 'code'])
def test_reference_mapping_must_be_explicit_unique_name_code_iso(mutation):
    _, _, html = sample()
    changes = {'duplicate': html + html, 'iso': html.replace('CN', 'PK'),
               'missing': html.replace('China', 'Other'), 'code': html.replace('5700', 'x')}
    with pytest.raises(ValueError, match='exact country'):
        country_reference(changes[mutation])


def test_review_rejects_corrupt_reference_before_pdf_or_output(tmp_path, monkeypatch):
    from review_fas_archive import review
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace())
    reference = tmp_path / 'reference'
    reference.mkdir()
    source = reference / 'source.html'
    source.write_text(sample()[2])
    receipt = {'source_url': 'https://www.census.gov/foreign-trade/schedules/c/countrycodes.html',
               'sha256': digest(source), 'http_status': 200}
    (reference / 'retrieval.json').write_text(json.dumps(receipt))
    source.write_text('changed')
    with pytest.raises(ValueError, match='reference origin/bytes'):
        # Index verification precedes the optional reference, so the existing
        # helper is patched only to keep this boundary test focused.
        from unittest.mock import patch
        with (patch('review_fas_archive.checked_response', return_value=(source, '2020', {})),
              patch('review_fas_archive.parse_index', return_value=[])):
            review(tmp_path, [], 2020, tmp_path / 'out.json', api_snapshot=source,
                   commodity_catalog=source, country_reference=reference)
    assert not (tmp_path / 'out.json').exists()


@pytest.mark.parametrize('matches,exit_code', [(True, 0), (False, 2)])
def test_cli_retains_negative_review_but_never_reports_it_as_passed(monkeypatch, capsys, matches, exit_code):
    import review_fas_archive
    monkeypatch.setattr(sys, 'argv', ['review', '--index', 'index', '--report', 'report',
                                     '--year', '2020', '--output', 'review.json'])
    monkeypatch.setattr(review_fas_archive, 'review', lambda *args, **kwargs: {
        'responses': [{}], 'distinct_pdf_payloads': 1, 'country_stock_values_match': matches})
    assert review_fas_archive.main() == exit_code
    output = capsys.readouterr()
    assert 'model_eligible=false' in output.out
    assert ('source admission blocked' in output.err) is (matches is False)
