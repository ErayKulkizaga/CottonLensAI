"""A valid PDF response can still be future data for the selected historical row."""
import json

import pytest
from cottonlens_ml.sources.fas_archive import (
    embargo_declaration,
    identity_check,
    parse_index,
    reconcile_upland,
    report_period_end,
    upland_stock_totals,
)

ID = '2248fde3-edca-41ba-99a5-0274a21e9a7c'


def row(**changes):
    return {'id': ID, 'fullFileName': '2020/06/04', 'fileType': 'ESRWeeklyReport',
            'fileExtension': 'pdf', 'weekEndingDate': '2020-06-04T00:00:00',
            'createdTime': '0001-01-01T00:00:00', **changes}


def report(period):
    return f'ESR WEEKLY REPORT\nWEEKLY HIGHLIGHTS\nThis summary is based on reports from exporters for the period {period}.'


def test_live_2026_pdf_for_2020_archive_row_is_rejected():
    records = parse_index(json.dumps([row()]), 2020)
    result = identity_check(records, ID, report('September 18-24, 2026'))
    assert result['content_identity_status'] == 'future_period_for_historical_row'
    assert result['pdf_report_period_end'] == '2026-09-24'
    assert not result['numeric_reconciliation_allowed'] and not result['model_eligible']
    assert not result['publication_timestamp_verified'] and not result['vintage_verified']


def test_matching_content_never_promotes_sentinel_timestamp():
    records = parse_index(json.dumps([row()]), 2020)
    result = identity_check(records, ID, report('May 29-June 4, 2020'))
    assert result['content_identity_status'] == 'historical_period_unverified'
    assert result['index_date_equals_pdf_period_end']
    assert not result['numeric_reconciliation_allowed'] and not result['index_date_semantics_verified']
    assert result['created_time_literal'] == '0001-01-01T00:00:00'
    assert not result['publication_timestamp_verified'] and not result['model_eligible']


def test_prior_week_is_not_rejected_by_assuming_index_labels_reporting_period():
    records = parse_index(json.dumps([row()]), 2020)
    result = identity_check(records, ID, report('May 22-28, 2020'))
    assert result['content_identity_status'] == 'historical_period_unverified'
    assert not result['index_date_equals_pdf_period_end']
    assert not result['numeric_reconciliation_allowed']


@pytest.mark.parametrize('changes', [
    {'id': 'bad'}, {'fullFileName': '2020/05/28'}, {'fileExtension': 'html'},
    {'id': 123},
    {'weekEndingDate': '2026-09-24T00:00:00'}, {'weekEndingDate': '2020-06-04T01:00:00'},
])
def test_invalid_index_identity_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        parse_index(json.dumps([row(**changes)]), 2020)


def test_duplicate_and_absent_selection_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        parse_index(json.dumps([row(), row()]), 2020)
    with pytest.raises(ValueError, match='selected archive'):
        identity_check(parse_index(json.dumps([row()]), 2020), 'unknown', report('May 29-June 4, 2020'))


def test_ambiguous_period_and_invalid_date_never_guessed():
    with pytest.raises(ValueError, match='unambiguous'):
        report_period_end(report('May 29-June 4, 2020') + report('September 18-24, 2026'))
    with pytest.raises(ValueError):
        report_period_end(report('June 1-31, 2020'))
    assert report_period_end(report('December 25, 2020-January 1, 2021')).isoformat() == '2021-01-01'


def test_actual_legacy_cam_first_page_without_modern_title_is_supported():
    legacy = 'This summary is based on reports from exporters for the period May 29-June 4, 2020.\nWheat: '
    assert report_period_end(legacy).isoformat() == '2020-06-04'
    with pytest.raises(ValueError, match='unambiguous'):
        report_period_end('An unrelated PDF dated June 4, 2020')


def test_report_specific_embargo_never_becomes_actual_publication_time():
    stamp = 'EMBARGOED UNTIL 8:30 AM\nJUNE 11, 2020'
    result = embargo_declaration(stamp)
    assert result['scheduled_release_date'] == '2020-06-11'
    assert result['local_time_literal'] == '8:30 AM' and result['timezone'] is None
    assert not result['publication_timestamp_verified'] and not result['model_eligible']
    actual_columns = ('Outstanding Export Sales on June 4, 2020 EMBARGOED UNTIL 8:30 AM '
                      'OF AGRICULTURE AGRICULTURAL WASHINGTON, D.C. 20250 SERVICE JUNE 11, 2020')
    assert embargo_declaration(actual_columns) == result
    for invalid in [stamp + '\n' + stamp, 'JUNE 11, 2020', stamp.replace('8:30', '25:00'),
                    stamp + '\nMAY 28, 2020']:
        with pytest.raises(ValueError):
            embargo_declaration(invalid)


def cotton_table(values='5115.6 4519.4 11609.7 10649.0 3027.7 239.6'):
    return (report('May 29-June 4, 2020') + '\nALL UPLAND COTTON MARKETING YEAR 08/01 - 07/31\n'
            '1000 RUNNING BALES AS OF JUNE 04, 2020\nTOTAL KNOWN & UNKNOWN : ' + values + '\n')


CATALOG = [{'commodityCode': 1404, 'commodityName': 'All Upland Cotton', 'unitId': 2}]


def api_row(**changes):
    return {'commodityCode': 1404, 'countryCode': 2010, 'unitId': 2,
            'weekEndingDate': '2020-06-04T00:00:00', 'outstandingSales': 5115630,
            'accumulatedExports': 11609744, 'nextMYOutstandingSales': 3027694, **changes}


def test_actual_printed_precision_reconciliation_is_content_only():
    result = reconcile_upland([api_row()], CATALOG, cotton_table())
    assert result['content_matches_at_printed_precision']
    assert result['comparisons']['outstandingSales']['difference_bales'] == 30
    assert not result['model_eligible'] and not result['vintage_verified']
    at_limit = reconcile_upland([api_row(outstandingSales=5115650)], CATALOG, cotton_table())
    assert at_limit['content_matches_at_printed_precision']
    mismatch = reconcile_upland([api_row(outstandingSales=5115651)], CATALOG, cotton_table())
    assert not mismatch['content_matches_at_printed_precision']


@pytest.mark.parametrize('changes', [{'unitId': 1}, {'commodityCode': 1301},
                                     {'outstandingSales': -1}, {'outstandingSales': 5115600.0}])
def test_wrong_units_commodity_or_stock_schema_rejected(changes):
    with pytest.raises(ValueError):
        reconcile_upland([api_row(**changes)], CATALOG, cotton_table())


def test_duplicate_countries_and_date_mismatches_rejected():
    with pytest.raises(ValueError, match='double-count'):
        reconcile_upland([api_row(), api_row()], CATALOG, cotton_table())
    with pytest.raises(ValueError, match='periods differ'):
        reconcile_upland([api_row()], CATALOG, cotton_table().replace('AS OF JUNE 04', 'AS OF JUNE 05'))


def test_pima_subtotals_missing_units_and_suppression_never_substituted():
    pima = 'AMERICAN PIMA COTTON MARKETING YEAR 08/01 - 07/31\nTOTAL KNOWN & UNKNOWN : 1.0 2.0 3.0 4.0 5.0 6.0\n'
    assert upland_stock_totals(pima + cotton_table())['totals']['outstandingSales'] == 5115600
    for invalid in [cotton_table().replace('1000 RUNNING BALES', 'METRIC TONS'),
                    cotton_table(values='* 4519.4 11609.7 10649.0 3027.7 239.6'),
                    cotton_table() + cotton_table(), cotton_table().replace('TOTAL KNOWN & UNKNOWN', 'TOTAL KNOWN')]:
        with pytest.raises(ValueError):
            upland_stock_totals(invalid)


def test_corrupt_response_bytes_and_wrong_source_identity_rejected(tmp_path):
    from cottonlens_ml.code_identity import digest
    from cottonlens_ml.sprint import freeze_record
    from review_fas_archive import checked_response

    (tmp_path / 'source.bin').write_bytes(b'archived evidence')
    receipt = {
        'kind': 'export_sales', 'sha256': digest(tmp_path / 'source.bin'),
        'source_url': f'https://apps.fas.usda.gov/esrqs/api/reports/GetPdfFile?Id={ID}',
        'retrieved_at': '2026-10-01T12:00:00+00:00',
    }
    freeze_record(tmp_path / 'retrieval.json', receipt)
    raw, key, _ = checked_response(tmp_path, 'GetPdfFile', 'Id')
    assert key == ID and raw.read_bytes() == b'archived evidence'
    with pytest.raises(ValueError, match='identity'):
        checked_response(tmp_path, 'GetArchivedWeeklyReportsList', 'selectedYear')
    raw.write_bytes(b'changed evidence')
    with pytest.raises(ValueError, match='corrupted'):
        checked_response(tmp_path, 'GetPdfFile', 'Id')


@pytest.mark.parametrize('publisher', ['USDAFAS', 'OTHER'])
def test_release_cover_requires_exact_publisher_and_unchanged_bytes(tmp_path, publisher):
    from cottonlens_ml.code_identity import digest
    from cottonlens_ml.sprint import freeze_record
    from review_fas_archive import checked_release_cover

    raw = tmp_path / 'govdelivery-wr06042020.pdf'
    raw.write_bytes(b'%PDF-synthetic identity fixture')
    freeze_record(tmp_path / 'retrieval.json', {
        'publisher_account': publisher, 'sha256': digest(raw),
        'source_url': 'https://content.govdelivery.com/attachments/USDAFAS/2020/06/10/report.pdf',
        'retrieved_at': '2026-10-01T13:00:00+00:00',
    })
    if publisher != 'USDAFAS':
        with pytest.raises(ValueError, match='identity'):
            checked_release_cover(tmp_path)
        return
    assert checked_release_cover(tmp_path)[0] == raw
    raw.write_bytes(b'changed')
    with pytest.raises(ValueError, match='corrupted'):
        checked_release_cover(tmp_path)


def test_complete_offline_review_never_promotes_embargo_or_numeric_match(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from cottonlens_ml.code_identity import digest
    from cottonlens_ml.sprint import freeze_record
    from review_fas_archive import review

    index, response, cover = [tmp_path / n for n in ['index', 'report', 'cover']]
    for folder in [index, response, cover]:
        folder.mkdir()
    (index / 'source.bin').write_text(json.dumps([row()]))
    (response / 'source.bin').write_bytes(b'%PDF-synthetic report')
    for folder, url in [
        (index, 'https://apps.fas.usda.gov/esrqs/api/reports/GetArchivedWeeklyReportsList?selectedYear=2020'),
        (response, f'https://apps.fas.usda.gov/esrqs/api/reports/GetPdfFile?Id={ID}'),
    ]:
        freeze_record(folder / 'retrieval.json', {'source_url': url, 'kind': 'export_sales',
                      'sha256': digest(folder / 'source.bin'), 'retrieved_at': '2026-10-01T13:00:00Z'})
    raw = cover / 'govdelivery-wr06042020.pdf'
    raw.write_bytes(b'%PDF-synthetic cover')
    freeze_record(cover / 'retrieval.json', {
        'publisher_account': 'USDAFAS', 'sha256': digest(raw), 'retrieved_at': '2026-10-01T13:00:00Z',
        'source_url': 'https://content.govdelivery.com/attachments/USDAFAS/2020/06/10/report.pdf'})
    def reader(path):
        texts = (['EMBARGOED UNTIL 8:30 AM JUNE 11, 2020', cotton_table()]
                 if path == raw else [cotton_table()])
        return SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda t=t: t) for t in texts], metadata={})
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace(PdfReader=reader, __version__='synthetic'))
    api, catalog = tmp_path / 'api.json', tmp_path / 'catalog.json'
    api.write_text(json.dumps([api_row()]))
    catalog.write_text(json.dumps(CATALOG))
    output = tmp_path / 'review.json'
    result = review(index, [response], 2020, output, api_snapshot=api, commodity_catalog=catalog,
                    release_cover=cover)
    assert result['responses'][0]['cotton_content_comparison']['content_matches_at_printed_precision']
    assert result['release_cover']['scheduled_release_date'] == '2020-06-11'
    assert result['release_cover']['report_period_end'] == '2020-06-04'
    assert not result['model_eligible'] and not result['publication_timestamp_verified']
    assert result['release_cover']['timezone'] is None
    assert not result['release_cover']['same_version_as_selected_archive_verified']
    before = output.read_bytes()
    assert review(index, [response], 2020, output, api_snapshot=api, commodity_catalog=catalog,
                  release_cover=cover) == result
    assert output.read_bytes() == before
