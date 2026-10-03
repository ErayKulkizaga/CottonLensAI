"""Synthetic reports test parsing/time boundaries; no network or market fits."""
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.oncall import audit_receipt, parse_report
from cottonlens_ml.sources.public import archive_observation, verify_observation


def modern(date='01/08/2016', release='January 14, 2016', footer='', second='0'):
    return f'''<p>Reported as of {date}</p><p>(In Contracts)</p><table>
    <tr><th>Unfixed Call Sales</th><th>Change From Previous Week</th>
    <th>Unfixed Call Purchases</th><th>Change From Previous Week</th>
    <th>At Close {date}</th><th>Change From Previous Week</th></tr>
    <tr><th>March 2016</th><td>1,000</td><td>-2*</td><td>500</td><td>3</td><td>2,000</td><td>5</td></tr>
    <tr><th>May 2016</th><td>0</td><td>0</td><td>0</td><td>0</td><td>{second}</td><td>0</td></tr>
    <tr><th>Totals</th><td>1,000</td><td>-2*</td><td>500</td><td>3</td><td>2,000</td><td>5</td></tr>
    </table><p>Merchants with futures positions of 100 or more contracts in one future.</p>
    <p>*Changes based on revised data for last week.</p>
    <p>Release after 3:30 p.m. Eastern time, {release}.<br>{footer}</p>'''


def test_modern_integer_totals_zero_oi_and_revision():
    report = parse_report(modern())
    assert report['totals']['unfixed_sales'] == 1000
    assert set(report['total_checks'].values()) == {'verified'}
    assert report['zero_oi_rows'] == 1
    assert report['contracts'][0]['revised_change_fields'] == ['sales_change']
    assert report['reporting_threshold_contracts'] == 100
    assert report['oi_header_matches_report_asof']
    assert report['printed_release_not_before_utc'] == '2016-01-14T20:30:00+00:00'
    assert report['release_bound_exclusive']
    assert not report['model_eligible'] and not report['historical_vintage_verified']
    assert report['published_at'] is None and report['available_by_historical'] is None


def test_delayed_release_uses_footer_and_dst_not_url_or_asof():
    report = parse_report(modern('10/03/2025', 'November 18, 2025', '11/18/2025'))
    assert report['as_of'] == '2025-10-03'
    assert report['printed_release_not_before_utc'] == '2025-11-18T20:30:00+00:00'
    summer = parse_report(modern('06/10/2016', 'June 16, 2016'))
    assert summer['printed_release_not_before_utc'] == '2016-06-16T19:30:00+00:00'


def test_legacy_pre_format_and_revised_changes():
    report = parse_report('''<pre>reported as of 10/12/01
    (In Contracts)
    |March '02 1,000 -2* 500 3 2,000 5
    |May '02 0 0 0 0 0 0
    |Totals: 1,000 -2* 500 3 2,000 5
    |Merchants with futures positions of 50 or more contracts in one future.
    |*Changes are based on revised data for last week.
    |Released after 3:00 p.m. Eastern Time
    |10/18/01
    </pre>''')
    assert report['contracts'][0]['delivery_month'] == '2002-03'
    assert report['reporting_threshold_contracts'] == 50
    assert report['printed_release_not_before_utc'] == '2001-10-18T19:00:00+00:00'
    assert report['revision_note_present']


@pytest.mark.parametrize('value', ['January 8, 2016', 'Jan 8, 2016', 'Jan. 8, 2016'])
def test_early_table_written_and_abbreviated_asof_dates(value):
    text = modern().replace('Reported as of 01/08/2016', f'Reported as of {value}')
    assert parse_report(text)['as_of'] == '2016-01-08'


def test_missing_is_not_zero_and_unknown_total_is_reported():
    report = parse_report(modern(second='—'))
    assert report['contracts'][1]['open_interest'] is None
    assert report['zero_oi_rows'] == 0 and report['missing_quantity_cells'] == 1
    assert report['total_checks']['open_interest'] == 'unknown_missing_quantity'


def test_ambiguous_dates_are_flagged_never_certified():
    text = modern('12/27/2019', 'January 3, 2020', '01/02/2020')
    report = parse_report(text.replace('At Close 12/27/2019', 'At Close 12/027/2019'))
    assert report['footer_date_conflict'] and not report['oi_header_matches_report_asof']
    assert report['printed_release_not_before_utc'] == '2020-01-03T20:30:00+00:00'
    assert not report['model_eligible']
    no_footer = parse_report(text.replace('Release after', 'Scheduled after'))
    assert no_footer['printed_release_not_before_utc'] is None


@pytest.mark.parametrize('old,new,error', [
    ('<td>1,000</td>', '<td>1,001</td>', 'total'),
    ('In Contracts', 'In Bales', 'units'),
    ('<td>500</td>', '<td>-500</td>', 'Negative'),
    ('<td>500</td>', '<td>500.5</td>', 'integer'),
    ('May 2016', 'March 2016', 'Unique'),
    ('May 2016', 'February 2016', 'delivery'),
    ('Unfixed Call Sales', 'Unfixed Call Purchases', 'semantics'),
    ('January 14, 2016', 'January 7, 2016', 'precedes'),
])
def test_invalid_reports_fail_closed(old, new, error):
    # Change one field; changing its printed total too would mask corruption.
    with pytest.raises(ValueError, match=error):
        parse_report(modern().replace(old, new, 1))


def test_verified_receipt_resume_corruption_and_observation_cutoff(tmp_path):
    class Response:
        status_code = 200
        def __init__(self):
            self.headers = {'Content-Type': 'text/html'}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, _):
            yield modern().encode()
    now = datetime(2026, 10, 2, tzinfo=UTC)
    receipt = archive_observation('cftc',
        'https://www.cftc.gov/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall010816.html',
        tmp_path / 'archive', session=SimpleNamespace(get=lambda *a, **kw: Response()), clock=lambda: now)
    result = audit_receipt(receipt)
    output = tmp_path / 'parsed.json'
    freeze_record(output, result)
    before = output.read_bytes()
    freeze_record(output, audit_receipt(receipt))
    assert output.read_bytes() == before
    assert read_record(output)['observed_available_at'] == '2026-10-02T00:00:00+00:00'
    with pytest.raises(ValueError, match='cutoff'):
        verify_observation(receipt, cutoff='2016-01-15T00:00:00Z')
    info = verify_observation(receipt)
    (tmp_path / 'archive' / info['source_file']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='corrupted'):
        audit_receipt(receipt)
