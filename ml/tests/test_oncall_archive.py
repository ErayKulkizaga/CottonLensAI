"""Offline inventory, cache and Tier-A chronology; no real fits or HTTP calls."""
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources import oncall_archive as archive
from cottonlens_ml.sources.public import archive_observation, verify_observation


def observed(root, url, text):
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
            yield text.encode()
    now = datetime(2026, 10, 2, tzinfo=UTC)
    return archive_observation('cftc', url, root,
        session=SimpleNamespace(get=lambda *a, **k: Response()), clock=lambda: now)


def test_inventory_both_year_formats_official_hosts_and_gaps(tmp_path):
    text = '''<a href="/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall010816.html">old</a>
    <a href="/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall012216.html">gap</a>
    <a href="/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall01072021.html">four-year</a>
    <a href="https://cftc-stg.ctacdev.com/deaoncall01072022.html">unreviewed host</a>
    <a href="/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall13072022.html">invalid month</a>
    <a href="/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall01072024.html">seen audit</a>
    <a href="/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall010816.html">duplicate</a>'''
    receipt = observed(tmp_path, archive.INDEX, text)
    first = archive.inventory(receipt)
    assert len(first['files']) == 3
    assert first['counts_by_nominal_year']['2021'] == 1
    assert first['counts_by_nominal_year']['2022'] == 0
    assert len(first['rejected_links']) == 2
    assert first['nominal_gaps_over_10_days'][0]['days'] == 14
    assert not first['model_eligible'] and not first['release_allowed']
    assert archive.inventory(receipt) == first
    with pytest.raises(ValueError, match='Pre-2024'):
        archive.inventory(receipt, last=2024)


def test_completed_acquisition_is_reused_and_corrupt_cache_rejected(tmp_path, monkeypatch):
    reuse = tmp_path / 'old'
    url = 'https://www.cftc.gov/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall010816.html'
    original = observed(reuse, url, 'synthetic source')
    root = tmp_path / 'new'
    freeze_record(root/'inventory.json', {'files': [{'source_url': url}]})
    monkeypatch.setattr(archive, 'archive_observation', lambda *a, **k: pytest.fail('Repeated HTTP request'))
    first = archive.acquire(root, reuse=[reuse])
    assert first == {'saved': 1, 'reused': 1, 'failed': 0, 'requested': 0, 'expected': 1, 'remaining': 0}
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    again = archive.acquire(root)
    assert again['requested'] == 0 and again['saved'] == 1
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert verify_observation(original)['observed_available_at'] == '2026-10-02T00:00:00+00:00'
    marker = read_record(root/'items'/(content_id(url)+'.json'))
    receipt = root/marker['receipt']
    body = verify_observation(receipt)
    (receipt.parent.parent/body['source_file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupted'):
        archive.acquire(root)


def test_repeated_failures_pause_and_are_not_retried_implicitly(tmp_path, monkeypatch):
    root = tmp_path / 'archive'
    urls = [f'https://www.cftc.gov/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall010{i}16.html' for i in range(1, 8)]
    freeze_record(root/'inventory.json', {'files': [{'source_url': url} for url in urls]})
    calls = []
    def fail(kind, url, *a, **k):
        calls.append(url)
        raise RuntimeError('synthetic failed response')
    monkeypatch.setattr(archive, 'archive_observation', fail)
    monkeypatch.setattr(archive.time, 'sleep', lambda n: None)
    first = archive.acquire(root)
    assert first['requested'] == 5 and first['remaining'] == 2
    assert not (root/'.writer-lock').exists()
    second = archive.acquire(root)
    assert second['requested'] == 2 and second['remaining'] == 0
    assert len(calls) == len(set(calls)) == 7
    with pytest.raises(ValueError, match='second'):
        archive.acquire(root, spacing=0)


def report(asof='2016-01-08', release='2016-01-14T20:30:00Z', **extra):
    return {'as_of': asof, 'printed_release_not_before_utc': release,
        'oi_header_matches_report_asof': True, 'footer_date_conflict': False,
        'reporting_threshold_contracts': 100, 'missing_quantity_cells': 0,
        'total_checks': {'all': 'verified'}, 'totals': {'unfixed_sales': 100, 'unfixed_purchases': 20, 'open_interest': 50},
        'source_sha256': 'a'*64, 'source_url': 'https://www.cftc.gov/sample',
        'revision_note_present': True, **extra}


def test_derive_causal_changes_gaps_masks_and_lower_bound_assumption():
    rows = [report(), report('2016-01-15', '2016-01-21T20:30:00Z'),
            report('2016-01-29', '2016-02-04T20:30:00Z'),
            report('2016-02-05', '2016-02-11T20:30:00Z', footer_date_conflict=True)]
    frame = archive.derive(rows)
    assert frame.assumed_day.iloc[0] == pd.Timestamp('2016-01-15')
    assert frame.net_share.iloc[0] == 1.6  # Position ratios need not be <=1.
    assert np.isnan(frame.net_share_change.iloc[0])
    assert frame.net_share_change.iloc[1] == 0
    assert np.isnan(frame.net_share_change.iloc[2])  # Missing week never compressed.
    assert np.isnan(frame.net_share.iloc[3])
    assert archive.derive(rows[:2]).equals(frame.iloc[:2].reset_index(drop=True))
    delayed = archive.derive([report(release='2016-03-01T20:30:00Z')])
    assert delayed.assumed_day.iloc[0] == pd.Timestamp('2016-03-02')


@pytest.mark.parametrize('extra,reason', [
    ({'oi_header_matches_report_asof': False}, 'oi_date'),
    ({'printed_release_not_before_utc': None}, 'footer_missing'),
    ({'reporting_threshold_contracts': 50}, 'threshold'),
    ({'missing_quantity_cells': 1}, 'quantity'),
    ({'totals': {'unfixed_sales': 100, 'unfixed_purchases': 20, 'open_interest': 0}}, 'nonpositive'),
])
def test_ambiguous_inputs_are_masked_not_origins(extra, reason):
    frame = archive.derive([report(**extra)])
    assert len(frame) == 1 and np.isnan(frame.net_share.iloc[0])
    assert reason in frame.input_mask_reason.iloc[0]


def test_duplicate_vintages_and_audit_input_are_rejected():
    with pytest.raises(ValueError, match='vintages'):
        archive.derive([report(), report()])
    with pytest.raises(ValueError, match='Audit'):
        archive.derive([report(asof='2024-01-05')])


def test_latest_date_is_unverified_delayed_assumption_not_deleted_quantity():
    records = [report(footer_date_conflict=True, footer_additional_dates=['2016-01-13']),
        report('2016-01-15', '2016-01-21T20:30:00Z', footer_date_conflict=True,
               footer_additional_dates=['2016-01-26'])]
    old = archive.derive(records)
    assert old.net_share.isna().all()
    new = archive.derive(records, footer_policy='latest_date_assumption')
    assert new.net_share.eq(1.6).all()
    assert new.assumed_day.tolist() == [pd.Timestamp('2016-01-15'), pd.Timestamp('2016-01-28')]
    assert new.net_share_change.iloc[1] == 0
    # This policy never repairs other source ambiguities or drops evaluation rows.
    new = archive.derive([report(oi_header_matches_report_asof=False,
        footer_date_conflict=True, footer_additional_dates=['2016-01-13'])],
        footer_policy='latest_date_assumption')
    assert new.net_share.isna().all() and len(new) == 1
    with pytest.raises(ValueError, match='supported footer'):
        archive.derive(records, footer_policy='guess_publication')


def test_compile_refuses_incomplete_acquisition(tmp_path):
    freeze_record(tmp_path/'inventory.json', {'files': [{'source_url': 'https://www.cftc.gov/sample'}]})
    with pytest.raises(ValueError, match='incomplete'):
        archive.compile_table(tmp_path, tmp_path/'out.csv')
    assert not (tmp_path/'out.csv').exists()


@pytest.mark.parametrize('footer_policy', ['mask_conflicts', 'latest_date_assumption'])
def test_compiled_packet_resumes_exactly_and_corrupt_raw_is_fatal(tmp_path, footer_policy):
    url = 'https://www.cftc.gov/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall010816.html'
    text = '''<p>Reported as of 01/08/2016 (In Contracts)</p><table>
    <tr><th>Unfixed Call Sales</th><th>Change</th><th>Unfixed Call Purchases</th>
    <th>Change</th><th>At Close 01/08/2016</th><th>Change</th></tr>
    <tr><th>March 2016</th><td>100</td><td>0</td><td>20</td><td>0</td><td>50</td><td>0</td></tr>
    <tr><th>Totals</th><td>100</td><td>0</td><td>20</td><td>0</td><td>50</td><td>0</td></tr>
    </table><p>Merchants with futures positions of 100 or more contracts</p>
    <p>Release after 3:30 p.m. Eastern time, January 14, 2016.</p>'''
    if footer_policy == 'latest_date_assumption':
        text = text.replace('January 14, 2016.</p>', 'January 14, 2016.<br>01/13/2016</p>')
    receipt = observed(tmp_path/'archive', url, text)
    freeze_record(tmp_path/'inventory.json', {'files': [{'source_url': url}]})
    freeze_record(tmp_path/'items'/(content_id(url)+'.json'), {'source_url': url, 'status': 'observed',
        'receipt': receipt.relative_to(tmp_path).as_posix()})
    table = tmp_path/'source.csv'
    first = archive.compile_table(tmp_path, table, footer_policy=footer_policy)
    before = {p: p.read_bytes() for p in (table, table.with_suffix('.manifest.json'))}
    assert first == archive.compile_table(tmp_path, table, footer_policy=footer_policy)
    assert before == {p: p.read_bytes() for p in before}
    assert first['model_eligible'] is False and first['first_version_verified'] is False
    assert first['rows'] == 1 and first['masked_reports'] == 0
    assert pd.read_csv(table).net_share.iloc[0] == 1.6
    body = verify_observation(receipt)
    (receipt.parent.parent/body['source_file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupted'):
        archive.compile_table(tmp_path, table, footer_policy=footer_policy)
    assert before == {p: p.read_bytes() for p in before}
