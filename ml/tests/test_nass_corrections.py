import hashlib
import json

import pytest
from cottonlens_ml.sources.nass_corrections import (
    crop_notices,
    notice_dates,
    reconcile_archives,
)


def test_notice_preserves_delay_and_revision_text_without_certifying_time():
    html = '''<table><tr><th>Report Name</th><th>Release Date</th><th>Correction Description</th>
    <th>Notification Date</th></tr><tr><td>Crop Progress</td><td>Apr 5, 2021</td>
    <td>Late report,<br>rescheduled for 5:00pm ET.</td><td>Apr 5, 2021</td></tr>
    <tr><td>Other</td><td>x</td><td>y</td><td>z</td></tr></table>'''
    records = crop_notices(html)
    assert len(records) == 1
    assert records[0]['description'] == 'Late report, rescheduled for 5:00pm ET.'
    assert 'published_at' not in records[0]


def test_changed_table_fails_closed():
    with pytest.raises(ValueError, match='headers missing'):
        crop_notices('<html>service unavailable</html>')


def test_rescheduling_uses_explicit_date_not_notification_date():
    assert notice_dates({'release_date_text': 'Nov 28, 2022',
        'description': 'Late report, rescheduled for Nov. 29, 2022 at 4:00pm ET.',
        'notification_date_text': 'Dec 1, 2022'}) == ['2022-11-28', '2022-11-29']


def test_reconciliation_verifies_payload_and_keeps_data_ineligible(tmp_path):
    html = b'''<table><tr><th>Report Name</th><th>Release Date</th><th>Correction Description</th>
    <th>Notification Date</th></tr><tr><td>Crop Progress</td><td>Nov 28, 2022</td>
    <td>Late report, rescheduled for Nov. 29, 2022.</td><td>Nov 29, 2022</td></tr></table>'''
    notices = tmp_path / 'notices'
    notices.mkdir()
    (notices / 'corrections.html').write_bytes(html)
    (notices / 'crop-progress-notices.json').write_text(json.dumps({
        'sha256': hashlib.sha256(html).hexdigest(), 'notices': crop_notices(html.decode())}))
    folder = tmp_path / 'nass_crop_progress/2022-11-29/snapshot'
    folder.mkdir(parents=True)
    report = b'Crop Progress\nReleased November 29, 2022, by USDA.'
    (folder / 'report.txt').write_bytes(report)
    (folder / 'release-page.html').write_bytes(b'official page')
    (folder / 'retrieval.json').write_text(json.dumps({
        'report_sha256': hashlib.sha256(report).hexdigest(),
        'release_page_sha256': hashlib.sha256(b'official page').hexdigest(),
        'release_page_date_field': '2022-11-29T12:00:00Z', 'release_page_url': 'fixture'}))
    result = reconcile_archives(notices, tmp_path)
    assert result['notices'][0]['matched_reports'][0]['date'] == '2022-11-29'
    assert result['model_eligible'] is False
    (folder / 'report.txt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        reconcile_archives(notices, tmp_path)
