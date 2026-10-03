"""Offline clock/version evidence must never silently admit historical features."""
import hashlib

import pytest
from cottonlens_ml.sources import ams_publications as ams
from cottonlens_ml.sprint import freeze_record, read_record
from review_ams_availability import (
    CORRECTED,
    LATEST_DOC,
    LISTING,
    PUBLISHED,
    audit,
    checked_request,
    publication_control,
)

DAY = '2025-07-31'
DOC = f'https://mymarketnews.ams.usda.gov/filerepo/sites/default/files/3004/{DAY}/1262140/ams_3004_01363.txt'
INDEX = ('Published Date(msec)\n3004 2025-07-31 12:45:31 MDT '
         f'1753987531803 {DAY} {DAY} {LATEST_DOC} (MP_CN001) Daily Spot Quotations\n')


def cache(root, url, payload):
    checksum = hashlib.sha256(payload).hexdigest()
    target = root / 'raw' / 'ams' / checksum / 'source.bin'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    key = hashlib.sha256(url.encode()).hexdigest()
    freeze_record(root / 'requests' / (key + '.json'), {'url': url, 'sha256': checksum})
    return target


def test_epoch_is_aware_and_not_a_historical_timezone_rule():
    result = publication_control(INDEX)
    assert result == {'report_date': DAY, 'published_at_utc': '2025-07-31T18:45:31.803000+00:00',
                      'epoch_ms': 1753987531803}


@pytest.mark.parametrize('text', [
    INDEX.replace('Published Date(msec)', 'Date'), INDEX.replace('3004 ', '3804 '),
    INDEX + INDEX.splitlines()[1], INDEX.replace(LATEST_DOC, 'https://example.com/report.txt'),
    INDEX.replace(f'{DAY} {DAY}', f'2025-07-30 {DAY}'),
])
def test_changed_control_identity_rejected(text):
    with pytest.raises(ValueError):
        publication_control(text)


def test_corrupt_evidence_bytes_rejected(tmp_path):
    target = cache(tmp_path, PUBLISHED, INDEX.encode())
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='corrupted'):
        checked_request(tmp_path, PUBLISHED)


def fixture(tmp_path):
    history, evidence = tmp_path / 'history', tmp_path / 'evidence'
    values = ['3004', 'MP_CN001', '(MP_CN001) Daily Spot Quotations, excerpts',
              '07-31-2025 01:45:31 pm', DAY, 'Final', f'<a href="{DOC}">view</a>']
    headers = ''.join(f'<th id="{key}">{value}</th>' for key, value in ams.HEADERS.items())
    cells = ''.join(f'<td headers="{key}">{value}</td>' for key, value in zip(ams.HEADERS, values))
    listing = f'<table id="filerepo-reports-table"><tr>{headers}</tr><tr>{cells}</tr></table>'.encode()
    stored = cache(history, LISTING, listing)
    freeze_record(history / 'publication-content-review.json', {
        'model_eligible': False, 'publication_timezone_verified': False, 'first_version_reviewed': False,
        'files': {stored.relative_to(history).as_posix(): hashlib.sha256(listing).hexdigest()},
        'reference_rows': 2, 'matched_dates': 1, 'missing_metadata_dates': ['2020-01-17'],
        'version_records': [{'report_date': '2020-10-28', 'published_local_naive': '2020-10-29T11:21:35',
                             'document_url': 'https://example.com/ams_3004_00174_01.txt',
                             'published_at': None, 'timezone_verified': False, 'model_eligible': False}],
    })
    cache(evidence, PUBLISHED, INDEX.encode())
    cache(evidence, CORRECTED, b'Header\n3804 unrelated report\nTotal: 1 reports\n')
    cache(evidence, LATEST_DOC, b'identical control bytes')
    cache(evidence, DOC, b'identical control bytes')
    return history, evidence


def test_control_and_absent_correction_do_not_admit_history(tmp_path):
    history, evidence = fixture(tmp_path)
    output = tmp_path / 'audit.json'
    result = audit(history, evidence, output)
    assert result['later_displayed_date_count'] == 1 and result['numeric_filename_suffix_count'] == 1
    assert result['delay_calendar_days_distribution'] == {'1': 1}
    assert not result['cotton_slug_in_correction_index']
    assert not result['absence_proves_no_corrections'] and not result['suffix_alone_proves_correction']
    control = result['latest_legacy_epoch_control']
    assert control['observed_display_offset_seconds'] == -18000
    assert not control['applies_to_historical_cohort'] and not control['first_version_verified']
    assert result['historical_utc_times_verified'] == 0 and not result['model_eligible']
    assert read_record(output) == result
    assert audit(history, evidence, output) == result  # Immutable offline replay.


def test_different_control_payload_cannot_certify_clock(tmp_path):
    history, evidence = fixture(tmp_path)
    # A separately acquired, internally checksummed file can still be the wrong version.
    receipt = evidence / 'requests' / (hashlib.sha256(DOC.encode()).hexdigest() + '.json')
    receipt.unlink()
    cache(evidence, DOC, b'different version')
    with pytest.raises(ValueError, match='differs'):
        audit(history, evidence, tmp_path / 'audit.json')
    assert not (tmp_path / 'audit.json').exists()
