"""Only actual successful retrievals establish observed availability, never history."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import requests
from cottonlens_ml.sources.public import archive_observation, verify_observation

URL = 'https://www.ams.usda.gov/mnreports/ams_3804.pdf'
NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)


class Response:
    status_code = 200

    def __init__(self, fail=False):
        self.fail = fail
        self.headers = {'Content-Type': 'application/pdf', 'Date': 'Mon, 01 Jan 2001 00:00:00 GMT'}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, _):
        yield b'%PDF-1.7 synthetic fixture'
        if self.fail:
            raise requests.ConnectionError('interrupted transfer')


def session(calls, *, fail=False):
    def get(url, **kwargs):
        calls.append(url)
        return Response(fail)
    return SimpleNamespace(get=get)


def clock(*values):
    return iter(values).__next__


def test_fresh_request_records_completion_not_old_http_date(tmp_path):
    calls = []
    first = archive_observation('ams', URL, tmp_path, session=session(calls),
                                clock=clock(NOW, NOW + timedelta(seconds=1)))
    second = archive_observation('ams', URL, tmp_path, session=session(calls),
                                 clock=clock(NOW + timedelta(days=1), NOW + timedelta(days=1, seconds=2)))
    before, after = verify_observation(first), verify_observation(second)
    assert calls == [URL, URL]  # Immutable bytes being cached never skips observation.
    assert first != second and before['source_sha256'] == after['source_sha256']
    assert before['observed_available_at'] == '2026-10-01T12:00:01+00:00'
    assert after['observed_available_at'] == '2026-10-02T12:00:02+00:00'
    assert before['published_at'] is None and not before['model_eligible']
    with pytest.raises(ValueError, match='cutoff'):
        verify_observation(first, cutoff=NOW)
    assert verify_observation(first, cutoff=NOW + timedelta(seconds=1)) == before


def test_partial_transfer_has_no_observation_or_completed_raw_bytes(tmp_path):
    with pytest.raises(requests.ConnectionError):
        archive_observation('ams', URL, tmp_path, session=session([], fail=True), clock=clock(NOW))
    assert not list(tmp_path.rglob('source.bin'))
    assert not list((tmp_path / 'observations').glob('*.json'))


@pytest.mark.parametrize('values', [(NOW, NOW - timedelta(seconds=1)), (NOW.replace(tzinfo=None),)])
def test_invalid_local_clock_cannot_create_observation(tmp_path, values):
    with pytest.raises(ValueError, match='clock|timezone'):
        archive_observation('ams', URL, tmp_path, session=session([]), clock=clock(*values))
    assert not list((tmp_path / 'observations').glob('*.json'))


def test_bad_snapshot_and_record_are_rejected(tmp_path):
    record = archive_observation('ams', URL, tmp_path, session=session([]), clock=clock(NOW, NOW))
    info = verify_observation(record)
    raw = tmp_path / info['source_file']
    raw.write_bytes(b'changed')
    with pytest.raises(ValueError, match='corrupted'):
        verify_observation(record)
    record.write_text('{}')
    with pytest.raises(ValueError, match='Corrupt completed'):
        verify_observation(record)


def test_observation_cannot_be_frozen_as_publication_evidence(tmp_path):
    from cottonlens_ml.sprint import freeze_record

    record = archive_observation('ams', URL, tmp_path, session=session([]), clock=clock(NOW, NOW))
    body = verify_observation(record)
    forged = record.parent / 'not-historical.json'
    freeze_record(forged, {**body, 'published_at': '2020-01-01T00:00:00Z',
                           'publication_timestamp_verified': True})
    with pytest.raises(ValueError, match='historical publication'):
        verify_observation(forged)
