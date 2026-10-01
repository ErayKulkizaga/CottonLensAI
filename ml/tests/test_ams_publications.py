"""Publication evidence, resume and corruption checks; no network or model fits."""
import hashlib
from pathlib import Path

import pytest
import requests
from cottonlens_ml.sources import ams_publications as ams
from cottonlens_ml.sprint import freeze_record

DAY = '2023-07-31'
DOC = f'https://mymarketnews.ams.usda.gov/filerepo/sites/default/files/3004/{DAY}/1029205/ams_3004_00863_01.txt'
URL = ams.INDEX + '?field_slug_id_value=3004'


def listing(*, slug='3004', day=DAY, link=DOC, clock='07-31-2023 02:20:30 pm', next_link=''):
    values = [slug, 'MP_CN001', '(MP_CN001) Daily Spot Quotations, excerpts',
              clock, day, 'Final', f'<a href="{link}">view report</a>']
    headers = ''.join(f'<th id="{key}">{value}</th>' for key, value in ams.HEADERS.items())
    cells = ''.join(f'<td headers="{key}">{value}</td>' for key, value in zip(ams.HEADERS, values))
    return (f'<table id="filerepo-reports-table"><tr>{headers}</tr><tr>{cells}</tr></table>'
            + (f'<a rel="next" href="{next_link}">Next</a>' if next_link else '')).encode()


def test_unzoned_clock_and_final_status_never_certify_publication():
    rows, next_url = ams.parse_listing(listing(), URL)
    assert rows[0]['published_local_naive'] == '2023-07-31T14:20:30'
    assert rows[0]['published_at'] is None
    assert not rows[0]['timezone_verified'] and not rows[0]['model_eligible']
    assert rows[0]['document_url'] == DOC and next_url is None


@pytest.mark.parametrize('changes', [
    {'slug': '3804'}, {'link': 'https://example.com/report.txt'},
    {'link': DOC.replace(DAY, '2023-07-30')},
    {'clock': '07-30-2023 02:20:30 pm'},
])
def test_changed_slug_link_or_date_rejected(changes):
    with pytest.raises(ValueError):
        ams.parse_listing(listing(**changes), URL)


def test_date_filter_and_pagination_cannot_silently_change():
    with pytest.raises(ValueError, match='date filter'):
        ams.parse_listing(listing(), URL + '&field_report_date_end_value=2023-07-30')
    for next_url in [URL + '&page=0', URL.replace('3004', '3804') + '&page=1']:
        with pytest.raises(ValueError):
            ams.parse_listing(listing(next_link=next_url), URL)
    rows, next_url = ams.parse_listing(listing(next_link=URL + '&page=1'), URL)
    assert len(rows) == 1 and next_url.endswith('page=1')


def test_legacy_uppercase_filename_retains_exact_embedded_report_date():
    legacy = f'https://mymarketnews.ams.usda.gov/filerepo/sites/default/files/3004/{DAY}/1029205/MP_CN00120230731.TXT'
    rows, _ = ams.parse_listing(listing(link=legacy), URL)
    assert rows[0]['document_url'] == legacy and not rows[0]['model_eligible']
    with pytest.raises(ValueError, match='versioned MMN report'):
        ams.parse_listing(listing(link=legacy.replace('MP_CN00120230731', 'MP_CN00120230730')), URL)


def fixture_table(tmp_path):
    payload = b'original synthetic report bytes'
    table = tmp_path / 'reference.csv'
    checksum = hashlib.sha256(payload).hexdigest()
    table.write_text(f'report_date,report_sha256\n{DAY},{checksum}\n', encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    freeze_record(manifest, {'table_sha256': hashlib.sha256(table.read_bytes()).hexdigest(), 'row_count': 1})
    return table, manifest, payload


def fake_archive(monkeypatch, payload):
    calls = []

    def fetch(kind, url, root, *, session=None):
        calls.append(url)
        raw = payload if url == DOC else listing()
        folder = Path(root) / kind / hashlib.sha256(raw).hexdigest()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'source.bin').write_bytes(raw)
        return folder

    monkeypatch.setattr(ams, 'archive', fetch)
    return calls


def test_byte_match_resume_and_corruption_guard(tmp_path, monkeypatch):
    table, manifest, payload = fixture_table(tmp_path)
    calls = fake_archive(monkeypatch, payload)
    output = tmp_path / 'review'
    result = ams.review_history(table, manifest, output)
    assert result['matched_dates'] == 1 and result['unmatched_dates'] == []
    assert not result['model_eligible'] and not result['first_version_reviewed']
    assert len(calls) == 2
    assert ams.review_history(table, manifest, output) == result
    assert len(calls) == 2
    stored = next(path for path in result['files'] if (output / path).read_bytes() == payload)
    (output / stored).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupted'):
        ams.review_history(table, manifest, output)


def test_interruption_reuses_listing_and_rejects_changed_diagnostic(tmp_path, monkeypatch):
    table, manifest, payload = fixture_table(tmp_path)
    output = tmp_path / 'review'
    calls = fake_archive(monkeypatch, payload)
    original = ams.archive

    def interrupted(kind, url, root, *, session=None):
        if url == DOC:
            raise RuntimeError('synthetic interruption')
        return original(kind, url, root, session=session)

    monkeypatch.setattr(ams, 'archive', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        ams.review_history(table, manifest, output)
    assert not (output / 'publication-content-review.json').exists()
    monkeypatch.setattr(ams, 'archive', original)
    assert ams.review_history(table, manifest, output)['matched_dates'] == 1
    assert len(calls) == 2  # One listing fetch plus one successful report fetch.
    table.write_text('changed diagnostic', encoding='utf-8')
    with pytest.raises(ValueError, match='table changed'):
        ams.review_history(table, manifest, output)


def test_different_version_preserved_without_claiming_equivalence(tmp_path, monkeypatch):
    table, manifest, _ = fixture_table(tmp_path)
    fake_archive(monkeypatch, b'different version')
    result = ams.review_history(table, manifest, tmp_path / 'review')
    assert result['matched_dates'] == 0 and result['unmatched_dates'] == [DAY]
    assert not result['version_records'][0]['matches_esmis_bytes']


def test_timeout_is_bounded_and_never_creates_completed_evidence(tmp_path, monkeypatch):
    calls = []
    pauses = []

    def unavailable(*args, **kwargs):
        calls.append(args)
        raise requests.ReadTimeout('synthetic network timeout')

    monkeypatch.setattr(ams, 'archive', unavailable)
    monkeypatch.setattr(ams.time, 'sleep', pauses.append)
    table, manifest, _ = fixture_table(tmp_path)
    output = tmp_path / 'review'
    with pytest.raises(requests.ReadTimeout):
        ams.review_history(table, manifest, output)
    assert len(calls) == 3 and pauses == [1, 2]
    assert not (output / 'publication-content-review.json').exists()
    assert not list(output.glob('requests/*.json'))


def test_transient_timeout_retries_then_preserves_resumable_download(tmp_path, monkeypatch):
    payload = b'report bytes'
    calls = fake_archive(monkeypatch, payload)
    original = ams.archive
    attempts = []

    def intermittent(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            raise requests.ConnectionError('synthetic transfer interruption')
        return original(*args, **kwargs)

    monkeypatch.setattr(ams, 'archive', intermittent)
    monkeypatch.setattr(ams.time, 'sleep', lambda _: None)
    result = ams.cached_archive(DOC, tmp_path)
    assert result.read_bytes() == payload and len(attempts) == 2
    assert ams.cached_archive(DOC, tmp_path) == result
    assert len(attempts) == 2 and calls == [DOC]


def test_browser_transport_streams_closes_and_keeps_denial_nonretryable(tmp_path):
    class TransportError(Exception):
        pass

    class Response:
        status_code = 200
        closed = False

        def __init__(self):
            self.headers = {'Content-Type': 'text/plain'}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise TransportError('denied')

        def iter_content(self, size):
            assert size == 65536
            yield b'versioned report'

        def close(self):
            self.closed = True

    response = Response()

    class Client:
        def get(self, url, **kwargs):
            assert url == DOC
            assert kwargs == {'timeout': (10, 60), 'stream': True, 'allow_redirects': False}
            return response

    transport = ams.PublicationTransport(Client(), TransportError)
    folder = ams.archive('ams', DOC, tmp_path, session=transport)
    assert (folder / 'source.bin').read_bytes() == b'versioned report'
    assert response.closed
    response.status_code = 403
    with pytest.raises(requests.HTTPError):
        ams.cached_archive(DOC, tmp_path / 'denied', session=transport)
    assert not list((tmp_path / 'denied').glob('requests/*.json'))


def test_stream_interruption_closes_response_and_cannot_freeze_partial_file(tmp_path, monkeypatch):
    class TransportError(Exception):
        pass

    responses = []

    class Response:
        status_code = 200
        closed = False

        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b'partial report'
            raise TransportError('synthetic interrupted stream')

        def close(self):
            self.closed = True

    class Client:
        def get(self, url, **kwargs):
            response = Response()
            responses.append(response)
            return response

    monkeypatch.setattr(ams.time, 'sleep', lambda _: None)
    with pytest.raises(requests.ConnectionError):
        ams.cached_archive(DOC, tmp_path, session=ams.PublicationTransport(Client(), TransportError))
    assert len(responses) == 3 and all(response.closed for response in responses)
    assert not list(tmp_path.rglob('source.bin')) and not list(tmp_path.glob('requests/*.json'))


def test_failed_persistent_connection_is_renewed_and_every_client_closed(tmp_path, monkeypatch):
    class TransportError(Exception):
        pass

    class Response:
        status_code = 200
        closed = False

        def __init__(self):
            self.headers = {}

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b'recovered report'

        def close(self):
            self.closed = True

    response = Response()

    class Client:
        closed = False

        def __init__(self, failed):
            self.failed = failed

        def get(self, url, **kwargs):
            if self.failed:
                raise TransportError('stale connection')
            return response

        def close(self):
            self.closed = True

    old, fresh = Client(True), Client(False)
    monkeypatch.setattr(ams.time, 'sleep', lambda _: None)
    with ams.PublicationTransport(old, TransportError, renew=lambda: fresh) as transport:
        path = ams.cached_archive(DOC, tmp_path, session=transport)
        assert path.read_bytes() == b'recovered report'
        assert old.closed and response.closed and not fresh.closed
    assert fresh.closed
