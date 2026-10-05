"""API contracts with synthetic responses; no secrets, live API calls or fits."""
import json

import pytest
import requests
from cottonlens_ml.sources.usda import USDAClient, normalize_nass


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.raw = json.dumps(payload).encode()
        self.status_code = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield self.raw


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def client(kind, session):
    return USDAClient(kind, session=session, credential='synthetic-api-secret', pause=lambda _: None)


def test_nass_limits_query_and_does_not_archive_key(tmp_path):
    session = Session([Response({'count': '0'}), Response({'data': []})])
    folder, _ = client('nass', session).cotton_by_year(2020, tmp_path, state='TX')
    assert session.calls[0][0].endswith('get_counts/')
    assert session.calls[1][1]['params']['key'] == 'synthetic-api-secret'
    assert session.calls[1][1]['params']['commodity_desc'] == 'COTTON'
    for path in tmp_path.rglob('*.json'):
        assert 'synthetic-api-secret' not in path.read_text()
    receipt = json.loads((folder / 'retrieval.json').read_text())
    assert not receipt['model_eligible']


def test_nass_ceiling_prevents_partial_data(tmp_path):
    session = Session([Response({'count': '50001'})])
    with pytest.raises(ValueError, match='50,000'):
        client('nass', session).cotton_by_year(2020, tmp_path)
    assert len(session.calls) == 1


def test_fas_header_and_catalog_not_hardcoded_cotton_code(tmp_path):
    session = Session([Response([{'commodityCode': 77, 'commodityName': 'Cotton, synthetic'},
                                 {'commodityCode': 99, 'commodityName': 'Other crop'}])])
    _, rows = client('fas', session).cotton_catalog(tmp_path)
    assert len(rows) == 1 and rows[0]['commodityCode'] == 77
    assert session.calls[0][1]['headers']['X-Api-Key'] == 'synthetic-api-secret'


def test_ams_auth_and_bounded_date_filter(tmp_path):
    session = Session([Response({'results': []})])
    api = client('ams', session)
    api.report(3804, '2020-01-01', '2020-01-31', tmp_path)
    assert session.calls[0][1]['auth'] == ('synthetic-api-secret', '')
    assert '01/01/2020:01/31/2020' in session.calls[0][1]['params']['q']
    with pytest.raises(ValueError, match='31 days'):
        api.report(3804, '2020-01-01', '2020-03-01', tmp_path)


def test_no_secret_in_transport_failure(tmp_path):
    failure = requests.ConnectionError('URL includes synthetic-api-secret')
    session = Session([failure, failure, failure])
    with pytest.raises(RuntimeError, match='suppressed') as error:
        client('nass', session).get('get_counts/', {}, tmp_path)
    assert 'synthetic-api-secret' not in str(error.value)
    assert error.value.__suppress_context__


def test_quota_retry_and_credential_echo_fail_closed(tmp_path):
    session = Session([Response({}, 429, {'Retry-After': '0'}), Response({'results': []})])
    client('ams', session).get('reports', {}, tmp_path)
    assert len(session.calls) == 2
    echo = Session([Response({'key': 'synthetic-api-secret'})])
    with pytest.raises(ValueError, match='echoed'):
        client('ams', echo).get('reports', {}, tmp_path)


def test_normalizer_preserves_suppression_and_does_not_use_load_time_as_publication():
    common = {'commodity_desc': 'COTTON', 'short_desc': 'COTTON - CONDITION', 'unit_desc': 'PCT',
              'year': '2020', 'week_ending': '2020-06-07', 'load_time': '2020-06-08 16:12:00',
              'agg_level_desc': 'NATIONAL'}
    frame = normalize_nass({'data': [{**common, 'Value': '(D)'}, {**common, 'Value': '1,234'}]})
    assert frame.value.isna().iloc[0] and frame.value.iloc[1] == 1234
    assert frame.suppression_code.iloc[0] == '(D)'
    assert frame.published_at.isna().all() and not frame.timestamp_verified.any()
