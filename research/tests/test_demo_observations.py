import json

import pytest

from research.demo_observations import append, readiness, records


def observation():
    return {'provider': 'generated', 'page': 'generated-ui', 'observed_at': '2020-01-01T10:00:00Z',
            'target_date': '2020-01-03', 'horizon': 1, 'forecast': 80., 'unit': 'cent/lb',
            'contract': 'UNKNOWN', 'price_field': 'UNKNOWN',
            'displayed_issue_time': 'UNKNOWN', 'origin_price': 'UNKNOWN'}


def test_first_observation_revisions_and_idempotence(tmp_path):
    value = observation()
    first = append(tmp_path, value)
    assert append(tmp_path, value) == first
    second = append(tmp_path, {**value, 'observed_at': '2020-01-02T10:00:00Z', 'forecast': 81.})
    assert second['revision'] == 1 and second['first_observed_at'] == value['observed_at']
    assert records(tmp_path)[0] == first
    assert readiness(tmp_path)['unknown_fields'] == ['contract', 'displayed_issue_time', 'origin_price', 'price_field']
    assert not readiness(tmp_path)['primary_evaluation_allowed']
    with pytest.raises(ValueError, match='Backdated'):
        append(tmp_path, {**value, 'forecast': 82.})


@pytest.mark.parametrize('patch', [{'forecast': float('nan')}, {'forecast': -1.},
    {'target_date': '2019-12-31'}, {'observed_at': '2020-01-01'}, {'horizon': True},
    {'contract': ''}, {'observed_at': '2099-01-01T00:00:00Z', 'target_date': '2099-01-02'}])
def test_bad_observation_rejected_before_publication(tmp_path, patch):
    with pytest.raises(ValueError):
        append(tmp_path, {**observation(), **patch})
    assert not list(tmp_path.rglob('*.json'))


def test_mutated_and_missing_chain_rejected(tmp_path):
    append(tmp_path, observation())
    append(tmp_path, {**observation(), 'forecast': 81.})
    path = tmp_path/'events/000000.json'
    original = path.read_bytes()
    value = json.loads(original)
    value['observation']['forecast'] = 99.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Corrupt'):
        records(tmp_path)
    path.write_bytes(original)
    path.rename(tmp_path/'preserved.json')
    with pytest.raises(ValueError, match='Incomplete'):
        records(tmp_path)
