import hashlib
import json

import pytest
from cottonlens_ml.sources.vintage import reconcile


def review(tmp_path, rows, provider='fas', expected=100, quantum=100):
    raw = json.dumps(rows if provider == 'fas' else {'data': rows}).encode()
    (tmp_path / 'api.json').write_bytes(raw)
    (tmp_path / 'report.txt').write_bytes(b'Synthetic report fixture')
    item = {'selector': {'commodityCode': 1404, 'weekEndingDate': '2020-05-28'},
            'field': 'weeklyExports', 'unit_id': 2, 'report_page': 1,
            'report_label': 'Synthetic exports', 'report_value': expected,
            'report_rounding_quantum': quantum}
    if provider == 'nass':
        item.update(selector={'short_desc': 'COTTON'}, field='Value')
    body = {'provider': provider, 'api_file': 'api.json', 'report_file': 'report.txt',
            'files': {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
                      for n in ('api.json', 'report.txt')}, 'observations': [item]}
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(body))
    return path


def row(country=1, value=111):
    return {'commodityCode': 1404, 'weekEndingDate': '2020-05-28',
            'countryCode': country, 'unitId': 2, 'weeklyExports': value}


def test_rounding_match_does_not_grant_publication_eligibility(tmp_path):
    result = reconcile(review(tmp_path, [row()]))
    assert result['all_values_match']
    assert not result['model_eligible'] and not result['publication_timestamp_verified']
    result = reconcile(review(tmp_path, [row(value=151)]))
    assert not result['all_values_match']


def test_country_aggregation_excludes_other_weeks(tmp_path):
    result = reconcile(review(tmp_path, [row(value=40), row(2, 60),
        {**row(3, 999), 'weekEndingDate': '2020-06-04'}], quantum=0))
    assert result['results'][0]['api_value'] == 100
    with pytest.raises(ValueError, match='duplicate'):
        reconcile(review(tmp_path, [row(), row()]))


def test_checksum_and_units_fail_closed(tmp_path):
    p = review(tmp_path, [row()])
    (tmp_path / 'report.txt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        reconcile(p)
    with pytest.raises(ValueError, match='unit'):
        reconcile(review(tmp_path, [row(), {**row(2), 'unitId': 9}]))


def test_nass_suppression_and_nonunique_rows_rejected(tmp_path):
    with pytest.raises(ValueError, match='numeric'):
        reconcile(review(tmp_path, [{'short_desc': 'COTTON', 'Value': '(D)'}], 'nass'))
    with pytest.raises(ValueError, match='exactly one'):
        reconcile(review(tmp_path, [{'short_desc': 'COTTON', 'Value': '5'}] * 2, 'nass'))
