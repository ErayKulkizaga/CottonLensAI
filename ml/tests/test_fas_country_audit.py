import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.sources.fas_country_audit import audit


def freeze(path, body):
    path.write_text(json.dumps({**body, 'record_id': manifest_id(body)}), encoding='utf-8')


def fixture(tmp_path, mutation=None):
    root, table, manifest = tmp_path / 'raw', tmp_path / 'weekly.csv', tmp_path / 'manifest.json'
    catalog = [{'commodityCode': 1404, 'commodityName': 'All Upland Cotton', 'unitId': 2}]
    def save(rows):
        raw = json.dumps(rows).encode()
        import hashlib
        checksum = hashlib.sha256(raw).hexdigest()
        path = root / 'request' / checksum / 'source.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return checksum
    catalog_sha = save(catalog)
    sources, csv_rows = {}, []
    for year in range(2010, 2025):
        week = f'{year}-07-01' if year < 2024 else '2023-12-28'
        row = {'commodityCode': 1404, 'countryCode': 1001, 'unitId': 2,
               'weekEndingDate': week + 'T00:00:00', 'weeklyExports': 10,
               'accumulatedExports': 20, 'outstandingSales': 100, 'grossNewSales': 5,
               'currentMYNetSales': -10, 'currentMYTotalCommitment': 120,
               'nextMYOutstandingSales': 5, 'nextMYNetSales': 0}
        rows = [row, {**row, 'countryCode': 1002, 'currentMYNetSales': 5}]
        if year == 2010:
            # Same boundary observation in an old MY must not be added.
            rows.append({**row, 'weekEndingDate': '2010-08-01T00:00:00', 'weeklyExports': 999})
            if mutation:
                mutation(rows)
        sources[str(year)] = save(rows)
        csv_rows.append({'week_date': week, 'currentMYNetSales': -5, 'weeklyExports': 20})
    with table.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    body = {'annual_sources': sources, 'catalog_sha256': catalog_sha,
            'table_sha256': digest(table), 'rows': len(csv_rows), 'model_eligible': False,
            'release_allowed': False, 'commodity': 'All Upland Cotton', 'unit': 'running_bales'}
    freeze(manifest, body)
    return root, manifest, table


def test_preserves_inputs_matches_totals_and_keeps_negative_sales(tmp_path):
    args = fixture(tmp_path)
    before = {p: digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    result = audit(*args)
    assert audit(*args) == result
    assert result['raw_rows'] == 31 and result['retained_rows'] == 30
    assert result['weeks'] == 15 and result['annual'][0]['other_marketing_year_rows'] == 1
    assert result['accounting']['negative_net_sales_rows'] == 15
    assert result['nonpositive_net_sales_total_weeks'] == 15
    assert result['accounting']['commitment_identity_mismatches'] == 0
    assert result['national_flow_totals_match_prior_table'] is True
    assert result['market_fits'] == 0 and result['model_eligible'] is False
    assert result['historical_publication_verified'] is False
    assert result['country_names_verified'] is False
    assert {p: digest(p) for p in before} == before


@pytest.mark.parametrize('field,value,match', [
    ('outstandingSales', None, 'Missing/boolean'),
    ('outstandingSales', True, 'Missing/boolean'),
    ('outstandingSales', 'NaN', 'Nonfinite'),
    ('unitId', 1, 'Mixed commodity'),
    ('countryCode', '1001', 'integer country'),
    ('weekEndingDate', '2010-07-01T00:00:00+00:00', 'calendar week'),
    ('weeklyExports', -1, 'Negative shipment'),
    ('currentMYNetSales', 900, 'national total mismatch'),
])
def test_invalid_data_rejected_even_when_new_bytes_have_matching_hash(tmp_path, field, value, match):
    args = fixture(tmp_path, lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError, match=match):
        audit(*args)


def test_duplicate_country_week_rejected(tmp_path):
    args = fixture(tmp_path, lambda rows: rows.append(rows[0].copy()))
    with pytest.raises(ValueError, match='Duplicate country/week'):
        audit(*args)


def test_missing_country_stays_unknown_not_zero(tmp_path):
    args = fixture(tmp_path, lambda rows: rows.pop(1))
    # Give this source its genuine national totals; no absent country imputation.
    table = args[2]
    table.write_text(table.read_text().replace('2010-07-01,-5,20', '2010-07-01,-10,10'))
    body = json.loads(args[1].read_text())
    body.pop('record_id')
    body['table_sha256'] = digest(table)
    freeze(args[1], body)
    result = audit(*args)
    assert result['countries'][1]['absent_weeks_not_imputed'] == 1


@pytest.mark.parametrize('target', ['raw', 'table', 'manifest'])
def test_checksum_change_stops_audit(tmp_path, target):
    args = fixture(tmp_path)
    if target == 'raw':
        checksum = json.loads(args[1].read_text())['annual_sources']['2010']
        path = args[0] / 'request' / checksum / 'source.json'
    else:
        path = args[2 if target == 'table' else 1]
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum|identity|corrupt'):
        audit(*args)


@pytest.mark.parametrize('corrupt', [False, True])
def test_cli_uses_stdlib_only_and_returns_error_without_partial_success(tmp_path, corrupt):
    root, manifest, table = fixture(tmp_path)
    if corrupt:
        table.write_text('changed', encoding='utf-8')
    cli = Path(__file__).resolve().parents[1] / 'review_fas_countries.py'
    completed = subprocess.run([sys.executable, '-S', str(cli), '--raw-root', str(root),
                               '--manifest', str(manifest), '--table', str(table)],
                              capture_output=True, text=True, check=False)
    if corrupt:
        assert completed.returncode == 2 and not completed.stdout
        assert 'checksum mismatch' in completed.stderr
    else:
        assert completed.returncode == 0
        assert json.loads(completed.stdout)['model_eligible'] is False
