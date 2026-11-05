"""Offline country-field inventory; numeric consistency never grants admission."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from cottonlens_ml.code_identity import digest, manifest_id

FIELDS = ('weeklyExports', 'accumulatedExports', 'outstandingSales', 'grossNewSales',
          'currentMYNetSales', 'currentMYTotalCommitment', 'nextMYOutstandingSales',
          'nextMYNetSales')


def number(value):
    if value is None or isinstance(value, bool):
        raise ValueError('Missing/boolean FAS value must not become zero')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('Invalid FAS numeric value') from exc
    if not result.is_finite():
        raise ValueError('Nonfinite FAS numeric value')
    return result


def audit(raw_root, manifest_path, table_path):
    """Read the prior frozen source/table; never create features or change bytes."""
    raw_root, manifest_path, table_path = map(Path, (raw_root, manifest_path, table_path))
    manifest = json.loads(manifest_path.read_bytes())
    body = {k: v for k, v in manifest.items() if k != 'record_id'}
    if manifest.get('record_id') != manifest_id(body):
        raise ValueError('Frozen manifest identity mismatch')
    if (body.get('model_eligible') is not False or body.get('release_allowed') is not False
            or body.get('unit') != 'running_bales' or body.get('commodity') != 'All Upland Cotton'):
        raise ValueError('Only the prior ineligible All Upland source is supported')
    sources = body['annual_sources']
    if sorted(sources) != [str(year) for year in range(2010, 2025)]:
        raise ValueError('Fifteen pinned marketing-year sources required')
    paths = defaultdict(list)
    for path in raw_root.glob('*/*/source.json'):
        if not path.resolve().is_relative_to(raw_root.resolve()):
            raise ValueError('Raw source escapes input root')
        paths[path.parent.name].append(path)

    inputs = {manifest_path: digest(manifest_path), table_path: digest(table_path)}
    if inputs[table_path] != body['table_sha256']:
        raise ValueError('Compiled table checksum mismatch')
    catalog_paths = paths[body['catalog_sha256']]
    if len(catalog_paths) != 1 or digest(catalog_paths[0]) != body['catalog_sha256']:
        raise ValueError('Pinned commodity catalog missing/corrupt/ambiguous')
    catalog = json.loads(catalog_paths[0].read_bytes())
    if not isinstance(catalog, list) or not all(isinstance(r, dict) for r in catalog):
        raise ValueError('Commodity catalog must contain objects')
    cotton = [r for r in catalog if r.get('commodityCode') == 1404]
    if len(cotton) != 1 or cotton[0].get('unitId') != 2 or cotton[0].get('commodityName') != 'All Upland Cotton':
        raise ValueError('Commodity/catalog unit mismatch')
    inputs[catalog_paths[0]] = body['catalog_sha256']

    annual, retained, accounting = [], [], Counter()
    for year, checksum in sources.items():
        candidates = paths[checksum]
        if len(candidates) != 1 or digest(candidates[0]) != checksum:
            raise ValueError(f'Pinned annual source missing/corrupt/ambiguous: {year}')
        path = candidates[0]
        inputs[path] = checksum
        rows = json.loads(path.read_bytes())
        if not isinstance(rows, list) or not rows:
            raise ValueError('Nonempty annual source required')
        keys, selected, outside, overlaps = set(), [], 0, 0
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError('Annual source rows must be objects')
            if row.get('commodityCode') != 1404 or row.get('unitId') != 2:
                raise ValueError('Mixed commodity or units')
            country = row.get('countryCode')
            if type(country) is not int or country < 0:
                raise ValueError('Unambiguous integer country code required')
            stamp = datetime.fromisoformat(row['weekEndingDate'])
            if stamp.tzinfo is not None or stamp.time() != datetime.min.time():
                raise ValueError('Unambiguous calendar week date required')
            day = stamp.date()
            key = (day, country)
            if key in keys:
                raise ValueError('Duplicate country/week')
            keys.add(key)
            values = {field: number(row.get(field)) for field in FIELDS}
            if values['weeklyExports'] < 0:
                raise ValueError('Negative shipment volume requires separate source review')
            # Preserve the legacy boundary rule, including its unknown first-version status.
            if day.year + int(day.month >= 8) != int(year):
                overlaps += 1
                continue
            if day.isoformat() >= '2024-01-01':
                outside += 1
                continue
            selected.append({'week': day.isoformat(), 'country': country, 'values': values})
            accounting['commitment_identity_mismatches'] += int(
                values['currentMYTotalCommitment'] != values['outstandingSales'] + values['accumulatedExports'])
            accounting['negative_net_sales_rows'] += int(values['currentMYNetSales'] < 0)
        retained.extend(selected)
        annual.append({'market_year': int(year), 'sha256': checksum, 'raw_rows': len(rows),
                       'retained_rows': len(selected), 'other_marketing_year_rows': overlaps,
                       'excluded_2024_plus_rows': outside})

    totals, country_weeks, negative = {}, defaultdict(set), Counter()
    all_keys = set()
    for row in retained:
        key = (row['week'], row['country'])
        if key in all_keys:
            raise ValueError('Cross-file country/week overlap')
        all_keys.add(key)
        total = totals.setdefault(row['week'], {f: Decimal(0) for f in FIELDS})
        for field, value in row['values'].items():
            total[field] += value
        country_weeks[row['country']].add(row['week'])
        negative[row['country']] += int(row['values']['currentMYNetSales'] < 0)
    with table_path.open(encoding='utf-8', newline='') as stream:
        compiled_rows = list(csv.DictReader(stream))
    compiled = {r['week_date']: r for r in compiled_rows}
    if len(compiled) != len(compiled_rows) or len(compiled) != body['rows'] or set(compiled) != set(totals):
        raise ValueError('Compiled/raw week coverage mismatch')
    for week, values in totals.items():
        for field in ('currentMYNetSales', 'weeklyExports'):
            if values[field] != number(compiled[week][field]):
                raise ValueError('Compiled/raw national total mismatch')
    if any(digest(path) != checksum for path, checksum in inputs.items()):
        raise ValueError('Source changed during read-only audit')
    return {'schema': 1, 'profile': 'fas-country-field-audit-v1', 'market_fits': 0,
            'audit_source_sha256': digest(Path(__file__)),
            'manifest_sha256': inputs[manifest_path], 'table_sha256': inputs[table_path],
            'catalog_sha256': body['catalog_sha256'], 'annual': annual,
            'raw_rows': sum(r['raw_rows'] for r in annual), 'retained_rows': len(retained),
            'weeks': len(totals), 'start': min(totals), 'end': max(totals),
            'numeric_fields': list(FIELDS), 'accounting': dict(accounting),
            'nonpositive_outstanding_total_weeks': sum(v['outstandingSales'] <= 0 for v in totals.values()),
            'nonpositive_net_sales_total_weeks': sum(v['currentMYNetSales'] <= 0 for v in totals.values()),
            'countries': [{'code': code, 'observed_weeks': len(weeks),
                           'absent_weeks_not_imputed': len(totals) - len(weeks),
                           'start': min(weeks), 'end': max(weeks),
                           'negative_net_sales_rows': negative[code]}
                          for code, weeks in sorted(country_weeks.items())],
            'national_flow_totals_match_prior_table': True, 'old_inputs_unchanged': True,
            'input_files_verified': len(inputs), 'model_eligible': False, 'release_allowed': False,
            'historical_publication_verified': False, 'first_version_verified': False,
            'country_names_verified': False,
            'limits': ['Country codes are not mapped to names without a pinned country catalog.',
                       'Absent country rows remain unknown, not zero.',
                       'Stock accounting is not a report-value or publication/vintage reconciliation.',
                       'No features, availability clocks, forecasts or performance results were created.']}
