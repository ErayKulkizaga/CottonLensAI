"""Offline reconciliation with official as-reported exports, never clock admission."""
import argparse
import csv
import io
import re
import zipfile
from datetime import date, time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import (
    digest,
    manifest_id,
    research_source_identity,
    safe_member,
)
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.public import utc_timestamp, validate_url
from cottonlens_ml.sources.wasde_regional import FIELDS, LEVELS
from cottonlens_ml.sources.wasde_regional import VERSION as CANDIDATE_VERSION

VERSION = 'wasde-as-reported-verification-v1'
COLUMNS = ['WasdeNumber', 'ReportDate', 'ReportTitle', 'Attribute', 'ReliabilityProjection',
    'Commodity', 'Region', 'MarketYear', 'ProjEstFlag', 'AnnualQuarterFlag', 'Value', 'Unit',
    'ReleaseDate', 'ReleaseTime', 'ForecastYear', 'ForecastMonth']
TITLE = 'World Cotton Supply and Use'
UNIT = 'Million 480-Pound Bales'


def csv_payload(raw, name):
    """Read the one reviewed bulk member without extracting arbitrary ZIP paths."""
    if name.endswith('.csv'):
        return raw, None
    if not name.endswith('.zip'):
        raise ValueError('Reviewed CSV or ZIP export required')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != 1 or not safe_member(members[0].filename)
                or not members[0].filename.endswith('.csv') or members[0].flag_bits & 1
                or members[0].file_size > 64 * 1024 * 1024):
            raise ValueError('One bounded safe CSV ZIP member required')
        return archive.read(members[0]), members[0].filename


def report_day(text):
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        return date.fromisoformat(text)
    if re.fullmatch(r'\d{2}/\d{2}/\d{4}', text):
        month, day, year = map(int, text.split('/'))
        return date(year, month, day)
    raise ValueError('Unreviewed release date format')


def parse_export(raw, name):
    payload, member = csv_payload(raw, name)
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != COLUMNS:
        raise ValueError('Official export schema differs')
    reports, rows = {}, 0
    for row in reader:
        rows += 1
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed CSV row')
        if row['Commodity'] != 'Cotton' or row['ReportTitle'] != TITLE:
            continue
        day = report_day(row['ReleaseDate'])
        if (row['ReportDate'] != day.strftime('%B %Y')
                or row['ForecastYear'] != str(day.year) or row['ForecastMonth'] != str(day.month)
                or row['AnnualQuarterFlag'] != 'Annual' or row['ReliabilityProjection']
                or row['Unit'] != UNIT or row['ProjEstFlag'] not in ('', 'Est.', 'Proj.')
                or not re.fullmatch(r'\d{2}:\d{2}:\d{2}(?:\.\d+)?', row['ReleaseTime'])
                or not row['WasdeNumber'].isdigit()):
            raise ValueError('Cotton report metadata/unit/clock format differs')
        # Syntax validation only. This local, timezone-free report clock is NOT
        # the CSV download time and cannot be promoted to available_at.
        time.fromisoformat(row['ReleaseTime'].split('.')[0])
        season = re.fullmatch(r'(\d{4})/(\d{2})', row['MarketYear'])
        if not season or int(season[2]) != (int(season[1]) + 1) % 100:
            raise ValueError('Unreviewed crop year')
        if (row['Region'], row['Attribute']) not in FIELDS.values():
            # Loss/unaccounted balancing items can legitimately be negative.
            # They are not any of the ten audited nonnegative level fields.
            continue
        try:
            value = Decimal(row['Value'])
        except InvalidOperation as exc:
            raise ValueError('Unknown numeric qualifier; never impute zero') from exc
        if not value.is_finite() or value < 0:
            raise ValueError('Finite nonnegative Cotton value required')
        key = day.strftime('%Y-%m-%d')
        reports.setdefault(key, []).append({**row, 'crop_year': int(season[1]), 'numeric': value})
    if not reports:
        raise ValueError('Export contains no Cotton reports')
    records = []
    for day, cells in sorted(reports.items()):
        crop = max(cell['crop_year'] for cell in cells)
        if (len({c['WasdeNumber'] for c in cells}) != 1
                or len({c['ReleaseTime'].split('.')[0] for c in cells}) != 1):
            raise ValueError('Mixed report identities/clocks on same date')
        selected = [cell for cell in cells if cell['crop_year'] == crop]
        values = {}
        for feature, (region, attribute) in FIELDS.items():
            matches = [c for c in selected if c['Region'] == region and c['Attribute'] == attribute]
            if len(matches) != 1:
                raise ValueError(f'Exactly one current-crop cell required: {day} {feature}')
            values[feature] = matches[0]['numeric']
        denominator = values['us_consumption'] + values['us_exports']
        if values['world_consumption'] <= 0 or denominator <= 0:
            raise ValueError('Positive ratios denominator required')
        values['world_stock_use'] = values['world_ending_stocks'] / values['world_consumption']
        values['world_production_use'] = values['world_production'] / values['world_consumption']
        values['us_stock_use'] = values['us_ending_stocks'] / denominator
        records.append({'report_date': day, 'crop_year': crop, 'values': {k: float(v) for k, v in values.items()},
            'wasde_number': int(cells[0]['WasdeNumber']), 'declared_report_local_time': cells[0]['ReleaseTime'],
            'csv_available_at': None, 'historical_model_eligible': False})
    return {'rows_read': rows, 'zip_member': member, 'records': records}


def compare(candidate, records):
    if (candidate.report_date.isna().any() or candidate.report_date.duplicated().any()
            or not candidate.report_date.is_monotonic_increasing):
        raise ValueError('Unique chronological candidate dates required')
    expected = pd.DataFrame([{'report_date': pd.Timestamp(r['report_date']),
        'crop_year': r['crop_year'], **r['values']} for r in records]).sort_values('report_date').reset_index(drop=True)
    if expected.report_date.duplicated().any() or not candidate.report_date.equals(expected.report_date):
        raise ValueError('Exact report dates required; silent intersection forbidden')
    if not np.array_equal(candidate.crop_year.to_numpy(), expected.crop_year.to_numpy()):
        raise ValueError('Crop-year identity mismatch')
    comparisons = []
    previous = None
    for i, row in expected.iterrows():
        valid_delta = previous is not None and row.crop_year == previous.crop_year and 1 <= (row.report_date - previous.report_date).days <= 62
        for name in LEVELS:
            for feature, value in [(name, row[name]), (name + '_revision', row[name] - previous[name] if valid_delta else np.nan)]:
                actual = candidate.loc[i, feature]
                matches = bool((pd.isna(actual) and pd.isna(value)) or
                    (np.isfinite(actual) and np.isfinite(value) and abs(actual - value) <= 1e-12))
                comparisons.append({'report_date': row.report_date.strftime('%Y-%m-%d'), 'crop_year': int(row.crop_year),
                    'feature': feature, 'candidate_value': actual, 'as_reported_value': value, 'matches': matches})
        previous = row
    return pd.DataFrame(comparisons)


def analyze(candidate_root, exports_root, output):
    candidate_root, exports_root, output = map(Path, (candidate_root, exports_root, output))
    if output.exists():
        raise ValueError('Immutable audit output; use a new directory')
    repo = Path(__file__).resolve().parents[4]
    code = research_source_identity(repo)
    inputs, paths = {}, {}
    def pin(root, name, expected=None):
        path = (root / name).resolve()
        if not safe_member(name) or not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe audit input path')
        sha = digest(path)
        if expected is not None and sha != expected:
            raise ValueError('Audit input checksum mismatch')
        key = ('candidate/' if root == candidate_root else 'exports/') + name
        inputs[key], paths[key] = sha, path
        return path
    complete = read_record(pin(candidate_root, 'complete.json'))
    manifest = read_record(pin(candidate_root, 'candidate-manifest.json', complete['files']['candidate-manifest.json']))
    if (complete.get('completed') is not True or complete.get('version') != CANDIDATE_VERSION
            or manifest.get('version') != CANDIDATE_VERSION
            or any(manifest.get(k) is not False for k in ('model_eligible', 'first_version_verified', 'publication_timestamp_verified'))):
        raise ValueError('Pinned unadmitted candidate required')
    table = pin(candidate_root, 'regional.csv', manifest['files']['regional.csv'])
    if digest(table) != complete['files']['regional.csv']:
        raise ValueError('Candidate completion checksum mismatch')
    exports = read_record(pin(exports_root, 'sources.json'))
    if exports.get('version') != 'wasde-as-reported-exports-v1' or exports.get('model_eligible') is not False:
        raise ValueError('Unadmitted official exports required')
    review_path = pin(exports_root, exports['link_review_file'], exports['link_review_sha256'])
    review = read_record(review_path)
    records, summaries = [], []
    for source in exports['sources']:
        validate_url('wasde', source['url'])
        if source['url'] not in review['links']:
            raise ValueError('Export URL is not in reviewed official links')
        raw = pin(exports_root, source['file'], source['sha256'])
        receipt_path = pin(exports_root, source['retrieval_file'], source['retrieval_sha256'])
        receipt = read_record(receipt_path)
        utc_timestamp(receipt['retrieved_at'])  # Ingestion, not historical availability.
        if receipt['source_url'] != source['url'] or receipt['sha256'] != source['sha256'] or receipt.get('model_eligible') is not False:
            raise ValueError('Export retrieval identity differs')
        parsed = parse_export(raw.read_bytes(), source['name'])
        records.extend([{**r, 'export_sha256': source['sha256']} for r in parsed['records']])
        summaries.append({'name': source['name'], 'sha256': source['sha256'], 'rows_read': parsed['rows_read'],
            'zip_member': parsed['zip_member'], 'reports': len(parsed['records'])})
    candidate = pd.read_csv(table, parse_dates=['report_date'])
    if (len(candidate) != manifest['rows'] or list(candidate.columns) !=
            ['report_date', 'crop_year', *LEVELS, *(name + '_revision' for name in LEVELS)]):
        raise ValueError('Candidate row count/schema differs')
    cells = compare(candidate, records)
    coverage = pd.DataFrame([{k: v for k, v in r.items() if k != 'values'} for r in records]).sort_values('report_date')
    identity = manifest_id({'version': VERSION, 'inputs': inputs, 'code': code})
    if any(digest(paths[name]) != sha for name, sha in inputs.items()) or research_source_identity(repo) != code:
        raise ValueError('Input/code changed during audit')
    output.mkdir(parents=True)
    cells.to_csv(output / 'comparison.csv', index=False)
    coverage.to_csv(output / 'coverage.csv', index=False)
    report = {'version': VERSION, 'identity': identity, 'code': code, 'inputs': inputs,
        'input_sha256': {'candidate_table': digest(table), 'candidate_manifest': digest(candidate_root / 'candidate-manifest.json'),
            'exports_manifest': digest(exports_root / 'sources.json'), 'link_review': digest(review_path)},
        'exports': summaries, 'reports': len(coverage), 'base_cells_compared': len(coverage) * len(FIELDS),
        'feature_cells_compared': len(cells), 'missing_cells_preserved': int(cells.as_reported_value.isna().sum()),
        'mismatches': int((~cells.matches).sum()), 'numeric_verified': bool(cells.matches.all()),
        'historical_model_eligible_rows': 0, 'model_eligible': False, 'fits': 0,
        'clock_scope': 'ReleaseDate/ReleaseTime describe report identity, not actual CSV delivery; no inferred available_at',
        'vintage_scope': 'Agreement with official as-reported exports supports these values; not authentication of first PDF/XML bytes',
        'files': {name: digest(output / name) for name in ['comparison.csv', 'coverage.csv']}}
    freeze_record(output / 'report.json', report)
    freeze_record(output / 'complete.json', {'version': VERSION, 'identity': identity, 'completed': True,
        'fits': 0, 'model_eligible': False, 'files': {name: digest(output / name) for name in ['comparison.csv', 'coverage.csv', 'report.json']}})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--exports-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze(args.candidate_root, args.exports_root, args.output)
    print(f"reports={report['reports']} cells={report['feature_cells_compared']} mismatches={report['mismatches']} admitted=0 fits=0")
    return 0 if report['numeric_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
