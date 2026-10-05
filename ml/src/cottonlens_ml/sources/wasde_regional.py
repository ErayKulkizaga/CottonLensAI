"""Offline regional WASDE candidate; preserves unverified availability/vintage status."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.wasde import cotton_rows

VERSION = 'wasde-regional-candidate-v1'
UNIT = '(Million 480-Pound Bales)'
FIELDS = {
    'world_production': ('World', 'Production'),
    'world_consumption': ('World', 'Domestic Use'),
    'world_ending_stocks': ('World', 'Ending Stocks'),
    'us_production': ('United States', 'Production'),
    'us_consumption': ('United States', 'Domestic Use'),
    'us_exports': ('United States', 'Exports'),
    'us_ending_stocks': ('United States', 'Ending Stocks'),
    'china_consumption': ('China', 'Domestic Use'),
    'india_production': ('India', 'Production'),
    'brazil_production': ('Brazil', 'Production'),
}
RATIOS = ['world_stock_use', 'world_production_use', 'us_stock_use']
LEVELS = [*FIELDS, *RATIOS]


def regional_values(rows, release_day):
    day = pd.Timestamp(release_day)
    if {r['report_month'] for r in rows} != {day.strftime('%B %Y')}:
        raise ValueError('Report month differs from dated release identity')
    years = {}
    for row in rows:
        season = row['marketing_year']
        match = re.fullmatch(r'(\d{4})/(\d{2})(?: Est\.| Proj\.)?', season)
        if not match or int(match[2]) != (int(match[1]) + 1) % 100:
            raise ValueError('Unreviewed marketing year')
        years[season] = int(match[1])
    if not years:
        raise ValueError('Nonempty regional report required')
    crop_year = max(years.values())
    selected = [r for r in rows if years[r['marketing_year']] == crop_year
                and r['forecast_month'] == day.strftime('%b')]
    values, qualifiers = {}, {}
    for name, (region, attribute) in FIELDS.items():
        cells = [r for r in selected if r['region'] == region and r['attribute'] == attribute]
        if len(cells) != 1 or cells[0]['units'] != UNIT:
            raise ValueError(f'One correctly dimensioned regional cell required: {name}')
        cell = cells[0]
        qualifiers[name] = cell.get('qualifier', 'numeric')
        if cell['value'] is None:
            if qualifiers[name] not in ['not_available', 'less_than_5000_bales']:
                raise ValueError('Unknown missing qualifier; never replace with zero')
            values[name] = np.nan
        else:
            value = float(cell['value'])
            if not np.isfinite(value) or value < 0:
                raise ValueError('Nonnegative finite regional levels required')
            values[name] = value
    for name, numerator, denominator in [
        ('world_stock_use', values['world_ending_stocks'], values['world_consumption']),
        ('world_production_use', values['world_production'], values['world_consumption']),
        ('us_stock_use', values['us_ending_stocks'], values['us_consumption'] + values['us_exports'])]:
        if np.isfinite(denominator) and denominator <= 0:
            raise ValueError('Positive stocks-to-use denominator required')
        values[name] = numerator / denominator
    return {'crop_year': crop_year, **values}, qualifiers


def add_revisions(frame):
    """Compare like crop years; preserve unknown/new-season deltas."""
    if frame.report_date.duplicated().any() or not frame.report_date.is_monotonic_increasing:
        raise ValueError('Unique chronological regional release dates required')
    result = frame.copy()
    valid = frame.crop_year.eq(frame.crop_year.shift()) & frame.report_date.diff().dt.days.between(1, 62)
    for name in LEVELS:
        result[name + '_revision'] = frame[name].diff().where(valid)
    return result


def compile_candidate(archive_root, world_table, output):
    root, world_table, output = map(Path, (archive_root, world_table, output))
    if output.exists():
        raise ValueError('Never overwrite a source archive or candidate package')
    parent_path = world_table.with_suffix('.manifest.json')
    parent = read_record(parent_path)
    if (digest(world_table) != parent['table_sha256'] or parent['model_eligible']
            or parent['publication_timestamp_verified'] or parent['first_version_verified']):
        raise ValueError('Pinned unverified-time parent required; no status promotion')
    inputs = {'parent_table': digest(world_table), 'parent_manifest': digest(parent_path)}
    proof_items = {}
    for name, expected in parent['numeric_evidence_sha256'].items():
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError('Numeric evidence checksum/path differs')
        proof = read_record(root / name)
        inputs[name] = expected
        for item in proof['releases']:
            if item['numeric_status'] != 'passed':
                raise ValueError('Unreconciled parent numeric evidence')
            proof_items[(name, item['url'])] = item
    records, provenance = [], []
    for source in parent['source_versions']:
        proof = proof_items[(source['numeric_evidence_file'], source['release_url'])]
        match = re.fullmatch(r'https://esmis\.nal\.usda\.gov/publication/world-agricultural-supply-and-demand-estimates/(\d{4}-\d{2}-\d{2})(?:-\d+)?', source['release_url'])
        if not match or not '2016-01-01' <= match[1] < '2024-01-01':
            raise ValueError('Pinned 2016-2023 dated release required')
        for field in ['xml_sha256', 'page_sha256']:
            sha = source[field]
            if not re.fullmatch('[a-f0-9]{64}', sha):
                raise ValueError('Source hash identity required')
            if digest(root / 'wasde' / sha / 'source.bin') != sha:
                raise ValueError('Raw regional source checksum mismatch')
        if proof.get('xml_sha256', source['xml_sha256']) != source['xml_sha256']:
            raise ValueError('Numeric review belongs to another XML version')
        xml = root / 'wasde' / source['xml_sha256'] / 'source.bin'
        values, qualifiers = regional_values(cotton_rows(xml.read_bytes()), match[1])
        records.append({'report_date': pd.Timestamp(match[1]), **values})
        provenance.append({**source, 'report_date': match[1], 'qualifiers': qualifiers})
    frame = pd.DataFrame(records).sort_values('report_date').reset_index(drop=True)
    duplicates = []
    for day, group in frame.groupby('report_date'):
        if len(group[['crop_year', *LEVELS]].drop_duplicates()) != 1:
            raise ValueError('Conflicting same-day regional versions; explicit vintage review required')
        if len(group) > 1:
            duplicates.append({'report_date': day.strftime('%Y-%m-%d'), 'copies': len(group)})
    frame = frame.drop_duplicates('report_date').reset_index(drop=True)
    # Collapse a later duplicate only when ALL regional levels agree, not just World ratios.
    same_month = frame.report_date.dt.to_period('M').eq(frame.report_date.shift().dt.to_period('M'))
    unchanged = frame[['crop_year', *LEVELS]].eq(frame[['crop_year', *LEVELS]].shift()).all(axis=1)
    collapsed = frame.loc[same_month & unchanged, 'report_date'].dt.strftime('%Y-%m-%d').tolist()
    frame = add_revisions(frame.loc[~(same_month & unchanged)].reset_index(drop=True))
    manifest = {'version': VERSION, 'kind': 'wasde_regional_candidate', 'source_tier': 'A_exploration_only',
        'model_eligible': False, 'release_allowed': False, 'first_version_verified': False,
        'publication_timestamp_verified': False, 'decision_clock': 'cotton-next-day-0015-v1',
        'clock_status': 'report_date only; no available_at invented', 'inputs': inputs,
        'rows': len(frame), 'levels': LEVELS, 'features': [*LEVELS, *[n + '_revision' for n in LEVELS]],
        'unit': 'million 480-pound bales for levels; ratios dimensionless',
        'revision_policy': 'same marketing year, previous report within 62 calendar days; no zero imputation',
        'ratio_policy': 'World stock/use: domestic use; US stock/use: domestic use + exports',
        'numeric_scope': 'XML extraction with existing parent review; no fresh full PDF reconciliation claimed',
        'source_versions': provenance, 'identical_same_day_copies': duplicates,
        'identical_later_same_month_copies': collapsed, 'known_no_report_months': ['2019-01'],
        'usage_status': 'No new source acquisition or redistribution-rights certification',
        'admission_blockers': ['specific-version historical availability', 'first/revised-vintage provenance',
                              'review of new field reconciliation and usage scope']}
    # Reject mutation during reads, before any completed package is written.
    if digest(world_table) != inputs['parent_table'] or digest(parent_path) != inputs['parent_manifest']:
        raise ValueError('Parent changed during compilation')
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / 'regional.csv', index=False, date_format='%Y-%m-%d')
    manifest['files'] = {'regional.csv': digest(output / 'regional.csv')}
    freeze_record(output / 'candidate-manifest.json', manifest)
    freeze_record(output / 'complete.json', {'version': VERSION, 'completed': True, 'fits': 0,
        'files': {name: digest(output / name) for name in ['regional.csv', 'candidate-manifest.json']},
        'model_eligible': False})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--world-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = compile_candidate(args.archive_root, args.world_table, args.output)
    print(json.dumps({'rows': manifest['rows'], 'model_eligible': False, 'fits': 0}))


if __name__ == '__main__':
    main()
