"""Import explicitly reviewed as-published releases; a calendar is not vintage evidence."""
import argparse
import json
from pathlib import Path

import pandas as pd

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.protocol import attach_releases


def load_package(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'publication-manifest.json').read_text(encoding='utf-8'))
    from cottonlens_ml.sources.public import SOURCES
    if manifest.get('kind') not in (*SOURCES, 'contract_curve') or manifest.get('vintage_policy') != 'as_published':
        raise ValueError('Supported source and as-published vintages required')
    for name, checksum in manifest['files'].items():
        if not safe_member(name) or digest(folder / name) != checksum:
            raise ValueError('Publication source checksum mismatch')
    if manifest.get('rows_file') not in manifest['files']:
        raise ValueError('Publication rows must be checksummed')
    rows = pd.read_parquet(folder / manifest['rows_file'])
    if 'available_at' in rows:
        from cottonlens_ml.sources.public import reviewed_upper_bound
        if manifest.get('availability_schema') != 'verified-availability-v1':
            raise ValueError('Explicit reviewed availability schema required')
        for row in rows.to_dict('records'):
            if row.get('availability_basis') == 'verified_upper_bound':
                if not pd.isna(row.get('published_at')):
                    raise ValueError('Upper bounds must not claim exact publication')
                receipt = {**row, 'published_at': None, 'available_by': row['available_at']}
                if row.get('source_file') not in manifest['files']:
                    raise ValueError('Availability source file must be checksummed')
                if manifest['files'][row['source_file']] != row['source_sha256']:
                    raise ValueError('Availability source hash mismatch')
                reviewed_upper_bound(receipt, manifest['files'], folder)
    elif manifest.get('availability_schema'):
        raise ValueError('Availability schema requires an availability clock')
    for column in ('publication_evidence_file', 'vintage_evidence_file'):
        if column not in rows or not rows[column].isin(manifest['files']).all():
            raise ValueError('Per-release publication and vintage evidence files required')
    if not rows.source_sha256.isin(manifest['files'].values()).all():
        raise ValueError('Per-row source hashes must match reviewed source files')
    features = manifest['features']
    if not features or any(not f.startswith(manifest['kind'] + '_') for f in features):
        raise ValueError('External features must use their source namespace')
    if manifest['kind'] == 'wasde':
        from cottonlens_ml.sources.public import utc_timestamp
        from cottonlens_ml.sources.wasde_admission import require_review
        for row in rows.to_dict('records'):
            clock = row['available_at'] if 'available_at' in row else row['published_at']
            require_review(row, manifest['files'], folder, features, utc_timestamp(clock))
    if (manifest['kind'] not in ('cftc', 'wasde', 'contract_curve')
            and (manifest.get('usage', {}).get('research_allowed') is not True or manifest['usage'].get('cost_tl') != 0)):
        raise ValueError('Verified zero-cost research usage required')
    return manifest, rows, features


def attach_package(history, manifest, rows, features):
    """Isolate source metadata, enforce freshness and keep a single decision clock."""
    if set(features) & set(history.columns):
        raise ValueError('Duplicate source feature names')
    if (history.date.isna().any() or history.date.duplicated().any()
            or not history.date.is_monotonic_increasing):
        raise ValueError('Unique chronological history required; never reorder feature origins silently')
    merged = attach_releases(history[['date']].copy(), rows, features,
        decision_clock=manifest.get('decision_clock', 'legacy-midnight-v1'))
    result = history.copy()
    clock = 'available_at' if 'available_at' in merged else 'published_at'
    age = (merged.decision_at - merged[clock]).dt.total_seconds() / 86400
    maximum = manifest.get('max_age_days')
    if maximum is not None:
        if type(maximum) is not int or not 1 <= maximum <= 366:
            raise ValueError('Invalid source freshness policy')
        merged.loc[age > maximum, features] = float('nan')
    for name in features:
        result[name] = merged[name].to_numpy()
    if manifest.get('kind'):
        age_name, missing_name = availability_features(manifest)
        if {age_name, missing_name} & set(features):
            raise ValueError('Source feature collides with reserved availability channel')
        result[age_name] = age.to_numpy()
        result[missing_name] = merged[features].isna().any(axis=1).astype(float).to_numpy()
    return result


def availability_features(manifest):
    return [manifest['kind'] + '_release_age_days', manifest['kind'] + '_unavailable']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, rows, features = load_package(args.package)
    history = pd.read_parquet(args.history)
    merged = attach_package(history, manifest, rows, features)
    if args.output.exists():
        raise ValueError('Never overwrite an existing information snapshot')
    args.output.mkdir(parents=True)
    merged.to_parquet(args.output / 'history.parquet', index=False)
    freeze_record(args.output / 'availability.json', {'source': manifest, 'features': features,
        'decision_clock': manifest.get('decision_clock', 'legacy-midnight-v1'),
        'max_age_days': manifest.get('max_age_days'),
        'first_available': str(rows['available_at' if 'available_at' in rows else 'published_at'].min()),
        'coverage_rows': int(merged[features].notna().all(axis=1).sum()),
        'history_sha256': digest(args.output / 'history.parquet'),
        'policy': 'Use a new frozen research namespace; shorter coverage cannot compete against full-cohort scores'})


if __name__ == '__main__':
    main()
