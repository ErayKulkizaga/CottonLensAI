"""Bind reviewed FAS values to a specific version and availability clock.

These checks validate the review contract, not the truth of external evidence.
Quarantined annual API snapshots cannot be approved by changing their flags.
"""
import json
import math
from numbers import Real
from pathlib import Path

from cottonlens_ml.code_identity import manifest_id
from cottonlens_ml.sources.public import utc_timestamp

SCHEMA = 'fas-specific-version-review-v1'
DECISION_CLOCK = 'cotton-next-day-0015-v1'


def reviewed_values(values, *, parquet_missing=False):
    """Canonical floats/nulls; missing cells are neither zero nor dropped origins."""
    result = {}
    for name, value in values.items():
        if value is None:
            result[name] = None
        elif isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError('FAS values must be numbers or explicit nulls')
        elif math.isnan(value) and parquet_missing:
            result[name] = None
        elif not math.isfinite(value):
            raise ValueError('FAS nonfinite values must not become missing evidence')
        else:
            result[name] = float(value)
    return result


def require_review(release, files, root, features, stamp):
    """The same receipt is checked during compilation and package loading."""
    name = release['vintage_evidence_file']
    if name not in files:
        raise ValueError('FAS vintage receipt must be checksummed')
    proof = json.loads((Path(root) / name).read_text(encoding='utf-8'))
    values = reviewed_values(release['values'] if 'values' in release else
                             {key: release[key] for key in features},
                             parquet_missing='values' not in release)
    source_file = proof.get('source_file')
    source_sha = release.get('source_sha256', files.get(release.get('source_file')))
    observed = utc_timestamp(release.get('observed_through'))
    if (proof.get('schema') != SCHEMA or proof.get('version_verified') is not True
            or proof.get('availability_verified') is not True
            or proof.get('value_review_verified') is not True
            or proof.get('basis') not in ('official_release_record', 'contemporaneous_archive_capture')
            or source_file not in files or files[source_file] != source_sha
            or proof.get('source_sha256') != source_sha
            or (release.get('source_file') is not None and release['source_file'] != source_file)
            or proof.get('vintage_id') != release['vintage_id']
            or proof.get('features') != list(features)
            or proof.get('values_sha256') != manifest_id(values)
            or proof.get('commodity_code') != 1404 or proof.get('unit') != 'running_bales'
            or utc_timestamp(proof.get('observed_through')) != observed or observed > stamp
            or utc_timestamp(proof.get('available_at')) != stamp
            or proof.get('availability_basis') != release.get('availability_basis', 'exact_publication')):
        raise ValueError('FAS specific-version/value/period/availability evidence mismatch')
    members = proof.get('evidence_files')
    if (not isinstance(members, list) or not members
            or any(not isinstance(member, str) or member not in files or member == name for member in members)
            or release['publication_evidence_file'] not in members):
        raise ValueError('FAS underlying value/publication evidence must be checksummed')
    # Explicitly prohibit treating the generated latest-snapshot panel as a release.
    raw = (Path(root) / source_file).read_bytes()
    try:
        source = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        source = None
    if isinstance(source, dict) and source.get('profile') == 'fas-country-quarantine-v1':
        raise ValueError('FAS quarantined panel is not an as-published source')
    return proof
