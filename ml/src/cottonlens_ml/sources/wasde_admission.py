"""WASDE-specific reviewed version contract; flags alone cannot approve a vintage.

Checks bind the reviewer assertion to exact values, source, vintage and clock.
They do not establish the truth of an underlying publisher/archive assertion.
"""
import json
from pathlib import Path

from cottonlens_ml.code_identity import manifest_id
from cottonlens_ml.sources.public import utc_timestamp

SCHEMA = 'wasde-specific-version-review-v1'


def require_review(release, files, root, features, stamp):
    name = release['vintage_evidence_file']
    if name not in files:
        raise ValueError('WASDE vintage receipt must be checksummed')
    proof = json.loads((Path(root) / name).read_text(encoding='utf-8'))
    values = release['values'] if 'values' in release else {key: release[key] for key in features}
    source_sha = release.get('source_sha256', files.get(release.get('source_file')))
    if (proof.get('schema') != SCHEMA or proof.get('version_verified') is not True
            or proof.get('availability_verified') is not True
            or proof.get('basis') not in ['official_release_record', 'contemporaneous_archive_capture']
            or proof.get('source_sha256') != source_sha
            or proof.get('vintage_id') != release['vintage_id']
            or proof.get('values_sha256') != manifest_id(values)
            or utc_timestamp(proof.get('available_at')) != stamp):
        raise ValueError('WASDE specific-version/value/availability evidence mismatch')
    evidence = proof.get('evidence_files')
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(member, str) or member not in files or member == name for member in evidence)):
        raise ValueError('WASDE underlying availability/vintage evidence must be checksummed')
    return proof
