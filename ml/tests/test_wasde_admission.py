import json

import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.research.publications import attach_package, load_package
from cottonlens_ml.sources.public import compile_review
from cottonlens_ml.sources.wasde_admission import SCHEMA


def fixture(tmp_path):
    (tmp_path / 'source.txt').write_text('Synthetic version A; not market evidence')
    (tmp_path / 'publication.txt').write_text('Synthetic contemporaneous release assertion')
    values, stamp = {'wasde_us_exports': 10.}, '2020-06-13T00:15:00Z'
    proof = {'schema': SCHEMA, 'basis': 'contemporaneous_archive_capture',
        'version_verified': True, 'availability_verified': True,
        'source_sha256': digest(tmp_path / 'source.txt'), 'vintage_id': 'A',
        'values_sha256': manifest_id(values), 'available_at': stamp, 'evidence_files': ['publication.txt']}
    (tmp_path / 'vintage.json').write_text(json.dumps(proof))
    review = {'kind': 'wasde', 'features': list(values), 'max_age_days': 45,
        'usage': {'cost_tl': 0, 'research_allowed': True, 'terms_url': 'https://www.usda.gov'},
        'files': {name: digest(tmp_path / name) for name in ['source.txt', 'publication.txt', 'vintage.json']},
        'releases': [{'values': values, 'observed_through': '2020-06-12T00:00:00Z',
            'published_at': stamp, 'timestamp_verified': True, 'vintage_id': 'A',
            'source_url': 'https://www.usda.gov/report.txt', 'source_file': 'source.txt',
            'publication_evidence_file': 'publication.txt', 'vintage_evidence_file': 'vintage.json'}]}
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(review))
    return path, review, proof


def save(path, review, proof):
    (path.parent / 'vintage.json').write_text(json.dumps(proof))
    review['files']['vintage.json'] = digest(path.parent / 'vintage.json')
    path.write_text(json.dumps(review))


def test_exact_boundary_and_changed_future_version_do_not_change_past_features(tmp_path):
    path, _, _ = fixture(tmp_path)
    folder = compile_review(path, tmp_path / 'package')
    manifest, rows, features = load_package(folder)
    manifest['decision_clock'] = 'cotton-next-day-0015-v1'
    history = pd.DataFrame({'date': pd.to_datetime(['2020-06-11', '2020-06-12'])})
    actual = attach_package(history, manifest, rows, features)
    assert pd.isna(actual.wasde_us_exports.iloc[0])
    assert actual.wasde_us_exports.iloc[1] == 10.
    future = rows.copy()
    future['published_at'] = pd.Timestamp('2020-06-13T00:15:01Z')
    future['wasde_us_exports'] = 999.
    pd.testing.assert_frame_equal(actual, attach_package(history, manifest, pd.concat([rows, future]), features))


@pytest.mark.parametrize('field,value', [('basis', 'calendar_schedule'), ('basis', 'pdf_creation_time'),
    ('basis', 'current_download_backdated'), ('version_verified', False),
    ('source_sha256', '0' * 64), ('values_sha256', '0' * 64), ('vintage_id', 'corrected-B'),
    ('available_at', '2020-06-13T00:14:59Z'), ('evidence_files', []), ('evidence_files', ['missing.txt'])])
def test_unproven_or_mismatched_evidence_cannot_become_a_publication_package(tmp_path, field, value):
    path, review, proof = fixture(tmp_path)
    proof[field] = value
    save(path, review, proof)
    with pytest.raises(ValueError, match='WASDE'):
        compile_review(path, tmp_path / 'package')
    assert not (tmp_path / 'package').exists()


def test_later_values_cannot_use_old_version_or_time_even_with_rechecksummed_rows(tmp_path):
    path, review, _ = fixture(tmp_path)
    folder = compile_review(path, tmp_path / 'package')
    manifest = json.loads((folder / 'publication-manifest.json').read_text())
    rows_path = folder / manifest['rows_file']
    rows = pd.read_parquet(rows_path)
    rows['wasde_us_exports'] = 999.
    rows.to_parquet(rows_path, index=False)
    manifest['files'][manifest['rows_file']] = digest(rows_path)
    (folder / 'publication-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='WASDE'):
        load_package(folder)
    review['releases'][0]['values']['wasde_us_exports'] = 999.
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='WASDE'):
        compile_review(path, tmp_path / 'other')
