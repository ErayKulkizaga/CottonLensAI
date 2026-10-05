"""Reviewed upper bounds delay availability; they never invent publication times."""
import json

import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.publications import attach_package, load_package
from cottonlens_ml.sources.public import compile_review


def reviewed_bound(tmp_path):
    (tmp_path / 'source.txt').write_text('Synthetic specific-version values; not market evidence')
    (tmp_path / 'archive.txt').write_text('Synthetic contemporaneous availability evidence')
    stamp = '2020-06-12T04:00:00Z'
    proof = {'schema': 'source-availability-upper-bound-v1',
             'basis': 'contemporaneous_archive_capture', 'version_verified': True,
             'timing_verified': True, 'source_sha256': digest(tmp_path / 'source.txt'),
             'vintage_id': 'specific-version', 'available_by': stamp,
             'evidence_files': ['archive.txt']}
    (tmp_path / 'availability.json').write_text(json.dumps(proof))
    release = {'values': {'ams_spot': 75.}, 'observed_through': '2020-06-11T00:00:00Z',
               'published_at': None, 'timestamp_verified': False,
               'available_by': stamp, 'availability_verified': True,
               'availability_basis': 'verified_upper_bound', 'vintage_id': 'specific-version',
               'source_url': 'https://www.ams.usda.gov/mnreports/cnddsq.pdf',
               'source_file': 'source.txt', 'publication_evidence_file': 'availability.json',
               'vintage_evidence_file': 'archive.txt'}
    review = {'kind': 'ams', 'features': ['ams_spot'], 'max_age_days': 2,
              'usage': {'cost_tl': 0, 'research_allowed': True, 'terms_url': 'https://www.ams.usda.gov/market-news'},
              'files': {name: digest(tmp_path / name) for name in ['source.txt', 'archive.txt', 'availability.json']},
              'releases': [release]}
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(review))
    return path, review, proof


def test_upper_bound_roundtrip_uses_later_clock_and_preserves_unknown_publication(tmp_path):
    path, _, _ = reviewed_bound(tmp_path)
    folder = compile_review(path, tmp_path / 'package')
    manifest, rows, features = load_package(folder)
    assert manifest['availability_schema'] == 'verified-availability-v1'
    assert rows.published_at.isna().all() and not rows.timestamp_verified.any()
    history = pd.DataFrame({'date': pd.date_range('2020-06-10', periods=7)})
    actual = attach_package(history, manifest, rows, features)
    assert actual.loc[actual.date <= '2020-06-11', 'ams_spot'].isna().all()
    assert actual.loc[actual.date.eq('2020-06-12'), 'ams_spot'].iloc[0] == 75.
    assert actual.loc[actual.date.eq('2020-06-12'), 'ams_release_age_days'].iloc[0] == pytest.approx(20 / 24)
    assert actual.loc[actual.date >= '2020-06-14', 'ams_spot'].isna().all()
    future = rows.copy()
    future['available_at'] = pd.Timestamp('2030-01-01', tz='UTC')
    future['ams_spot'] = 999.
    pd.testing.assert_frame_equal(actual, attach_package(history, manifest, pd.concat([rows, future]), features))


@pytest.mark.parametrize('mutation', ['schedule', 'wrong_hash', 'wrong_vintage', 'missing_raw_evidence', 'unverified', 'wrong_time'])
def test_upper_bound_rejects_unbound_or_unverified_evidence(tmp_path, mutation):
    path, review, proof = reviewed_bound(tmp_path)
    if mutation == 'schedule':
        proof['basis'] = 'scheduled_release'
    elif mutation == 'wrong_hash':
        proof['source_sha256'] = '0' * 64
    elif mutation == 'wrong_vintage':
        proof['vintage_id'] = 'another-version'
    elif mutation == 'missing_raw_evidence':
        proof['evidence_files'] = []
    elif mutation == 'unverified':
        proof['timing_verified'] = False
    else:
        proof['available_by'] = '2020-06-11T04:00:00Z'
    (tmp_path / 'availability.json').write_text(json.dumps(proof))
    review['files']['availability.json'] = digest(tmp_path / 'availability.json')
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='evidence'):
        compile_review(path, tmp_path / 'package')
    assert not (tmp_path / 'package').exists()


def test_upper_bound_cannot_claim_exact_publication_and_requires_explicit_timezone(tmp_path):
    path, review, _ = reviewed_bound(tmp_path)
    review['releases'][0]['timestamp_verified'] = True
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='not an exact'):
        compile_review(path, tmp_path / 'package')
    review['releases'][0]['timestamp_verified'] = False
    review['releases'][0]['available_by'] = '2020-06-12'
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='timezone'):
        compile_review(path, tmp_path / 'package')


def test_load_rechecks_bound_receipt_and_join_rejects_forged_clock(tmp_path):
    path, _, _ = reviewed_bound(tmp_path)
    folder = compile_review(path, tmp_path / 'package')
    manifest, rows, features = load_package(folder)
    bad = rows.copy()
    bad['timestamp_verified'] = True
    with pytest.raises(ValueError, match='exact publication'):
        attach_package(pd.DataFrame({'date': pd.to_datetime(['2020-06-12'])}), manifest, bad, features)
    proof = json.loads((folder / 'availability.json').read_text())
    proof['available_by'] = '2010-01-01T00:00:00Z'
    (folder / 'availability.json').write_text(json.dumps(proof))
    manifest['files']['availability.json'] = digest(folder / 'availability.json')
    (folder / 'publication-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='mismatch'):
        load_package(folder)


def test_mixed_exact_and_bounded_versions_preserve_their_time_semantics(tmp_path):
    path, review, _ = reviewed_bound(tmp_path)
    exact = {**review['releases'][0], 'values': {'ams_spot': 70.},
             'availability_basis': 'exact_publication', 'timestamp_verified': True,
             'published_at': '2020-06-11T12:00:00Z'}
    review['releases'].append(exact)
    path.write_text(json.dumps(review))
    manifest, rows, features = load_package(compile_review(path, tmp_path / 'package'))
    history = pd.DataFrame({'date': pd.to_datetime(['2020-06-10', '2020-06-11', '2020-06-12'])})
    result = attach_package(history, manifest, rows, features)
    assert result.ams_spot.iloc[1:].tolist() == [70., 75.]
    assert rows.loc[rows.availability_basis.eq('exact_publication'), 'published_at'].notna().all()
    assert rows.loc[rows.availability_basis.eq('verified_upper_bound'), 'published_at'].isna().all()


def test_equal_cutoff_cannot_use_release_and_unknown_basis_fails_closed(tmp_path):
    path, review, proof = reviewed_bound(tmp_path)
    stamp = '2020-06-12T00:00:00Z'
    proof['available_by'] = stamp
    (tmp_path / 'availability.json').write_text(json.dumps(proof))
    review['files']['availability.json'] = digest(tmp_path / 'availability.json')
    review['releases'][0]['available_by'] = stamp
    path.write_text(json.dumps(review))
    manifest, rows, features = load_package(compile_review(path, tmp_path / 'package'))
    history = pd.DataFrame({'date': pd.to_datetime(['2020-06-11', '2020-06-12'])})
    result = attach_package(history, manifest, rows, features)
    assert pd.isna(result.ams_spot.iloc[0]) and result.ams_spot.iloc[1] == 75.
    review['releases'][0]['availability_basis'] = 'arbitrary_lag'
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='Unsupported availability'):
        compile_review(path, tmp_path / 'other-package')
