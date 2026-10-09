"""Synthetic receipt/clock tests only; no historical FAS source is admitted."""
import json

import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.research.fas_country_alignment import attach_matched
from cottonlens_ml.research.publications import load_package
from cottonlens_ml.sources.fas_admission import DECISION_CLOCK, SCHEMA, reviewed_values
from cottonlens_ml.sources.public import compile_review

NATIONAL = ['export_sales_national_exports']
COUNTRY = ['export_sales_country_5700_export_share', 'export_sales_country_5700_exports_4w']


def fixture(tmp_path, *, bounded=False):
    (tmp_path / 'source.txt').write_text('Synthetic version A; not real FAS evidence', encoding='utf-8')
    (tmp_path / 'publication.txt').write_text('Synthetic version-bound release assertion', encoding='utf-8')
    values = dict(zip(NATIONAL + COUNTRY, (20., .5, None)))
    stamp, observed = '2020-06-13T00:15:00Z', '2020-06-11T00:00:00Z'
    basis = 'verified_upper_bound' if bounded else 'exact_publication'
    proof = {'schema': SCHEMA, 'basis': 'contemporaneous_archive_capture',
        'version_verified': True, 'availability_verified': True, 'value_review_verified': True,
        'source_file': 'source.txt', 'source_sha256': digest(tmp_path / 'source.txt'),
        'vintage_id': 'synthetic-A', 'features': list(values),
        'values_sha256': manifest_id(values), 'observed_through': observed,
        'available_at': stamp, 'availability_basis': basis,
        'commodity_code': 1404, 'unit': 'running_bales', 'evidence_files': ['publication.txt']}
    release = {'values': values, 'observed_through': observed, 'published_at': stamp,
        'timestamp_verified': True, 'vintage_id': 'synthetic-A',
        'source_url': 'https://apps.fas.usda.gov/synthetic-only.txt', 'source_file': 'source.txt',
        'publication_evidence_file': 'publication.txt', 'vintage_evidence_file': 'vintage.json'}
    names = ['source.txt', 'publication.txt', 'vintage.json']
    if bounded:
        availability = {'schema': 'source-availability-upper-bound-v1',
            'basis': 'contemporaneous_archive_capture', 'version_verified': True,
            'timing_verified': True, 'source_sha256': digest(tmp_path / 'source.txt'),
            'vintage_id': 'synthetic-A', 'available_by': stamp, 'evidence_files': ['publication.txt']}
        (tmp_path / 'availability.json').write_text(json.dumps(availability), encoding='utf-8')
        names.append('availability.json')
        release.update({'published_at': None, 'timestamp_verified': False,
            'availability_verified': True, 'available_by': stamp, 'availability_basis': basis,
            'publication_evidence_file': 'availability.json'})
        proof['evidence_files'].append('availability.json')
    (tmp_path / 'vintage.json').write_text(json.dumps(proof), encoding='utf-8')
    review = {'kind': 'export_sales', 'features': list(values), 'max_age_days': 3,
        'decision_clock': DECISION_CLOCK,
        'usage': {'cost_tl': 0, 'research_allowed': True, 'terms_url': 'https://www.usda.gov'},
        'files': {name: digest(tmp_path / name) for name in names}, 'releases': [release]}
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(review), encoding='utf-8')
    return path, review, proof


def save(path, review, proof):
    (path.parent / 'vintage.json').write_text(json.dumps(proof), encoding='utf-8')
    review['files']['vintage.json'] = digest(path.parent / 'vintage.json')
    path.write_text(json.dumps(review), encoding='utf-8')


def history():
    return pd.DataFrame({'date': pd.to_datetime(['2020-06-11', '2020-06-12', '2020-06-15', '2020-06-16']),
                         'close': [60., 61., 62., 63.], 'target_t1': [.1, .2, -.2, .0]})


@pytest.mark.parametrize('bounded', [False, True])
def test_exact_cutoff_missing_cells_and_freshness_keep_origins_and_matched_masks(tmp_path, bounded):
    path, _, _ = fixture(tmp_path, bounded=bounded)
    folder = compile_review(path, tmp_path / 'package')
    manifest, rows, features = load_package(folder)
    assert manifest['decision_clock'] == DECISION_CLOCK and list(features) == NATIONAL + COUNTRY
    before = history()
    actual, groups = attach_matched(before, folder, national_features=NATIONAL, country_features=COUNTRY)
    pd.testing.assert_frame_equal(actual[before.columns], before)
    assert pd.isna(actual[NATIONAL[0]].iloc[0])
    assert actual[NATIONAL[0]].iloc[1] == 20.  # equal 00:15 cutoff is eligible
    assert actual[NATIONAL[0]].iloc[2] == 20.  # exactly max-age three days
    assert pd.isna(actual[NATIONAL[0]].iloc[3])
    assert actual[COUNTRY[1]].isna().all()  # no null->zero or origin filtering
    assert actual[COUNTRY[1] + '_missing'].tolist() == [1., 1., 1., 1.]
    assert actual[NATIONAL[0] + '_missing'].tolist() == [1., 0., 0., 1.]
    shared = set(groups['national_control']) & set(groups['country_candidate'])
    assert set(groups['country_candidate']) - shared == set(COUNTRY)
    assert all(f + '_missing' in shared for f in features)
    assert rows.published_at.isna().all() == bounded
    assert actual.export_sales_vintage_id.iloc[1] == 'synthetic-A'
    assert actual.export_sales_available_at.iloc[1] == pd.Timestamp('2020-06-13T00:15:00Z')
    assert actual.export_sales_observed_through.iloc[1] == pd.Timestamp('2020-06-11T00:00:00Z')
    assert 'export_sales_vintage_id' not in groups['country_candidate']


@pytest.mark.parametrize('bounded', [False, True])
def test_one_second_after_cutoff_cannot_enter_earlier_origin(tmp_path, bounded):
    path, review, proof = fixture(tmp_path, bounded=bounded)
    stamp = '2020-06-13T00:15:01Z'
    proof['available_at'] = stamp
    if bounded:
        review['releases'][0]['available_by'] = stamp
        availability = json.loads((tmp_path / 'availability.json').read_text())
        availability['available_by'] = stamp
        (tmp_path / 'availability.json').write_text(json.dumps(availability))
        review['files']['availability.json'] = digest(tmp_path / 'availability.json')
    else:
        review['releases'][0]['published_at'] = stamp
    save(path, review, proof)
    actual, _ = attach_matched(history(), compile_review(path, tmp_path / 'package'),
                               national_features=NATIONAL, country_features=COUNTRY)
    assert pd.isna(actual[NATIONAL[0]].iloc[1])


@pytest.mark.parametrize('field,value', [
    ('schema', 'old-calendar-policy'), ('basis', 'calendar_schedule'),
    ('basis', 'pdf_creation_time'), ('basis', 'current_download_backdated'),
    ('version_verified', False), ('availability_verified', False), ('value_review_verified', False),
    ('source_sha256', '0' * 64), ('source_file', 'publication.txt'),
    ('vintage_id', 'later-correction'), ('values_sha256', '0' * 64), ('features', list(reversed(NATIONAL + COUNTRY))),
    ('observed_through', '2020-06-10T00:00:00Z'), ('available_at', '2020-06-12T00:15:00Z'),
    ('availability_basis', 'verified_upper_bound'), ('commodity_code', 1403), ('unit', '480_pound_bales'),
    ('evidence_files', []), ('evidence_files', ['missing.txt']), ('evidence_files', ['vintage.json']),
])
def test_unbound_value_version_units_period_or_clock_fails_before_package_write(tmp_path, field, value):
    path, review, proof = fixture(tmp_path)
    proof[field] = value
    save(path, review, proof)
    with pytest.raises(ValueError, match='FAS'):
        compile_review(path, tmp_path / 'package')
    assert not (tmp_path / 'package').exists()
    assert not list(tmp_path.glob('*.pending'))


@pytest.mark.parametrize('field,value', [('decision_clock', None), ('decision_clock', 'legacy-midnight-v1'),
                                       ('max_age_days', True), ('max_age_days', 0)])
def test_explicit_clock_and_freshness_required(tmp_path, field, value):
    path, review, proof = fixture(tmp_path)
    review[field] = value
    save(path, review, proof)
    with pytest.raises(ValueError, match='FAS'):
        compile_review(path, tmp_path / 'package')


@pytest.mark.parametrize('field,value', [('observed_through', '2020-06-14T00:00:00Z'),
                                       ('observed_through', '2020-06-11'),
                                       ('published_at', '2020-06-13')])
def test_future_observation_or_timezone_free_source_time_rejected(tmp_path, field, value):
    path, review, proof = fixture(tmp_path)
    review['releases'][0][field] = value
    save(path, review, proof)
    with pytest.raises(ValueError, match='beyond|timezone'):
        compile_review(path, tmp_path / 'package')


@pytest.mark.parametrize('value', [True, '20', float('nan'), float('inf'), -float('inf')])
def test_nonfinite_or_coerced_source_values_not_hidden_as_missing(tmp_path, value):
    path, review, proof = fixture(tmp_path)
    review['releases'][0]['values'][NATIONAL[0]] = value
    save(path, review, proof)
    with pytest.raises(ValueError, match='FAS'):
        compile_review(path, tmp_path / 'package')


@pytest.mark.parametrize('mutation', ['value', 'null_to_zero', 'period', 'clock', 'empty', 'manifest_clock'])
def test_loader_revalidates_receipt_even_if_changed_rows_are_rechecksummed(tmp_path, mutation):
    path, _, _ = fixture(tmp_path)
    folder = compile_review(path, tmp_path / 'package')
    manifest, rows, _ = load_package(folder)
    if mutation == 'value':
        rows[NATIONAL[0]] = 999.
    elif mutation == 'null_to_zero':
        rows[COUNTRY[1]] = 0.
    elif mutation == 'period':
        rows['observed_through'] = pd.Timestamp('2020-06-10T00:00:00Z')
    elif mutation == 'clock':
        rows['published_at'] = pd.Timestamp('2020-06-12T00:15:00Z')
    elif mutation == 'empty':
        rows = rows.iloc[:0]
    else:
        manifest['decision_clock'] = 'legacy-midnight-v1'
    file = folder / manifest['rows_file']
    rows.to_parquet(file, index=False)
    manifest['files'][manifest['rows_file']] = digest(file)
    (folder / 'publication-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='FAS'):
        load_package(folder)


def test_quarantined_panel_cannot_become_a_release_by_flipping_flags(tmp_path):
    path, review, proof = fixture(tmp_path)
    source = tmp_path / 'source.txt'
    source.write_text(json.dumps({'profile': 'fas-country-quarantine-v1', 'model_eligible': True,
                                 'release_allowed': True, 'availability_policy': 'verified'}))
    review['files']['source.txt'] = digest(source)
    proof['source_sha256'] = digest(source)
    save(path, review, proof)
    with pytest.raises(ValueError, match='quarantined panel'):
        compile_review(path, tmp_path / 'package')


def test_changed_future_release_and_unordered_source_rows_leave_past_features_identical(tmp_path):
    path, review, proof = fixture(tmp_path)
    before_folder = compile_review(path, tmp_path / 'before')
    before, groups = attach_matched(history(), before_folder, national_features=NATIONAL, country_features=COUNTRY)
    late = {**review['releases'][0], 'values': dict(zip(NATIONAL + COUNTRY, (999., .99, 777.))),
            'published_at': '2030-01-01T00:15:00Z', 'vintage_id': 'synthetic-later',
            'vintage_evidence_file': 'later.json'}
    late_proof = {**proof, 'available_at': late['published_at'], 'vintage_id': late['vintage_id'],
                  'values_sha256': manifest_id(late['values'])}
    (tmp_path / 'later.json').write_text(json.dumps(late_proof))
    review['files']['later.json'] = digest(tmp_path / 'later.json')
    review['releases'].insert(0, late)
    path.write_text(json.dumps(review))
    after, after_groups = attach_matched(history(), compile_review(path, tmp_path / 'after'),
                                        national_features=NATIONAL, country_features=COUNTRY)
    pd.testing.assert_frame_equal(after, before)
    assert groups == after_groups


@pytest.mark.parametrize('mutation', ['reverse', 'duplicate', 'missing_date', 'intraday', 'timezone',
                                    'collision', 'provenance_collision'])
def test_history_may_not_be_reordered_filtered_or_have_masks_overwritten(tmp_path, mutation):
    path, _, _ = fixture(tmp_path)
    folder = compile_review(path, tmp_path / 'package')
    frame = history()
    if mutation == 'reverse':
        frame = frame.iloc[::-1]
    elif mutation == 'duplicate':
        frame.loc[1, 'date'] = frame.date.iloc[0]
    elif mutation == 'missing_date':
        frame.loc[1, 'date'] = pd.NaT
    elif mutation == 'intraday':
        frame.loc[1, 'date'] += pd.Timedelta(hours=1)
    elif mutation == 'timezone':
        frame['date'] = frame.date.dt.tz_localize('UTC')
    elif mutation == 'collision':
        frame[COUNTRY[0] + '_missing'] = 0.
    else:
        frame['export_sales_vintage_id'] = 'user-data-must-not-be-overwritten'
    with pytest.raises(ValueError):
        attach_matched(frame, folder, national_features=NATIONAL, country_features=COUNTRY)


@pytest.mark.parametrize('national,country', [([], COUNTRY), (NATIONAL, []),
    (NATIONAL + COUNTRY[:1], COUNTRY), (NATIONAL, COUNTRY[:1]), (COUNTRY[:1], NATIONAL + COUNTRY[1:])])
def test_groups_must_partition_declared_source_features(tmp_path, national, country):
    path, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='partition'):
        attach_matched(history(), compile_review(path, tmp_path / 'package'),
                       national_features=national, country_features=country)


def test_explicit_zero_and_negative_net_sales_remain_values():
    assert reviewed_values({'export_sales_sales': -2, 'export_sales_exports': 0}) == {
        'export_sales_sales': -2., 'export_sales_exports': 0.}
