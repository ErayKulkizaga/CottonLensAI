"""Synthetic checks only; no real-market fitting or publication approval."""
import json

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research import wasde_regional_pilot as pilot
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import Preprocessor
from cottonlens_ml.sources.wasde_regional import LEVELS, add_revisions


def reports(dates=('2018-12-11', '2019-02-08', '2019-05-10')):
    rows = pd.DataFrame({'report_date': pd.to_datetime(dates), 'crop_year': [2018, 2018, 2019]})
    for index, name in enumerate(LEVELS):
        rows[name] = np.array([1., 2., 3.]) + index
    return add_revisions(rows)


def history(dates=None):
    if dates is None:
        dates = pd.bdate_range('2018-12-10', '2019-05-20')
    frame = pd.DataFrame({'date': pd.to_datetime(dates)})
    for index, name in enumerate(pilot.FEATURE_NAMES):
        frame[name] = np.arange(len(frame), dtype=float) + index
    frame['cotton_close'] = 70.
    return frame


def test_clock_first_decision_weekend_and_additional_observation():
    source = reports(('2018-12-11', '2019-02-09', '2019-05-10'))
    dates = ['2018-12-10', '2018-12-11', '2018-12-12', '2019-02-08', '2019-02-11', '2019-02-12']
    frame = pilot.align_assumed(history(dates), source)
    assert pd.isna(frame.numeric_D0_world_production.iloc[0])
    assert frame.numeric_D0_world_production.iloc[1] == 1.
    assert pd.isna(frame.numeric_D1_world_production.iloc[1])
    assert frame.numeric_D1_world_production.iloc[2] == 1.
    assert frame.numeric_D0_world_production.iloc[3] == 1.
    assert frame.numeric_D0_world_production.iloc[4] == 2.
    assert frame.numeric_D1_world_production.iloc[4] == 1.
    assert frame.numeric_D1_world_production.iloc[5] == 2.
    for delay in pilot.DELAYS:
        field = frame[f'regional_D{delay}_assumed_available_at']
        assert (field.dropna() <= frame.loc[field.notna(), 'regional_decision_time']).all()
    assert pilot.CLOCK['publication_verified'] is False


def test_future_source_mutation_cannot_change_past_features():
    source = reports()
    original = pilot.align_assumed(history(), source)
    changed = source[['report_date', 'crop_year', *LEVELS]].copy()
    changed.loc[2, LEVELS] += 100
    altered = pilot.align_assumed(history(), add_revisions(changed))
    past = original.date < pd.Timestamp('2019-05-10')
    pd.testing.assert_frame_equal(original.loc[past], altered.loc[past])


def test_future_cotton_rows_cannot_change_past_alignment():
    frame = history()
    original = pilot.align_assumed(frame, reports())
    shorter = pilot.align_assumed(frame.iloc[:70], reports())
    pd.testing.assert_frame_equal(original.iloc[:70], shorter)


def test_common_expiry_is_not_extended_by_delay():
    frame = history(pd.bdate_range('2018-12-10', periods=60))
    result = pilot.align_assumed(frame, reports())
    # The next report is beyond this inspection slice; age is anchored at row 1.
    assert result.numeric_D0_world_production.iloc[41] == 1.
    assert result.numeric_D1_world_production.iloc[41] == 1.
    assert pd.isna(result.numeric_D0_world_production.iloc[42])
    assert pd.isna(result.numeric_D1_world_production.iloc[42])
    assert result.regional_D0_age_sessions.iloc[42] == 41


def test_inputs_rows_and_unknown_revisions_preserved():
    source, frame = reports(), history()
    a, b = source.copy(deep=True), frame.copy(deep=True)
    result = pilot.align_assumed(frame, source)
    pd.testing.assert_frame_equal(source, a)
    pd.testing.assert_frame_equal(frame, b)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    first = result.date.eq(pd.Timestamp('2018-12-11'))
    reset = result.date.eq(pd.Timestamp('2019-05-10'))
    for name in LEVELS:
        assert result.loc[first | reset, 'numeric_D0_' + name + '_revision'].isna().all()


@pytest.mark.parametrize('delay', pilot.DELAYS)
def test_control_has_identical_automatic_missing_indicators(delay):
    result = pilot.align_assumed(history(), reports())
    control = pilot.groups()[f'mask_D{delay}']
    numeric = pilot.groups()[f'numeric_D{delay}']
    left = Preprocessor.fit(result.iloc[:90], control).transform(result)
    right = Preprocessor.fit(result.iloc[:90], numeric).transform(result)
    assert left.shape == right.shape
    np.testing.assert_array_equal(left[:, len(control):], right[:, len(numeric):])
    assert not np.array_equal(left[:, :len(control)], right[:, :len(numeric)])
    for name in pilot.NUMERIC:
        a, b = result[f'mask_D{delay}_{name}'], result[f'numeric_D{delay}_{name}']
        assert a.isna().equals(b.isna())
        assert a.dropna().eq(0.).all()


@pytest.mark.parametrize('defect', ['duplicate', 'future', 'negative', 'infinite', 'revision_zero', 'intraday'])
def test_bad_source_fails_closed(defect):
    source = reports()
    if defect == 'duplicate':
        source.loc[1, 'report_date'] = source.loc[0, 'report_date']
    elif defect == 'future':
        source.loc[2, 'report_date'] = pd.Timestamp('2024-01-01')
    elif defect == 'negative':
        source.loc[0, 'world_production'] = -1.
    elif defect == 'infinite':
        source.loc[0, 'world_production'] = np.inf
    elif defect == 'revision_zero':
        source.loc[0, 'world_production_revision'] = 0.
    elif defect == 'intraday':
        source.loc[0, 'report_date'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        pilot.align_assumed(history(), source)


def test_completed_payload_corruption_rejected(tmp_path):
    (tmp_path / 'payload.csv').write_text('value\n1\n')
    freeze_record(tmp_path / 'complete.json', {'completed': True, 'fits': 0, 'model_eligible': False,
                  'files': {'payload.csv': digest(tmp_path / 'payload.csv')}})
    pilot.completed_files(tmp_path)
    (tmp_path / 'payload.csv').write_text('value\n2\n')
    with pytest.raises(ValueError, match='payload changed'):
        pilot.completed_files(tmp_path)


@pytest.mark.parametrize('member', ['../outside', 'C:/private/file', 'folder\\file'])
def test_unsafe_completion_member_rejected(tmp_path, member):
    freeze_record(tmp_path / 'complete.json', {'completed': True, 'fits': 0, 'model_eligible': False,
                  'files': {member: 'not-a-checksum'}})
    with pytest.raises(ValueError):
        pilot.completed_files(tmp_path)


def test_prepare_completion_excludes_writer_lock_and_never_overwrites(tmp_path, monkeypatch):
    repo, candidate, numeric, official, reference = [tmp_path / name for name in ['repo', 'candidate', 'numeric', 'official', 'reference']]
    names = {repo / 'research': ['registry.json', 'trials.json'], repo / 'ml': ['uv.lock'],
             repo / 'research/evidence': ['wasde-regional-verification-20261008.json', 'wasde-as-reported-20261008.json'],
             candidate: ['complete.json', 'candidate-manifest.json', 'regional.csv'],
             numeric: ['complete.json', 'report.json', 'coverage.csv'],
             official: ['complete.json', 'report.json', 'coverage.csv', 'comparison.csv'],
             reference: ['ready.json', 'history.parquet']}
    for folder, members in names.items():
        folder.mkdir(parents=True, exist_ok=True)
        for name in members:
            (folder / name).write_text('synthetic')
    monkeypatch.setattr(pilot, 'validate', lambda _: None)
    monkeypatch.setattr(pilot, 'load_registry', lambda _: ({'records': []}, {'records': []}))
    monkeypatch.setattr(pilot, 'candidate_rows', lambda *args: reports())
    monkeypatch.setattr(pilot, 'registered_audits', lambda *args: None)
    monkeypatch.setattr(pilot, 'verify_history', lambda _: ({'identity': {'profile': 'full-year-v1'}}, history()))
    monkeypatch.setattr(pilot, 'make_design', lambda *args: (history(), {'groups': pilot.groups()}, []))
    output = tmp_path / 'output'
    before = {path: digest(path) for path in tmp_path.rglob('*') if path.is_file()}
    registration = pilot.prepare(repo, output, reference, candidate, numeric, official)
    complete = read_record(output / 'complete.json')
    assert complete['completed_experiment'] is False
    assert complete['fits'] == 0 and registration['model_eligible'] is False
    assert not (output / 'ready.json').exists()
    assert not (output / '.writer-lock').exists()
    assert all('.writer-lock' not in name for name in complete['files'])
    assert all(digest(output / name) == sha for name, sha in complete['files'].items())
    assert all(digest(path) == sha for path, sha in before.items())
    saved = digest(output / 'complete.json')
    with pytest.raises(ValueError, match='Preserve existing'):
        pilot.prepare(repo, output, reference, candidate, numeric, official)
    assert digest(output / 'complete.json') == saved


def test_existing_partial_output_is_not_completed(tmp_path):
    (tmp_path / 'history.parquet').write_text('partial')
    with pytest.raises(FileNotFoundError):
        pilot.completed_files(tmp_path)


def test_rechecksummed_audit_cannot_replace_registered_review(tmp_path):
    evidence = tmp_path / 'research/evidence'
    evidence.mkdir(parents=True)
    numeric, official = tmp_path / 'numeric', tmp_path / 'official'
    for folder in (numeric, official):
        folder.mkdir()
        (folder / 'report.json').write_text('original report')
    (evidence / 'wasde-regional-verification-20261008.json').write_text(json.dumps(
        {'completion': {'files': {'report.json': digest(numeric / 'report.json')}}}))
    (evidence / 'wasde-as-reported-20261008.json').write_text(json.dumps(
        {'payload_sha256': {'audit/report.json': digest(official / 'report.json')}}))
    pilot.registered_audits(tmp_path, numeric, official)
    (numeric / 'report.json').write_text('forged report with a new self-consistent checksum')
    with pytest.raises(ValueError, match='registered numeric evidence'):
        pilot.registered_audits(tmp_path, numeric, official)


def test_existing_regional_columns_are_not_overwritten():
    frame = history()
    frame['regional_decision_time'] = pd.NaT
    with pytest.raises(ValueError, match='never overwrite'):
        pilot.align_assumed(frame, reports())


