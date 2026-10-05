"""Causal alignment, strict source identity and shared runner boundaries; no fits."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ledger import freeze_record
from cottonlens_ml.research.oncall_exploration import (
    add_oncall,
    dispatch,
    group_names,
    prepare,
    read_oncall,
)
from cottonlens_ml.sources.oncall_archive import FIELDS


def reports():
    return pd.DataFrame({'report_date': pd.to_datetime(['2020-01-03', '2020-01-10', '2020-01-17']),
        'assumed_day': pd.to_datetime(['2020-01-10', '2020-01-17', '2020-01-24']),
        'net_share': [.2, .3, .4], 'sales_share': [.5, .6, .7],
        'net_share_change': [np.nan, .1, .1], 'input_mask_reason': ['', '', '']})


def history(end='2020-03-01'):
    return pd.DataFrame({'date': pd.bdate_range('2020-01-01', end)})


@pytest.mark.parametrize('status,identity', [
    (None, True), ('blocked', True), ('passed', False), ('passed', True),
])
def test_pilot_requires_reviewed_readiness_before_shared_runner(tmp_path, monkeypatch, status, identity):
    from cottonlens_ml.research import ams_exploration
    folder = tmp_path/'experiments/research-oncall'
    freeze_record(folder/'ready.json', {'identity': {'source_evidence': {'table_sha256': 'table'}}})
    if status is not None:
        freeze_record(folder/'source-readiness.json', {'status': status,
            'ready_sha256': digest(folder/'ready.json') if identity else 'changed',
            'table_sha256': 'table'})
    calls = []
    monkeypatch.setattr(ams_exploration, 'dispatch_information', lambda *args: calls.append(args))
    args = SimpleNamespace(stage='pilot', drive_root=tmp_path, experiment='research-oncall')
    if status == 'passed' and identity:
        dispatch(args)
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match='no fits started'):
            dispatch(args)
        assert not calls


def test_future_data_cannot_change_past_features_and_time_is_not_compressed():
    h, rows = history(), reports()
    before = add_oncall(h, rows)
    changed = rows.copy()
    changed.loc[2, list(FIELDS)] = 999
    after = add_oncall(h, changed)
    mask = h.date <= '2020-01-24'
    pd.testing.assert_frame_equal(before.loc[mask], after.loc[mask])
    assert before.loc[h.date == '2020-01-10', 'oncall_missing_L1'].iloc[0] == 1
    # First report has unknown change; level itself becomes available after assumption.
    assert before.loc[h.date == '2020-01-13', 'oncall_net_share_L1'].iloc[0] == .2
    assert before.loc[h.date == '2020-01-20', 'oncall_net_share_L6'].iloc[0] == .2
    assert len(before) == len(h)
    assert before.loc[h.date >= '2020-02-11', 'oncall_missing_L1'].eq(1).all()
    pd.testing.assert_frame_equal(before.iloc[:20], add_oncall(h.iloc[:20], rows))


def test_late_older_release_never_overwrites_newer_known_report():
    rows = reports()
    rows.loc[0, 'assumed_day'] = pd.Timestamp('2020-01-28')
    result = add_oncall(history(), rows)
    assert result.loc[result.date == '2020-01-29', 'oncall_net_share_L1'].iloc[0] == .4


def test_masked_new_report_blocks_unbounded_carry_forward():
    rows = reports()
    rows.loc[2, list(FIELDS)] = np.nan
    rows.loc[2, 'input_mask_reason'] = 'conflicting_footer_dates'
    result = add_oncall(history(), rows)
    assert result.loc[result.date == '2020-01-24', 'oncall_net_share_L1'].iloc[0] == .3
    assert result.loc[result.date == '2020-01-27', 'oncall_missing_L1'].iloc[0] == 1
    assert result.loc[result.date == '2020-01-27', 'oncall_age_L1'].iloc[0] == 6


def pin(table, rows, **extra):
    rows.to_csv(table, index=False)
    freeze_record(table.with_suffix('.manifest.json'), {'schema': 'oncall-tier-a-table-v1',
        'table_sha256': digest(table), 'rows': len(rows), 'fields': list(FIELDS),
        'model_eligible': False, 'release_allowed': False, 'publication_timestamp_verified': False,
        'first_version_verified': False, **extra})


def test_table_resume_bad_bytes_formula_and_wrong_admission(tmp_path):
    table = tmp_path/'oncall.csv'
    pin(table, reports())
    loaded, _ = read_oncall(table)
    assert len(loaded) == 3
    table.write_text('corrupt')
    with pytest.raises(ValueError, match='Pinned Tier-A'):
        read_oncall(table)
    rows = reports()
    rows.loc[1, 'net_share_change'] = 777
    bad = tmp_path/'wrong.csv'
    pin(bad, rows)
    with pytest.raises(ValueError, match='causal changes'):
        read_oncall(bad)
    eligible = tmp_path/'not-tier-a.csv'
    pin(eligible, reports(), model_eligible=True)
    with pytest.raises(ValueError, match='Pinned Tier-A'):
        read_oncall(eligible)


def test_v2_delay_matches_every_pinned_footer_date_and_keeps_gate_closed(tmp_path):
    rows = reports()
    rows['source_sha256'], rows['source_url'] = 'a'*64, 'https://www.cftc.gov/sample'
    rows.loc[1, 'assumed_day'] = pd.Timestamp('2020-02-01')
    versions = [{'as_of': row.report_date.strftime('%Y-%m-%d'), 'source_sha256': row.source_sha256,
        'source_url': row.source_url, 'printed_release_not_before_utc': None,
        'footer_additional_dates': ['2020-01-30'] if i == 1 else []}
        for i, row in enumerate(rows.itertuples())]
    extra = {'schema': 'oncall-tier-a-table-v2', 'footer_policy': 'latest_date_assumption',
             'parsed_versions': versions}
    table = tmp_path/'v2.csv'; pin(table, rows, **extra)
    loaded, source = read_oncall(table)
    assert loaded.assumed_day.iloc[1] == pd.Timestamp('2020-02-01')
    assert source['release_allowed'] is False and source['publication_timestamp_verified'] is False
    # Even a checksum-updated packet must not make the late report available early.
    rows.loc[1, 'assumed_day'] = pd.Timestamp('2020-01-17')
    early = tmp_path/'early.csv'; pin(early, rows, **extra)
    with pytest.raises(ValueError, match='delay/source'):
        read_oncall(early)
    bad = tmp_path/'no-assumption.csv'; pin(bad, rows, **{**extra, 'footer_policy': 'verified'})
    with pytest.raises(ValueError, match='unverified latest-date'):
        read_oncall(bad)


def test_clearance_hydrates_from_existing_mirror_without_compute(tmp_path, monkeypatch):
    from cottonlens_ml.research import ams_exploration
    from cottonlens_ml.research.mirror import Mirror
    origin = tmp_path/'origin'; target = tmp_path/'drive/experiments/research-oncall'
    freeze_record(origin/'ready.json', {'identity': {'source_evidence': {'table_sha256': 'table'}}})
    freeze_record(origin/'source-readiness.json', {'status': 'passed', 'table_sha256': 'table',
        'ready_sha256': digest(origin/'ready.json')})
    mirror = Mirror(origin, target)
    mirror.enqueue(['ready.json', 'source-readiness.json']); mirror.flush()
    calls = []
    monkeypatch.setattr(ams_exploration, 'dispatch_information', lambda *args: calls.append(args))
    dispatch(SimpleNamespace(stage='pilot', drive_root=tmp_path/'local',
        mirror_root=tmp_path/'drive', experiment='research-oncall'))
    assert len(calls) == 1
    assert not list((tmp_path/'local').rglob('completed/*.json'))


@pytest.mark.parametrize('column,value', [
    ('report_date', pd.Timestamp('2024-01-05')),
    ('assumed_day', pd.Timestamp('2020-01-04')),
    ('sales_share', -1.),
])
def test_invalid_audit_time_or_domain_is_rejected(tmp_path, column, value):
    rows = reports()
    rows.loc[0, column] = value
    table = tmp_path/'bad.csv'
    pin(table, rows)
    with pytest.raises(ValueError, match='Chronological'):
        read_oncall(table)


def test_identical_origins_control_columns_and_release_stay_closed(tmp_path):
    groups = group_names()
    assert len(groups) == 7
    for lag in (1, 2, 6):
        assert groups[f'oncall_L{lag}'][:-3] == groups[f'missing_L{lag}']
    with pytest.raises(ValueError, match='cannot lock/export'):
        dispatch(SimpleNamespace(stage='export', drive_root=tmp_path, experiment='research-synthetic'))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('version', [1, 2])
def test_prepare_preserves_all_year_origins_and_resumes_without_fitting(tmp_path, monkeypatch, version):
    from cottonlens_ml.config import FEATURE_NAMES
    from cottonlens_ml.research import models
    from cottonlens_ml.research.ledger import read_record
    from cottonlens_ml.research.protocol import full_year_manifest, mature
    dates = pd.bdate_range('2010-01-01', '2023-12-29')
    h = pd.DataFrame({'date': dates, 'cotton_close': 80., 'cotton_session_index': np.arange(len(dates))})
    for horizon in (1, 5):
        h[f'target_date_{horizon}'] = h.date.shift(-horizon)
        h[f'target_return_{horizon}'] = 0.
    for name in FEATURE_NAMES:
        h[name] = np.nan
    parent = tmp_path/'reference'
    parent.mkdir()
    h.to_parquet(parent/'history.parquet', index=False)
    freeze_record(parent/'ready.json', {'history_sha256': digest(parent/'history.parquet')})
    table = tmp_path/'oncall.csv'
    source = reports()
    source['report_date'] = source.report_date-pd.DateOffset(years=10)
    source['assumed_day'] = source.assumed_day-pd.DateOffset(years=10)
    evidence = {'calendar_policy': 'synthetic conservative assumption'}
    if version == 2:
        source['source_sha256'], source['source_url'] = 'a'*64, 'https://www.cftc.gov/sample'
        evidence.update({'schema': 'oncall-tier-a-table-v2', 'footer_policy': 'latest_date_assumption',
            'parsed_versions': [{'as_of': row.report_date.strftime('%Y-%m-%d'),
                'source_sha256': row.source_sha256, 'source_url': row.source_url,
                'printed_release_not_before_utc': None, 'footer_additional_dates': []}
                for row in source.itertuples()]})
    pin(table, source, **evidence)
    monkeypatch.setattr(models, 'fit_predict', lambda *a, **k: pytest.fail('Prepare cannot fit'))
    folder = tmp_path/'experiment'
    repo = Path(__file__).resolve().parents[2]
    prepare(repo, folder, parent, table)
    first = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    prepare(repo, folder, parent, table)
    assert first == {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    ready = read_record(folder/'ready.json')
    assert ready['identity']['profile'] == f'oncall-exploration-v{version}'
    split = ready['identity']['split']
    reference = full_year_manifest(h)
    assert [f['origins'] for f in split['folds']] == [f['origins'] for f in reference['folds']]
    assert len(split['folds']) == 8 and split['purge_observations'] == 5 and split['refit_cadence'] == 21
    assert ready['identity']['release_allowed'] is False
    assert ready['identity']['gate_evaluation_allowed'] is False
    for fold in split['folds']:
        assert len(fold['inner']) == 3
        for block in fold['inner']:
            assert mature(h, block['cutoff']).target_date_5.max() < pd.Timestamp(block['cutoff'])
