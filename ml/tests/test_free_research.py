"""No fits: free-source evidence, compact reporting and ablation contracts."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.research.ablation import admission, groups
from cottonlens_ml.research.ledger import Ledger
from cottonlens_ml.research.protocol import feature_groups, feature_history
from cottonlens_ml.research.publications import attach_package, load_package
from cottonlens_ml.sources.public import compile_review, validate_url
from cottonlens_ml.sprint import writer
from test_sprint import history


def test_compact_summary_never_reads_fit_payload(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path, {'data': 'one'})
    def operation(workspace):
        (workspace / 'model.json').write_text('{}')
        return {'data_seconds': .25}
    record = ledger.run({'seed': 42}, operation)
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if 'completed' in path.parts or 'payloads' in path.parts:
            raise AssertionError('Reporting opened a fit record')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    summary = ledger.summary()
    assert summary['durably_saved'] == summary['timing_receipts'] == 1
    assert summary['timing']['data_seconds'] == .25
    assert not summary['payloads_verified_by_status']
    assert record['files']


def test_writer_ownership_is_visible_and_never_stolen(tmp_path):
    with writer(tmp_path):
        owner = json.loads((tmp_path / '.writer-lock/owner.json').read_text())
        assert owner['pid'] > 0 and owner['host'] and owner['token']
        with pytest.raises(RuntimeError, match='writer'), writer(tmp_path):
            pytest.fail('Second writer acquired lock')
    assert not (tmp_path / '.writer-lock').exists()


def test_upload_retry_does_not_repeat_completed_fit(tmp_path, monkeypatch):
    import cottonlens_ml.research.ledger as module
    ledger = Ledger(tmp_path, {'data': 'stable'})
    calls = []
    def operation(workspace):
        calls.append(1)
        (workspace / 'model.json').write_text('{}')
        return {'predictions': [0.]}
    copy = module.shutil.copyfile
    def unavailable(*args):
        raise OSError('Drive unavailable')
    monkeypatch.setattr(module.shutil, 'copyfile', unavailable)
    with pytest.raises(OSError, match='unavailable'):
        ledger.run({}, operation)
    assert not list((tmp_path / 'completed').glob('*.json'))
    monkeypatch.setattr(module.shutil, 'copyfile', copy)
    ledger.run({}, operation)
    assert calls == [1]
    assert ledger.summary()['durably_saved'] == 1


def test_age_features_are_causal_and_groups_are_independent():
    frame = feature_history(history().head(300))
    frame.loc[10:15, 'cotton_ret_1'] = np.nan
    groups_before = groups(frame, feature_groups(frame))
    assert frame.loc[15, 'age_cotton_ret_1'] == 6
    earlier = frame.loc[:199].copy()
    frame.loc[200:, 'cotton_ret_1'] = np.nan
    groups(frame, feature_groups(frame))
    pd.testing.assert_frame_equal(earlier, frame.loc[:199])
    assert len(groups_before['base']) == 24
    assert all('volume' not in f for f in groups_before['cotton_no_volume'])
    assert 'age_cotton_ret_1' not in groups_before['expanded']


def test_admission_requires_matched_eight_fold_inner_evidence():
    baseline = {i: 1. for i in range(8)}
    assert admission(baseline, {i: .99 for i in range(8)})['admitted']
    assert not admission(baseline, {i: .999 for i in range(8)})['admitted']
    assert not admission({1: 1}, {1: .5})['admitted']
    with pytest.raises(ValueError, match='Matched'):
        admission(baseline, {1: 1})


def reviewed(tmp_path):
    raw = tmp_path / 'report.txt'
    raw.write_text('Synthetic source: published 2010-01-06 12:00 UTC, original vintage')
    review = {'kind': 'ams', 'features': ['ams_spot'], 'max_age_days': 2,
              'usage': {'cost_tl': 0, 'research_allowed': True, 'terms_url': 'https://www.ams.usda.gov/market-news'},
              'files': {'report.txt': digest(raw)}, 'releases': [{
                  'values': {'ams_spot': 75.}, 'observed_through': '2010-01-06T00:00:00Z',
                  'published_at': '2010-01-06T12:00:00Z', 'vintage_id': 'original',
                  'source_url': 'https://www.ams.usda.gov/mnreports/cnddsq.pdf',
                  'source_file': 'report.txt', 'publication_evidence_file': 'report.txt',
                  'vintage_evidence_file': 'report.txt', 'timestamp_verified': True}]}
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(review))
    return path, review


def test_review_package_roundtrip_freshness_and_future_release(tmp_path):
    path, _review = reviewed(tmp_path)
    target = compile_review(path, tmp_path / 'package')
    manifest, rows, features = load_package(target)
    frame = history().head(15)
    values = attach_package(frame, manifest, rows, features)
    assert values.loc[values.date < '2010-01-06', 'ams_spot'].isna().all()
    assert values.loc[values.date.eq('2010-01-06'), 'ams_spot'].iloc[0] == 75
    assert values.loc[values.date >= '2010-01-09', 'ams_spot'].isna().all()
    future = rows.copy()
    future['published_at'] = pd.Timestamp('2030-01-01', tz='UTC')
    future['ams_spot'] = 999
    again = attach_package(frame, manifest, pd.concat([rows, future]), features)
    pd.testing.assert_frame_equal(values, again)
    with pytest.raises(ValueError, match='immutable'):
        compile_review(path, target)


@pytest.mark.parametrize('mutation,match', [
    ('unverified', 'review'), ('naive_time', 'timezone'), ('future_observation', 'beyond'), ('paid', 'zero-cost')])
def test_review_fails_closed(tmp_path, mutation, match):
    path, review = reviewed(tmp_path)
    release = review['releases'][0]
    if mutation == 'unverified':
        release['timestamp_verified'] = False
    elif mutation == 'naive_time':
        release['published_at'] = '2010-01-06'
    elif mutation == 'future_observation':
        release['observed_through'] = '2010-01-07T00:00:00Z'
    else:
        review['usage']['cost_tl'] = 1
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match=match):
        compile_review(path, tmp_path / 'package')
    assert not (tmp_path / 'package').exists()


@pytest.mark.parametrize('url', ['http://www.ams.usda.gov/a', 'https://evil.example/a',
    'https://www.ams.usda.gov/a?api_key=secret', 'https://user:secret@www.ams.usda.gov/a'])
def test_source_urls_are_public_and_credential_free(url):
    with pytest.raises(ValueError, match='Public HTTPS'):
        validate_url('ams', url)


def test_free_notebook_preserves_old_experiment_and_uses_dynamic_progress():
    notebook = Path(__file__).parents[1] / 'notebooks/cottonlens_free_research_colab.ipynb'
    cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    code = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
    for cell in code:
        ast.parse(cell)
    combined = '\n'.join(code)
    assert 'research-v2-tf-placement' not in combined
    assert "'free-data-v1'" in combined
    assert 'decision_pattern=namespace' in combined and 'group_count * 4' in combined
    assert "STAGE = 'status'" in combined


def test_two_publication_sources_have_no_metadata_collision(tmp_path):
    path, _review = reviewed(tmp_path)
    target = compile_review(path, tmp_path / 'package')
    manifest, rows, names = load_package(target)
    frame = attach_package(history().head(15), manifest, rows, names)
    second = rows.rename(columns={'ams_spot': 'wasde_stocks'})
    result = attach_package(frame, {'max_age_days': 30}, second, ['wasde_stocks'])
    assert not any(c.endswith(('_x', '_y')) for c in result.columns)
    assert 'ams_spot' in result and 'wasde_stocks' in result
