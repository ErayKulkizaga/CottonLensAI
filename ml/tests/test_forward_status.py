"""Read-only forward evidence audit; synthetic inputs, no network or fitting."""
import json

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.prospective import (
    baseline_lock,
    baseline_status,
    record_baselines,
)


def inputs(root, observed='2026-10-03T00:05Z'):
    market = root / 'market'
    frame = pd.DataFrame({'date': pd.bdate_range('2026-07-01', '2026-10-02')})
    frame['close'] = 80 * np.exp(np.sin(np.arange(len(frame))) * .01)
    path = market / 'objects' / 'synthetic.parquet'
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    receipt = {'symbol': 'CT=F', 'source_file': 'objects/synthetic.parquet',
               'source_sha256': digest(path), 'observed_available_at': observed}
    freeze_record(market / 'observations' / (content_id(receipt) + '.json'), receipt)
    baseline_lock(root / 'forward', now='2026-10-02T23:00Z')
    return [(frame.rename(columns={'close': 'cotton_close'}), receipt)]


def hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob('*') if p.is_file()}


def replace_record(path, changes):
    body = {**read_record(path), **changes}
    path.write_text(json.dumps({**body, 'record_id': content_id(body)}), encoding='utf-8')


def test_status_verifies_published_receipt_and_is_read_only(tmp_path):
    snapshots = inputs(tmp_path)
    record_baselines(tmp_path / 'forward', snapshots, now='2026-10-03T00:20Z')
    before = hashes(tmp_path)
    result = baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')
    assert result['published_origins'] == 1 and result['missing_origins'] == 0
    assert result['origins'][0]['pre_cutoff_snapshots'] == 1
    assert result['status'] == 'collecting'
    assert 'horizons' not in result and 'mae' not in str(result)
    assert hashes(tmp_path) == before


@pytest.mark.parametrize(('observed', 'reason'), [
    ('2026-10-03T00:05Z', 'publication_window_missed'),
    ('2026-10-03T00:19Z', 'no_archived_pre_cutoff_input'),
])
def test_missing_diagnosis_does_not_backfill(tmp_path, observed, reason):
    snapshots = inputs(tmp_path, observed)
    record_baselines(tmp_path / 'forward', snapshots, now='2026-10-03T01:00Z')
    before = hashes(tmp_path)
    result = baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T02:00Z')
    assert result['status'] == 'awaiting_forward_predictions'
    assert result['registered_origins'] == 1 and result['published_origins'] == 0
    assert result['origins'][0]['diagnosis'] == reason
    assert hashes(tmp_path) == before


@pytest.mark.parametrize('changes', [
    {'recorded_at': '2026-10-03T00:30Z'},
    {'recorded_at': '2026-10-03T00:14Z'},
    {'decision_at': '2026-10-03T00:16Z'},
    {'state': 'arbitrary'},
    {'current_price': 1.},
    {'input_file': '../market/objects/synthetic.parquet'},
    {'predictions': {}},
    {'source_receipt_id': 'missing'},
])
def test_rehashed_inconsistent_record_is_not_valid_evidence(tmp_path, changes):
    snapshots = inputs(tmp_path)
    record_baselines(tmp_path / 'forward', snapshots, now='2026-10-03T00:20Z')
    replace_record(tmp_path / 'forward/baseline-origins/2026-10-02.json', changes)
    with pytest.raises(ValueError):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_published_record_cannot_claim_late_receipt(tmp_path):
    snapshots = inputs(tmp_path, '2026-10-03T00:19Z')
    record_baselines(tmp_path / 'forward', snapshots, now='2026-10-03T01:00Z')
    other = tmp_path / 'other'
    ontime = inputs(other)
    record_baselines(other / 'forward', ontime, now='2026-10-03T00:20Z')
    published = read_record(other / 'forward/baseline-origins/2026-10-02.json')['predictions']
    replace_record(tmp_path / 'forward/baseline-origins/2026-10-02.json',
                   {'state': 'published', 'recorded_at': '2026-10-03T00:20Z',
                    'predictions': published, 'missing_reason': None})
    with pytest.raises(ValueError, match='cutoff'):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_source_corruption_is_rejected(tmp_path):
    snapshots = inputs(tmp_path)
    record_baselines(tmp_path / 'forward', snapshots, now='2026-10-03T00:20Z')
    (tmp_path / 'market/objects/synthetic.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='snapshot checksum'):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_unregistered_origin_and_window_boundary_are_visible(tmp_path):
    inputs(tmp_path)
    result = baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T00:20Z')
    assert result['pending_publication_origins'] == ['2026-10-02']
    assert not result['unregistered_origins']
    result = baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T00:30Z')
    assert result['unregistered_origins'] == ['2026-10-02']


def test_absent_store_does_not_create_lock(tmp_path):
    root = tmp_path / 'absent'
    assert baseline_status(root / 'forward', root / 'market')['status'] == 'not_locked'
    assert not root.exists()


def test_changed_locked_recipe_is_rejected(tmp_path):
    inputs(tmp_path)
    replace_record(tmp_path / 'forward/baseline-lock.json', {'decision_utc': '01:15'})
    with pytest.raises(ValueError, match='lock policy'):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_rehashed_input_is_bound_to_original_receipt(tmp_path):
    snapshots = inputs(tmp_path)
    record_baselines(tmp_path / 'forward', snapshots, now='2026-10-03T00:20Z')
    target = tmp_path / 'forward/baseline-inputs/2026-10-02.parquet'
    frame = pd.read_parquet(target)
    frame.loc[0, 'cotton_close'] += 1
    frame.to_parquet(target, index=False)
    replace_record(tmp_path / 'forward/baseline-origins/2026-10-02.json', {'input_sha256': digest(target)})
    with pytest.raises(ValueError, match='match selected source receipt'):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_rehashed_variance_is_bound_to_recipe(tmp_path):
    snapshots = inputs(tmp_path)
    record_baselines(tmp_path / 'forward', snapshots, now='2026-10-03T00:20Z')
    path = tmp_path / 'forward/baseline-origins/2026-10-02.json'
    predictions = read_record(path)['predictions']
    predictions['5']['variance'] *= 2
    replace_record(path, {'predictions': predictions})
    with pytest.raises(ValueError, match='locked baseline recipe'):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_in_progress_writer_prevents_unstable_audit(tmp_path):
    inputs(tmp_path)
    (tmp_path / '.writer-lock').mkdir()
    with pytest.raises(RuntimeError, match='stable copy'):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_no_weekend_origins_are_invented(tmp_path):
    inputs(tmp_path)
    result = baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-05T01:00Z')
    assert result['unregistered_origins'] == ['2026-10-02']


def test_receipt_with_current_day_bar_is_not_completed_input(tmp_path):
    inputs(tmp_path, '2026-10-02T23:05Z')
    with pytest.raises(ValueError, match='uncompleted daily bar'):
        baseline_status(tmp_path / 'forward', tmp_path / 'market', now='2026-10-03T01:00Z')


def test_status_cli_never_collects_or_writes(tmp_path, monkeypatch, capsys):
    from cottonlens_ml.research import live

    inputs(tmp_path)
    before = hashes(tmp_path)
    monkeypatch.setattr('sys.argv', ['live', '--store', str(tmp_path), '--status'])
    monkeypatch.setattr(live, 'collect', lambda *a, **k: pytest.fail('status must not collect'))
    live.main()
    assert json.loads(capsys.readouterr().out)['published_origins'] == 0
    assert hashes(tmp_path) == before


def test_status_cli_fails_closed_for_corruption(tmp_path, monkeypatch, capsys):
    from cottonlens_ml.research import live

    inputs(tmp_path)
    (tmp_path / 'market/objects/synthetic.parquet').write_bytes(b'changed')
    monkeypatch.setattr('sys.argv', ['live', '--store', str(tmp_path), '--status'])
    with pytest.raises(SystemExit) as exc:
        live.main()
    assert exc.value.code == 2 and 'no collection or training' in capsys.readouterr().err
