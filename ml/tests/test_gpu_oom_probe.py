from types import SimpleNamespace

import gpu_oom_probe as probe
import pytest
from cottonlens_ml.sprint import freeze_record


def test_denied_before_output_or_tensorflow(tmp_path, monkeypatch):
    def deny(gpu):
        raise RuntimeError('Colab required')
    monkeypatch.setattr(probe, 'device', deny)
    with pytest.raises(RuntimeError, match='Colab required'):
        probe.run(tmp_path/'probe')
    assert not (tmp_path/'probe').exists()


def test_process_order_cache_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'device', lambda gpu: 'generated-fixture')
    calls = []
    root = tmp_path/'probe'
    def process(args, **kwargs):
        stage = args[-1]
        calls.append(stage)
        freeze_record(root/f'{stage}.json', {'stage': stage, 'generated_fixture': True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probe.subprocess, 'run', process)
    result = probe.run(root)
    assert calls == ['exhaust', 'recover'] and result['synthetic_fits'] == 0
    assert probe.run(root)['cached'] and len(calls) == 2
    (root/'recover.json').write_bytes(b'broken')
    with pytest.raises(ValueError, match='Corrupt'):
        probe.run(root)
    assert len(calls) == 2


def test_failed_child_preserves_partial_and_cannot_autoretry(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'device', lambda gpu: 'generated-fixture')
    def fail(*args, **kwargs):
        raise probe.subprocess.CalledProcessError(1, 'generated-worker')
    monkeypatch.setattr(probe.subprocess, 'run', fail)
    with pytest.raises(probe.subprocess.CalledProcessError):
        probe.run(tmp_path/'probe')
    assert not (tmp_path/'probe/complete.json').exists()
    with pytest.raises(ValueError, match='Partial'):
        probe.run(tmp_path/'probe')


def test_worker_preserves_completed_stage_before_tensorflow_import(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'device', lambda gpu: 'generated-fixture')
    (tmp_path/'exhaust.json').write_bytes(b'preserve')
    with pytest.raises(ValueError, match='preserve'):
        probe.worker(tmp_path, 'L4', 'exhaust')
