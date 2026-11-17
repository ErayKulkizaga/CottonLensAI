"""Local control-flow tests only; subprocess stubs do not prove GPU behavior."""
from types import SimpleNamespace

import gpu_restart_probe as probe
import pytest


def test_local_restart_probe_stops_before_query_or_output(tmp_path, monkeypatch):
    def deny():
        raise RuntimeError('Colab required')
    monkeypatch.setattr(probe, 'require_colab_training', deny)
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **k: pytest.fail('Unexpected device query'))
    with pytest.raises(RuntimeError, match='Colab required'):
        probe.run(tmp_path/'probe')
    assert not (tmp_path/'probe').exists()


def test_wrong_device_and_partial_output_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'require_colab_training', lambda: None)
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **k: 'NVIDIA T4')
    with pytest.raises(RuntimeError, match='no fallback'):
        probe.run(tmp_path/'probe', 'L4')
    assert not (tmp_path/'probe').exists()
    monkeypatch.setattr(probe, 'device', lambda gpu: 'generated-device-fixture')
    with pytest.raises(ValueError, match='Partial restart'):
        probe.run(tmp_path)


@pytest.mark.parametrize('exit_code', [73, 1, -9])
def test_orchestration_cache_and_unexpected_exit(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(probe, 'device', lambda gpu: 'generated-device-fixture')
    root = tmp_path/'probe'
    calls = []
    def operation(args, **kwargs):
        calls.append(args)
        if '--stage' not in args:
            (root/'baseline').mkdir()
            (root/'baseline/step1.keras').write_bytes(b'generated-byte-fixture')
        elif args[-1] == 'crash':
            return SimpleNamespace(returncode=exit_code)
        else:
            (root/'resumed.keras').write_bytes(b'generated-resumed-fixture')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probe.subprocess, 'run', operation)
    if exit_code != 73:
        with pytest.raises(RuntimeError, match='worker exit'):
            probe.run(root)
        assert len(calls) == 2 and not (root/'complete.json').exists()
        return
    result = probe.run(root)
    assert len(calls) == 3 and result['market_fits'] == 0
    assert probe.run(root)['cached'] and len(calls) == 3
    (root/'resumed.keras').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt restart'):
        probe.run(root)
    assert len(calls) == 3


def test_worker_refuses_resumed_output_before_tensorflow_import(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'device', lambda gpu: 'generated-device-fixture')
    (tmp_path/'resumed.keras').write_bytes(b'preserve')
    with pytest.raises(ValueError, match='Preserve existing'):
        probe.worker(tmp_path, 'L4', 'resume')
    assert (tmp_path/'resumed.keras').read_bytes() == b'preserve'
