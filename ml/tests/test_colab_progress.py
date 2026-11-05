"""Notebook supervision checks: subprocesses print/sleep only; no model fits."""
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('colab_progress', Path(__file__).parents[1] / 'colab_progress.py')
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


def options(tmp_path, updates):
    return {'cwd': tmp_path, 'env': os.environ.copy(), 'drive_logs': tmp_path / 'drive',
            'decisions': tmp_path / 'decisions', 'scratch': tmp_path, 'update': updates.append,
            'interval': .03, 'sync_interval': .02}


def test_silent_child_has_heartbeat_and_mirrored_log(tmp_path):
    updates = []
    result = progress.supervise([sys.executable, '-u', '-c',
        "import time; time.sleep(.3); print('A elasticnet T+5 fold=6 candidate=24/24 inner=0.99'); time.sleep(.15)"],
        **options(tmp_path, updates))
    assert len(updates) > 3
    assert any('Starting or loading' in message for message in updates)
    assert any('elasticnet T+5 fold 6/8' in message for message in updates)
    assert 'candidate=24/24' in Path(result['local_log']).read_text()
    # The short synthetic heartbeat allows supervise to return with final
    # Drive sync still pending. Verify eventual mirroring, not thread timing.
    target = Path(result['drive_log'])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if target.exists() and 'candidate=24/24' in target.read_text():
            break
        time.sleep(.01)
    assert target.exists() and 'candidate=24/24' in target.read_text(), updates[-1]


def test_failure_preserves_trace(tmp_path):
    with pytest.raises(RuntimeError, match='deliberate failure'):
        progress.supervise([sys.executable, '-c', "raise ValueError('deliberate failure')"],
                            **options(tmp_path, []))
    assert list(tmp_path.glob('cottonlens-*.log'))


def test_planned_pilot_pause_is_visible_as_pause(tmp_path):
    updates = []
    progress.supervise([sys.executable, '-u', '-c',
        "print('PILOT planned_pause: completed fits remain saved; rerun the same pilot to resume.')"],
        **options(tmp_path, updates))
    assert 'Stage paused as planned' in updates[-1]
    assert 'Stage completed successfully' not in updates[-1]


def test_interrupt_stops_child(tmp_path, monkeypatch):
    children = []
    original = subprocess.Popen
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(progress.subprocess, 'Popen', spawn)
    opts = options(tmp_path, [])
    opts['update'] = interrupt
    with pytest.raises(KeyboardInterrupt):
        progress.supervise([sys.executable, '-c', 'import time; time.sleep(30)'], **opts)
    assert children[0].poll() is not None


def test_blocked_drive_does_not_block_child_or_heartbeat(tmp_path, monkeypatch):
    entered, release, left = threading.Event(), threading.Event(), threading.Event()
    def blocked_copy(*_):
        entered.set()
        release.wait(5)
        left.set()
        raise OSError('Drive unavailable')
    monkeypatch.setattr(progress.shutil, 'copyfile', blocked_copy)
    updates = []
    try:
        result = progress.supervise([sys.executable, '-c', 'import time; time.sleep(.2)'],
                                    **options(tmp_path, updates))
        assert entered.is_set()
        assert result['returncode'] == 0
        assert len(updates) > 3
        assert 'still pending' in updates[-1]
    finally:
        release.set()
        left.wait(2)


def test_lock_requires_confirmation_and_no_live_writer(tmp_path):
    lock = tmp_path / '.writer-lock'
    lock.mkdir()
    evidence = tmp_path / 'saved.json'
    evidence.write_text('preserved')
    proc = tmp_path / 'proc'
    process = proc / '123'
    process.mkdir(parents=True)
    (process / 'cmdline').write_bytes(b'python\0-m\0cottonlens_ml.research.engine\0')
    with pytest.raises(RuntimeError, match='Confirm'):
        progress.recover_empty_lock(tmp_path, proc=proc)
    with pytest.raises(RuntimeError, match='still running'):
        progress.recover_empty_lock(tmp_path, proc=proc, confirmed_stopped=True)
    (process / 'cmdline').write_bytes(b'other\0')
    progress.recover_empty_lock(tmp_path, proc=proc, confirmed_stopped=True)
    assert not lock.exists()
    assert evidence.read_text() == 'preserved'


def test_nonempty_lock_is_never_deleted(tmp_path):
    lock = tmp_path / '.writer-lock'
    lock.mkdir()
    (lock / 'owner.json').write_text('preserve')
    with pytest.raises(OSError):
        progress.recover_empty_lock(tmp_path, confirmed_stopped=True, proc=tmp_path / 'proc')
    assert (lock / 'owner.json').read_text() == 'preserve'


def test_owned_recovery_preserves_evidence_and_requires_exact_owner(tmp_path):
    import json
    token = 'a' * 32
    lock = tmp_path / '.writer-lock'
    lock.mkdir()
    owner = {'token': token, 'pid': 123}
    (lock / 'owner.json').write_text(json.dumps(owner))
    kwargs = {'confirmed_stopped': True, 'proc': tmp_path / 'proc'}
    with pytest.raises(ValueError):
        progress.recover_owned_lock(tmp_path, owner_token='../outside', **kwargs)
    with pytest.raises(ValueError, match='owner changed'):
        progress.recover_owned_lock(tmp_path, owner_token='b' * 32, **kwargs)
    assert lock.exists()
    progress.recover_owned_lock(tmp_path, owner_token=token, **kwargs)
    assert not lock.exists()
    assert json.loads((tmp_path / 'recovered-locks' / (token + '.json')).read_text()) == owner
