"""Real subprocess/concurrent-call faults with byte fixtures, never model fits."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from cottonlens_ml.research.ledger import Ledger, read_record


def test_concurrent_direct_calls_cannot_duplicate_a_job(tmp_path):
    ledger = Ledger(tmp_path, {'data': 'generated-bytes'})
    started, release = Event(), Event()
    calls = []

    def operation(workspace):
        calls.append(1)
        started.set()
        assert release.wait(30)
        (workspace/'model.bin').write_bytes(b'generated checkpoint')
        return {'predictions': [0.]}

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(ledger.run, {}, operation)
        try:
            assert started.wait(10)
            with pytest.raises(RuntimeError, match='writer'):
                ledger.run({}, operation)
            program = '''
import sys
from pathlib import Path
from cottonlens_ml.research.ledger import Ledger
try:
    Ledger(Path(sys.argv[1]), {'data':'generated-bytes'}).run({}, lambda p: {})
except RuntimeError as error:
    assert 'writer' in str(error)
else:
    raise AssertionError('Duplicate process acquired ledger')
'''
            rival = subprocess.run([sys.executable, '-c', program, str(tmp_path)],
                                   capture_output=True, text=True, timeout=20, check=False)
            assert rival.returncode == 0, rival.stderr
        finally:
            release.set()
        record = first.result(timeout=10)
    assert ledger.run({}, operation) == record
    assert calls == [1]
    assert ledger.summary()['execution_attempts_started'] == 1
    assert ledger.summary()['execution_attempts_unresolved'] == 0


def test_hard_exit_preserves_started_attempt_and_never_steals_lock(tmp_path):
    program = '''
import json, os, sys
from pathlib import Path
from cottonlens_ml.research.ledger import Ledger
def operation(workspace):
    Path(sys.argv[1], 'child.json').write_text(json.dumps({'pid':os.getpid(), 'workspace':str(workspace)}))
    (workspace/'partial.bin').write_bytes(b'incomplete generated checkpoint')
    os._exit(73)
Ledger(Path(sys.argv[1]), {'data':'hard-exit-fixture'}).run({}, operation)
'''
    process = subprocess.Popen([sys.executable, '-c', program, str(tmp_path)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        _, error = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    assert process.returncode == 73, error
    ledger = Ledger(tmp_path, {'data': 'hard-exit-fixture'})
    summary = ledger.summary()
    assert summary['durably_saved'] == 0
    assert summary['execution_attempts_started'] == summary['execution_attempts_unresolved'] == 1
    start = read_record(next((tmp_path/'execution-attempts/started').glob('*.json')))
    assert start['state'] == 'started_new_compute' and not start['optimizer_resume_claimed']
    partial = Path(json.loads((tmp_path/'child.json').read_text())['workspace'])/'partial.bin'
    assert partial.read_bytes() == b'incomplete generated checkpoint'
    calls = []

    def new_attempt(workspace):
        calls.append(1)
        (workspace/'model.bin').write_bytes(b'new generated attempt')
        return {'predictions': [0.]}

    with pytest.raises(RuntimeError, match='writer'):
        ledger.run({}, new_attempt)
    assert not calls
    # Test-only recovery: the exact owner process has exited authoritatively.
    # Production never deletes a stale lock based on elapsed time or a PID guess.
    lock = tmp_path/'execution/.writer-lock'
    owner = json.loads((lock/'owner.json').read_text())
    assert owner['pid'] == json.loads((tmp_path/'child.json').read_text())['pid']
    assert process.poll() == 73
    if sys.platform == 'win32':
        # Windows venv python.exe may be a launcher with a different PID from
        # the interpreter. Query the actual lock owner's process without signals.
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, owner['pid'])
        if handle:
            try:
                code = ctypes.c_ulong()
                assert kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
                assert code.value != 259  # STILL_ACTIVE
            finally:
                kernel.CloseHandle(ctypes.c_void_p(handle))
        else:
            assert ctypes.get_last_error() == 87  # No such process.
    else:
        assert owner['pid'] == process.pid
    (lock/'owner.json').unlink()
    lock.rmdir()
    record = ledger.run({}, new_attempt)
    assert record['run_id'] != start['run_id'] and calls == [1]
    assert ledger.run({}, new_attempt) == record and calls == [1]
    assert ledger.summary()['execution_attempts_unresolved'] == 1
    assert partial.read_bytes() == b'incomplete generated checkpoint'


def test_oom_like_failure_never_changes_recipe_or_retries_automatically(tmp_path):
    ledger = Ledger(tmp_path, {'data': 'generated-oom-fixture'})
    spec = {'batch_size': 256, 'seed': 1701}
    calls = []

    class InjectedResourceExhausted(RuntimeError):
        pass

    def operation(workspace):
        calls.append(1)
        (workspace/'partial.bin').write_bytes(b'partial generated bytes')
        raise InjectedResourceExhausted('injected OOM; no actual GPU allocation')

    with pytest.raises(InjectedResourceExhausted, match='injected OOM'):
        ledger.run(spec, operation)
    assert calls == [1] and spec == {'batch_size': 256, 'seed': 1701}
    failure = read_record(next((tmp_path/'attempts').glob('*.json')))
    start = read_record(next((tmp_path/'execution-attempts/started').glob('*.json')))
    assert failure['run_id'] == start['run_id']
    assert failure['specification'] == {**spec, 'repeat_reason': None}
    assert not failure['optimizer_resume_claimed'] and not failure['locally_completed']
    assert ledger.summary()['durably_saved'] == ledger.summary()['execution_attempts_unresolved'] == 0


def test_corrupt_staged_checkpoint_stops_without_refitting(tmp_path, monkeypatch):
    from cottonlens_ml.research import ledger as module
    ledger = Ledger(tmp_path, {'data': 'generated-staged-fixture'})
    calls = []
    def operation(workspace):
        calls.append(workspace)
        (workspace/'model.bin').write_bytes(b'generated checkpoint')
        return {'predictions': [0.]}
    original = module.shutil.copyfile
    def unavailable(*args):
        raise OSError('injected upload failure')
    monkeypatch.setattr(module.shutil, 'copyfile', unavailable)
    with pytest.raises(OSError, match='upload failure'):
        ledger.run({}, operation)
    (calls[0]/'model.bin').write_bytes(b'corrupt checkpoint')
    monkeypatch.setattr(module.shutil, 'copyfile', original)
    with pytest.raises(ValueError, match='Corrupt locally completed'):
        ledger.run({}, operation)
    assert len(calls) == 1 and ledger.summary()['durably_saved'] == 0


def test_summary_rejects_mutated_execution_journal(tmp_path):
    ledger = Ledger(tmp_path, {'data': 'generated-journal-fixture'})
    ledger.run({}, lambda workspace: {'predictions': [0.]})
    marker = next((tmp_path/'execution-attempts/computed').glob('*.json'))
    marker.write_text('{}')
    with pytest.raises(ValueError):
        ledger.summary()
