"""Notebook-only process supervision; never changes training code or checkpoint identities."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path

PROGRESS = re.compile(r'^([\w-]+) (\w+) T\+(\d+) fold=(\d+) candidate=(\d+)/(\d+) inner=([0-9.]+)')


def engine_processes(proc=Path('/proc')):
    active = []
    for directory in proc.glob('[0-9]*'):
        try:
            args = (directory / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        if b'cottonlens_ml.research.engine' in args:
            active.append(directory.name)
    return active


def recover_empty_lock(experiment, *, confirmed_stopped=False, proc=Path('/proc')):
    """Legacy locks have no owner/heartbeat: external-runtime confirmation is required."""
    lock = Path(experiment) / '.writer-lock'
    if not lock.exists():
        return
    if not confirmed_stopped:
        raise RuntimeError('Confirm the old runtime was deleted before recovering its lock')
    active = engine_processes(proc)
    if active:
        raise RuntimeError(f'Research still running on this runtime: {active}')
    lock.rmdir()  # Only an empty lock. Never recursively remove experiment evidence.


def recover_owned_lock(experiment, *, owner_token, confirmed_stopped=False, proc=Path('/proc')):
    """Explicit recovery after the previous runtime is stopped; never age-based."""
    if not isinstance(owner_token, str) or not re.fullmatch(r'[0-9a-f]{32}', owner_token):
        raise ValueError('Expected the exact hexadecimal owner token')
    lock = Path(experiment) / '.writer-lock'
    owner_path = lock / 'owner.json'
    if not confirmed_stopped or engine_processes(proc):
        raise RuntimeError('Confirm the old runtime stopped and stop all local research writers')
    owner = json.loads(owner_path.read_text(encoding='utf-8'))
    if owner.get('token') != owner_token or {p.name for p in lock.iterdir()} != {'owner.json'}:
        raise ValueError('Lock owner changed or unexpected lock contents')
    # Keep the ownership evidence without deleting any experiment or model files.
    archived = Path(experiment) / 'recovered-locks'
    archived.mkdir(exist_ok=True)
    destination = archived / (str(owner_token) + '.json')
    if destination.exists():
        raise ValueError('Owner already recovered; inspect lock manually')
    owner_path.rename(destination)
    lock.rmdir()


def process_sample(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        cpu = (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK')
        return cpu, Path(f'/proc/{pid}/wchan').read_text().strip()
    except (OSError, ValueError, IndexError, AttributeError):
        return None, 'unknown'


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def supervise(command, *, cwd, env, drive_logs, decisions, total=64, update=print,
              decision_pattern='*.json', fold_count=8,
              interval=3., sync_interval=30., scratch=None):
    """Local stdout file prevents pipe backpressure; Drive I/O never blocks heartbeat."""
    run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    local = Path(scratch or tempfile.gettempdir()) / f'cottonlens-{run_id}.log'
    target = Path(drive_logs) / local.name
    state = {'saved': None, 'sync_at': None, 'error': None}
    state_lock = threading.Lock()
    changed, finished, synced = threading.Event(), threading.Event(), threading.Event()
    last_line_at = started = time.monotonic()
    recent = deque(maxlen=30)
    last_progress = 'Starting or loading completed checkpoints'

    def sync():
        while True:
            changed.wait(sync_interval)
            changed.clear()
            final_copy = finished.is_set()
            try:
                saved = len(list(Path(decisions).glob(decision_pattern)))
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_suffix('.log.pending')
                shutil.copyfile(local, pending)
                pending.replace(target)
                with state_lock:
                    state.update(saved=saved, sync_at=time.monotonic(), error=None)
            except Exception as exc:  # noqa: BLE001 - report sync errors without stopping the training heartbeat
                with state_lock:
                    state['error'] = f'{type(exc).__name__}: {exc}'
            if final_copy:
                synced.set()
                return

    child = None
    try:
        # The reader sees local writes immediately; no mounted-Drive file is held open.
        with local.open('w', encoding='utf-8') as output, local.open(encoding='utf-8', errors='replace') as reader:
            child = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                     stdout=output, stderr=subprocess.STDOUT, text=True)
            thread = threading.Thread(target=sync, daemon=True)
            thread.start()
            changed.set()
            previous_cpu, previous_at = None, started
            while True:
                for line in reader:
                    recent.append(line.rstrip())
                    last_line_at = time.monotonic()
                    match = PROGRESS.match(line)
                    if match:
                        _, family, horizon, fold, candidate, count, score = match.groups()
                        last_progress = f'{family} T+{horizon} fold {fold}/{fold_count} candidate {candidate}/{count} inner={score}'
                    elif line.startswith(('FIT ', 'CACHE ', 'STAGE ', 'PILOT ')):
                        last_progress = line.strip()
                now = time.monotonic()
                cpu, wait = process_sample(child.pid)
                cpu_pct = None if cpu is None or previous_cpu is None else 100 * max(0., cpu - previous_cpu) / max(.001, now - previous_at)
                previous_cpu, previous_at = cpu, now
                with state_lock:
                    snapshot = dict(state)
                saved = snapshot['saved']
                filled = 0 if saved is None else min(24, int(24 * saved / max(1, total)))
                bar = '?' * 24 if saved is None else '#' * filled + '-' * (24 - filled)
                sync_age = 'pending' if snapshot['sync_at'] is None else f'{now - snapshot["sync_at"]:.0f}s ago'
                idle = now - last_line_at
                alert = ' | CHECK: no new output for 10+ minutes' if idle >= 600 else ''
                cpu_text = 'pending' if cpu_pct is None else f'{cpu_pct:.1f}%'
                update(f'[{bar}] saved={saved}/{total} | {last_progress}\n'
                       f'PID {child.pid} | elapsed {(now-started)/60:.1f} min | last output {idle:.0f}s ago | '
                       f'process CPU {cpu_text} | wait={wait} | Drive log sync {sync_age}{alert}\n'
                       f'local log: {local} | Drive log: {target}'
                       + (f'\nDrive sync error: {snapshot["error"]}' if snapshot['error'] else ''))
                code = child.poll()
                if code is not None:
                    recent.extend(line.rstrip() for line in reader)
                    break
                time.sleep(interval)
    except BaseException:
        if child is not None:
            stop_child(child)
        raise
    finally:
        finished.set()
        changed.set()
        synced.wait(timeout=min(5., max(.01, interval)))
    if code:
        raise RuntimeError(f'Stage exited {code}. Local log: {local}\n' + '\n'.join(recent))
    with state_lock:
        sync_error = state['error']
    sync_status = 'complete' if synced.is_set() and not sync_error else (sync_error or 'still pending')
    outcome = ('Stage paused as planned; resume the same pilot.'
               if any(line.startswith('PILOT planned_pause:') for line in recent)
               else 'Stage completed successfully.')
    update(f'{outcome} Local log: {local}. Drive log: {target}. '
           f'Final log sync: {sync_status}. Completed checkpoints remain in the experiment folder.')
    return {'returncode': code, 'local_log': str(local), 'drive_log': str(target)}
