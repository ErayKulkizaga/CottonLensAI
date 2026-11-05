"""Approved twice-daily consoleless runner; execute using pythonw, never a shell."""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution(directory, expected):
    body = json.loads((directory/'execution-manifest.json').read_bytes())
    identity = {k: v for k, v in body.items() if k != 'execution_source_id'}
    if fingerprint(identity) != expected or body['execution_source_id'] != expected:
        raise ValueError('Execution source identity changed')
    for name, checksum in body['files'].items():
        p = directory/name
        if (not p.resolve().is_relative_to(directory.resolve()) or p.is_symlink() or sha(p) != checksum):
            raise ValueError('Frozen daily runner changed')
    code = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()
            and p.suffix.lower() in ('.py', '.pyw', '.ps1', '.bat', '.cmd', '.pyd', '.dll', '.exe')
            and '__pycache__' not in p.parts}
    if code != set(body['files']):
        raise ValueError('Unmanifested executable code in daily runtime')
    return body


def verify_core_before_import(study, source, expected):
    raw = json.loads((study/'source-manifest.json').read_bytes())
    record_id = raw.pop('record_id', None)
    if record_id != fingerprint(raw) or raw['source_id'] != expected or fingerprint({k: v for k, v in raw.items() if k != 'source_id'}) != expected:
        raise ValueError('Core manifest identity changed before import')
    known = {**raw['ml']['files'], **raw['helpers']}
    for name, checksum in known.items():
        p = source/name
        if p.is_symlink() or not p.resolve().is_relative_to(source.resolve()) or sha(p) != checksum:
            raise ValueError('Core source changed before import')
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()
              and p.suffix.lower() in ('.py', '.pyw', '.ps1', '.toml', '.lock', '.txt', '.pyd', '.dll', '.exe', '.bat', '.cmd')
              and '__pycache__' not in p.parts}
    if actual != set(known):
        raise ValueError('Unmanifested core code or dependency file before import')


def phase_at(stamp):
    if stamp.hour != 0:
        return 'outside_window'
    if stamp.minute < 15:
        return 'capture'
    if stamp.minute < 30:
        return 'publish'
    return 'outside_window'


def operate(core, study, source, authorization, requested='auto'):
    stamp = core.now(); phase = 'status' if requested == 'status' else phase_at(stamp)
    ready, design = core.verify(study, source)
    core.authorize(study, ready, design, authorization)
    auth = core.read_record(authorization)
    if (auth.get('human_authorization_sha256') != core.digest(study/'human-authorization.json')
            or core.read_record(study/'human-authorization.json')['human_user_answer'] != 'Evet, yalnız günlük ve konsolsuz görev'):
        raise ValueError('Explicit daily-consoleless consent is missing or changed')
    locked = core.read_record(study/'cohort-lock.json')
    if locked['identity'] != ready['identity']:
        raise ValueError('Daily runner has a different prospective lock')
    if phase == 'status':
        return {'phase': phase, 'status': 'VERIFIED_NO_SIDE_EFFECT', 'study_status': core.status(study, source)}
    if phase == 'outside_window':
        return {'phase': phase, 'status': 'SKIPPED_OUTSIDE_WINDOW', 'new_fits': 0, 'new_forecasts': 0}
    origin = (stamp.normalize()-core.pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    cal = core.calendar(study)
    if core.pd.Timestamp(origin).date() > cal.end:
        raise ValueError('Official calendar extension required before further daily recording')
    if origin < locked['first_origin'] or not cal.session(core.pd.Timestamp(origin).date()):
        return {'phase': phase, 'origin': origin, 'status': 'SKIPPED_NON_COHORT_SESSION', 'new_fits': 0, 'new_forecasts': 0}
    slots = core.slots(cal, locked['first_origin'])
    core.target_date(cal, origin)  # No fit/publication when the next session is not yet supported.
    label_only = len(slots) == 252 and origin == core.target_date(cal, slots[-1])
    if origin not in slots and not label_only:
        return {'phase': phase, 'origin': origin, 'status': 'STUDY_ORIGIN_BUDGET_ENDED', 'new_fits': 0, 'new_forecasts': 0}
    if phase == 'capture':
        observation = core.capture(study, source)
        if observation['latest_completed_date'] != origin:
            raise ValueError('Source did not supply the expected completed origin')
        if label_only:
            return {'phase': phase, 'origin': origin, 'status': 'FINAL_TARGET_CAPTURED',
                    'observation_id': observation['observation_id'], 'new_fits': 0, 'new_forecasts': 0}
        remaining = (core.decision(origin)-core.now()).total_seconds()/60
        if remaining <= 0:
            raise ValueError('Capture missed the model/selection cutoff; no late fit or publication')
        previous_permission = os.environ.get('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR')
        os.environ['COTTONLENS_ALLOW_LOCAL_CPU_TABULAR'] = '1'
        try:
            fitted = core.fit_for_origin(study, source, authorization, origin, minutes=min(30., remaining))
        finally:
            if previous_permission is None:
                os.environ.pop('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR', None)
            else:
                os.environ['COTTONLENS_ALLOW_LOCAL_CPU_TABULAR'] = previous_permission
        return {'phase': phase, 'origin': origin, 'status': 'CAPTURED_AND_STATE_READY',
                'observation_id': observation['observation_id'], 'state': fitted, 'new_forecasts': 0}
    if label_only:
        return {'phase': phase, 'origin': origin, 'status': 'NO_FORECAST_ON_FINAL_TARGET_CAPTURE', 'new_fits': 0, 'new_forecasts': 0}
    result = core.publish(study, source)
    if result['origin'] != origin:
        raise ValueError('Unexpected origin publication')
    path = study/'forecasts'/origin
    return {'phase': phase, 'origin': origin, 'status': result['status'], 'new_fits': 0,
            'forward_record_id': core.content_id(core.read_record(path/'forecast.json')),
            'forecast_sha256': core.digest(path/'forecast.json'),
            'publication_receipt_sha256': core.digest(path/'publication-receipt.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'authorization', 'expected-execution-source', 'expected-core-source'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--phase', choices=('auto', 'status', 'backup'), default='auto')
    args = parser.parse_args(argv)
    study, source, authorization = Path(args.study), Path(args.source), Path(args.authorization)
    run_id = uuid.uuid4().hex; directory = study/'daily-runs'/run_id
    directory.mkdir(parents=True, exist_ok=False)
    # pythonw has no console/stdout. Every result and traceback goes to files.
    started = datetime.now(UTC).isoformat(); result = None; failure = None; code = 1; execution = None; permission_verified = False
    os.environ.update({'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8'})
    sys.dont_write_bytecode = True
    with (directory/'run.log').open('w', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            execution = verify_execution(Path(__file__).resolve().parent, args.expected_execution_source)
            if execution['core_source_id'] != args.expected_core_source:
                raise ValueError('Execution manifest points to a different core')
            verify_core_before_import(study, source, args.expected_core_source)
            sys.path[:0] = [str(source/'ml/src'), str(source)]
            from research import ct_manual_forward as core
            if core.source(source)['source_id'] != args.expected_core_source:
                raise ValueError('Frozen forecasting core changed')
            permission = core.read_record(authorization)
            if (permission.get('human_authorization_sha256') != core.digest(study/'human-authorization.json')
                    or core.read_record(study/'human-authorization.json')['human_user_answer'] != 'Evet, yalnız günlük ve konsolsuz görev'):
                raise ValueError('Explicit daily consent is missing or changed')
            permission_verified = True
            with core.writer(study/'operations'):
                result = operate(core, study, source, authorization, 'status' if args.phase == 'backup' else args.phase)
            code = 0
        except Exception as exc:  # noqa: BLE001 - process boundary records full traceback and a nonzero exit.
            failure = {'type': type(exc).__name__, 'message': str(exc)}
            traceback.print_exc()
        finally:
            body = {'run_id': run_id, 'started_at': started, 'completed_at': datetime.now(UTC).isoformat(),
                    'execution_source_id': args.expected_execution_source, 'core_source_id': args.expected_core_source,
                    'phase_requested': args.phase, 'result': result, 'error': failure, 'exit_code': code,
                    'runner_sha256': sha(Path(__file__)), 'console_requested': False}
            # Independent small receipt, readable without importing the ML stack.
            (directory/'receipt.json').write_text(json.dumps({**body, 'record_id': fingerprint(body)}, indent=2, allow_nan=False), encoding='utf-8')
    phase = (result or {}).get('phase', phase_at(datetime.fromisoformat(started)))
    skipped = (result or {}).get('status', '').startswith(('SKIPPED_', 'STUDY_ORIGIN_', 'NO_FORECAST_'))
    if permission_verified and execution is not None and (args.phase == 'backup' or (args.phase == 'auto' and phase in ('capture', 'publish') and not skipped)):
        with (directory/'backup.log').open('w', encoding='utf-8', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                from research.ct_forward_backup import backup
                result_backup = backup(study, {**execution, 'runtime_path': str(Path(__file__).resolve().parent)},
                                       run_id, 'manual' if args.phase == 'backup' else phase)
                archive_body = {'status': result_backup['status'], 'backup_id': result_backup['backup_id'],
                                'release': result_backup['release'], 'archive_sha256': result_backup['archive_sha256'],
                                'execution_source_id': args.expected_execution_source, 'source_files_deleted': 0}
            except Exception as exc:  # noqa: BLE001 - preserve failure trace without a GUI console or source deletion.
                traceback.print_exc(); code = 1
                archive_body = {'status': 'BACKUP_FAILED_SOURCE_PRESERVED', 'error': type(exc).__name__+': '+str(exc),
                                'execution_source_id': args.expected_execution_source, 'source_files_deleted': 0}
            (directory/'backup-receipt.json').write_text(json.dumps({**archive_body, 'record_id': fingerprint(archive_body)}, indent=2), encoding='utf-8')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
