"""Immutable result ledger, atomic completion and single-writer ownership."""
import json
import shutil
import time
import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.cohort import content_id
from cottonlens_ml.sprint import freeze_record, read_record, writer

REPEAT_REASONS = {'reproduction', 'new_seed', 'bugfix_verification'}


class FitBudgetReached(RuntimeError):
    """A planned pause before computation; not a failed or interrupted fit."""


class Ledger:
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity

    def key(self, specification):
        return content_id({'identity': self.identity, 'specification': specification})

    def receipt(self, record):
        """Small, immutable report projection; never copy predictions or fitted state."""
        body = {name: record[name] for name in ('experiment_id', 'completed_at',
                'compute_seconds', 'checkpoint_copy_seconds', 'elapsed_seconds')}
        body['data_seconds'] = record['result'].get('data_seconds', 0)
        freeze_record(self.root / 'receipts' / (record['experiment_id'] + '.json'), body)
        index_path = self.root / 'summary-index.json'
        index = read_record(index_path) if index_path.exists() else {
            'keys': [], 'timing': {k: 0. for k in ('compute_seconds', 'checkpoint_copy_seconds', 'data_seconds')}}
        if body['experiment_id'] not in index['keys']:
            index['keys'].append(body['experiment_id'])
            for name in index['timing']:
                index['timing'][name] += body[name]
            pending = index_path.with_name('.summary-' + uuid.uuid4().hex + '.pending')
            freeze_record(pending, index)
            pending.replace(index_path)

    def summary(self):
        """Legacy records stay readable without a hidden full-ledger migration."""
        completed = {p.stem for p in (self.root / 'completed').glob('*.json')}
        index_path = self.root / 'summary-index.json'
        index = read_record(index_path) if index_path.exists() else {'keys': [], 'timing': {}}
        if not set(index['keys']).issubset(completed):
            raise ValueError('Summary references missing completed records')
        # Remote metadata restore carries small receipts, not a mutable summary index.
        # Rebuild the read-only projection without loading fitted payloads/results.
        for path in (self.root/'receipts').glob('*.json'):
            if path.stem not in index['keys']:
                receipt = read_record(path)
                if path.stem not in completed or receipt['experiment_id'] != path.stem:
                    raise ValueError('Receipt references missing completed record')
                index['keys'].append(path.stem)
                for name in ('compute_seconds','checkpoint_copy_seconds','data_seconds'):
                    index['timing'][name] = index['timing'].get(name, 0.) + receipt[name]
        def execution_ids(stage):
            result = set()
            for path in (self.root/'execution-attempts'/stage).glob('*.json'):
                event = read_record(path)
                if event['run_id'] != path.stem:
                    raise ValueError('Execution journal run identity mismatch')
                result.add(path.stem)
            return result
        starts, computed = execution_ids('started'), execution_ids('computed')
        failed = {read_record(p).get('run_id') for p in (self.root/'attempts').glob('*.json')}
        return {'durably_saved': len(completed), 'timing_receipts': len(index['keys']),
                'unindexed_legacy_fits': len(completed) - len(index['keys']),
                'payloads_verified_by_status': False, 'timing': index['timing'],
                'failed_or_interrupted_attempts': len(list((self.root / 'attempts').glob('*.json'))),
                'execution_attempts_started': len(starts),
                'execution_attempts_unresolved': len(starts - computed - failed)}

    def run(self, specification, operation, *, repeat=None, before_compute=None, admission=None):
        # Protect cache checks, fitting, upload and mutable summary updates even
        # when a caller forgets its campaign-level writer. Separate workers must
        # use separate ledgers. A hard-kill lock is never stolen automatically.
        with writer(self.root/'execution'):
            return self._run_owned(specification, operation, repeat=repeat,
                                   before_compute=before_compute, admission=admission)

    def _run_owned(self, specification, operation, *, repeat=None, before_compute=None, admission=None):
        # New admitted namespaces validate before either cache reuse or fitting.
        # Historical namespaces remain readable; campaign migration is explicit.
        if self.identity.get('admission_policy') == 1:
            if admission is None:
                raise ValueError('Required data admission is missing')
            admission(self.identity, specification)
        if repeat is not None and repeat not in REPEAT_REASONS:
            raise ValueError('Explicit valid repetition reason required')
        spec = {**specification, 'repeat_reason': repeat}
        key = self.key(spec)
        result_path = self.root / 'completed' / f'{key}.json'
        if result_path.exists():
            record = read_record(result_path)
            if record['identity'] != self.identity or record['specification'] != spec:
                raise ValueError('Completed record identity/specification mismatch')
            for name, checksum in record['files'].items():
                if not safe_member(name) or digest(self.root / name) != checksum:
                    raise ValueError(f'Corrupt completed payload: {name}')
            self.receipt(record)
            print(f'CACHE restored verified fit {key[:12]}', flush=True)
            return record
        started = time.monotonic()
        run_id = uuid.uuid4().hex
        spool = Path('/content/cottonlens-research-work') / content_id(str(self.root.resolve())) / key
        if not Path('/content').is_dir():
            # Real small CPU fits separately enforce explicit permission and the thread limit.
            spool = self.root / ('local-work' if self.identity.get('profile') in ('price-only-median-t1-v1', 'full-year-v1', 'ams-exploration-v1', 'fas-exploration-v1', 'nass-exploration-v1', 'wasde-exploration-v1', 'cftc-exploration-v1', 'fx-exploration-v1', 'crop-exploration-v1', 'weather-exploration-v1', 'recency-pilot-v1', 'availability-clock-pilot-v1', 'return-path-pilot-v1', 'agri-transfer-pilot-v1', 'wasde-text-pilot-v1', 'fundamental-joint-pilot-v1', 'wasde-regional-t1-pilot-v1', 'nass-regional-t1-pilot-v1', 'weak-signal-control-v1', 'paired-price-loss-control-v1', 'named-label-control-t5-v1', 'named-capacity-control-t5-v1', 'named-history-source-control-t5-v1', 'named-weak-signal-control-t5-v1', 'named-cftc-source-control-t5-v1', 'named-risk-label-control-t1-v1', 'named-risk-history-control-t1-v1', 'named-risk-loss-control-t1-v1') else 'synthetic-work') / key
        staged_path = spool / 'locally-completed.json'
        try:
            if staged_path.exists():
                staged = read_record(staged_path)
                run_id = staged['run_id']
                workspace = spool / run_id
                for name, checksum in staged['files'].items():
                    if not safe_member(name) or digest(workspace / name) != checksum:
                        raise ValueError('Corrupt locally completed fit; refusing silent retraining')
                body, compute_seconds = staged['result'], staged['compute_seconds']
                print(f'CACHE locally_complete {key[:12]} retrying upload without training', flush=True)
            else:
                if before_compute is not None:
                    before_compute()
                # Any new estimator job requires admission, including legacy
                # namespaces. Completed/staged historical jobs stay readable.
                if 'recipe' in specification or 'profile' in self.identity or 'source_id' in self.identity:
                    from cottonlens_ml.research.admission import ExecutionAdmission
                    if not isinstance(admission, ExecutionAdmission):
                        raise ValueError('New estimator fit requires verified data admission')
                    admission(self.identity, specification)
                elif self.identity.get('admission_policy') == 1:
                    # The budget hook may yield control or stage files. Recheck
                    # immediately before execution, including any changed bytes.
                    admission(self.identity, specification)
                workspace = spool / run_id
                workspace.mkdir(parents=True)
                freeze_record(self.root/'execution-attempts/started'/f'{run_id}.json', {
                    'run_id': run_id, 'experiment_id': key, 'specification': spec,
                    'state': 'started_new_compute', 'optimizer_resume_claimed': False,
                    'started_at': datetime.now(UTC).isoformat()})
                print(f'FIT running {key[:12]} role={specification.get("role", "unknown")}', flush=True)
                scope = (admission.training_scope(self.identity, specification)
                         if admission is not None else nullcontext())
                with scope:
                    body = operation(workspace)
                compute_seconds = time.monotonic() - started
                staged = {'run_id': run_id, 'result': body, 'compute_seconds': compute_seconds,
                          'files': {p.relative_to(workspace).as_posix(): digest(p)
                                    for p in workspace.rglob('*') if p.is_file()}}
                freeze_record(staged_path, staged)
            # Also publish this event after a verified staged restore: a hard
            # exit may occur between staging bytes and recording computation.
            freeze_record(self.root/'execution-attempts/computed'/f'{run_id}.json', {
                'run_id': run_id, 'experiment_id': key, 'state': 'locally_computed',
                'staged_sha256': digest(staged_path)})
            destination = self.root / 'payloads' / key / run_id
            print(f'FIT locally_complete {key[:12]} uploading checkpoint', flush=True)
            copying_started = time.monotonic()
            files = {}
            for source in workspace.rglob('*'):
                if source.is_file():
                    target = destination / source.relative_to(workspace)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pending = target.with_name('.' + target.name + '.pending')
                    shutil.copyfile(source, pending)
                    if digest(source) != digest(pending):
                        raise ValueError('Drive checkpoint copy mismatch')
                    if target.exists():
                        if digest(target) != digest(source):
                            raise ValueError('Conflicting remote checkpoint; refusing overwrite')
                        pending.unlink()
                    else:
                        pending.rename(target)
                    files[target.relative_to(self.root).as_posix()] = digest(target)
            record = {'specification': spec, 'experiment_id': key, 'identity': self.identity, 'run_id': run_id,
                      'result': body, 'files': files, 'payload_root': destination.relative_to(self.root).as_posix(),
                      'compute_seconds': compute_seconds, 'checkpoint_copy_seconds': time.monotonic() - copying_started,
                      'completed_at': datetime.now(UTC).isoformat(), 'elapsed_seconds': time.monotonic() - started}
            freeze_record(result_path, record)
            self.receipt(record)
            print(f'FIT durably_saved {key[:12]}', flush=True)
            return record
        except FitBudgetReached:
            raise
        except (Exception, KeyboardInterrupt) as exc:
            freeze_record(self.root / 'attempts' / f'{uuid.uuid4().hex}.json', {'experiment_id': key,
                'run_id': run_id, 'optimizer_resume_claimed': False,
                'locally_completed': staged_path.exists(),
                'specification': spec, 'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': time.monotonic() - started})
            raise

    def results(self):
        return [read_record(p) for p in sorted((self.root / 'completed').glob('*.json'))]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


__all__ = ['Ledger', 'freeze_record', 'read_record', 'writer']
