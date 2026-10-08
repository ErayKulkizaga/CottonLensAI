"""Immutable result ledger, atomic completion and single-writer ownership."""
import json
import shutil
import time
import uuid
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
        return {'durably_saved': len(completed), 'timing_receipts': len(index['keys']),
                'unindexed_legacy_fits': len(completed) - len(index['keys']),
                'payloads_verified_by_status': False, 'timing': index['timing'],
                'failed_or_interrupted_attempts': len(list((self.root / 'attempts').glob('*.json')))}

    def run(self, specification, operation, *, repeat=None, before_compute=None):
        if repeat is not None and repeat not in REPEAT_REASONS:
            raise ValueError('Explicit valid repetition reason required')
        spec = {**specification, 'repeat_reason': repeat}
        key = self.key(spec)
        result_path = self.root / 'completed' / f'{key}.json'
        if result_path.exists():
            record = read_record(result_path)
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
            spool = self.root / ('local-work' if self.identity.get('profile') in ('full-year-v1', 'ams-exploration-v1', 'fas-exploration-v1', 'nass-exploration-v1', 'wasde-exploration-v1', 'cftc-exploration-v1', 'fx-exploration-v1', 'crop-exploration-v1', 'weather-exploration-v1', 'recency-pilot-v1', 'availability-clock-pilot-v1', 'return-path-pilot-v1', 'agri-transfer-pilot-v1', 'wasde-text-pilot-v1', 'fundamental-joint-pilot-v1', 'wasde-regional-t1-pilot-v1') else 'synthetic-work') / key
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
                workspace = spool / run_id
                workspace.mkdir(parents=True)
                print(f'FIT running {key[:12]} role={specification.get("role", "unknown")}', flush=True)
                body = operation(workspace)
                compute_seconds = time.monotonic() - started
                staged = {'run_id': run_id, 'result': body, 'compute_seconds': compute_seconds,
                          'files': {p.relative_to(workspace).as_posix(): digest(p)
                                    for p in workspace.rglob('*') if p.is_file()}}
                freeze_record(staged_path, staged)
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
            record = {'specification': spec, 'experiment_id': key, 'identity': self.identity,
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
                'specification': spec, 'state': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': time.monotonic() - started})
            raise

    def results(self):
        return [read_record(p) for p in sorted((self.root / 'completed').glob('*.json'))]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


__all__ = ['Ledger', 'freeze_record', 'read_record', 'writer']
