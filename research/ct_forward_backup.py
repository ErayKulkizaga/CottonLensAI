"""Append-only private GitHub deltas with fresh byte verification; never delete sources."""

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from research.storage import file_sha, pack, verify


def fingerprint(value):
    from hashlib import sha256
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    body = json.loads(path.read_bytes()); claimed = body.pop('record_id', None)
    if claimed != fingerprint(body):
        raise ValueError('Backup journal checksum changed')
    return body


def gh(execution, *args, check=True):
    result = subprocess.run([execution['gh_path'], *args], capture_output=True, text=True, check=False,
        stdin=subprocess.DEVNULL, timeout=90,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if check and result.returncode:
        raise RuntimeError('Private GitHub backup command failed: '+result.stderr[:1000])
    return result


def files(study, run_id):
    result = {}
    for p in study.rglob('*'):
        relative = p.relative_to(study); parts = relative.parts
        if (not p.is_file() or parts[0] in ('backups', 'operations') or '__pycache__' in parts
                or '.writer-lock' in parts or p.name.endswith('.pending')
                or (parts[:2] == ('daily-runs', run_id) and p.name.startswith('backup'))):
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(study.resolve()):
            raise ValueError('Backup cannot read outside its exact study root')
        result[relative.as_posix()] = file_sha(p)
    return result


def backup(study, execution, run_id, phase, *, gh_call=None):
    if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id) or phase not in ('capture', 'publish', 'manual'):
        raise ValueError('Explicit bounded run identifier and phase required')
    if (study/'.writer-lock').exists():
        raise RuntimeError('Core writer active; do not snapshot partial scientific state')
    if execution.get('backup_repository') != 'ErayKulkizaga/CottonLensAI' or file_sha(Path(execution['gh_path'])) != execution['gh_sha256']:
        raise ValueError('Unapproved backup repository or GitHub CLI executable')
    call = gh_call or gh; repository = execution['backup_repository']
    privacy = call(execution, 'api', 'repos/'+repository, '--jq', '.private')
    if privacy.stdout.strip() != 'true':
        raise ValueError('Refuse backup to a public or unverified repository')
    base = study/'backups'; head = base/'head.json'; head_hash = file_sha(head) if head.exists() else None
    previous = read(head) if head.exists() else {'source_files': {}, 'backup_id': None}
    current = files(study, run_id)
    if set(previous['source_files'])-set(current):
        raise ValueError('Previously preserved study files disappeared; no silent deletion mirror')
    changed = {n: h for n, h in current.items() if previous['source_files'].get(n) != h}
    attempt = base/'attempts'/run_id; capture = attempt/'capture'; bundle = capture/'bundle'
    bundle.mkdir(parents=True, exist_ok=False)
    stamp = datetime.now(UTC); tag = f'ct-forward-records-{stamp.year}-q{(stamp.month-1)//3+1}'
    for name, checksum in changed.items():
        dest = bundle/'study'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(study/name, dest)
        if file_sha(dest) != checksum:
            raise ValueError('Scientific input changed during backup capture')
    runtime = Path(execution['runtime_path'])
    for name, checksum in execution['files'].items():
        src = runtime/name; dest = bundle/'runtime'/name
        if not src.resolve().is_relative_to(runtime.resolve()) or file_sha(src) != checksum:
            raise ValueError('Runtime changed during backup capture')
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dest)
    runtime_manifest = (runtime/'execution-manifest.json').read_bytes()
    runtime_body = json.loads(runtime_manifest)
    if (runtime_body['execution_source_id'] != execution['execution_source_id']
            or fingerprint({k: v for k, v in runtime_body.items() if k != 'execution_source_id'}) != execution['execution_source_id']):
        raise ValueError('Runtime manifest changed during capture')
    (bundle/'runtime/execution-manifest.json').write_bytes(runtime_manifest)
    delivery = {'parent_backup_id': previous['backup_id'], 'run_id': run_id, 'phase': phase,
                'execution_source_id': execution['execution_source_id'], 'core_source_id': execution['core_source_id'],
                'source_files': current, 'delta_files': changed, 'created_at': stamp.isoformat(),
                'deletions_performed': 0, 'model_publication': False}
    (bundle/'delivery.json').write_text(json.dumps(delivery, indent=2)+'\n', encoding='utf-8')
    manifest = pack(capture, 'bundle', attempt/'packed')
    if manifest['excluded']:
        raise ValueError('Backup contains withheld/unreviewed material; do not claim a complete delivery')
    prefix = 'CT-forward-'+stamp.strftime('%Y%m%dT%H%M%SZ')+'-'+phase+'-'+run_id[:8]
    assets = attempt/'assets'; assets.mkdir()
    archive = assets/(prefix+'.zip'); sidecar = assets/(prefix+'-manifest.json')
    shutil.copyfile(attempt/'packed/evidence.zip', archive)
    manifest['archive']['name'] = archive.name
    sidecar.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    found = call(execution, 'release', 'view', tag, '--repo', repository, '--json', 'tagName', check=False)
    if found.returncode:
        if 'release not found' not in found.stderr.lower() and '404' not in found.stderr:
            raise RuntimeError('Cannot verify private backup Release: '+found.stderr[:1000])
        call(execution, 'release', 'create', tag, '--repo', repository,
             '--target', execution['repository_commit'], '--title', 'Cotton forward research records '+tag,
             '--notes', 'Private append-only source/model/forecast evidence. No source deletion or public model deployment.')
    call(execution, 'release', 'upload', tag, str(archive), str(sidecar), '--repo', repository)
    fresh = attempt/'fresh-download'
    call(execution, 'release', 'download', tag, '--repo', repository, '--dir', str(fresh),
         '--pattern', archive.name, '--pattern', sidecar.name)
    if file_sha(fresh/archive.name) != file_sha(archive) or file_sha(fresh/sidecar.name) != file_sha(sidecar):
        raise ValueError('Fresh GitHub object differs from the exact uploaded bytes')
    verified = verify(fresh/archive.name, fresh/sidecar.name)
    if current != files(study, run_id) or (file_sha(head) if head.exists() else None) != head_hash:
        raise ValueError('Study or backup parent changed during delivery; preserve attempt, do not advance journal')
    result = {'status': 'private_delta_fresh_download_verified', 'run_id': run_id, 'phase': phase,
              'parent_backup_id': previous['backup_id'], 'release': tag, 'assets': [archive.name, sidecar.name],
              'archive_sha256': file_sha(archive), 'manifest_sha256': file_sha(sidecar),
              'verified_files': verified['files'], 'delta_scientific_files': len(changed),
              'source_files': current, 'source_files_deleted': 0, 'core_source_id': execution['core_source_id'],
              'execution_source_id': execution['execution_source_id']}
    result['backup_id'] = fingerprint(result)
    body = {**result, 'record_id': fingerprint(result)}
    (attempt/'receipt.json').write_text(json.dumps(body, indent=2)+'\n', encoding='utf-8')
    pending = base/('.head-'+run_id+'.pending')
    pending.write_text(json.dumps(body, indent=2)+'\n', encoding='utf-8')
    pending.replace(head)  # Only the journal head changes; every prior receipt/object remains immutable.
    return result
