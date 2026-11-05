"""Dependency-free, immutable evidence snapshots; never delete source files.

Drive placeholders may report a false size. Identity and size come from bytes read.
Use pack, then download from GitHub into a fresh directory and run verify.
"""
import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(name):
    p = PurePosixPath(name)
    return (bool(name) and not p.is_absolute() and '\\' not in name
            and ':' not in name and all(v not in ('', '.', '..') for v in name.split('/')))


def check_content(data, suffix, depth=0):
    """Fail closed on credentials/unreviewed databases; do not rewrite evidence."""
    if suffix.lower() in {'.db', '.sqlite', '.sqlite3', '.clixml', '.pem', '.key'}:
        raise ValueError('private_or_unreviewed_database')
    patterns = (
        rb'gh[pousr]_[A-Za-z0-9]{30,}', rb'github_pat_[A-Za-z0-9_]{30,}',
        rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        rb'(?i)[?&](?:api_key|apikey|access_token|token|key)=[A-Za-z0-9_+/=.-]{16,}',
        rb'''(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)["']?\s*[:=]\s*["'][A-Za-z0-9_+/=.-]{20,}["']''',
    )
    if any(re.search(pattern, data) for pattern in patterns):
        raise ValueError('possible_credential_keep_local_for_review')
    if suffix.lower() in {'.zip', '.keras', '.npz'}:
        if depth >= 5:
            raise ValueError('archive_depth_limit')
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [i.filename for i in z.infolist() if not i.is_dir()]
            if len(names) != len({n.casefold() for n in names}):
                raise ValueError('duplicate_archive_members')
            for member in names:
                if not safe(member) or any(p.startswith('.env') for p in member.split('/')):
                    raise ValueError('unsafe_or_private_archive_member')
                check_content(z.read(member), Path(member).suffix, depth + 1)


def pack(root, selected, target, resume=False):
    root = root.resolve(strict=True)
    scope = (root / selected).resolve(strict=True)
    if not scope.is_relative_to(root) or scope == root or not scope.is_dir():
        raise ValueError('Select one directory strictly inside source root')
    target = target.resolve()
    if target.is_relative_to(root):
        raise ValueError('Snapshot must be outside source root')
    binding = {'root': str(root), 'scope': selected}
    if resume:
        saved = json.loads((target / 'capture-root.json').read_bytes())
        if saved != binding:
            raise ValueError('Resume source identity mismatch')
        if (target / 'manifest.json').exists():
            verify(target / 'evidence.zip', target / 'manifest.json')
            return json.loads((target / 'manifest.json').read_bytes())
        if (target / 'evidence.zip').exists():
            raise ValueError('Interrupted archive assembly: retain archive and use a new namespace')
    else:
        target.mkdir(parents=True, exist_ok=False)
        (target / 'capture-root.json').write_text(json.dumps(binding), encoding='utf-8')
    files, excluded = [], []
    objects = target / 'objects'
    objects.mkdir(exist_ok=resume)
    captured = set()
    journal = target / 'capture-index.jsonl'
    if resume and journal.exists():
        for line in journal.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if not safe(row['path']) or row['path'] in captured:
                raise ValueError('Invalid capture journal')
            obj = target / ('quarantine' if 'reason' in row else 'objects') / row['sha256']
            if file_sha(obj) != row['sha256'] or obj.stat().st_size != row['bytes']:
                raise ValueError('Captured object changed; do not resume')
            (excluded if 'reason' in row else files).append(row)
            captured.add(row['path'])
    # Never trust reported placeholder sizes. Read once, then use actual bytes.
    # Stream discovery: a cloud tree need not be enumerated completely before
    # the first useful byte copy. Journaled copies survive interrupted sessions.
    for path in scope.rglob('*'):
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name in captured:
            continue
        if not safe(name) or path.is_symlink() or not path.resolve().is_relative_to(scope):
            raise ValueError('Unsafe source path')
        data = path.read_bytes()
        checksum = sha(data)
        row = {'path': name, 'sha256': checksum, 'bytes': len(data)}
        reason = None
        if any(p.startswith('.env') for p in path.parts):
            reason = 'private_environment_file'
        else:
            try:
                check_content(data, path.suffix)
            except (ValueError, zipfile.BadZipFile) as error:
                reason = str(error)
        if reason:
            quarantine = target / 'quarantine'
            quarantine.mkdir(exist_ok=True)
            (quarantine / checksum).write_bytes(data)
            excluded.append({**row, 'reason': reason, 'source_retained': True})
            row = excluded[-1]
        else:
            obj = objects / checksum
            if not obj.exists():
                obj.write_bytes(data)
            files.append(row)
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        captured.add(name)
        (target / 'progress.json').write_text(json.dumps({'verified_files': len(files),
            'excluded_files': len(excluded), 'last_path': name}), encoding='utf-8')
    manifest = {'schema': 1, 'scope': selected, 'files': files, 'excluded': excluded,
                'deleted_files': 0, 'source_bytes_unchanged': True,
                'reported_placeholder_sizes_used': False}
    archive = target / 'evidence.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(objects.iterdir()):
            z.write(path, 'objects/' + path.name)
        z.writestr('inventory.json', json.dumps(manifest, sort_keys=True))
    manifest['archive'] = {'name': archive.name, 'sha256': file_sha(archive),
                           'bytes': archive.stat().st_size}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def verify(archive, manifest):
    m = json.loads(manifest.read_bytes())
    if file_sha(archive) != m['archive']['sha256'] or archive.stat().st_size != m['archive']['bytes']:
        raise ValueError('Archive identity mismatch')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        expected = {'inventory.json'} | {'objects/' + f['sha256'] for f in m['files']}
        if len(names) != len({n.casefold() for n in names}) or set(names) != expected:
            raise ValueError('Archive membership mismatch')
        inner = json.loads(z.read('inventory.json'))
        if inner != {k: v for k, v in m.items() if k != 'archive'}:
            raise ValueError('Embedded inventory mismatch')
        paths = [r['path'] for r in m['files']]
        if len(paths) != len({p.casefold() for p in paths}):
            raise ValueError('Duplicate logical paths')
        for row in m['files']:
            if not safe(row['path']):
                raise ValueError('Unsafe logical path')
            data = z.read('objects/' + row['sha256'])
            if sha(data) != row['sha256'] or len(data) != row['bytes']:
                raise ValueError('Payload identity mismatch')
            check_content(data, Path(row['path']).suffix)
    return {'status': 'verified', 'files': len(m['files']),
            'logical_bytes': sum(f['bytes'] for f in m['files']),
            'excluded': len(m['excluded']), 'archive_sha256': m['archive']['sha256'],
            'model_deserialization_performed': False, 'deleted_files': 0}


def restore(archive, manifest, target):
    result = verify(archive, manifest)
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=False)
    m = json.loads(manifest.read_bytes())
    with zipfile.ZipFile(archive) as z:
        for row in m['files']:
            path = target / row['path']
            if not path.resolve().is_relative_to(target):
                raise ValueError('Restoration escapes destination')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(z.read('objects/' + row['sha256']))
            if file_sha(path) != row['sha256']:
                raise ValueError('Restored file readback mismatch')
    return {**result, 'restored_files': len(m['files']), 'source_files_deleted': 0}


def split(archive, manifest, target, part_bytes=128 * 1024 * 1024):
    """Bound remote assets while retaining the original logical path/hash mapping."""
    verify(archive, manifest)
    target.mkdir(parents=True, exist_ok=False)
    original = json.loads(manifest.read_bytes())
    batches, batch, seen, size = [], [], set(), 0
    for row in original['files']:
        extra = 0 if row['sha256'] in seen else row['bytes']
        if extra > part_bytes:
            raise ValueError('Single object exceeds partition budget; retain original')
        if batch and size + extra > part_bytes:
            batches.append(batch)
            batch, seen, size = [], set(), 0
            extra = row['bytes']
        batch.append(row)
        seen.add(row['sha256'])
        size += extra
    if batch:
        batches.append(batch)
    parts = []
    with zipfile.ZipFile(archive) as source:
        for number, rows in enumerate(batches, 1):
            stem = f'part-{number:03d}'
            body = {**{k: v for k, v in original.items() if k != 'archive'}, 'files': rows}
            path = target / (stem + '.zip')
            with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
                for checksum in sorted({r['sha256'] for r in rows}):
                    z.writestr('objects/' + checksum, source.read('objects/' + checksum))
                z.writestr('inventory.json', json.dumps(body, sort_keys=True))
            body['archive'] = {'name': path.name, 'sha256': file_sha(path), 'bytes': path.stat().st_size}
            part_manifest = target / (stem + '-manifest.json')
            part_manifest.write_text(json.dumps(body, indent=2) + '\n', encoding='utf-8')
            verify(path, part_manifest)
            parts.append({'archive': body['archive'], 'manifest': part_manifest.name,
                          'manifest_sha256': file_sha(part_manifest), 'files': len(rows)})
    result = {'schema': 1, 'parent_manifest_sha256': file_sha(manifest), 'parts': parts,
              'logical_files': len(original['files']), 'excluded_files': len(original['excluded'])}
    (target / 'parts.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--scope', required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    v = commands.add_parser('verify')
    v.add_argument('--archive', type=Path, required=True)
    v.add_argument('--manifest', type=Path, required=True)
    s = commands.add_parser('split')
    s.add_argument('--archive', type=Path, required=True)
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--target', type=Path, required=True)
    r = commands.add_parser('restore')
    r.add_argument('--archive', type=Path, required=True)
    r.add_argument('--manifest', type=Path, required=True)
    r.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        result = pack(args.root, args.scope, args.target, resume=args.resume)
    elif args.command == 'split':
        result = split(args.archive, args.manifest, args.target)
    elif args.command == 'restore':
        result = restore(args.archive, args.manifest, args.target)
    else:
        result = verify(args.archive, args.manifest)
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
