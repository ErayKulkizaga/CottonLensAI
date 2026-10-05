"""Restore selected public evidence with SHA-256 checks, without overwriting or executing it."""
import argparse
import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

HEX = re.compile(r'^[0-9a-f]{64}$')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    return (bool(name) and not path.is_absolute() and '..' not in path.parts
            and '\\' not in name and ':' not in name and str(path) == name)


def restore(archive_dir, destination, prefix):
    archive_dir, destination = Path(archive_dir), Path(destination).resolve()
    if not prefix or not safe_path(prefix.rstrip('/')):
        raise ValueError('An explicit safe relative prefix is required')
    summary = json.loads((archive_dir / 'archive-summary.json').read_text(encoding='utf-8'))
    manifest_path = archive_dir / summary['manifest']['name']
    if not safe_path(summary['manifest']['name']) or manifest_path.parent != archive_dir:
        raise ValueError('Unsafe manifest name')
    if digest(manifest_path) != summary['manifest']['sha256']:
        raise ValueError('Manifest checksum mismatch')
    with gzip.open(manifest_path, 'rt', encoding='utf-8') as stream:
        manifest = json.load(stream)
    if manifest.get('schema') != 1:
        raise ValueError('Unsupported archive schema')
    selected = [f for f in manifest['files'] if f['path'] == prefix.rstrip('/')
                or f['path'].startswith(prefix.rstrip('/') + '/')]
    if not selected:
        raise ValueError('No files match the requested prefix')
    assets = {a['name']: a for a in manifest['assets']}
    targets, used = set(), set()
    # Validate all destinations/assets before creating anything.
    for item in selected:
        if not safe_path(item['path']) or not HEX.fullmatch(item['sha256']):
            raise ValueError('Unsafe evidence path or digest')
        target = (destination / item['path']).resolve()
        if not target.is_relative_to(destination) or target.exists() or target in targets:
            raise ValueError('Unsafe, duplicate or existing destination')
        targets.add(target)
        name = item['asset']
        if not safe_path(name) or '/' in name or name not in assets:
            raise ValueError('Unsafe archive asset')
        if item['member'] != 'objects/' + item['sha256']:
            raise ValueError('Unexpected object path')
        if name not in used:
            if digest(archive_dir / name) != assets[name]['sha256']:
                raise ValueError('Archive asset checksum mismatch')
            used.add(name)
    archives = {}
    sanitized = 0
    try:
        for item in selected:
            name = item['asset']
            if name not in archives:
                archives[name] = zipfile.ZipFile(archive_dir / name)
            payload = archives[name].read(item['member'])
            if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
                raise ValueError('Evidence object checksum mismatch')
            target = destination / item['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(payload)
            sanitized += bool(item['sanitized'])
    finally:
        for archive in archives.values():
            archive.close()
    return {'status': 'restored', 'files': len(selected), 'sanitized_copies': sanitized,
            'warning': 'Public copies do not certify historical availability or authorize cache reuse. '
                       'Sanitized copies have different original hashes. No models were executed.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(restore(args.archive_dir, args.destination, args.prefix), indent=2))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f'Evidence restore stopped: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
