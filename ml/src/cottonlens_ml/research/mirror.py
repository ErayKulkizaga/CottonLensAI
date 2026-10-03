"""Retryable local-first immutable copies. Readback is not a cloud sync receipt."""
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.ledger import freeze_record, read_record


class Mirror:
    def __init__(self, local, target):
        self.local, self.target = Path(local), Path(target)
        self._receipt_stamps = {}
        if self.local.resolve() == self.target.resolve():
            raise ValueError('Mirror must have a separate destination')

    def enqueue(self, members):
        files = {}
        for name in members:
            if not safe_member(name):
                raise ValueError('Unsafe mirror member')
            files[name] = digest(self.local / name)
        key = content_id(files)
        freeze_record(self.local / 'transfer-queue' / (key + '.json'), {'files': files})
        return key

    def fit(self, record):
        key = record['experiment_id']
        names = ['ledger/' + name for name in record['files']]
        # Payloads precede the completed marker; readers never select partial fits.
        names += [f'ledger/receipts/{key}.json', f'ledger/completed/{key}.json']
        self.enqueue(names)
        self.flush()

    def flush(self):
        copied, pending = 0, 0
        for path in sorted((self.local / 'transfer-queue').glob('*.json')):
            receipt = self.local / 'transfer-receipts' / path.name
            if receipt.exists():
                stat = receipt.stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
                if self._receipt_stamps.get(receipt) != stamp:
                    read_record(receipt)
                    self._receipt_stamps[receipt] = stamp
                continue
            body = read_record(path)
            try:
                started = time.monotonic()
                package = self.local / 'transfer-packages' / (path.stem + '.zip')
                package.parent.mkdir(parents=True, exist_ok=True)
                if not package.exists():
                    temporary = package.with_suffix('.pending')
                    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                        for name, checksum in body['files'].items():
                            if digest(self.local/name) != checksum:
                                raise ValueError('Local queued content changed')
                            archive.write(self.local/name, name)
                        archive.writestr('.transfer-manifest.json', json.dumps(body['files'], sort_keys=True))
                    temporary.rename(package)
                checksum = digest(package)
                prepared = time.monotonic()
                target = self.target/'packages'/(checksum+'.zip')
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if digest(target) != checksum:
                        raise ValueError('Mirror package conflict')
                else:
                    temporary = target.with_name('.'+uuid.uuid4().hex+'.pending')
                    shutil.copyfile(package, temporary)
                    if digest(temporary) != checksum:
                        raise ValueError('Mirror package checksum mismatch')
                    temporary.rename(target)
                if digest(target) != checksum:
                    raise ValueError('Mirror package readback mismatch')
                freeze_record(receipt, {'queue_id': path.stem, 'package_sha256': checksum,
                    'copy_readback_verified': True, 'cloud_synchronization_confirmed': False,
                    'package_seconds': prepared-started,
                    'copy_readback_seconds': time.monotonic()-prepared})
                stat = receipt.stat()
                self._receipt_stamps[receipt] = (stat.st_size, stat.st_mtime_ns)
                copied += 1
            except OSError:
                pending += 1
        return {'new_verified_copy_batches': copied, 'pending_batches': pending,
                'cloud_synchronization_confirmed': False}

    def publish_metadata(self, members):
        """Append an immutable catalogue for small preparation/output/report batches."""
        if any(name.startswith('ledger/') for name in members):
            raise ValueError('Metadata catalogue cannot publish training ledger payloads')
        key = self.enqueue(members)
        self.flush()
        receipt = self.local / 'transfer-receipts' / (key + '.json')
        if not receipt.exists():
            raise OSError('Metadata transfer pending; preserve local work and retry')
        body = read_record(receipt)
        freeze_record(self.target / 'metadata-catalog' / (key + '.json'),
            {'package_sha256': body['package_sha256']})

    def hydrate_metadata(self):
        """Fetch catalogued result batches without scanning training packages."""
        packages = set()
        for path in sorted((self.target / 'metadata-catalog').glob('*.json')):
            checksum = read_record(path)['package_sha256']
            if (not isinstance(checksum, str) or len(checksum) != 64
                    or any(c not in '0123456789abcdef' for c in checksum)):
                raise ValueError('Invalid metadata catalogue package identity')
            packages.add(checksum + '.zip')
        return self.hydrate(metadata_only=True, package_names=sorted(packages))

    def hydrate(self, *, metadata_only=False, package_names=None):
        """Read each small package once, verify locally, then publish completion last."""
        restored = 0
        if package_names is not None and any(not safe_member(name) or Path(name).name != name for name in package_names):
            raise ValueError('Unsafe mirror package name')
        sources = (sorted((self.target/'packages').glob('*.zip')) if package_names is None
                   else [self.target / 'packages' / name for name in package_names])
        for source in sources:
            if not safe_member(source.name) or source.suffix != '.zip':
                raise ValueError('Unsafe mirror package name')
            receipt = self.local/('hydrated-metadata' if metadata_only else 'hydrated-packages')/(source.stem+'.json')
            if receipt.exists():
                read_record(receipt)
                continue
            package = self.local/'transfer-downloads'/source.name
            package.parent.mkdir(parents=True, exist_ok=True)
            if not package.exists():
                shutil.copyfile(source, package)
            if digest(package) != source.stem:
                raise ValueError('Corrupt mirror package; preserve completed work')
            with zipfile.ZipFile(package) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError('Duplicate mirror archive member')
                files = json.loads(archive.read('.transfer-manifest.json'))
                if set(names) != set(files)|{'.transfer-manifest.json'}:
                    raise ValueError('Unexpected mirror archive member')
                # Immutable payloads first; completed experiment markers last.
                for name in sorted(files, key=lambda n: '/completed/' in n):
                    if not safe_member(name):
                        raise ValueError('Unsafe mirror archive member')
                    if metadata_only and name.startswith('ledger/payloads/'):
                        continue
                    target = self.local/name
                    content = archive.read(name)
                    import hashlib
                    if hashlib.sha256(content).hexdigest() != files[name]:
                        raise ValueError('Mirror member checksum mismatch')
                    if target.exists():
                        if digest(target) != files[name]:
                            raise ValueError('Conflicting local evidence; no overwrite')
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_name('.'+uuid.uuid4().hex+'.pending')
                        temporary.write_bytes(content)
                        temporary.rename(target)
            freeze_record(receipt, {'package_sha256': source.stem, 'members_verified': len(files)})
            restored += 1
        return {'restored_packages': restored, 'metadata_only': metadata_only}
