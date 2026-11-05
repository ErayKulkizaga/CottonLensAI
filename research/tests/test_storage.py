import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('storage', Path(__file__).parents[1] / 'storage.py')
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


class StorageTests(unittest.TestCase):
    def test_exact_roundtrip_and_dedup_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source/checkpoints'
            source.mkdir(parents=True)
            (source / 'a.json').write_bytes(b'{"model": 1}')
            (source / 'b.json').write_bytes(b'{"model": 1}')
            m = storage.pack(root / 'source', 'checkpoints', root / 'snapshot')
            self.assertEqual(len(m['files']), 2)
            self.assertEqual(len(list((root / 'snapshot/objects').iterdir())), 1)
            receipt = storage.verify(root / 'snapshot/evidence.zip', root / 'snapshot/manifest.json')
            self.assertEqual(receipt['status'], 'verified')
            self.assertEqual((source / 'a.json').read_bytes(), b'{"model": 1}')
            self.assertEqual(receipt['deleted_files'], 0)

    def test_credential_and_database_are_retained_not_uploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source/models'
            source.mkdir(parents=True)
            (source / 'key.json').write_bytes(b'{"api_key":"' + b'A' * 32 + b'"}')
            (source / 'tracking.sqlite').write_bytes(b'private database')
            m = storage.pack(root / 'source', 'models', root / 'snapshot')
            self.assertEqual(len(m['excluded']), 2)
            self.assertEqual(len(m['files']), 0)
            self.assertTrue((source / 'tracking.sqlite').exists())

    def test_nested_archive_credential_rejected(self):
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, 'w') as z:
            z.writestr('data.json', b'{"api_key":"' + b'A' * 32 + b'"}')
        with self.assertRaises(ValueError):
            storage.check_content(payload.getvalue(), '.zip')

    def test_scope_escape_and_in_source_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/models').mkdir(parents=True)
            (root / 'other').mkdir()
            with self.assertRaises(ValueError):
                storage.pack(root / 'source', '../other', root / 'snapshot')
            with self.assertRaises(ValueError):
                storage.pack(root / 'source', 'models', root / 'source/backup')

    def test_archive_and_coherently_rehashed_manifest_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/models').mkdir(parents=True)
            (root / 'source/models/a.json').write_bytes(b'{}')
            m = storage.pack(root / 'source', 'models', root / 'snapshot')
            manifest = root / 'snapshot/manifest.json'
            m['files'][0]['bytes'] = 9
            manifest.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                storage.verify(root / 'snapshot/evidence.zip', manifest)

    def test_duplicate_casefold_and_traversal_archive_rejected(self):
        for names in [('A.json', 'a.json'), ('../a.json',)]:
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, 'w') as z:
                for name in names:
                    z.writestr(name, '{}')
            with self.assertRaises(ValueError):
                storage.check_content(payload.getvalue(), '.zip')

    def test_partition_preserves_all_paths_and_verified_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/models').mkdir(parents=True)
            for name, data in [('a.json', b'{}'), ('b.json', b'[]'), ('c.json', b'{}')]:
                (root / 'source/models' / name).write_bytes(data)
            m = storage.pack(root / 'source', 'models', root / 'snapshot')
            result = storage.split(root / 'snapshot/evidence.zip', root / 'snapshot/manifest.json',
                                   root / 'parts', part_bytes=2)
            files = []
            for part in result['parts']:
                body = json.loads((root / 'parts' / part['manifest']).read_bytes())
                files.extend(body['files'])
                storage.verify(root / 'parts' / part['archive']['name'], root / 'parts' / part['manifest'])
            self.assertEqual(files, m['files'])

    def test_partition_oversized_object_keeps_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/models').mkdir(parents=True)
            (root / 'source/models/a.json').write_bytes(b'{"x":123}')
            storage.pack(root / 'source', 'models', root / 'snapshot')
            with self.assertRaises(ValueError):
                storage.split(root / 'snapshot/evidence.zip', root / 'snapshot/manifest.json',
                              root / 'parts', part_bytes=2)
            self.assertTrue((root / 'source/models/a.json').exists())

    def test_restore_reopens_exact_bytes_and_refuses_existing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/models').mkdir(parents=True)
            (root / 'source/models/a.json').write_bytes(b'{"weights":[1,2]}')
            storage.pack(root / 'source', 'models', root / 'snapshot')
            archive, manifest = root / 'snapshot/evidence.zip', root / 'snapshot/manifest.json'
            result = storage.restore(archive, manifest, root / 'restored')
            self.assertEqual(result['restored_files'], 1)
            self.assertEqual((root / 'restored/models/a.json').read_bytes(), b'{"weights":[1,2]}')
            with self.assertRaises(FileExistsError):
                storage.restore(archive, manifest, root / 'restored')

    def test_interrupted_capture_resumes_verified_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/models').mkdir(parents=True)
            for name in ['a.json', 'b.json']:
                (root / 'source/models' / name).write_bytes(b'{}')
            original = storage.check_content
            calls = 0

            def interrupt(data, suffix):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('simulated interruption')
                return original(data, suffix)

            with patch.object(storage, 'check_content', interrupt), self.assertRaises(OSError):
                storage.pack(root / 'source', 'models', root / 'snapshot')
            m = storage.pack(root / 'source', 'models', root / 'snapshot', resume=True)
            self.assertEqual(len(m['files']), 2)
            self.assertTrue((root / 'source/models/a.json').exists())

    def test_resume_rejects_source_change_and_corrupt_captured_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/models').mkdir(parents=True)
            (root / 'source/models/a.json').write_bytes(b'{}')
            storage.pack(root / 'source', 'models', root / 'snapshot')
            with self.assertRaises(ValueError):
                storage.pack(root / 'source', '.', root / 'snapshot', resume=True)
            (root / 'snapshot/manifest.json').unlink()
            (root / 'snapshot/evidence.zip').unlink()
            for obj in (root / 'snapshot/objects').iterdir():
                obj.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                storage.pack(root / 'source', 'models', root / 'snapshot', resume=True)


if __name__ == '__main__':
    unittest.main()
