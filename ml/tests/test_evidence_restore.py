"""Archives are untrusted input; restoring evidence must not replace user files."""
import gzip
import hashlib
import json
import zipfile

import pytest
from evidence_restore import digest, restore


def archive(root, *, path='output/experiment/predictions.json', corrupt_object=False):
    root.mkdir()
    data = b'{"loss": 1.0}'
    checksum = hashlib.sha256(data).hexdigest()
    asset = root/'evidence-part-01.zip'
    with zipfile.ZipFile(asset, 'w') as z:
        z.writestr('objects/'+checksum, data if not corrupt_object else b'corrupt')
    manifest = {'schema': 1, 'assets': [{'name': asset.name, 'sha256': digest(asset)}],
                'files': [{'path': path, 'sha256': checksum, 'bytes': len(data), 'asset': asset.name,
                           'member': 'objects/'+checksum, 'sanitized': False}]}
    with gzip.open(root/'evidence-manifest.json.gz', 'wt') as f:
        json.dump(manifest, f)
    summary = {'manifest': {'name': 'evidence-manifest.json.gz',
                           'sha256': digest(root/'evidence-manifest.json.gz')}}
    (root/'archive-summary.json').write_text(json.dumps(summary))


def test_restore_verified_bytes_and_refuse_overwrite(tmp_path):
    source, destination = tmp_path/'archive', tmp_path/'recovered'
    archive(source)
    assert restore(source, destination, 'output/experiment/')['files'] == 1
    p = destination/'output/experiment/predictions.json'
    before = p.read_bytes()
    with pytest.raises(ValueError, match='existing destination'):
        restore(source, destination, 'output/experiment/')
    assert p.read_bytes() == before


@pytest.mark.parametrize('damage', ['asset', 'manifest', 'object'])
def test_checksum_failure_does_not_write_object(tmp_path, damage):
    source, destination = tmp_path/'archive', tmp_path/'recovered'
    archive(source, corrupt_object=damage == 'object')
    if damage != 'object':
        name = 'evidence-part-01.zip' if damage == 'asset' else 'evidence-manifest.json.gz'
        with (source/name).open('ab') as f:
            f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        restore(source, destination, 'output/experiment/')
    assert not (destination/'output/experiment/predictions.json').exists()


@pytest.mark.parametrize('path', ['output/experiment/../../../escape', 'output/experiment/C:/escape'])
def test_unsafe_manifest_path_is_rejected(tmp_path, path):
    source, destination = tmp_path/'archive', tmp_path/'recovered'
    archive(source, path=path)
    with pytest.raises(ValueError, match='Unsafe evidence path'):
        restore(source, destination, 'output/experiment/')
    assert not destination.exists()


def test_prefix_cannot_include_sibling_experiment(tmp_path):
    source, destination = tmp_path/'archive', tmp_path/'recovered'
    archive(source, path='output/experiment-other/predictions.json')
    with pytest.raises(ValueError, match='No files match'):
        restore(source, destination, 'output/experiment')
