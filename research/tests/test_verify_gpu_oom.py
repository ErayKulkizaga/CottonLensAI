import io
import json
import zipfile

import pytest

from research.verify_gpu_oom import sha, verify


def fixture(tmp_path, fault=None):
    source = tmp_path/'source.zip'
    names = ('gpu_oom_probe.py', 'gpu_restart_probe.py', 'a100_preflight.py')
    with zipfile.ZipFile(source, 'w') as z:
        for name in names:
            z.writestr('ml/'+name, b'generated source fixture; never executed')
    def encoded(body):
        return json.dumps({**body, 'record_id': sha(json.dumps(body, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode())}).encode()
    common = {'schema': 1, 'hardware': 'NVIDIA L4, generated fixture',
              'device': '/job:localhost/device:GPU:0', 'tensorflow': '2.20.0', 'fits': 0}
    exhausted = {**common, 'stage': 'exhaust', 'exception_type': 'tensorflow.errors.ResourceExhaustedError',
                 'logical_memory_limit_mb': 128, 'requested_shape': [8192, 8192], 'requested_bytes': 268435456}
    if fault == 'limit':
        exhausted['logical_memory_limit_mb'] = 512
    if fault == 'exception':
        exhausted['exception_type'] = 'MemoryError'
    a = encoded(exhausted)
    recovered = {**common, 'stage': 'recover', 'values': [1., 2.], 'prior_exhaust_sha256': sha(a)}
    if fault == 'values':
        recovered['values'] = [0., 0.]
    if fault == 'cpu':
        recovered['device'] = '/device:CPU:0'
    if fault == 'binding':
        recovered['prior_exhaust_sha256'] = 'wrong'
    b = encoded(recovered)
    complete = {'schema': 1, 'source': {n: sha(b'generated source fixture; never executed') for n in names},
        'requested_gpu': 'L4', 'hardware': common['hardware'], 'market_fits': 0, 'synthetic_fits': 0,
        'whole_session_restart_verified': False, 'stochastic_resume_verified': False, 'market_skill': False,
        'files': {'exhaust.json': sha(a), 'recover.json': sha(b)}}
    if fault == 'source':
        complete['source'][names[0]] = 'wrong'
    if fault == 'scope':
        complete['whole_session_restart_verified'] = True
    payload = {'complete.json': encoded(complete), 'exhaust.json': a, 'recover.json': b}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        for n, raw in payload.items():
            z.writestr(n, raw)
    archive = tmp_path/'result.zip'
    archive.write_bytes(buffer.getvalue())
    checksum = tmp_path/'checksum.json'
    checksum.write_text(json.dumps({'archive_sha256': sha(buffer.getvalue()),
        'source_archive_sha256': sha(source.read_bytes()), 'members': {n: sha(v) for n, v in payload.items()}}))
    return archive, checksum, source


def test_valid_control_and_archive_tamper(tmp_path):
    args = fixture(tmp_path)
    result = verify(*args)
    assert result['new_fits'] == 0 and not result['physical_vram_exhaustion_verified']
    args[0].write_bytes(args[0].read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify(*args)


@pytest.mark.parametrize('fault', ['limit', 'exception', 'values', 'cpu', 'binding', 'source', 'scope'])
def test_semantic_fault_rejected_even_with_consistent_rehashed_records(tmp_path, fault):
    with pytest.raises(ValueError):
        verify(*fixture(tmp_path, fault))
