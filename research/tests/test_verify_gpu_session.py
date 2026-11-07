import json
import zipfile

import numpy as np
import pytest

from research import verify_gpu_session as verifier


def encoded(body):
    return json.dumps({**body, 'record_id': verifier.sha(json.dumps(body,
        sort_keys=True, separators=(',', ':')).encode())}).encode()


def fixture(tmp_path, fault=None):
    generator = np.random.default_rng(1701)
    traces, rng = {}, {}
    for i in (1, 2):
        traces[f'trace{i}'] = {'indexes': generator.permutation(64)[:16].tolist(), 'mask': np.zeros((16, 5)).tolist()}
        rng[f'rng{i}.json'] = {'numpy': json.loads(json.dumps(generator.bit_generator.state)),
                              'tensorflow_alg': 1, 'tensorflow_state': [i, 0, 0]}
    if fault == 'mask':
        traces['trace2']['mask'][0][0] = .5
    payload = {'step1.keras': b'1', 'reference2.keras': b'2', 'resumed2.keras': b'2'}
    payload.update({n: encoded(v) for n, v in rng.items()})
    payload['reference.json'] = encoded({**traces, 'prediction': [[1.]]})
    payload['resumed.json'] = encoded({'trace2': traces['trace2'], 'rng2': rng['rng2.json'],
                                     'prediction': [[1.]], 'variable_devices': ['/device:GPU:0']})
    prior = {'prior-evidence/'+n: b'generated prior receipt' for n in verifier.PRIOR}
    source = tmp_path/'source.zip'
    with zipfile.ZipFile(source, 'w') as z:
        z.writestr('ml/gpu_session_probe.py', b'generated source, never executed')
        for n, v in prior.items():
            z.writestr(n, v)
    src = verifier.sha(b'generated source, never executed')
    prepared = {'source_sha256': src, 'boot_identity': 'a'*64, 'requested_gpu': 'L4',
                'hardware': 'NVIDIA L4, generated', 'tensorflow': '2.20.0', 'numpy': '2.1.3',
                'train_on_batch_calls': 2, 'variable_devices': ['/device:GPU:0'],
                'files': {n: verifier.sha(v) for n, v in payload.items() if n not in ('resumed2.keras', 'resumed.json')}}
    payload['prepared.json'] = encoded(prepared)
    payload['started.json'] = encoded({'source_sha256': src, 'boot_identity': 'a'*64})
    payload['resume-started.json'] = encoded({'source_sha256': src, 'boot_identity': 'b'*64})
    done = {'source_sha256': src, 'before_boot': 'a'*64, 'after_boot': 'b'*64,
            'prepared_sha256': verifier.sha(payload['prepared.json']), 'local_marker_absent_at_resume': True,
            'synthetic_train_on_batch_calls': 3, 'market_fits': 0, 'market_skill': False,
            'requested_gpu': 'L4', 'hardware_before': prepared['hardware'], 'hardware_after': prepared['hardware'],
            'absolute_tolerance': 1e-6, 'files': {n: verifier.sha(payload[n]) for n in verifier.FILES}}
    if fault == 'boot':
        done['after_boot'] = done['before_boot']
    if fault == 'marker':
        done['local_marker_absent_at_resume'] = False
    if fault == 'source':
        done['source_sha256'] = 'wrong'
    if fault == 'fits':
        done['market_fits'] = 1
    payload['complete.json'] = encoded(done)
    payload.update(prior)
    if fault == 'prior':
        payload['prior-evidence/'+verifier.PRIOR[0]] = b'changed'
    archive = tmp_path/'result.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for n, v in payload.items():
            z.writestr(n, v)
    checksum = tmp_path/'checksum.json'
    checksum.write_text(json.dumps({'archive_sha256': verifier.sha(archive.read_bytes()),
        'source_archive_sha256': verifier.sha(source.read_bytes()),
        'members': {n: verifier.sha(v) for n, v in payload.items()}}))
    return archive, checksum, source


def fake_native(raw):
    # Scope is receipt semantics, not an HDF5 or TensorFlow control.
    return {'generated': True}, [np.asarray(0.), np.asarray(0.), np.asarray(int(raw))]


def test_valid_receipt_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, 'native', fake_native)
    monkeypatch.setattr(verifier, 'independent_replay', lambda *a: {'mock_receipt_scope_only': True})
    result = verifier.verify(*fixture(tmp_path))
    assert result['new_fits'] == 0 and not result['remote_runtime_attestation']
    assert not result['universal_dropout_rng_verified']


@pytest.mark.parametrize('fault', ['boot', 'marker', 'source', 'fits', 'prior', 'mask'])
def test_rehashed_semantic_defect_rejected(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(verifier, 'native', fake_native)
    monkeypatch.setattr(verifier, 'independent_replay', lambda *a: {'mock_receipt_scope_only': True})
    with pytest.raises(ValueError):
        verifier.verify(*fixture(tmp_path, fault))
