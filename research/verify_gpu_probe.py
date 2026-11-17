"""Verify returned GPU probe hashes and independently replay Adam with NumPy.

Never imports TensorFlow, executes model config or fits an estimator. Hardware
is reported by the hash-bound probe; this is not remote GPU attestation.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def members(data, expected):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        infos = z.infolist()
        if (len(infos) != len(expected) or {i.filename for i in infos} != set(expected)
                or any(i.file_size > 10_000_000 for i in infos)):
            raise ValueError('Unexpected, duplicate or oversized archive member')
        return {n: z.read(n) for n in expected}


def verify(archive, checksum, source):
    data = Path(archive).read_bytes()
    manifest = json.loads(Path(checksum).read_text())
    if sha(data) != manifest['archive_sha256'] or sha(Path(source).read_bytes()) != manifest['source_archive_sha256']:
        raise ValueError('Result/source archive checksum mismatch')
    payload = members(data, ('complete.json', 'step1.keras', 'step2.keras'))
    if {n: sha(v) for n, v in payload.items()} != manifest['members']:
        raise ValueError('Result member checksum mismatch')
    record = json.loads(payload['complete.json'])
    body = {k: v for k, v in record.items() if k != 'record_id'}
    if sha(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) != record['record_id']:
        raise ValueError('Completed record identity mismatch')
    with zipfile.ZipFile(source) as z:
        if sha(z.read('ml/a100_preflight.py')) != record['source_id']:
            raise ValueError('Executed probe source mismatch')
    if (record['market_fits'] != 0 or record['synthetic_train_on_batch_calls'] != 3
            or record['seed'] != 1701 or record['market_skill'] or record['G1_G2_admission_granted']
            or not record['optimizer_resume_checked'] or not record['native_reload_checked']):
        raise ValueError('Unexpected probe scope/result')
    if not record['variable_devices'] or any('GPU:' not in d for d in record['variable_devices']):
        raise ValueError('GPU variable evidence missing')
    import h5py
    import numpy as np
    names = ['layers/dense/vars/0', 'layers/dense/vars/1', *[f'optimizer/vars/{i}' for i in range(6)]]
    shapes = [(5, 1), (1,), (), (), (5, 1), (5, 1), (1,), (1,)]
    states = []
    configs = []
    for filename in ('step1.keras', 'step2.keras'):
        if sha(payload[filename]) != record['files'][filename]:
            raise ValueError('Native model checksum mismatch')
        native = members(payload[filename], ('metadata.json', 'config.json', 'model.weights.h5'))
        config = json.loads(native['config.json'])
        layers = config['config']['layers']
        if ([layer['class_name'] for layer in layers] != ['InputLayer', 'Dense']
                or layers[-1]['config']['units'] != 1 or layers[-1]['config']['activation'] != 'linear'
                or config['compile_config']['optimizer']['class_name'] != 'Adam'):
            raise ValueError('Unexpected architecture/optimizer')
        configs.append(config['compile_config']['optimizer']['config'])
        with h5py.File(io.BytesIO(native['model.weights.h5']), 'r') as f:
            arrays = []
            for name, shape in zip(names, shapes, strict=True):
                if not isinstance(f.get(name, getlink=True), h5py.HardLink) or f[name].shape != shape:
                    raise ValueError('Unexpected native state layout')
                value = np.asarray(f[name][()])
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite native state')
                arrays.append(value)
            states.append(arrays)
    if configs[0] != configs[1]:
        raise ValueError('Optimizer recipe changed across resume')
    w, b, step, lr, mw, vw, mb, vb = states[0]
    if int(step) != 1 or int(states[1][2]) != 2:
        raise ValueError('Wrong optimizer resume step')
    cfg = configs[0]
    if (cfg['beta_1'] != .9 or cfg['beta_2'] != .999 or cfg['epsilon'] != 1e-7
            or cfg['amsgrad'] or cfg['weight_decay'] is not None):
        raise ValueError('Unsupported Adam recipe')
    rng = np.random.default_rng(1701)
    x = rng.normal(size=(256, 5)).astype('float32')
    y = (2*x[:, :1]-x[:, 1:2]).astype('float32')
    residual = x@w+b-y
    gradients = [np.float32(2/256)*(x.T@residual), np.float32(2)*residual.mean(axis=0)]
    alpha = float(lr)*np.sqrt(1-.999**2)/(1-.9**2)
    expected = [None, None, np.asarray(2), lr, None, None, None, None]
    for parameter, m, v, g, indexes in [(w, mw, vw, gradients[0], (0, 4, 5)), (b, mb, vb, gradients[1], (1, 6, 7))]:
        m2 = .9*m+.1*g
        v2 = .999*v+.001*np.square(g)
        p2 = parameter-alpha*m2/(np.sqrt(v2)+1e-7)
        for index, value in zip(indexes, (p2, m2, v2), strict=True):
            expected[index] = value
    maximum = 0.
    for actual, wanted in zip(states[1], expected, strict=True):
        maximum = max(maximum, float(np.max(np.abs(actual-wanted))))
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-6)
    return {'schema': 1, 'status': 'hashes_and_independent_adam_replay_verified',
            'archive_sha256': sha(data), 'source_archive_sha256': manifest['source_archive_sha256'],
            'probe_record_id': record['record_id'], 'hardware_reported': record['hardware'],
            'requested_gpu': record['requested_gpu'], 'tensorflow_reported': record['tensorflow'],
            'probe_seconds_reported': record['elapsed_seconds'], 'independent_max_abs_error': maximum,
            'absolute_tolerance': 1e-6, 'new_fits': 0, 'tensorflow_imported': False,
            'market_skill': False, 'G1_G2_admission_granted': False,
            'verifier_sha256': sha(Path(__file__).read_bytes())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'checksum', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.archive, args.checksum, args.source)
    if args.output.exists():
        raise ValueError('Preserve verification output; use a new namespace')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))
