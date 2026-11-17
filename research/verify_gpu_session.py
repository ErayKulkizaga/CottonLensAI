"""Offline native/session/RNG verification; no TensorFlow or fitted computation."""
import argparse
import json
import zipfile
from pathlib import Path

from research.session_math import independent_replay
from research.verify_gpu_probe import members, sha
from research.verify_gpu_restart import native, record

FILES = ('prepared.json', 'step1.keras', 'reference2.keras', 'resumed2.keras',
         'rng1.json', 'rng2.json', 'reference.json', 'resumed.json')
PRIOR = ('l4-technical-verification-20261106.json', 'l4-restart-verification-20261106.json',
         'l4-oom-verification-20261107.json')


def verify(archive, checksum, source):
    import numpy as np
    raw = Path(archive).read_bytes()
    manifest = json.loads(Path(checksum).read_text(encoding='utf-8'))
    if sha(raw) != manifest['archive_sha256'] or sha(Path(source).read_bytes()) != manifest['source_archive_sha256']:
        raise ValueError('Session archive/source checksum mismatch')
    names = (*FILES, 'complete.json', 'started.json', 'resume-started.json', *('prior-evidence/'+n for n in PRIOR))
    payload = members(raw, names)
    hashes = {n: sha(v) for n, v in payload.items()}
    if hashes != manifest['members']:
        raise ValueError('Session member checksum mismatch')
    values = {n: {k: v for k, v in record(payload[n]).items() if k != 'record_id'} for n in (*FILES[:1], 'rng1.json', 'rng2.json', 'reference.json',
                                              'resumed.json', 'complete.json', 'started.json', 'resume-started.json')}
    done, before = values['complete.json'], values['prepared.json']
    with zipfile.ZipFile(source) as z:
        expected = sha(z.read('ml/gpu_session_probe.py'))
        for name in PRIOR:
            if payload['prior-evidence/'+name] != z.read('prior-evidence/'+name):
                raise ValueError('Prior verified evidence changed')
    if (done['source_sha256'] != expected or before['source_sha256'] != expected
            or any(values[n]['source_sha256'] != expected for n in ('started.json', 'resume-started.json'))
            or done['files'] != {n: hashes[n] for n in FILES}
            or before['files'] != {n: hashes[n] for n in FILES if n not in ('prepared.json', 'resumed2.keras', 'resumed.json')}
            or done['prepared_sha256'] != hashes['prepared.json']):
        raise ValueError('Frozen session source or closure mismatch')
    if (done['before_boot'] != before['boot_identity'] or done['before_boot'] == done['after_boot']
            or done['before_boot'] != values['started.json']['boot_identity']
            or done['after_boot'] != values['resume-started.json']['boot_identity']
            or any(len(done[k]) != 64 for k in ('before_boot', 'after_boot'))
            or done['local_marker_absent_at_resume'] is not True):
        raise ValueError('Distinct runtime witnesses required')
    if (done['market_fits'] != 0 or done['market_skill'] is not False
            or done['synthetic_train_on_batch_calls'] != 3 or before['train_on_batch_calls'] != 2
            or before['tensorflow'] != '2.20.0' or before['numpy'] != '2.1.3'
            or done['requested_gpu'] != 'L4' or before['requested_gpu'] != 'L4'
            or any('NVIDIA L4,' not in done[k] for k in ('hardware_before', 'hardware_after'))
            or done['hardware_before'] != before['hardware'] or done['absolute_tolerance'] != 1e-6):
        raise ValueError('Unexpected session recipe/environment')
    reference, resumed = values['reference.json'], values['resumed.json']
    if reference['trace2'] != resumed['trace2'] or values['rng2.json'] != resumed['rng2']:
        raise ValueError('Random continuation differs')
    generator = np.random.default_rng(1701)
    for index in (1, 2):
        np.testing.assert_array_equal(generator.permutation(64)[:16], reference[f'trace{index}']['indexes'])
        if generator.bit_generator.state != values[f'rng{index}.json']['numpy']:
            raise ValueError('NumPy RNG continuation differs')
        mask = np.asarray(reference[f'trace{index}']['mask'], dtype='float32')
        if mask.shape != (16, 5) or not np.isin(mask, np.asarray([0., 1/.75], dtype='float32')).all():
            raise ValueError('Unsupported stochastic mask')
    for name in ('rng1.json', 'rng2.json'):
        state = values[name]
        if state['tensorflow_alg'] != 1 or len(state['tensorflow_state']) != 3:
            raise ValueError('Explicit Philox state required')
    states = {name: native(payload[name]) for name in ('step1.keras', 'reference2.keras', 'resumed2.keras')}
    if not states['step1.keras'][0] == states['reference2.keras'][0] == states['resumed2.keras'][0]:
        raise ValueError('Native model recipe changed')
    if [int(states[n][1][2]) for n in states] != [1, 2, 2]:
        raise ValueError('Wrong optimizer step')
    errors = []
    for a, b in zip(states['reference2.keras'][1], states['resumed2.keras'][1], strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        errors.append(float(np.max(np.abs(a-b))))
    np.testing.assert_allclose(reference['prediction'], resumed['prediction'], rtol=0, atol=1e-6)
    for devices in (before['variable_devices'], resumed['variable_devices']):
        if not devices or any('GPU:' not in d.upper() for d in devices):
            raise ValueError('GPU placement witness required')
    independent = independent_replay(states, reference)
    return {'schema': 1, 'status': 'native_session_and_explicit_rng_receipts_verified',
            'archive_sha256': sha(raw), 'source_archive_sha256': manifest['source_archive_sha256'],
            'native_max_abs_difference': max(errors), 'new_fits': 0, 'tensorflow_imported': False,
            'reported_train_on_batch_calls': 3, 'prior_checks_retrained': False,
            'independent_numpy_replay': independent,
            'independent_replay_sha256': sha(Path(__file__).with_name('session_math.py').read_bytes()),
            'remote_runtime_attestation': False, 'universal_dropout_rng_verified': False,
            'market_skill': False, 'verifier_sha256': sha(Path(__file__).read_bytes())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'checksum', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior session verification')
    result = verify(args.archive, args.checksum, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))
