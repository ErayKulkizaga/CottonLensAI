"""Verify native restart evidence without TensorFlow or executable deserialization."""
import argparse
import io
import json
import tempfile
import zipfile
from pathlib import Path

from research.verify_gpu_probe import members, sha
from research.verify_gpu_probe import verify as verify_baseline

NAMES = ('baseline/complete.json', 'baseline/step1.keras', 'baseline/step2.keras',
         'complete.json', 'interrupted.keras', 'resume.json', 'resumed.keras', 'worker-exit.json')


def record(data):
    value = json.loads(data)
    body = {k: v for k, v in value.items() if k != 'record_id'}
    if sha(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) != value['record_id']:
        raise ValueError('Restart record identity mismatch')
    return value


def semantic_config(value):
    # Keras changes this in-memory alias ID on save/load; it is not a recipe.
    if isinstance(value, dict):
        return {k: semantic_config(v) for k, v in value.items() if k != 'shared_object_id'}
    if isinstance(value, list):
        return [semantic_config(v) for v in value]
    return value


def native(data):
    import h5py
    import numpy as np
    payload = members(data, ('metadata.json', 'config.json', 'model.weights.h5'))
    names = ['layers/dense/vars/0', 'layers/dense/vars/1', *[f'optimizer/vars/{i}' for i in range(6)]]
    shapes = [(5, 1), (1,), (), (), (5, 1), (5, 1), (1,), (1,)]
    arrays = []
    with h5py.File(io.BytesIO(payload['model.weights.h5']), 'r') as f:
        for name, shape in zip(names, shapes, strict=True):
            node = f
            for part in name.split('/'):
                if not isinstance(node.get(part, getlink=True), h5py.HardLink):
                    raise TypeError('External/soft native state link rejected')
                node = node[part]
            if not isinstance(node, h5py.Dataset) or node.shape != shape:
                raise ValueError('Unexpected restart state layout')
            value = np.asarray(node[()])
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite restart state')
            arrays.append(value)
    return semantic_config(json.loads(payload['config.json'])), arrays


def verify(archive, checksum, source):
    import numpy as np
    data = Path(archive).read_bytes()
    manifest = json.loads(Path(checksum).read_text())
    if sha(data) != manifest['archive_sha256'] or sha(Path(source).read_bytes()) != manifest['source_archive_sha256']:
        raise ValueError('Restart archive/source checksum mismatch')
    payload = members(data, NAMES)
    if {n: sha(v) for n, v in payload.items()} != manifest['members']:
        raise ValueError('Restart member checksum mismatch')
    complete, worker, resumed = (record(payload[n]) for n in ('complete.json', 'worker-exit.json', 'resume.json'))
    baseline = record(payload['baseline/complete.json'])
    with zipfile.ZipFile(source) as z:
        expected_source = {'probe': sha(z.read('ml/gpu_restart_probe.py')),
                           'baseline': sha(z.read('ml/a100_preflight.py'))}
    if complete['source'] != expected_source or baseline['source_id'] != expected_source['baseline']:
        raise ValueError('Restart executed source mismatch')
    if complete['files'] != {n: sha(v) for n, v in payload.items() if n != 'complete.json'}:
        raise ValueError('Restart completion file closure mismatch')
    if (complete['worker_exit_code'] != 73 or complete['synthetic_train_on_batch_calls'] != 4
            or complete['market_fits'] != 0 or complete['market_skill']
            or any(complete[k] for k in ('whole_colab_session_restart_verified', 'stochastic_rng_resume_verified',
                                        'real_gpu_oom_verified'))):
        raise ValueError('Unexpected restart scope')
    if (worker['checkpoint_sha256'] != sha(payload['interrupted.keras'])
            or worker['stage'] != 'native_checkpoint_saved_before_process_exit'
            or worker['pid'] <= 0 or not worker['variable_devices']
            or any('GPU:' not in d for d in worker['variable_devices'])
            or not complete['hardware'] == worker['hardware'] == baseline['hardware']
            or complete['requested_gpu'] != baseline['requested_gpu']
            or resumed['optimizer_step'] != 2 or resumed['absolute_tolerance'] != 1e-6
            or not resumed['native_state_and_prediction_match']
            or resumed['tensorflow'] != baseline['tensorflow']):
        raise ValueError('Restart worker or resume record mismatch')
    states = {n: native(payload[n]) for n in
              ('baseline/step1.keras', 'baseline/step2.keras', 'interrupted.keras', 'resumed.keras')}
    errors = []
    for before, after in [('baseline/step1.keras', 'interrupted.keras'), ('baseline/step2.keras', 'resumed.keras')]:
        if states[before][0] != states[after][0]:
            raise ValueError('Native restart recipe changed')
        maximum = 0.
        for a, b in zip(states[before][1], states[after][1], strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(a-b))))
        errors.append(maximum)
    # Repackage only the original baseline bytes, with no invented model/record.
    # The existing independent NumPy calculator checks its actual Adam step.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = {n: payload['baseline/'+n] for n in ('complete.json', 'step1.keras', 'step2.keras')}
        with zipfile.ZipFile(root/'baseline.zip', 'w') as z:
            for name, value in base.items():
                z.writestr(name, value)
        (root/'checksums.json').write_text(json.dumps({
            'archive_sha256': sha((root/'baseline.zip').read_bytes()),
            'source_archive_sha256': manifest['source_archive_sha256'], 'members': {n: sha(v) for n, v in base.items()}}))
        independent = verify_baseline(root/'baseline.zip', root/'checksums.json', source)
    return {'schema': 1, 'status': 'native_restart_state_and_independent_adam_verified',
            'archive_sha256': sha(data), 'source_archive_sha256': manifest['source_archive_sha256'],
            'completion_record_id': complete['record_id'], 'hardware_reported': complete['hardware'],
            'worker_exit_reported': 73, 'before_exit_state_max_abs_error': errors[0],
            'resumed_state_max_abs_error': errors[1], 'baseline_adam_max_abs_error': independent['independent_max_abs_error'],
            'absolute_tolerance': 1e-6, 'new_fits': 0, 'tensorflow_imported': False,
            'whole_session_restart_verified': False, 'stochastic_rng_resume_verified': False,
            'real_gpu_oom_verified': False, 'market_skill': False,
            'verifier_sha256': sha(Path(__file__).read_bytes()),
            'baseline_verifier_sha256': independent['verifier_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'checksum', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing restart verification')
    result = verify(args.archive, args.checksum, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))
