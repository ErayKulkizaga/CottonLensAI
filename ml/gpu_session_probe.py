"""Two-runtime L4 synthetic probe with explicit NumPy/TF generator state."""
import argparse
import hashlib
import json
from pathlib import Path

from a100_preflight import sha
from cottonlens_ml.sprint import freeze_record, read_record, writer
from gpu_restart_probe import device

MARKER = Path('/content/cottonlens-session-marker-20261107.json')


def boot_identity():
    value = Path('/proc/sys/kernel/random/boot_id').read_bytes().strip()
    if not value:
        raise RuntimeError('Runtime boot identity unavailable')
    return hashlib.sha256(value).hexdigest()


def verify_files(root, record):
    for name, wanted in record['files'].items():
        target = (root/name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target) != wanted:
            raise ValueError('Corrupt session checkpoint; no retraining')


def phase(root, boot, marker_exists, source):
    if (root/'complete.json').exists():
        receipt = read_record(root/'complete.json')
        if receipt['source_sha256'] != source:
            raise ValueError('Session source changed')
        verify_files(root, receipt)
        return 'cached'
    if (root/'prepared.json').exists():
        receipt = read_record(root/'prepared.json')
        if receipt['source_sha256'] != source:
            raise ValueError('Session source changed')
        verify_files(root, receipt)
        if boot == receipt['boot_identity'] or marker_exists:
            raise ValueError('Delete the Colab runtime, reconnect L4 and run this notebook again')
        return 'resume'
    if (root/'started.json').exists() or marker_exists:
        raise ValueError('Incomplete attempt preserved; review before retry')
    return 'prepare'


def run(output, gpu='L4'):
    hardware = device(gpu)  # Reject local/CI before TF import or filesystem writes.
    root = Path(output)
    source, boot = sha(Path(__file__)), boot_identity()
    with writer(root):
        stage = phase(root, boot, MARKER.exists(), source)
        if stage == 'cached':
            return {'status': 'cached', 'new_train_on_batch_calls': 0}
        if stage == 'prepare':
            freeze_record(root/'started.json', {'source_sha256': source, 'boot_identity': boot})
            with MARKER.open('x', encoding='utf-8') as handle:
                json.dump({'source_sha256': source, 'boot_identity': boot}, handle)
        elif (root/'resume-started.json').exists():
            raise ValueError('Interrupted resume preserved; no automatic retraining')
        else:
            freeze_record(root/'resume-started.json', {'source_sha256': source, 'boot_identity': boot})
        import numpy as np
        import tensorflow as tf
        if tf.__version__ != '2.20.0' or np.__version__ != '2.1.3':
            raise RuntimeError('Pinned TF2.20.0/NumPy2.1.3 required')
        tf.keras.utils.set_random_seed(1701)
        tf.config.experimental.enable_op_determinism()
        tf.config.set_soft_device_placement(False)
        if len(tf.config.list_physical_devices('GPU')) != 1:
            raise RuntimeError('Exactly one GPU required')
        x = (np.arange(320, dtype='float32').reshape(64, 5)-160)/100
        y = 2*x[:, :1]-x[:, 1:2]

        def rng_state(numpy_rng, tf_rng):
            return {'numpy': numpy_rng.bit_generator.state,
                    'tensorflow_state': tf_rng.state.numpy().tolist(), 'tensorflow_alg': int(tf_rng.algorithm)}

        def step(model, numpy_rng, tf_rng):
            indexes = numpy_rng.permutation(64)[:16]
            mask = tf.cast(tf_rng.uniform((16, 5), dtype=tf.float32) >= .25, tf.float32)/.75
            masked = tf.convert_to_tensor(x[indexes])*mask
            model.train_on_batch(masked, y[indexes])
            return {'indexes': indexes.tolist(), 'mask': mask.numpy().tolist()}

        def check_gpu(model):
            devices = [str(getattr(getattr(v, 'value', v), 'device', '')) for v in model.trainable_variables]
            if not devices or any('GPU:' not in name.upper() for name in devices):
                raise RuntimeError('Model fell back from GPU')
            return devices

        def file_hashes(names):
            return {n: sha(root/n) for n in names}

        with tf.device('/GPU:0'):
            if stage == 'prepare':
                nr = np.random.default_rng(1701)
                tr = tf.random.Generator.from_seed(1701, alg='philox')
                model = tf.keras.Sequential([tf.keras.layers.Input((5,)), tf.keras.layers.Dense(1,
                    kernel_initializer=tf.keras.initializers.Constant(.1), bias_initializer='zeros')])
                model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='mse', jit_compile=False)
                trace1 = step(model, nr, tr)
                devices = check_gpu(model)
                model.save(root/'step1.keras')
                freeze_record(root/'rng1.json', rng_state(nr, tr))
                trace2 = step(model, nr, tr)
                model.save(root/'reference2.keras')
                freeze_record(root/'rng2.json', rng_state(nr, tr))
                freeze_record(root/'reference.json', {'trace1': trace1, 'trace2': trace2,
                    'prediction': model(x, training=False).numpy().tolist()})
                freeze_record(root/'prepared.json', {'schema': 1, 'source_sha256': source,
                    'boot_identity': boot, 'hardware': hardware, 'requested_gpu': gpu,
                    'tensorflow': tf.__version__, 'numpy': np.__version__, 'variable_devices': devices,
                    'train_on_batch_calls': 2, 'market_fits': 0,
                    'files': file_hashes(['step1.keras', 'reference2.keras', 'rng1.json', 'rng2.json', 'reference.json'])})
                return {'status': 'prepared_delete_runtime_then_rerun_same_notebook', 'new_train_on_batch_calls': 2}
            before = read_record(root/'prepared.json')
            if before['requested_gpu'] != gpu:
                raise ValueError('Requested device changed between sessions')
            state = read_record(root/'rng1.json')
            nr = np.random.default_rng()
            nr.bit_generator.state = state['numpy']
            tr = tf.random.Generator.from_state(state['tensorflow_state'], alg=state['tensorflow_alg'])
            model = tf.keras.models.load_model(root/'step1.keras')
            trace = step(model, nr, tr)
            reference = read_record(root/'reference.json')
            if trace != reference['trace2'] or rng_state(nr, tr) != read_record(root/'rng2.json'):
                raise ValueError('Random batch/mask or generator continuation differs')
            expected = tf.keras.models.load_model(root/'reference2.keras')
            actual_vars, expected_vars = model.weights+model.optimizer.variables, expected.weights+expected.optimizer.variables
            if len(actual_vars) != len(expected_vars) or int(model.optimizer.iterations.numpy()) != 2:
                raise ValueError('Optimizer continuation differs')
            maximum = 0.
            for actual, wanted in zip(actual_vars, expected_vars, strict=True):
                if not np.isfinite(actual.numpy()).all() or not np.isfinite(wanted.numpy()).all():
                    raise ValueError('Nonfinite native state')
                np.testing.assert_allclose(actual.numpy(), wanted.numpy(), rtol=0, atol=1e-6)
                maximum = max(maximum, float(np.max(np.abs(actual.numpy()-wanted.numpy()))))
            np.testing.assert_allclose(model(x, training=False).numpy(), reference['prediction'], rtol=0, atol=1e-6)
            model.save(root/'resumed2.keras')
            freeze_record(root/'resumed.json', {'trace2': trace, 'rng2': rng_state(nr, tr),
                'prediction': model(x, training=False).numpy().tolist(), 'variable_devices': check_gpu(model)})
        freeze_record(root/'complete.json', {'schema': 1, 'source_sha256': source,
            'prepared_sha256': sha(root/'prepared.json'), 'before_boot': before['boot_identity'], 'after_boot': boot,
            'local_marker_absent_at_resume': True, 'hardware_before': before['hardware'], 'hardware_after': hardware,
            'requested_gpu': gpu, 'native_max_abs_difference': maximum, 'absolute_tolerance': 1e-6,
            'synthetic_train_on_batch_calls': 3, 'market_fits': 0, 'market_skill': False,
            'scope': 'Explicit NumPy PCG64 and TF Philox mask generator; no universal Keras dropout RNG claim',
            'remote_runtime_attestation': False,
            'files': file_hashes(['prepared.json', 'step1.keras', 'reference2.keras', 'resumed2.keras',
                                  'rng1.json', 'rng2.json', 'reference.json', 'resumed.json'])})
        return {'status': 'complete', 'new_train_on_batch_calls': 1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('L4',), default='L4')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.gpu), indent=2))
