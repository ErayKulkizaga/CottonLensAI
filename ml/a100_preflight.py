"""Colab A100 technical checkpoint probe. Generated data only; no market skill."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from cottonlens_ml.runtime_guard import require_colab_training
from cottonlens_ml.sprint import freeze_record, read_record


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, expected_gpu='A100'):
    require_colab_training()
    output = Path(output)
    source_id = sha(Path(__file__))
    hardware = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    devices = {'A100': 'A100', 'L4': 'L4', 'G4': 'RTX PRO 6000'}
    if expected_gpu not in devices:
        raise ValueError('Unsupported requested GPU')
    if not hardware or any(devices[expected_gpu] not in line for line in hardware.splitlines()):
        raise RuntimeError(f'{expected_gpu} required; no silent device substitution')
    completed = output/'complete.json'
    if completed.exists():
        record = read_record(completed)
        if record['source_id'] != source_id:
            raise ValueError('Probe source changed; use a new namespace')
        for name, expected in record['files'].items():
            if sha(output/name) != expected:
                raise ValueError('Corrupt probe checkpoint; refusing retraining')
        return {**record, 'cached': True}
    if output.exists():
        raise ValueError('Incomplete probe preserved; inspect before deliberate retry')
    output.mkdir(parents=True)
    started = time.monotonic()
    # Imports happen only after Colab/device policy checks.
    import numpy as np
    import tensorflow as tf
    tf.keras.utils.set_random_seed(1701)
    tf.config.experimental.enable_op_determinism()
    tf.config.set_soft_device_placement(False)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('TensorFlow GPU unavailable')
    rng = np.random.default_rng(1701)
    x = rng.normal(size=(256, 5)).astype('float32')
    y = (2*x[:, :1]-x[:, 1:2]).astype('float32')
    with tf.device('/GPU:0'):
        model = tf.keras.Sequential([tf.keras.layers.Input((5,)), tf.keras.layers.Dense(1)])
        model.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='mse', jit_compile=False)
        model.train_on_batch(x, y)
        devices = [str(getattr(getattr(v, 'value', v), 'device', '')) for v in model.trainable_variables]
        if not devices or any('GPU:' not in name.upper() for name in devices):
            raise RuntimeError('Trainable variables fell back from GPU')
        model.save(output/'step1.keras')
        model.train_on_batch(x, y)
        expected = [v.numpy().copy() for v in model.weights + model.optimizer.variables]
        prediction = model(x, training=False).numpy()
        restored = tf.keras.models.load_model(output/'step1.keras')
        restored.train_on_batch(x, y)
        actual = [v.numpy() for v in restored.weights + restored.optimizer.variables]
        if len(actual) != len(expected):
            raise ValueError('Optimizer state was not restored')
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        np.testing.assert_allclose(restored(x, training=False).numpy(), prediction, rtol=0, atol=1e-6)
        restored.save(output/'step2.keras')
        native = tf.keras.models.load_model(output/'step2.keras')
        np.testing.assert_allclose(native(x, training=False).numpy(), prediction, rtol=0, atol=1e-6)
    record = {'schema': 1, 'source_id': source_id, 'kind': 'synthetic_technical_not_learning_power',
              'market_fits': 0, 'synthetic_train_on_batch_calls': 3, 'seed': 1701,
              'hardware': hardware, 'requested_gpu': expected_gpu, 'tensorflow': tf.__version__, 'numpy': np.__version__,
              'variable_devices': devices, 'elapsed_seconds': time.monotonic()-started,
              'optimizer_resume_checked': True, 'native_reload_checked': True,
              'market_skill': False, 'G1_G2_admission_granted': False,
              'files': {name: sha(output/name) for name in ('step1.keras', 'step2.keras')}}
    freeze_record(completed, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('A100', 'L4', 'G4'), default='A100')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.gpu), indent=2))
