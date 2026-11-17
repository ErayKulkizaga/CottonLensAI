"""Bounded Colab native worker-exit probe; generated data, never market data."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from a100_preflight import sha
from cottonlens_ml.runtime_guard import require_colab_training
from cottonlens_ml.sprint import freeze_record, read_record


def device(gpu):
    require_colab_training()
    names = {'A100': 'A100', 'L4': 'L4', 'G4': 'RTX PRO 6000'}
    if gpu not in names:
        raise ValueError('Explicit supported GPU required')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                        '--format=csv,noheader'], text=True).strip()
    if not hardware or any(names[gpu] not in row for row in hardware.splitlines()):
        raise RuntimeError('Requested GPU unavailable; no fallback')
    return hardware


def worker(root, gpu, stage):
    hardware = device(gpu)
    if stage not in ('crash', 'resume'):
        raise ValueError('Unknown worker stage')
    if stage == 'resume' and any((root/n).exists() for n in ('resumed.keras', 'resume.json')):
        raise ValueError('Preserve existing resumed worker output')
    import numpy as np
    import tensorflow as tf
    tf.keras.utils.set_random_seed(1701)
    tf.config.experimental.enable_op_determinism()
    tf.config.set_soft_device_placement(False)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('TensorFlow GPU unavailable')
    with tf.device('/GPU:0'):
        if stage == 'crash':
            model = tf.keras.models.load_model(root/'baseline/step1.keras')
            target = root/'interrupted.keras'
            if target.exists() or (root/'worker-exit.json').exists():
                raise ValueError('Preserve interrupted worker output')
            model.save(target)
            variables = [str(getattr(getattr(v, 'value', v), 'device', '')) for v in model.trainable_variables]
            if not variables or any('GPU:' not in v.upper() for v in variables):
                raise RuntimeError('Worker fell back from GPU')
            if int(model.optimizer.iterations.numpy()) != 1:
                raise ValueError('Wrong pre-exit optimizer step')
            freeze_record(root/'worker-exit.json', {'pid': os.getpid(), 'hardware': hardware,
                          'stage': 'native_checkpoint_saved_before_process_exit',
                          'checkpoint_sha256': sha(target), 'variable_devices': variables})
            os._exit(73)  # Intentionally bypass Python cleanup, not a caught exception.
        if stage != 'resume':
            raise ValueError('Unknown worker stage')
        witness = read_record(root/'worker-exit.json')
        if sha(root/'interrupted.keras') != witness['checkpoint_sha256']:
            raise ValueError('Interrupted checkpoint changed')
        model = tf.keras.models.load_model(root/'interrupted.keras')
        rng = np.random.default_rng(1701)
        x = rng.normal(size=(256, 5)).astype('float32')
        y = (2*x[:, :1]-x[:, 1:2]).astype('float32')
        model.train_on_batch(x, y)
        expected = tf.keras.models.load_model(root/'baseline/step2.keras')
        actual_state = model.weights + model.optimizer.variables
        expected_state = expected.weights + expected.optimizer.variables
        if len(actual_state) != len(expected_state) or int(model.optimizer.iterations.numpy()) != 2:
            raise ValueError('Optimizer state did not resume')
        for actual, wanted in zip(actual_state, expected_state, strict=True):
            if not np.isfinite(actual.numpy()).all() or not np.isfinite(wanted.numpy()).all():
                raise ValueError('Nonfinite native optimizer/model state')
            np.testing.assert_allclose(actual.numpy(), wanted.numpy(), rtol=0, atol=1e-6)
        np.testing.assert_allclose(model(x, training=False).numpy(), expected(x, training=False).numpy(),
                                   rtol=0, atol=1e-6)
        model.save(root/'resumed.keras')
        freeze_record(root/'resume.json', {'optimizer_step': 2, 'absolute_tolerance': 1e-6,
                      'native_state_and_prediction_match': True, 'tensorflow': tf.__version__})


def run(output, gpu='L4'):
    hardware = device(gpu)
    root = Path(output)
    source = {'probe': sha(Path(__file__)), 'baseline': sha(Path(__file__).with_name('a100_preflight.py'))}
    if (root/'complete.json').exists():
        record = read_record(root/'complete.json')
        if record['source'] != source or record['requested_gpu'] != gpu:
            raise ValueError('Restart probe identity changed')
        for name, checksum in record['files'].items():
            target = (root/name).resolve()
            if not target.is_relative_to(root.resolve()) or sha(target) != checksum:
                raise ValueError('Corrupt restart probe payload; no retraining')
        return {**record, 'cached': True}
    if root.exists():
        raise ValueError('Partial restart probe preserved; use a reviewed new attempt')
    root.mkdir(parents=True)
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('a100_preflight.py')),
                    '--output', str(root/'baseline'), '--gpu', gpu], check=True, timeout=600)
    common = [sys.executable, '-B', str(Path(__file__).resolve()), '--output', str(root), '--gpu', gpu]
    child = subprocess.run([*common, '--stage', 'crash'], check=False, timeout=120)
    if child.returncode != 73:
        raise RuntimeError('Expected explicit worker exit was not observed')
    subprocess.run([*common, '--stage', 'resume'], check=True, timeout=600)
    record = {'schema': 1, 'source': source, 'requested_gpu': gpu, 'hardware': hardware,
              'worker_exit_code': 73, 'synthetic_train_on_batch_calls': 4, 'market_fits': 0,
              'market_skill': False, 'whole_colab_session_restart_verified': False,
              'stochastic_rng_resume_verified': False, 'real_gpu_oom_verified': False,
              'files': {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    freeze_record(root/'complete.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('A100', 'L4', 'G4'), default='L4')
    parser.add_argument('--stage', choices=('crash', 'resume'))
    args = parser.parse_args()
    if args.stage:
        worker(args.output, args.gpu, args.stage)
    else:
        print(run(args.output, args.gpu))
