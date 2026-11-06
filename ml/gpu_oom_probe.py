"""Colab-only bounded GPU allocator OOM and fresh-process recovery; no fits."""
import argparse
import subprocess
import sys
from pathlib import Path

from a100_preflight import sha
from cottonlens_ml.sprint import freeze_record, read_record
from gpu_restart_probe import device

LIMIT_MB = 128
SHAPE = (8192, 8192)  # 256 MiB float32; no attempt to exhaust physical VRAM.


def worker(root, gpu, stage):
    hardware = device(gpu)
    target = root/f'{stage}.json'
    if stage not in ('exhaust', 'recover') or target.exists():
        raise ValueError('Explicit new worker stage required; preserve existing evidence')
    import tensorflow as tf
    physical = tf.config.list_physical_devices('GPU')
    if len(physical) != 1:
        raise RuntimeError('Exactly one TensorFlow GPU required')
    tf.config.set_soft_device_placement(False)
    tf.config.set_logical_device_configuration(
        physical[0], [tf.config.LogicalDeviceConfiguration(memory_limit=LIMIT_MB)])
    # Force runtime initialization before the expected allocation failure.
    with tf.device('/GPU:0'):
        small = tf.constant([1., 2.], dtype=tf.float32)
        if 'GPU:0' not in small.device.upper() or small.numpy().tolist() != [1., 2.]:
            raise RuntimeError('GPU placement/recovery computation failed')
        if stage == 'exhaust':
            try:
                value = tf.zeros(SHAPE, dtype=tf.float32)
                value.numpy()  # Synchronize if the backend defers allocation.
            except tf.errors.ResourceExhaustedError:
                freeze_record(target, {'schema': 1, 'stage': stage,
                    'exception_type': 'tensorflow.errors.ResourceExhaustedError',
                    'logical_memory_limit_mb': LIMIT_MB, 'requested_shape': list(SHAPE),
                    'requested_bytes': 4*SHAPE[0]*SHAPE[1], 'hardware': hardware,
                    'device': small.device, 'tensorflow': tf.__version__, 'fits': 0})
                return
            raise RuntimeError('Expected allocator OOM not observed; no success receipt')
        witness = read_record(root/'exhaust.json')
        if witness['exception_type'] != 'tensorflow.errors.ResourceExhaustedError':
            raise ValueError('Verified exhaustion receipt required before recovery')
        freeze_record(target, {'schema': 1, 'stage': stage, 'hardware': hardware,
            'device': small.device, 'tensorflow': tf.__version__, 'values': small.numpy().tolist(),
            'fits': 0, 'prior_exhaust_sha256': sha(root/'exhaust.json')})


def run(output, gpu='L4'):
    hardware = device(gpu)  # Colab guard runs before filesystem or TensorFlow.
    root = Path(output)
    source = {name: sha(Path(__file__).with_name(name)) for name in
              ('gpu_oom_probe.py', 'gpu_restart_probe.py', 'a100_preflight.py')}
    if (root/'complete.json').exists():
        record = read_record(root/'complete.json')
        if record['source'] != source or record['requested_gpu'] != gpu:
            raise ValueError('OOM probe identity changed')
        if set(record['files']) != {'exhaust.json', 'recover.json'}:
            raise ValueError('Complete OOM evidence required')
        for name, checksum in record['files'].items():
            if sha(root/name) != checksum:
                raise ValueError('Corrupt OOM evidence; no rerun')
        return {**record, 'cached': True}
    if root.exists():
        raise ValueError('Partial OOM attempt preserved; review before a new namespace')
    root.mkdir(parents=True)
    for stage in ('exhaust', 'recover'):
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
            '--output', str(root), '--gpu', gpu, '--stage', stage], check=True, timeout=180)
        read_record(root/f'{stage}.json')
    record = {'schema': 1, 'source': source, 'requested_gpu': gpu, 'hardware': hardware,
        'scope': 'Real GPU allocator with explicit128MiB logical limit; not physical VRAM exhaustion',
        'whole_session_restart_verified': False, 'stochastic_resume_verified': False,
        'market_skill': False, 'market_fits': 0, 'synthetic_fits': 0,
        'files': {name: sha(root/name) for name in ('exhaust.json', 'recover.json')}}
    freeze_record(root/'complete.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', choices=('A100', 'L4', 'G4'), default='L4')
    parser.add_argument('--stage', choices=('exhaust', 'recover'))
    args = parser.parse_args()
    if args.stage:
        worker(args.output, args.gpu, args.stage)
    else:
        print(run(args.output, args.gpu))
