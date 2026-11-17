"""Offline semantic verification of returned bounded-OOM receipts; no TensorFlow."""
import argparse
import json
import zipfile
from pathlib import Path

from research.verify_gpu_probe import members, sha
from research.verify_gpu_restart import record


def verify(archive, checksum, source):
    data = Path(archive).read_bytes()
    manifest = json.loads(Path(checksum).read_text(encoding='utf-8'))
    if (sha(data) != manifest['archive_sha256']
            or sha(Path(source).read_bytes()) != manifest['source_archive_sha256']):
        raise ValueError('OOM result/source checksum mismatch')
    payload = members(data, ('complete.json', 'exhaust.json', 'recover.json'))
    hashes = {n: sha(v) for n, v in payload.items()}
    if hashes != manifest['members']:
        raise ValueError('OOM member checksum mismatch')
    complete, exhausted, recovered = (record(payload[n]) for n in payload)
    with zipfile.ZipFile(source) as z:
        expected = {name: sha(z.read('ml/'+name)) for name in
                    ('gpu_oom_probe.py', 'gpu_restart_probe.py', 'a100_preflight.py')}
    if complete['source'] != expected:
        raise ValueError('OOM executed source mismatch')
    if complete['files'] != {n: h for n, h in hashes.items() if n != 'complete.json'}:
        raise ValueError('OOM completion closure mismatch')
    if (any(r['schema'] != 1 for r in (complete, exhausted, recovered))
            or complete['requested_gpu'] != 'L4' or 'NVIDIA L4,' not in complete['hardware']
            or complete['market_fits'] != 0 or complete['synthetic_fits'] != 0
            or any(complete[k] is not False for k in
                   ('whole_session_restart_verified', 'stochastic_resume_verified', 'market_skill'))):
        raise ValueError('Unexpected OOM scope or hardware')
    if (exhausted['stage'] != 'exhaust'
            or exhausted['exception_type'] != 'tensorflow.errors.ResourceExhaustedError'
            or exhausted['logical_memory_limit_mb'] != 128
            or exhausted['requested_shape'] != [8192, 8192]
            or exhausted['requested_bytes'] != 268435456
            or exhausted['requested_bytes'] <= exhausted['logical_memory_limit_mb']*1024**2):
        raise ValueError('Expected bounded allocation exhaustion required')
    if (recovered['stage'] != 'recover' or recovered['values'] != [1., 2.]
            or recovered['prior_exhaust_sha256'] != hashes['exhaust.json']
            or exhausted['fits'] != 0 or recovered['fits'] != 0
            or not complete['hardware'] == exhausted['hardware'] == recovered['hardware']
            or not exhausted['tensorflow'] == recovered['tensorflow'] == '2.20.0'
            or any(not r['device'].upper().endswith('/DEVICE:GPU:0') for r in (exhausted, recovered))):
        raise ValueError('Fresh-process recovery witness mismatch')
    return {'schema': 1, 'status': 'bounded_oom_and_recovery_receipts_verified',
            'archive_sha256': sha(data), 'source_archive_sha256': manifest['source_archive_sha256'],
            'hardware_reported': complete['hardware'], 'tensorflow_reported': exhausted['tensorflow'],
            'logical_memory_limit_mb': 128, 'requested_bytes': 268435456,
            'recovery_values': recovered['values'], 'new_fits': 0, 'tensorflow_imported': False,
            'physical_vram_exhaustion_verified': False, 'remote_hardware_attestation': False,
            'whole_session_restart_verified': False, 'stochastic_resume_verified': False,
            'market_skill': False, 'verifier_sha256': sha(Path(__file__).read_bytes())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'checksum', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous OOM verification')
    result = verify(args.archive, args.checksum, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))
