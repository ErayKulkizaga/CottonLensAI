"""Offline, zero-fit replay of the strict source gate against its frozen parent."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(bundle, parent, output):
    bundle, parent, output = (p.resolve() for p in (bundle, parent, output))
    if output.exists():
        raise ValueError('Preserve previous replay output')
    reference = json.loads((bundle/'parent-reference.json').read_text())
    for row in reference['files']:
        p = (parent/row['path']).resolve()
        if not p.is_relative_to(parent) or sha(p) != row['sha256']:
            raise ValueError('Parent payload missing or changed')
    source = bundle/'source'
    for name, digest in json.loads((bundle/'source-manifest.json').read_text()).items():
        p = (source/name).resolve()
        if not p.is_relative_to(source) or sha(p) != digest:
            raise ValueError('Frozen source changed')
    output.mkdir(parents=True)
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(str(source/p) for p in ('ml/src', '.')),
           'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR', None)
    bootstrap = '''import runpy,sys,socket
def deny(*a,**k):raise RuntimeError('Offline zero-fit replay')
class OfflineSocket(socket.socket):
 def connect(self,*a,**k):return deny()
 def connect_ex(self,*a,**k):return deny()
socket.socket=OfflineSocket;socket.getaddrinfo=deny
from cottonlens_ml import runtime_guard
runtime_guard.require_training=deny;runtime_guard.require_colab_training=deny
sys.argv=sys.argv[1:];runpy.run_module(sys.argv[0],run_name='__main__')'''

    def run(module, args):
        subprocess.run([sys.executable, '-B', '-c', bootstrap, module, *map(str, args)],
                       cwd=source, env=env, check=True, stdout=subprocess.PIPE, text=True)

    old = parent/'bundle'
    common = ['--recovery', old/'normalized-recovery', '--merged', old/'reference/prior-merged.json',
              '--contract', old/'reference/contract.json']
    full = old/'full-daily/verified-full-daily-overlay.json'
    run('research.ams_cell_readiness', [*common, '--full-daily', full, '--output', output/'strict-readiness'])
    run('research.verify_ams_cell_readiness', ['--root', output/'strict-readiness', *common,
                                             '--full', full, '--output', output/'strict-independent.json'])
    history = old/'normalized-recovery/inputs/parent/inputs/history.parquet'
    contract = old/'reference/contract.json'
    overlay = output/'strict-readiness/witness-overlay.json'
    args = ['--history', history, '--overlay', overlay, '--contract', contract]
    run('research.ams_contract_feature_impact', [*args, '--output', output/'strict-feature-impact'])
    run('research.verify_ams_contract_feature_impact', ['--root', output/'strict-feature-impact', *args,
                                                      '--output', output/'strict-feature-independent.json'])
    expected = json.loads((bundle/'replay-expected.json').read_text())
    for name, digest in expected.items():
        if sha(output/name) != digest:
            raise ValueError('Strict replay differs: '+name)
    result = {'status': 'strict_source_replay_byte_identical', 'new_fits': 0,
              'network_disabled': True, 'parent_members_checked': len(reference['files']),
              'outputs_verified': expected, 'replay_script_sha256': sha(Path(__file__))}
    (output/'replay-receipt.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'parent', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.bundle, args.parent, args.output), indent=2))
