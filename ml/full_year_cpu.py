"""Thin launcher for the existing engine with the isolated CPU environment."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('setup', 'status', 'prepare', 'pilot-plan', 'pilot', 'compare', 'report', 'track', 'reproduce'))
    parser.add_argument('--registration-root', type=Path)
    parser.add_argument('--decision-contract', type=Path)
    parser.add_argument('--cpu-environment', type=Path)
    parser.add_argument('--drive-root', type=Path)
    parser.add_argument('--market-file', type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--mirror-root', type=Path)
    parser.add_argument('--max-minutes', type=float, default=60.)
    parser.add_argument('--python', default='3.12.14')
    parser.add_argument('--experiment')
    parser.add_argument('--profile', choices=('full-year-v1', 'ams-exploration-v1', 'fas-exploration-v1', 'nass-exploration-v1', 'wasde-exploration-v1', 'cftc-exploration-v1', 'fx-exploration-v1', 'crop-exploration-v1', 'weather-exploration-v1', 'oncall-exploration-v1', 'oncall-exploration-v2', 'recency-pilot-v1', 'availability-clock-pilot-v1', 'return-path-pilot-v1', 'agri-transfer-pilot-v1', 'wasde-regional-t1-pilot-v1'), default='full-year-v1')
    parser.add_argument('--ams-table', type=Path)
    parser.add_argument('--ams-publications', type=Path)
    parser.add_argument('--fas-table', type=Path)
    parser.add_argument('--nass-table', type=Path)
    parser.add_argument('--nass-audit-root', type=Path)
    parser.add_argument('--wasde-table', type=Path)
    parser.add_argument('--cftc-table', type=Path)
    parser.add_argument('--fx-table', type=Path)
    parser.add_argument('--crop-table', type=Path)
    parser.add_argument('--weather-table', type=Path)
    parser.add_argument('--oncall-table', type=Path)
    args = parser.parse_args()
    if args.experiment is None:
        args.experiment = {'wasde-regional-t1-pilot-v1': 'research-wasde-regional-t1-pilot-v1', 'availability-clock-pilot-v1': 'research-availability-clock-pilot-v1','ams-exploration-v1':'research-ams-exploration-v1',
            'fas-exploration-v1':'research-fas-exploration-v1',
            'nass-exploration-v1':'research-nass-exploration-v1',
            'wasde-exploration-v1':'research-wasde-exploration-v1',
            'cftc-exploration-v1':'research-cftc-exploration-v1',
            'fx-exploration-v1':'research-fx-exploration-v1',
            'crop-exploration-v1':'research-crop-exploration-v1',
            'weather-exploration-v1':'research-weather-exploration-v1',
            'oncall-exploration-v1':'research-oncall-exploration-v1',
            'recency-pilot-v1':'research-recency-pilot-v1',
            'return-path-pilot-v1':'research-return-path-pilot-v1',
            'agri-transfer-pilot-v1':'research-agri-transfer-pilot-v1',
            'oncall-exploration-v2':'research-oncall-exploration-v2'}.get(args.profile,'research-full-year-v1-r2')
    environment = args.cpu_environment or repo/'output/full-year-cpu-env'
    env = os.environ.copy()
    env.update({'PYTHONPATH': str(repo/'ml/src'), 'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'UV_PROJECT_ENVIRONMENT': str(environment)})
    if args.stage == 'setup':
        command = [sys.executable, '-m', 'uv', 'sync', '--project', str(repo/'ml'), '--locked',
                   '--only-group', 'cpu', '--python', args.python]
    else:
        python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not python.exists():
            raise RuntimeError('Run setup first; CPU packages are separate from backend and GPU environments')
        command = [str(python), '-m', 'cottonlens_ml.research.engine', '--repo', str(repo),
                   '--drive-root', str(args.drive_root or repo/'output/full-year'), '--experiment', args.experiment,
                   '--profile', args.profile, '--stage', args.stage, '--max-minutes', str(min(args.max_minutes, 30.) if args.profile in ('availability-clock-pilot-v1', 'wasde-regional-t1-pilot-v1') else args.max_minutes)]
        if args.registration_root:
            command += ['--registration-root', str(args.registration_root)]
        if args.decision_contract:
            command += ['--decision-contract', str(args.decision_contract)]
        if args.market_file:
            command += ['--market-file', str(args.market_file)]
        if args.reference_root:
            command += ['--reference-root', str(args.reference_root)]
        if args.mirror_root:
            command += ['--mirror-root', str(args.mirror_root)]
        if args.ams_table:
            command += ['--ams-table', str(args.ams_table)]
        if args.ams_publications:
            command += ['--ams-publications', str(args.ams_publications)]
        if args.fas_table:
            command += ['--fas-table', str(args.fas_table)]
        if args.nass_table:
            command += ['--nass-table',str(args.nass_table)]
        if args.nass_audit_root:
            command += ['--nass-audit-root',str(args.nass_audit_root)]
        if args.weather_table:
            command += ['--weather-table',str(args.weather_table)]
        if args.oncall_table:
            command += ['--oncall-table',str(args.oncall_table)]
        if args.crop_table:
            command += ['--crop-table',str(args.crop_table)]
        if args.fx_table:
            command += ['--fx-table',str(args.fx_table)]
        if args.cftc_table:
            command += ['--cftc-table',str(args.cftc_table)]
        if args.wasde_table:
            command += ['--wasde-table',str(args.wasde_table)]
        if args.stage in ('pilot','reproduce'):
            env['COTTONLENS_ALLOW_LOCAL_CPU_TABULAR'] = '1'
    subprocess.run(command, cwd=repo, env=env, check=True)


if __name__ == '__main__':
    main()
