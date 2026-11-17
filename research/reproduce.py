"""Rebuild every Phase 2 headline result from cached public data and compare it with the committed evidence.

Each step re-runs the original scorer into a temporary folder and compares every number in its JSON output with the
committed evidence file (relative tolerance 1e-9). Exit code 0 only if all steps match.
Inputs are the cached public data under output/ (re-downloadable with the pull scripts named in each evidence file).
Usage: python research/reproduce.py            (run from the repository root)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'research/evidence'
O = ROOT / 'output'
CN = 'research/crop_nowcast'
MOM_ENV = {'COTTONLENS_MOMENTUM_YEARS': '2002-2025', 'COTTONLENS_MOMENTUM_FIRST_TRAIN': '2000'}

STEPS = [
    ('Momentum record 2002-2025 (12/12 cells)', [f'{CN}/momentum.py', O / 'crop-nowcast-full/nass', '{out}'], MOM_ENV,
     'momentum-full-2002-2025-20261116.json'),
    ('Forward lock 2026', [f'{CN}/lock_2026.py', O / 'crop-nowcast-full/nass', O / 'crop-nowcast-2026/nass', '{out}'], {},
     'forward-lock-2026-20261116.json'),
    ('Robustness + published-style comparison', [f'{CN}/robustness.py', O / 'crop-nowcast-full/nass', '{out}'], MOM_ENV,
     'robustness-momentum-20261116.json'),
    ('T46 corn/soy', [f'{CN}/t46_crops.py', 'score', O / 'crop-nowcast-t46', '{out}'], {}, 't46-corn-soy-20261116.json'),
    ('T47 stage A 1995-1999', [f'{CN}/t47_blind_old.py', 'score', O / 'crop-nowcast-t47a', O / 'crop-nowcast-full/nass', '{out}'],
     {}, 't47-stageA-1995-1999-20261116.json'),
    ('Error mechanism 2012-2025', [f'{CN}/error_mechanism.py', O / 'crop-nowcast-t39/nass', '{out}'], {},
     'error-mechanism-2012-2025-20261116.json'),
    ('T48 mechanism 1995-2011', [f'{CN}/t48_mechanism.py', 'score', O / 'crop-nowcast-t48', '{out}'], {},
     't48-mechanism-1995-2011-20261116.json'),
    ('T51 Texas area: anchoring 7.0% vs USDA 11.4%', [f'{CN}/t51_area_production.py', O / 't49', '{out}'], {},
     't51-area-production-20261116.json'),
    ('T50 next-revision direction (67%, p 0.012)', [f'{CN}/t50_revision_direction.py', '{out}'], {},
     't50-revision-direction-20261116.json'),
    # shortcut: pools the committed per-era results (t47-rolling, t45-blind); the two eras themselves are not re-run here
    ('Pooled blind 1987-2011 (88/128, p 0.007)', [f'{CN}/pooled_blind.py', '{out}'], {},
     'pooled-blind-1987-2011-20261116.json'),
]


def diff(a, b, path=''):
    """List of mismatches between two JSON values."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f'{path}/{k}: missing' for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += diff(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in diff(x, y, f'{path}[{i}]')]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a != a and b != b:  # both NaN
            return []
        return [] if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)) else [f'{path}: {a!r} != {b!r}']
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, cmd, env, ref in STEPS:
            out = Path(tmp) / ref
            args = [sys.executable, *[str(c).replace('{out}', str(out)) for c in cmd]]
            r = subprocess.run(args, cwd=ROOT, env={**os.environ, **env}, capture_output=True, text=True)
            if r.returncode or not out.exists():
                print(f'FAIL  {name}: exit {r.returncode}\n{r.stderr[-800:]}')
                bad += 1
                continue
            d = diff(json.loads(out.read_text(encoding='utf-8')), json.loads((E / ref).read_text(encoding='utf-8')))
            print(f"{'OK   ' if not d else 'DIFF '} {name}  ({ref}){'' if not d else f': {len(d)} differences, e.g. ' + d[0]}")
            bad += bool(d)
    print('ALL MATCH' if not bad else f'{bad} step(s) differ')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
