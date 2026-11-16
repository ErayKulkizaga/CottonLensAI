"""T45: merge report-parsed 2000-2011 forecasts with QuickStats finals into nass-style files (no values printed).
Usage: python t45_assemble.py t45_root
"""
import json
import sys
from pathlib import Path


def main(root):
    t = Path(root)
    (t / 'nass').mkdir(exist_ok=True)
    for level in ('state', 'us'):
        fc = json.loads((t / 'forecasts' / f'{level}_forecasts.json').read_text())['data']
        fin = [r for r in json.loads((t / 'nass_finals' / f'{level}_production.json').read_text())['data']
               if r['reference_period_desc'] == 'YEAR']
        (t / 'nass' / f'{level}_production.json').write_text(json.dumps({'data': fc + fin}), encoding='utf-8')
        print(level, 'forecast rows', len(fc), 'final rows', len(fin))


if __name__ == '__main__':
    main(sys.argv[1])
