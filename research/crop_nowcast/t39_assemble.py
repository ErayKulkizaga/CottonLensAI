"""T39: assemble dev + sealed (2022+) inputs into one tree without touching the dev files.

Dev rows (<= 2021) are copied unchanged from the dev tree; sealed pulls only add rows for years >= 2022.
Layout written under t39_root: nass/*.json (+ june_acreage.json), fsa/fsa_cotton.csv, power/, ndvi/.
Usage: python t39_assemble.py dev_root t39_root
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.june_acreage import parse as parse_june

SEALED_FROM = 2022


def main(dev_root, t39_root):
    dev, t = Path(dev_root), Path(t39_root)
    (t / 'nass').mkdir(parents=True, exist_ok=True)
    for f in sorted((dev / 'nass').glob('*.json')):
        if f.name == 'june_acreage.json':
            continue
        rows = json.loads(f.read_text())['data'] if f.name != 'receipt.json' else None
        if rows is None:
            continue
        assert all(int(r['year']) < SEALED_FROM for r in rows), f.name
        sealed = t / 'nass_sealed' / f.name
        new = [r for r in json.loads(sealed.read_text())['data'] if int(r['year']) >= SEALED_FROM] if sealed.exists() else []
        (t / 'nass' / f.name).write_text(json.dumps({'data': rows + new}, separators=(',', ':')), encoding='utf-8')
        print(f.name, len(rows), '+', len(new))
    june = json.loads((dev / 'nass' / 'june_acreage.json').read_text())
    for p in sorted((t / 'june_raw').glob('acrg06*.txt')):
        year = 2000 + int(p.stem[-2:])
        assert year >= SEALED_FROM and str(year) not in june['planted_june']
        june['planted_june'][str(year)] = parse_june(p.read_text(encoding='latin-1'), year)
    (t / 'nass' / 'june_acreage.json').write_text(json.dumps(june, indent=1), encoding='utf-8')
    (t / 'fsa').mkdir(exist_ok=True)
    d = (dev / 'fsa' / 'fsa_cotton.csv').read_text(encoding='utf-8').splitlines()
    s = (t / 'fsa_sealed.csv').read_text(encoding='utf-8').splitlines()
    assert d[0] == s[0] and all(int(x.split(',')[0]) >= SEALED_FROM for x in s[1:])
    (t / 'fsa' / 'fsa_cotton.csv').write_text('\n'.join(d + s[1:]) + '\n', encoding='utf-8')
    for sub in ('power', 'ndvi'):  # dev files copied; sealed files already sit in t39_root/<sub>_sealed
        (t / sub).mkdir(exist_ok=True)
        for f in [*(dev / sub).glob('*'), *(t / f'{sub}_sealed').glob('*')]:
            if f.is_file():
                shutil.copy2(f, t / sub / f.name)
    print('june years', sorted(june['planted_june']), 'fsa rows', len(d) - 1, '+', len(s) - 1)


if __name__ == '__main__':
    main(*sys.argv[1:3])
