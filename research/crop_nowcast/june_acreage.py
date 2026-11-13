"""Parse USDA June Acreage reports (released ~30 June, before every cutoff here) into
state upland-cotton planted area for the report year. Development years only.

Source: ESMIS (esmis.nal.usda.gov/publication/acreage), plain-text releases.
Table 'Cotton Area Planted and Harvested by Type', Upland block, current-year
planted column (2nd number on each state line).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

STATES = {'Alabama': 'AL', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Florida': 'FL', 'Georgia': 'GA',
          'Kansas': 'KS', 'Louisiana': 'LA', 'Mississippi': 'MS', 'Missouri': 'MO', 'New Mexico': 'NM',
          'North Carolina': 'NC', 'Oklahoma': 'OK', 'South Carolina': 'SC', 'Tennessee': 'TN', 'Texas': 'TX',
          'Virginia': 'VA', 'United States': 'US'}
SEALED_FROM = 2022


def parse(text, year):
    lines = text.splitlines()
    # the table itself (not the contents entry): an 'Upland' line follows within a few lines
    up = next(j for i, ln in enumerate(lines) if ln.startswith('Cotton Area Planted and Harvested by Type')
              for j in range(i, min(i + 15, len(lines))) if lines[j].startswith('Upland'))
    out = {}
    for ln in lines[up + 1:up + 40]:
        m = re.match(r'^([A-Za-z ]+?)\s*\.+\s*:\s*([\d,.]+)\s+([\d,.]+)', ln)
        if not m:
            if out and ln.strip() and not ln.startswith(' '):
                break
            continue
        if m.group(1) in STATES:
            out[STATES[m.group(1)]] = float(m.group(3).replace(',', '')) * 1000
        if m.group(1) == 'United States':
            break
    return out


def main(raw_dir, out_json):
    res, receipt = {}, {}
    for p in sorted(Path(raw_dir).glob('*.txt')):
        b = p.read_bytes()
        t = b.decode('latin-1')
        m = re.search(r'(20\d\d)\.txt$', p.name) or re.search(r'acrg06(\d\d)', p.name)
        year = int(m.group(1)) if len(m.group(1)) == 4 else 2000 + int(m.group(1))
        if year >= SEALED_FROM:
            raise SystemExit(f'sealed year file present: {p.name}')
        res[year] = parse(t, year)
        receipt[p.name] = {'year': year, 'sha256': hashlib.sha256(b).hexdigest(), 'states': len(res[year])}
    Path(out_json).write_text(json.dumps({'source': 'https://esmis.nal.usda.gov/publication/acreage', 'files': receipt,
                                          'planted_june': res}, indent=1), encoding='utf-8')
    for y in sorted(res):
        print(y, len(res[y]), res[y].get('TX'), res[y].get('US'))


if __name__ == '__main__':
    main(*sys.argv[1:3])
