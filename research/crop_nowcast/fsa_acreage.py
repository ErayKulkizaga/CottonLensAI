"""FSA monthly crop acreage (planted and failed acres reported by farmers) for upland cotton.

Source: https://www.fsa.usda.gov/.../crop-acreage-data (monthly 'as of' files, 2011+).
Each file is a snapshot of farmer filings to its 'as of' date. Only files dated
in-season (Aug-Oct) of development years 2011-2021 are read here; the sealed
years 2022-2025 are refused. Assumption (recorded in the ledger): a file dated
'as of D' is public before USDA's Crop Production report of the same month.
Usage: python fsa_acreage.py receipt.json raw_dir out.csv
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

SEALED_FROM = 2022
# 2012+ headers, plus the 2011 layout ('Crop Name', 'Planted', 'Failed'; no irrigation column)
COLS = {'state code': 'st', 'county code': 'co', 'crop': 'crop', 'crop type': 'ctype',
        'irrigation practice': 'irr', 'planted acres': 'planted', 'failed acres': 'failed',
        'crop name': 'crop', 'crop type name': 'ctype', 'planted': 'planted', 'failed': 'failed',
        'planted total': 'planted', 'failed total': 'failed',
        'state_fsa_code': 'st', 'county_fsa_code': 'co'}  # 2012-2014 (FSA state codes equal FIPS)  # Oct 2011 headers contain line breaks


def cotton_rows(rows):
    """rows: iterable of dicts with the COLS values. Returns {(st, co, irr): [planted, failed]} for upland cotton."""
    out = {}
    for r in rows:
        crop, ctype = str(r.get('crop') or '').upper(), str(r.get('ctype') or '').upper()
        if 'COTTON' not in crop or 'ELS' in crop or 'ELS' in ctype or 'PIMA' in ctype:
            continue
        if not ('UPLAND' in crop or 'UPLAND' in ctype):
            continue
        k = (str(r['st']).zfill(2), str(r['co']).zfill(3), str(r.get('irr') or 'N')[:1].upper())
        acc = out.setdefault(k, [0.0, 0.0])
        acc[0] += float(r.get('planted') or 0)
        acc[1] += float(r.get('failed') or 0)
    return out


def read_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        it = ws.iter_rows(values_only=True)
        for _ in range(12):  # header is within the first rows of the county-level sheet
            head = next(it, None)
            if head is None:
                break
            names = [' '.join(str(h).split()).lower() if h is not None else '' for h in head]
            idx = {COLS[n]: i for i, n in enumerate(names) if n in COLS}
            if {'st', 'co', 'crop', 'planted', 'failed'} <= idx.keys():
                yield from ({k: row[i] for k, i in idx.items()} for row in it if row and row[idx['crop']])
                return
    raise ValueError(f'no county-level sheet with failed acres in {path}')


def month_of(label):
    m = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', label)
    return {'Jul': 'aug', 'Aug': 'aug', 'Sep': 'sep', 'Oct': 'oct'}.get(m.group(1)) if m else None


def main(receipt_json, raw_dir, out_csv):
    rec = json.loads(Path(receipt_json).read_text())
    lines = ['year,month,label,st,co,irr,planted,failed']
    def day(label):  # day of the 'as of' date; files without one (e.g. 'August 2014') count as earliest
        m = re.search(r'(?:August|September|October|July)\s+(\d{1,2})', label)
        return int(m.group(1)) if m else 0

    first = {}
    for name, meta in rec.items():
        key = (meta['year'], month_of(meta['label']))
        if key[1] and (key not in first or day(meta['label']) < day(rec[first[key]]['label'])):
            first[key] = name
    for name, meta in sorted(rec.items()):
        if meta['year'] >= SEALED_FROM and '--unseal' not in sys.argv:  # T39 prereg pushed
            raise SystemExit(f'sealed year file present: {name}')
        mon = month_of(meta['label'])
        if mon is None or first[(meta['year'], mon)] != name:  # keep one snapshot per month (the earliest)
            continue
        try:
            z = zipfile.ZipFile(Path(raw_dir) / name)
        except zipfile.BadZipFile:
            print('SKIP damaged archive', name, flush=True)
            continue
        inner = next(n for n in z.namelist() if n.lower().endswith('.xlsx'))
        try:
            agg = cotton_rows(read_xlsx(io.BytesIO(z.read(inner))))
        except ValueError:
            print('SKIP unknown layout', name, flush=True)
            continue
        for (st, co, irr), (p, f) in sorted(agg.items()):
            lines.append(f"{meta['year']},{mon},\"{meta['label']}\",{st},{co},{irr},{p:.2f},{f:.2f}")
        print(meta['year'], mon, len(agg), round(sum(v[0] for v in agg.values())), round(sum(v[1] for v in agg.values())), flush=True)
    Path(out_csv).write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
