"""Index of every cached input file behind the Phase 2 results: source, SHA-256, size, retrieval time, timing rule.

Writes docs/research_package/data/data_versions.csv. Retrieval times come from the pull receipts where they exist,
otherwise from the file's modification time (marked 'mtime').
Usage: python research/package/data_versions.py
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / 'output'
RULES = {
    'nass': 'NASS QuickStats. Monthly forecasts are used only for their own report month; finals (YEAR) are used only as targets.',
    'report': 'USDA Crop Production text report (ESMIS). A value is usable from its release date; the release date is parsed from the report.',
    'ocr': 'Scanned USDA Crop Production report, OCR on Kaggle; rows gated by production = area x yield.',
}
SETS = [
    ('crop-nowcast-full/nass', 'nass', 'Cotton upland state/US production, 2000-2025 (merged: reports 2000-2011 + QuickStats)'),
    ('crop-nowcast-2026/nass', 'nass', 'Cotton 2026 forecasts (forward lock input)'),
    ('crop-nowcast-t39/nass', 'nass', 'Cotton QuickStats 2008-2025: production, harvested area, yield, planted, county'),
    ('crop-nowcast-t45/cp_raw', 'report', 'Crop Production text reports 2000-2013'),
    ('crop-nowcast-t47a/cp_raw', 'report', 'Crop Production text reports 1995-1999'),
    ('crop-nowcast-t46/corn/quickstats', 'nass', 'Corn for grain QuickStats 2000-2025'),
    ('crop-nowcast-t46/soy/quickstats', 'nass', 'Soybeans QuickStats 2000-2025'),
    ('crop-nowcast-t47a/finals', 'nass', 'Cotton upland finals 1985-1999'),
    ('crop-nowcast-t48', 'nass', 'Cotton area/yield forecasts parsed 1995-2011 and QuickStats finals'),
]


def receipt_times(folder):
    out = {}
    for r in folder.glob('receipt.json'):
        d = json.loads(r.read_text(encoding='utf-8'))
        for k, v in {**d.get('queries', {}), **d.get('files', {})}.items():
            if isinstance(v, dict) and v.get('retrieved_utc'):
                out[k] = v['retrieved_utc']
        for k, v in d.items():
            if isinstance(v, dict) and v.get('retrieved_utc'):
                out[k] = v['retrieved_utc']
    return out


def main():
    rows = []
    for rel, kind, desc in SETS:
        folder = O / rel
        if not folder.exists():
            rows.append([rel, '(missing locally)', '', '', '', kind, desc, RULES[kind]])
            continue
        times = receipt_times(folder)
        for p in sorted(x for x in folder.rglob('*') if x.is_file()):
            stem = p.stem.replace('_production', '').replace('_forecasts', '')
            t = times.get(p.stem) or times.get(stem)
            when = t or datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ') + ' (mtime)'
            rows.append([rel, p.relative_to(folder).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size,
                         when, kind, desc, RULES[kind]])
    out = ROOT / 'docs/research_package/data/data_versions.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'file', 'sha256', 'bytes', 'retrieved_utc', 'kind', 'description', 'timing_rule'])
        w.writerows(rows)
    print(out.name, len(rows), 'files')


if __name__ == '__main__':
    main()
