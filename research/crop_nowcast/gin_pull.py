"""NASS Cotton Ginnings (ESMIS text reports): all-cotton running bales ginned by state, current crop (dev years only).

Rows: asof, release, state, bales (empty = withheld (D)). Values are as first published in that report.
Usage: python gin_pull.py raw_dir out.csv [first last]
"""
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.cond_pull import BASE, STATES, get

SEALED_FROM = 2022


def parse(text):
    """-> (asof date, release date, {state: bales or None}) from the 'All' block of the state table, or None."""
    r = re.search(r'Released ([A-Za-z]+ \d{1,2}, \d{4})', text)
    new = re.search(r'Running Bales Ginned by Crop - States and United States: ([A-Za-z]+ \d{1,2}), (?:\d{4}-)?(\d{4})', text)
    old = re.search(r'As of ([A-Za-z]+ \d{1,2}), Crop Years \d{4}-(\d{4})', text)  # 2009 layout, state abbreviations
    t = new or old
    if not t or not r:
        return None
    asof = dt.datetime.strptime(f'{t.group(1)}, {t.group(2)}', '%B %d, %Y').replace(tzinfo=dt.timezone.utc).date()
    rel = dt.datetime.strptime(r.group(1), '%B %d, %Y').replace(tzinfo=dt.timezone.utc).date()
    block = text[t.end():]
    block = block[block.index('\nAll '):]
    block = block[:block.index('\n---' if new else '\nUS ')]
    abbr = set(STATES.values())
    out = {}
    for line in block.splitlines():
        m = re.match(r'^([A-Za-z ]+?)\s*\.*:\s*(.+)$', line)
        name = m.group(1).strip() if m else None
        st = STATES.get(name) or (name if name in abbr else None)
        if st:
            last = m.group(2).split()[-1]  # current crop is the last column; (D) or 1/ = withheld
            out[st] = int(last.replace(',', '')) if re.fullmatch(r'[\d,]+', last) else None
    return asof, rel, out


def main(raw_dir, out_csv, first=2009, last=2021):
    first, last = int(first), int(last)
    if last >= SEALED_FROM and '--unseal' not in sys.argv:  # T39 prereg pushed
        raise SystemExit('Sealed years need a pushed preregistration')
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    lines = ['asof,release,state,bales']
    for y in range(first, last + 1):
        for mon in (11, 12):
            page = get(f'{BASE}/publication/cotton-ginnings?date={y}-{mon:02d}').decode('utf-8', 'ignore')
            for href in sorted(set(re.findall(r'href="(/sites/default/release-files/[^"]+\.txt)"', page))):
                dest = raw / href.split('/')[-1]
                if not dest.exists():
                    dest.write_bytes(get(BASE + href))
                res = parse(dest.read_bytes().decode('latin-1'))
                if not res or res[0].year != y:
                    continue
                asof, rel, rows = res
                lines += [f"{asof},{rel},{st},{'' if v is None else v}" for st, v in rows.items()]
        print(y, flush=True)
    uniq = [lines[0], *sorted(set(lines[1:]))]
    Path(out_csv).write_text('\n'.join(uniq) + '\n', encoding='utf-8')
    print('rows', len(uniq) - 1)


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
