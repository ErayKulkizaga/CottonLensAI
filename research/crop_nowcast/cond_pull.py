"""NASS weekly cotton condition by state from ESMIS Crop Progress text reports (dev years only).

Each report is released on the Monday after its 'Week Ending' Sunday; a week's values are
usable only from that release date. Rows: week_ending, release, state, vp, p, f, g, e (percent).
Usage: python cond_pull.py raw_dir out.csv [first last]
"""
import datetime as dt
import re
import sys
import time
import urllib.request
from pathlib import Path

SEALED_FROM = 2022
BASE = 'https://esmis.nal.usda.gov'
STATES = {'Alabama': 'AL', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Georgia': 'GA', 'Kansas': 'KS',
          'Louisiana': 'LA', 'Mississippi': 'MS', 'Missouri': 'MO', 'New Mexico': 'NM', 'North Carolina': 'NC',
          'Oklahoma': 'OK', 'South Carolina': 'SC', 'Tennessee': 'TN', 'Texas': 'TX', 'Virginia': 'VA', 'Florida': 'FL'}


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 research-client'})
            return urllib.request.urlopen(req, timeout=120).read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def parse(text):
    """Return (week_ending date, {state: (vp, p, f, g, e)}) for the cotton condition table, or None."""
    m = re.search(r'Cotton Condition - Selected States: Week Ending ([A-Za-z]+ \d{1,2}, \d{4})', text)
    if not m:
        return None
    week = dt.datetime.strptime(m.group(1), "%B %d, %Y").replace(tzinfo=dt.timezone.utc).date()
    out = {}
    for line in text[m.end():m.end() + 4000].splitlines():
        r = re.match(r'^([A-Za-z ]+?)\s*\.+:\s*(.+)$', line)
        if not r:
            if out and line.startswith('-'):
                break
            continue
        name = r.group(1).strip()
        if name not in STATES:
            continue
        vals = [0.0 if v == '-' else float(v) for v in r.group(2).split()[:5]]
        if len(vals) == 5:
            out[STATES[name]] = vals
    return week, out


def main(raw_dir, out_csv, first=2011, last=2021):
    first, last = int(first), int(last)
    if last >= SEALED_FROM:
        raise SystemExit('Sealed years need a pushed preregistration')
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    lines = ['week_ending,release,state,vp,p,f,g,e']
    for y in range(first, last + 1):
        for mon in range(5, 11):
            page = get(f'{BASE}/publication/crop-progress?date={y}-{mon:02d}').decode('utf-8', 'ignore')
            for href in sorted(set(re.findall(r'href="(/sites/default/release-files/[^"]+\.txt)"', page))):
                dest = raw / href.split('/')[-1]
                if not dest.exists():
                    dest.write_bytes(get(BASE + href))
                res = parse(dest.read_bytes().decode('latin-1'))
                if not res or res[0].year != y:
                    continue
                week, rows = res
                for st, v in rows.items():
                    lines.append(f"{week},{week + dt.timedelta(days=1)},{st}," + ','.join(f'{x:g}' for x in v))
            print(y, mon, flush=True)
    uniq = [lines[0], *sorted(set(lines[1:]))]
    Path(out_csv).write_text('\n'.join(uniq) + '\n', encoding='utf-8')
    print('rows', len(uniq) - 1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
