"""T45: USDA upland cotton production forecasts (Aug-Dec) by state and US from ESMIS Crop Production text reports.

Takes the last number of each state line in the "Upland" block of the table "Cotton: Area Harvested, Yield, and
Production by Type, State, and United States" (= the current forecast; reports give 1,000 bales, stored as bales). Output rows mimic NASS
QuickStats so state_model.nass_state reads them: reference_period_desc = 'YEAR - <MON> FORECAST'.
Usage: python cp_pull.py raw_dir out_dir first last
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.cond_pull import BASE, STATES, get

ABBR = set(STATES.values())
MON = {8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


def parse(text):
    """-> (release date, {state or 'US': bales}) or None."""
    r = re.search(r'Released ([A-Za-z]+ \d{1,2}, \d{4})', text)
    t = re.search(r'Cotton:?\s+Area Harvested, Yield, and Production by Type', text)
    if not r or not t:
        return None
    rel = dt.datetime.strptime(r.group(1), '%B %d, %Y').replace(tzinfo=dt.timezone.utc).date()
    block = text[t.end():]
    block = block[block.index('\nUpland'):]
    out = {}
    for line in block.splitlines()[1:]:
        if line.startswith(('Amer', 'All')):
            break
        m = re.match(r'^\s*([A-Za-z ]+?)\s*\.*\s*:(.*)$', line)  # 'AL     :' (to 2008) or 'Alabama ....:' (2009+)
        if not m:
            continue
        name = m.group(1).strip()
        st = 'US' if name in ('US', 'United States') else STATES.get(name, name if name in ABBR else None)
        nums = [x for x in m.group(2).split() if re.fullmatch(r'[\d,]+(?:\.\d+)?', x)]
        if st and nums:
            out[st] = round(float(nums[-1].replace(',', '')) * 1000)  # reports give 1,000 bales, e.g. '2,160.0'
    return rel, out


def main(raw_dir, out_dir, first, last):
    raw, out = Path(raw_dir), Path(out_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = {'state': [], 'us': []}
    for y in range(int(first), int(last) + 1):
        for mon, mon_name in MON.items():
            page = get(f'{BASE}/publication/crop-production?date={y}-{mon:02d}').decode('utf-8', 'ignore')
            for href in sorted(set(re.findall(r'href="(/sites/default/release-files/[^"]+\.txt)"', page))):
                if '/796144/' in href:  # current-report placeholder linked on every page
                    continue
                dest = raw / href.split('/')[-1]
                if not dest.exists():
                    dest.write_bytes(get(BASE + href))
                res = parse(dest.read_bytes().decode('latin-1'))
                if not res or res[0].year != y or res[0].month != mon:
                    continue
                rel, vals = res
                for st, v in vals.items():
                    rec = {'year': str(y), 'reference_period_desc': f'YEAR - {mon_name} FORECAST', 'Value': f'{v:,}',
                           'prodn_practice_desc': 'ALL PRODUCTION PRACTICES', 'release': str(rel)}
                    rows['us' if st == 'US' else 'state'].append({**rec, 'state_alpha': st})
        print(y, len(rows['state']), len(rows['us']), flush=True)
    for k, r in rows.items():
        (out / f'{k}_forecasts.json').write_text(json.dumps({'data': r}), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:5])
