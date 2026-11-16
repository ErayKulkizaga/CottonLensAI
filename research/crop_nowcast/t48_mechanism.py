"""T48: confirm the error mechanism found on 2012-2025 on 1995-2011 (preregistered: docs/T48_MECHANISM_PREREG_20261116.md).

  parse  root            area / yield / production forecasts from cached 1995-2011 Crop Production text reports
  finals root            QuickStats upland harvested area and yield finals 1995-2011 ('YEAR' rows), key from NASS_API_KEY
  score  root out.json   the four pre-declared hypotheses
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.cond_pull import STATES
from research.crop_nowcast.t47_blind_old import clean

RAW = [ROOT / 'output/crop-nowcast-t47a/cp_raw', ROOT / 'output/crop-nowcast-t45/cp_raw']
MON = {8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
MONTHS = list(MON.values())
ABBR = set(STATES.values())


def rows(text):
    """Upland block -> {state: (area_cur 1000 ac, yield_cur lb, prod_cur 1000 bales)}; row kept only if
    prod x 480 / (area x yield) is within 3% of 1 (same check as T47 amendment 2)."""
    r = re.search(r'Released ([A-Za-z]+) (\d{1,2}), (\d{4})', text)
    t = re.search(r'Cotton:?\s+Area Harvested, Yield, and Production by Type', text)
    if not r or not t:
        return None
    import datetime as dt
    rel = dt.datetime.strptime(' '.join(r.groups()), '%B %d %Y').date()
    block = text[t.end():]
    block = block[block.index('\nUpland'):]
    out = {}
    for line in block.splitlines()[1:]:
        if line.startswith(('Amer', 'All')):
            break
        m = re.match(r'^\s*([A-Za-z ]+?)\s*\.*\s*:(.*)$', line)
        if not m:
            continue
        name = m.group(1).strip()
        st = 'US' if name in ('US', 'United States') else STATES.get(name, name if name in ABBR else None)
        n = [float(x.replace(',', '')) for x in m.group(2).split() if re.fullmatch(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', x)]
        if not st or len(n) < 6:
            continue
        a, p = n[1], n[-1]
        # current yield = the yield column that satisfies the check (Aug has no previous-month yield column);
        # scan from the right so a later month's yield wins over the previous year's
        for y in reversed(n[2:-2]):
            if a > 0 and y > 0 and abs(p * 480 / (a * y) - 1) < 0.03:
                out[st] = (a, y, p)
                break
    return rel, out


def cmd_parse(root):
    recs, seen = [], set()
    for d in RAW:
        for p in sorted(d.glob('CropProd-*.txt')):
            res = rows(clean(p.read_bytes().decode('latin-1')).replace('Yield, and, Production', 'Yield, and Production'))
            if not res or res[0].month not in MON or not 1995 <= res[0].year <= 2011:
                continue
            rel, o = res
            key = (rel.year, MON[rel.month])
            if key in seen:
                continue
            seen.add(key)
            for st, (a, y, pr) in o.items():
                recs.append({'state': st, 'year': rel.year, 'month': MON[rel.month], 'area': a * 1000, 'yield': y,
                             'prod': pr * 1000, 'source': p.name})
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(out / 'forecasts_area_yield.csv', index=False)
    f = pd.DataFrame(recs)
    print('reports', len(seen), 'rows', len(f), 'TX rows', int((f.state == 'TX').sum()), 'US rows', int((f.state == 'US').sum()))


def cmd_finals(root):
    from research.crop_nowcast.nass_pull import fetch
    key = __import__('research.crop_nowcast.nass_pull', fromlist=['nass_key']).nass_key()
    out = []
    for stat, unit in (('AREA HARVESTED', 'ACRES'), ('YIELD', 'LB / ACRE'), ('PRODUCTION', '480 LB BALES')):
        for agg in ('STATE', 'NATIONAL'):
            for y in range(1995, 2012):
                d = json.loads(fetch(dict(agg_level_desc=agg, statisticcat_desc=stat, unit_desc=unit,
                                          reference_period_desc='YEAR'), key, y))['data']
                for r in d:
                    if r['reference_period_desc'] == 'YEAR' and r.get('prodn_practice_desc') == 'ALL PRODUCTION PRACTICES':
                        out.append({'state': 'US' if agg == 'NATIONAL' else r['state_alpha'], 'year': y, 'stat': stat,
                                    'v': float(r['Value'].replace(',', ''))})
    pd.DataFrame(out).to_csv(Path(root) / 'finals_area_yield.csv', index=False)
    print('final rows', len(out))


def cmd_score(root, out_json):
    f = pd.read_csv(Path(root) / 'forecasts_area_yield.csv')
    fin = pd.read_csv(Path(root) / 'finals_area_yield.csv').pivot_table(index=['state', 'year'], columns='stat', values='v')
    fin.columns = ['area_f', 'prod_f', 'yield_f']  # AREA HARVESTED, PRODUCTION, YIELD (alphabetical)
    x = f.join(fin, on=['state', 'year']).dropna(subset=['area_f', 'yield_f', 'prod_f'])
    x['area_e'] = np.log(x.area_f / x.area)
    x['yield_e'] = np.log(x.yield_f / x['yield'])
    x['total'] = np.log(x.prod_f / x['prod'])
    p = x.pivot_table(index=['state', 'year'], columns='month', values='prod')
    rev = {}
    for i, m in enumerate(MONTHS[1:], 1):
        prev = p[MONTHS[i - 1]]
        for k in range(i - 2, -1, -1):
            prev = prev.fillna(p[MONTHS[k]])
        rev[m] = np.log(p[m] / prev)
    x['r'] = [rev[m].get((s, y), np.nan) if m in rev else np.nan for s, y, m in zip(x.state, x.year, x.month)]
    res = {}
    for reg in ('TX', 'US'):
        for m in MONTHS:
            g = x[(x.state == reg) & (x.month == m)]
            v = g.total.var()
            s = {'n': len(g), 'mae_area': float(100 * g.area_e.abs().mean()), 'mae_yield': float(100 * g.yield_e.abs().mean()),
                 'mae_total': float(100 * g.total.abs().mean()),
                 'var_share_area': float(np.cov(g.area_e, g.total)[0, 1] / v) if len(g) > 2 else np.nan,
                 'var_share_yield': float(np.cov(g.yield_e, g.total)[0, 1] / v) if len(g) > 2 else np.nan}
            gr = g.dropna(subset=['r'])
            if len(gr) > 4:
                s['corr_r_area'] = float(np.corrcoef(gr.r, gr.area_e)[0, 1])
                s['corr_r_yield'] = float(np.corrcoef(gr.r, gr.yield_e)[0, 1])
            res[f'{reg}_{m}'] = s
            print(reg, m, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
    h1 = sum(res[f'TX_{m}']['var_share_area'] > 0.5 for m in ('SEP', 'OCT', 'NOV'))
    h2 = res['TX_DEC']['mae_area'] >= 0.75 * res['TX_AUG']['mae_area']
    h3 = sum(res[f'TX_{m}'].get('corr_r_area', -9) > res[f'TX_{m}'].get('corr_r_yield', 9) for m in ('SEP', 'OCT', 'NOV', 'DEC'))
    h4 = sum(res[f'US_{m}']['var_share_yield'] > 0.5 for m in ('OCT', 'NOV', 'DEC'))
    res['H1_TX_area_dominates_SepOctNov'] = {'count': int(h1), 'pass': bool(h1 >= 2)}
    res['H2_TX_area_error_not_learned'] = {'dec_over_aug': res['TX_DEC']['mae_area'] / res['TX_AUG']['mae_area'], 'pass': bool(h2)}
    res['H3_TX_momentum_in_area'] = {'count': int(h3), 'pass': bool(h3 >= 3)}
    res['H4_US_yield_dominates_OctNovDec'] = {'count': int(h4), 'pass': bool(h4 >= 2)}
    res['all_pass'] = bool(h1 >= 2 and h2 and h3 >= 3 and h4 >= 2)
    for k in [k for k in res if k.startswith('H')]:
        print(k, res[k])
    print('ALL PASS', res['all_pass'])
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'parse': cmd_parse, 'finals': cmd_finals, 'score': cmd_score}[cmd](*a)
