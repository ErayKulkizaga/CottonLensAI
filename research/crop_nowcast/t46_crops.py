"""T46: the frozen revision-momentum rule (momentum.py, unchanged) applied to USDA corn-for-grain and soybean forecasts.

Pre-declared in docs/T46_CORN_SOY_PREREG_20261116.md. Subcommands:
  parse  raw_dir root          2000-2011 Sep-Nov (and Aug) forecasts from cached Crop Production text reports
  pull   root first last       QuickStats state/US production (forecasts + finals), key from NASS_API_KEY
  check  root                  parser vs QuickStats on the overlap years 2012-2013 (counts only, no values printed)
  build  root                  merge into <root>/<crop>/nass/{state,us}_production.json
  score  root out.json         run the rule once and write the result
"""
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CROPS = {
    'corn': dict(title=r'Corn for Grain:?\s+Area Harvested, Yield, and Production',
                 qs=dict(commodity_desc='CORN', class_desc='ALL CLASSES', util_practice_desc='GRAIN',
                         statisticcat_desc='PRODUCTION', unit_desc='BU')),
    'soy': dict(title=r'Soybeans for Beans:?\s+Area Harvested, Yield, and Production',
                qs=dict(commodity_desc='SOYBEANS', class_desc='ALL CLASSES', util_practice_desc='ALL UTILIZATION PRACTICES',
                        statisticcat_desc='PRODUCTION', unit_desc='BU')),
}
SCORED = ('SEP', 'OCT', 'NOV')
MON = {8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV'}
ST = dict(zip(
    'Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana '
    'Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska '
    'Nevada NewHampshire NewJersey NewMexico NewYork NorthCarolina NorthDakota Ohio Oklahoma Oregon Pennsylvania RhodeIsland '
    'SouthCarolina SouthDakota Tennessee Texas Utah Vermont Virginia Washington WestVirginia Wisconsin Wyoming'.split(),
    'AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI '
    'SC SD TN TX UT VT VA WA WV WI WY'.split()))
ABBR = set(ST.values())


GROUPED = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?'


def last_value(nums):
    """Last column = current production. Amendment 2: wide numbers can fuse with the previous column
    ('10,089,22210,917,000'); then split so both halves are valid groupings and the right half is closest to
    current harvested area (nums[1]) x current yield (nums[-2])."""
    tok = nums[-1]
    if re.fullmatch(GROUPED, tok):
        return float(tok.replace(',', ''))
    target = float(nums[1].replace(',', '')) * float(nums[-2].replace(',', ''))
    cands = [float(tok[i:].replace(',', '')) for i in range(1, len(tok))
             if re.fullmatch(GROUPED, tok[:i]) and re.fullmatch(GROUPED, tok[i:])]
    if not cands:
        raise ValueError(f'unsplittable fused number {tok!r}')
    return min(cands, key=lambda c: abs(c - target))


def parse_table(text, title):
    """-> (release date, {state or 'US': bushels}) or None. Last number of each row = current production (1,000 bu)."""
    r = re.search(r'Released ([A-Za-z]+ \d{1,2}, \d{4})', text)
    if not r:
        return None
    rel = dt.datetime.strptime(r.group(1), '%B %d, %Y').date()
    for t in re.finditer(title, text):  # skip table-of-contents hits: the real table has a 'State' header soon after
        head = text[t.end():t.end() + 1500]
        if not re.search(r'^\s*State\s*:', head, re.M):
            continue
        out = {}
        for line in text[t.end():].splitlines()[1:]:
            m = re.match(r'^\s*([A-Za-z .]+?)\s*\.*\s*:(.*)$', line)  # 'IA      :' (to 2008) or 'Iowa .....:' (2009+)
            if not m:
                continue
            name = m.group(1).strip().rstrip('.').strip()
            st = 'US' if name in ('US', 'United States') else ST.get(name.replace(' ', ''), name if name in ABBR else None)
            nums = [x for x in m.group(2).split() if re.fullmatch(r'[\d,]+(?:\.\d+)?', x)]
            if st and nums:
                out[st] = round(last_value(nums) * 1000)
            if st == 'US':
                return rel, out
        return None
    return None


def cmd_parse(raw_dir, root):
    rows = {c: {'state': [], 'us': []} for c in CROPS}
    for p in sorted(Path(raw_dir).glob('CropProd-*.txt')):
        text = p.read_bytes().decode('latin-1')
        for c, cfg in CROPS.items():
            res = parse_table(text, cfg['title'])
            if not res or res[0].month not in MON or not 2000 <= res[0].year <= 2013:
                continue
            rel, vals = res
            for st, v in vals.items():
                rec = {'year': str(rel.year), 'reference_period_desc': f'YEAR - {MON[rel.month]} FORECAST', 'Value': f'{v:,}',
                       'prodn_practice_desc': 'ALL PRODUCTION PRACTICES', 'release': str(rel), 'state_alpha': st,
                       'source': p.name}
                rows[c]['us' if st == 'US' else 'state'].append(rec)
    for c, r in rows.items():
        d = Path(root) / c / 'report_forecasts'
        d.mkdir(parents=True, exist_ok=True)
        for lvl, x in r.items():
            (d / f'{lvl}_forecasts.json').write_text(json.dumps({'data': x}), encoding='utf-8')
        yrs = sorted({int(x['year']) for x in r['us']})
        print(c, 'state rows', len(r['state']), 'us rows', len(r['us']), 'years', yrs[0] if yrs else None, '-', yrs[-1] if yrs else None)


def cmd_pull(root, first, last):
    import hashlib
    import time
    import urllib.error

    from research.crop_nowcast.nass_pull import API, fetch
    key = os.environ.get('NASS_API_KEY')
    if not key:
        raise SystemExit('NASS_API_KEY not set')
    for c, cfg in CROPS.items():
        d = Path(root) / c / 'quickstats'
        d.mkdir(parents=True, exist_ok=True)
        receipt = {'api': API, 'params': cfg['qs'], 'years': [int(first), int(last)], 'files': {}}
        for lvl, agg in (('state', 'STATE'), ('us', 'NATIONAL')):
            rows, hashes = [], {}
            for y in range(int(first), int(last) + 1):
                try:
                    raw = fetch(dict(cfg['qs'], agg_level_desc=agg, domain_desc='TOTAL', source_desc='SURVEY'), key, y)
                except urllib.error.HTTPError as exc:
                    if exc.code == 400:
                        hashes[y] = None
                        continue
                    raise
                part = json.loads(raw).get('data', [])
                if any(int(r['year']) != y for r in part):
                    raise SystemExit(f'{c} {lvl} {y}: other years in response')
                rows += part
                hashes[y] = hashlib.sha256(raw).hexdigest()
            body = json.dumps({'data': rows}, separators=(',', ':')).encode()
            (d / f'{lvl}_production.json').write_bytes(body)
            receipt['files'][lvl] = {'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest(), 'per_year': hashes,
                                     'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            print(c, lvl, len(rows), flush=True)
        text = json.dumps(receipt, indent=1)
        if key in text:
            raise SystemExit('key leaked')
        (d / 'receipt.json').write_text(text, encoding='utf-8')


def _qs(root, c, lvl):
    from research.crop_nowcast.state_model import nass_state
    return nass_state(Path(root) / c / 'quickstats', f'{lvl}_production')


def _rep(root, c, lvl):
    import pandas as pd
    d = pd.DataFrame(json.loads((Path(root) / c / 'report_forecasts' / f'{lvl}_forecasts.json').read_text())['data'])
    d['v'] = pd.to_numeric(d.Value.str.replace(',', ''))
    d['year'] = d.year.astype(int)
    return d


def cmd_check(root):
    """Gate: on 2012-2013, >= 98% of report cells must equal QuickStats exactly and the median ratio must be 1."""
    ok = True
    for c in CROPS:
        for lvl in ('state', 'us'):
            q, r = _qs(root, c, lvl), _rep(root, c, lvl)
            q = q[q.reference_period_desc.str.endswith('FORECAST') & q.year.between(2012, 2013)]
            r = r[r.year.between(2012, 2013)]
            k = ['year', 'reference_period_desc'] + (['state_alpha'] if lvl == 'state' else [])
            j = r[k + ['v']].merge(q[k + ['v']], on=k, suffixes=('_rep', '_qs'))
            eq = int((j.v_rep == j.v_qs).sum())
            share = eq / max(1, len(j))
            ratio = float((j.v_rep / j.v_qs).median()) if len(j) else float('nan')
            good = len(j) > 0 and share >= 0.98 and abs(ratio - 1) < 1e-9
            ok &= good
            print(f'{c} {lvl}: matched cells {len(j)}, equal {eq} ({100 * share:.1f}%), report-only {len(r) - len(j)}, '
                  f'median ratio ok={abs(ratio - 1) < 1e-9}  {"PASS" if good else "FAIL"}')
    print('CHECK', 'PASS' if ok else 'FAIL')
    return ok


def cmd_build(root):
    import pandas as pd
    for c in CROPS:
        out = Path(root) / c / 'nass'
        out.mkdir(parents=True, exist_ok=True)
        for lvl in ('state', 'us'):
            q = json.loads((Path(root) / c / 'quickstats' / f'{lvl}_production.json').read_text())['data']
            if lvl == 'state':  # amendment 1: drop 'OT' (Other States) aggregate rows, final = 0 from 2015 -> log(0)
                q = [x for x in q if x.get('state_alpha') in ABBR]
            q = [x for x in q if x['reference_period_desc'] == 'YEAR'
                 or (x['reference_period_desc'].endswith('FORECAST') and int(x['year']) >= 2012)]
            rep = [x for x in json.loads((Path(root) / c / 'report_forecasts' / f'{lvl}_forecasts.json').read_text())['data']
                   if int(x['year']) <= 2011]
            (out / f'{lvl}_production.json').write_text(json.dumps({'data': q + rep}), encoding='utf-8')
            n = pd.DataFrame(q + rep)
            print(c, lvl, 'final rows', int((n.reference_period_desc == 'YEAR').sum()),
                  'forecast rows', int(n.reference_period_desc.str.endswith('FORECAST').sum()))


def perm_p(diff, n=20000, seed=42):
    """One-sided sign-flip p for mean(diff) > 0 (diff = |USDA error| - |rule error|)."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(n, len(diff)))
    return float(((s * diff).mean(1) >= diff.mean() - 1e-12).mean())


def cmd_score(root, out_json):
    os.environ.setdefault('COTTONLENS_MOMENTUM_YEARS', '2002-2025')
    os.environ.setdefault('COTTONLENS_MOMENTUM_FIRST_TRAIN', '2000')
    from research.crop_nowcast import momentum as M
    from research.crop_nowcast.stack import state_w
    res, pooled = {}, {}
    for c in CROPS:
        nd = Path(root) / c / 'nass'
        res[c] = {}
        for m in SCORED:
            s = M.predict(M.with_revision(M.frame(nd, 'state'), m)).dropna(subset=['final'])
            us = M.frame(nd, 'us').set_index('year')[[m, 'final']].rename(columns={m: 'usda_us'})
            j = s.groupby('year')[['corr', 'usda']].sum().join(us).dropna()
            j['us_hat'] = j.usda_us * j['corr'] / j.usda
            e_rule, e_usda = 100 * (j.us_hat / j.final - 1), 100 * (j.usda_us / j.final - 1)
            diff = (e_usda.abs() - e_rule.abs()).to_numpy()
            r = {'US_by_year': {int(y): [round(a, 2), round(b, 2)] for y, a, b in zip(j.index, e_rule, e_usda)},
                 'US': float(e_rule.abs().mean()), 'US_usda': float(e_usda.abs().mean()),
                 'US_better': int((diff > 0).sum()), 'US_worse': int((diff < 0).sum()),
                 'state_w': state_w(s, 'corr'), 'state_w_usda': state_w(s, 'usda'), 'n_years': len(j),
                 'perm_p_one_sided': perm_p(diff)}
            r['cell_pass'] = bool(r['US'] < r['US_usda'] and r['state_w'] < r['state_w_usda'])
            res[c][m] = r
            pooled[(c, m)] = dict(zip(j.index.astype(int), diff))
            print(f"{c} {m}: US {r['US']:.2f} vs USDA {r['US_usda']:.2f} (better {r['US_better']} worse {r['US_worse']}, "
                  f"p {r['perm_p_one_sided']:.3f}) | state_w {r['state_w']:.2f} vs {r['state_w_usda']:.2f}  cell_pass={r['cell_pass']}")
        res[c]['crop_pass'] = sum(res[c][m]['cell_pass'] for m in SCORED) >= 2
    # year-block permutation: one sign per year, shared by all six cells (cells of a year are correlated)
    years = sorted(set().union(*[set(v) for v in pooled.values()]))
    mat = np.array([[pooled[k].get(y, np.nan) for k in pooled] for y in years])
    obs = np.nanmean(mat)
    rng = np.random.default_rng(42)
    sims = np.array([np.nanmean(mat * rng.choice([-1.0, 1.0], size=(len(years), 1))) for _ in range(20000)])
    res['pooled_year_block_p'] = float((sims >= obs - 1e-12).mean())
    res['pooled_mean_gain_pts'] = float(obs)
    res['primary_pass'] = bool(res['corn']['crop_pass'] and res['soy']['crop_pass'])
    res['secondary_pass'] = res['pooled_year_block_p'] < 0.05
    print('corn crop_pass', res['corn']['crop_pass'], '| soy crop_pass', res['soy']['crop_pass'])
    print(f"PRIMARY {res['primary_pass']} | pooled mean gain {obs:.3f} pts, year-block p {res['pooled_year_block_p']:.4f} "
          f"-> SECONDARY {res['secondary_pass']}")
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'parse': cmd_parse, 'pull': cmd_pull, 'check': cmd_check, 'build': cmd_build, 'score': cmd_score}[cmd](*a)
