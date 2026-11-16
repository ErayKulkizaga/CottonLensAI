"""T53 (docs/T53_AREA_PRODUCTION_LONG_PREREG_20261116.md): Texas area correction -> production, longest history.

  pull  root        Texas cotton condition 1986-1999 and Texas harvested-area finals 1985-1994 (QuickStats)
  run   root out    build the 1985-2025 Texas frame and score exactly as T51-C
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M
from research.crop_nowcast.nass_pull import fetch, nass_key
from research.crop_nowcast.state_model import nass_state
from research.crop_nowcast.t47_blind_old import ocr_rows
from research.crop_nowcast.t49_anchoring import pvp_at, qs
from research.crop_nowcast.t51_area_production import block_p, ols

O = ROOT / 'output'
MONTHS = ('SEP', 'OCT', 'NOV', 'DEC')
NUM = {9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}


def cmd_pull(root):
    key = nass_key()
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    cond = []
    for y in range(1986, 2000):
        for unit in ('PCT POOR', 'PCT VERY POOR'):
            for r in qs(key, commodity_desc='COTTON', statisticcat_desc='CONDITION', unit_desc=unit, agg_level_desc='STATE',
                        state_alpha='TX', year=str(y)):
                if r.get('week_ending') and r['Value'].strip().replace('.', '').isdigit():
                    cond.append({'crop': 'COTTON', 'state': 'TX', 'year': y, 'unit': unit, 'week_ending': r['week_ending'],
                                 'v': float(r['Value'])})
    pd.DataFrame(cond).to_csv(out / 'tx_condition_1986_1999.csv', index=False)
    fin = []
    for y in range(1985, 1995):
        d = json.loads(fetch(dict(agg_level_desc='STATE', statisticcat_desc='AREA HARVESTED', unit_desc='ACRES',
                                  reference_period_desc='YEAR'), key, y))['data']
        fin += [{'year': y, 'v': float(r['Value'].replace(',', ''))} for r in d
                if r.get('state_alpha') == 'TX' and r.get('prodn_practice_desc') == 'ALL PRODUCTION PRACTICES']
    pd.DataFrame(fin).to_csv(out / 'tx_area_final_1985_1994.csv', index=False)
    print('condition rows', len(cond), 'area finals', len(fin))


def frame(root):
    gate = set(json.loads((O / 'crop-nowcast-t47b/gate.json').read_text())['pass'])
    rows = []
    for p in sorted((O / 'crop-nowcast-t47b/ocr').glob('CropProd-*.txt')):
        mm, dd, yy = re.match(r'CropProd-(\d\d)-(\d\d)-(\d{4})', p.name).groups()
        mon = NUM.get(int(mm))
        if not mon or f'{yy} {mon}' not in gate:
            continue
        o = ocr_rows(p.read_text(encoding='utf-8'))
        if 'TX' in o:
            rows.append({'year': int(yy), 'month': mon, 'area': o['TX'][0] * 1000})
    old = pd.read_csv(O / 'crop-nowcast-t48/forecasts_area_yield.csv')
    old = old[(old.state == 'TX') & old.month.isin(MONTHS)][['year', 'month', 'area']]
    h = nass_state(O / 'crop-nowcast-t39/nass', 'state_harvested')
    h = h[h.state_alpha == 'TX']
    new = h[h.reference_period_desc.str.endswith('FORECAST')].assign(month=lambda x: x.reference_period_desc.str[7:10])
    new = new[new.month.isin(MONTHS) & (new.year >= 2012)][['year', 'month', 'v']].rename(columns={'v': 'area'})
    a = pd.concat([pd.DataFrame(rows), old[old.year <= 2011], new])
    f85 = pd.read_csv(Path(root) / 'tx_area_final_1985_1994.csv').set_index('year').v
    fo = pd.read_csv(O / 'crop-nowcast-t48/finals_area_yield.csv')
    fo = fo[(fo.state == 'TX') & (fo.stat == 'AREA HARVESTED')].set_index('year').v
    fn = h[h.reference_period_desc == 'YEAR'].set_index('year').v
    fin = pd.concat([f85, fo[fo.index <= 2011], fn[fn.index >= 2012]])
    a['final'] = a.year.map(fin)
    cond = pd.concat([pd.read_csv(Path(root) / 'tx_condition_1986_1999.csv'), pd.read_csv(O / 't49/condition.csv')])
    a['pvp'] = [pvp_at(cond, 'COTTON', 'TX', y, {'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}[m]) for y, m in zip(a.year, a.month)]
    a = a.dropna(subset=['area', 'final', 'pvp']).drop_duplicates(['year', 'month'])
    a['e'] = np.log(a.final / a.area)
    p1 = M.frame(O / 'crop-nowcast-t47ab/nass', 'state')
    p2 = M.frame(O / 'crop-nowcast-full/nass', 'state')
    prod = pd.concat([p1[(p1.state == 'TX') & (p1.year <= 1999)], p2[(p2.state == 'TX') & (p2.year >= 2000)]]).set_index('year')
    return a, prod


def cmd_run(root, out_json):
    a, prod = frame(root)
    rec = []
    for m in MONTHS:
        g = a[a.month == m]
        for y in sorted(g.year.unique()):
            tr = g[g.year < y]
            if tr.year.nunique() < 8 or y not in prod.index:
                continue
            P, Pf = prod.loc[y, m], prod.loc[y, 'final']
            if np.isnan(P) or np.isnan(Pf):
                continue
            ca, cb = ols(tr.pvp, tr.e)
            te = g[g.year == y].iloc[0]
            eh = ca + cb * te.pvp
            rec.append({'year': int(y), 'month': m, 'area_err_usda': 100 * abs(np.expm1(te.e)),
                        'area_err_adj': 100 * abs(np.expm1(te.e - eh)), 'prod_err_usda': 100 * (P / Pf - 1),
                        'prod_err_adj': 100 * (P * np.exp(eh) / Pf - 1)})
    r = pd.DataFrame(rec)

    def summary(x):
        bm = {m: {'seasons': int((x.month == m).sum()), 'prod_usda': float(x[x.month == m].prod_err_usda.abs().mean()),
                  'prod_adjusted': float(x[x.month == m].prod_err_adj.abs().mean()),
                  'area_usda': float(x[x.month == m].area_err_usda.mean()), 'area_adjusted': float(x[x.month == m].area_err_adj.mean())}
              for m in MONTHS if (x.month == m).any()}
        gain = (x.prod_err_usda.abs() - x.prod_err_adj.abs()).groupby(x.year).mean()
        return {'by_month': bm, 'months_better': int(sum(v['prod_adjusted'] < v['prod_usda'] for v in bm.values())),
                'year_block_p': block_p(gain), 'seasons': [int(x.year.min()), int(x.year.max())]}

    res = {'all': summary(r), 'before_2008': summary(r[r.year < 2008]), 'from_2008': summary(r[r.year >= 2008])}
    res['pass'] = bool(res['all']['months_better'] >= 3 and res['all']['year_block_p'] < 0.05)
    res['data_seasons'] = {m: sorted(a[a.month == m].year.unique().tolist()) for m in MONTHS}
    res['rows'] = rec
    print(json.dumps({k: v for k, v in res.items() if k not in ('rows', 'data_seasons')}, indent=1))
    print('data seasons per month', {m: len(v) for m, v in res['data_seasons'].items()})
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    {'pull': cmd_pull, 'run': cmd_run}[sys.argv[1]](*sys.argv[2:])
