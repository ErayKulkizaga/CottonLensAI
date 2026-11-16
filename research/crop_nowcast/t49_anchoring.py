"""T49: anchoring of USDA's in-season harvested area against public crop condition (docs/T49_ANCHORING_PREREG_20261116.md).

  pull  root        QuickStats: cotton condition OK/KS/TX 2000-2025; winter wheat harvested area (forecasts + finals)
                    and condition for KS/OK/TX 2012-2025. Key from NASS_API_KEY.
  run   root out    S0 (Texas cotton, rolling), S1 (OK+KS cotton, leave-one-season-out), S2 (winter wheat, LOSO)
"""
import json
import os
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.m1_data import get
from research.crop_nowcast.t46_crops import perm_p  # noqa: F401  (same seeded sign-flip as elsewhere)

COTTON_M = {'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
WHEAT_M = {'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8}
API = 'https://quickstats.nass.usda.gov/api/api_GET/?'


def qs(key, **q):
    try:
        return json.loads(get(API + urllib.parse.urlencode(dict(q, key=key, format='JSON', source_desc='SURVEY'))))['data']
    except (SystemExit, json.JSONDecodeError):
        return []


def cmd_pull(root):
    key = os.environ['NASS_API_KEY']
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    cond = []
    for crop, states, years, extra in (('COTTON', ('OK', 'KS', 'TX'), range(2000, 2026), {}),
                                       ('WHEAT', ('KS', 'OK', 'TX'), range(2012, 2026), {'class_desc': 'WINTER'})):
        for st in states:
            for y in years:
                for unit in ('PCT POOR', 'PCT VERY POOR'):
                    for r in qs(key, commodity_desc=crop, statisticcat_desc='CONDITION', unit_desc=unit, agg_level_desc='STATE',
                                state_alpha=st, year=str(y), **extra):
                        if r.get('week_ending') and r['Value'].strip().replace('.', '').isdigit():
                            cond.append({'crop': crop, 'state': st, 'year': y, 'unit': unit, 'week_ending': r['week_ending'],
                                         'v': float(r['Value'])})
        print(crop, 'condition rows so far', len(cond), flush=True)
    pd.DataFrame(cond).to_csv(out / 'condition.csv', index=False)
    area = []
    for st in ('KS', 'OK', 'TX'):
        for y in range(2012, 2026):
            for r in qs(key, commodity_desc='WHEAT', class_desc='WINTER', statisticcat_desc='AREA HARVESTED', unit_desc='ACRES',
                        agg_level_desc='STATE', state_alpha=st, year=str(y)):
                if r.get('prodn_practice_desc', 'ALL PRODUCTION PRACTICES') == 'ALL PRODUCTION PRACTICES' and \
                        r.get('util_practice_desc', 'ALL UTILIZATION PRACTICES') == 'ALL UTILIZATION PRACTICES':
                    area.append({'state': st, 'year': y, 'ref': r['reference_period_desc'], 'v': float(r['Value'].replace(',', ''))})
    pd.DataFrame(area).to_csv(out / 'wheat_area.csv', index=False)
    a = pd.DataFrame(area)
    print('wheat area rows', len(a), 'refs', sorted(a.ref.unique()) if len(a) else [])


def pvp_at(cond, crop, state, year, month):
    c = cond[(cond.crop == crop) & (cond.state == state) & (cond.year == year)]
    ref = pd.Timestamp(year, month, 1)
    c = c[pd.to_datetime(c.week_ending) <= ref - pd.Timedelta(days=1)]
    if c.empty:
        return np.nan
    last = c[c.week_ending == c.week_ending.max()]
    return float(last.v.sum()) / 100 if set(last.unit) == {'PCT POOR', 'PCT VERY POOR'} else np.nan


def cotton_rows(cond, states):
    from research.crop_nowcast.state_model import nass_state
    old = pd.read_csv(ROOT / 'output/crop-nowcast-t48/forecasts_area_yield.csv')
    old = old[old.state.isin(states) & old.month.isin(COTTON_M) & (old.year >= 2000)][['state', 'year', 'month', 'area']]
    h = nass_state(ROOT / 'output/crop-nowcast-t39/nass', 'state_harvested')
    h = h[h.state_alpha.isin(states)].rename(columns={'state_alpha': 'state'})
    new = h[h.reference_period_desc.str.endswith('FORECAST')].assign(month=lambda x: x.reference_period_desc.str[7:10])
    new = new[new.month.isin(COTTON_M) & (new.year >= 2012)][['state', 'year', 'month', 'v']].rename(columns={'v': 'area'})
    fo = pd.read_csv(ROOT / 'output/crop-nowcast-t48/finals_area_yield.csv')
    fo = fo[fo.state.isin(states) & (fo.stat == 'AREA HARVESTED') & (fo.year < 2012)].set_index(['state', 'year']).v
    fn = h[(h.reference_period_desc == 'YEAR') & (h.year >= 2012)].set_index(['state', 'year']).v
    fin = pd.concat([fo, fn])
    d = pd.concat([old, new])
    d['final'] = [fin.get((s, y), np.nan) for s, y in zip(d.state, d.year)]
    d['pvp'] = [pvp_at(cond, 'COTTON', s, y, COTTON_M[m]) for s, y, m in zip(d.state, d.year, d.month)]
    return d.dropna()


def wheat_rows(root, cond):
    a = pd.read_csv(Path(root) / 'wheat_area.csv')
    fin = a[a.ref == 'YEAR'].set_index(['state', 'year']).v
    f = a[a.ref.str.endswith('FORECAST')].assign(month=lambda x: x.ref.str[7:10])
    f = f[f.month.isin(WHEAT_M)].rename(columns={'v': 'area'})[['state', 'year', 'month', 'area']]
    f['final'] = [fin.get((s, y), np.nan) for s, y in zip(f.state, f.year)]
    f['pvp'] = [pvp_at(cond, 'WHEAT', s, y, WHEAT_M[m]) for s, y, m in zip(f.state, f.year, f.month)]
    return f.dropna()


def fit(tr):
    X = np.c_[np.ones(len(tr)), tr.pvp]
    return np.linalg.lstsq(X, tr.e.to_numpy(), rcond=None)[0]


def evaluate(d, months, scheme, first_scored=None, min_train=8):
    d = d.copy()
    d['e'] = np.log(d.final / d.area)
    out, pooled = {}, []
    for m in months:
        g = d[d.month == m]
        preds = []
        for y in sorted(g.year.unique()):
            tr = g[g.year < y] if scheme == 'rolling' else g[g.year != y]
            if scheme == 'rolling' and (tr.year.nunique() < min_train or (first_scored and y < first_scored)):
                continue
            a, b = fit(tr)
            te = g[g.year == y]
            preds.append(te.assign(e_hat=a + b * te.pvp))
        if not preds:
            continue
        p = pd.concat(preds)
        p['gain'] = p.e.abs() - (p.e - p.e_hat).abs()
        pooled.append(p)
        out[m] = {'rows': int(len(p)), 'seasons': int(p.year.nunique()), 'mae_usda_pct': float(100 * p.e.abs().mean()),
                  'mae_adjusted_pct': float(100 * (p.e - p.e_hat).abs().mean()),
                  'corr_pvp_e': float(np.corrcoef(g.pvp, g.e)[0, 1])}
    allp = pd.concat(pooled)
    rng = np.random.default_rng(42)
    yrs = sorted(d.year.unique())
    bs = []
    for _ in range(5000):
        s = d.set_index('year').loc[rng.choice(yrs, len(yrs))].reset_index()
        if s.pvp.std() > 0:
            bs.append(np.corrcoef(s.pvp, s.e)[0, 1])
    corr = float(np.corrcoef(d.pvp, d.e)[0, 1])
    by_year = allp.groupby('year').gain.mean()
    obs = float(by_year.mean())
    sims = np.array([np.mean(by_year.to_numpy() * rng.choice([-1.0, 1.0], len(by_year))) for _ in range(20000)])
    res = {'months': out, 'pooled_corr': corr, 'pooled_corr_upper95_one_sided': float(np.percentile(bs, 95)),
           'mean_gain_log_pts': 100 * obs, 'year_block_p': float((sims >= obs - 1e-12).mean()),
           'months_better': int(sum(v['mae_adjusted_pct'] < v['mae_usda_pct'] for v in out.values())), 'n_months': len(out)}
    res['pass_sign'] = res['pooled_corr_upper95_one_sided'] < 0
    res['pass_prediction'] = res['months_better'] >= min(3, len(out)) and res['year_block_p'] < 0.05
    res['pass'] = bool(res['pass_sign'] and res['pass_prediction'])
    return res


def cmd_run(root, out_json):
    cond = pd.read_csv(Path(root) / 'condition.csv')
    res = {}
    res['S0_texas_cotton_rolling'] = evaluate(cotton_rows(cond, ['TX']), COTTON_M, 'rolling', first_scored=2008)
    res['S1_ok_ks_cotton_loso'] = evaluate(cotton_rows(cond, ['OK', 'KS']), COTTON_M, 'loso')
    res['S2_winter_wheat_loso'] = evaluate(wheat_rows(root, cond), WHEAT_M, 'loso')
    for k, v in res.items():
        print(k, '| corr', round(v['pooled_corr'], 3), 'upper95', round(v['pooled_corr_upper95_one_sided'], 3),
              '| months better', v['months_better'], '/', v['n_months'], '| gain', round(v['mean_gain_log_pts'], 2),
              'p', round(v['year_block_p'], 4), '| PASS', v['pass'])
        for m, x in v['months'].items():
            print('   ', m, x)
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    {'pull': cmd_pull, 'run': cmd_run}[sys.argv[1]](*sys.argv[2:])
