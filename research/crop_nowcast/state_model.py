"""State and US production nowcast vs USDA's monthly forecast (development years only).

State features = county features weighted by each county's PRIOR-year planted area.
Production_hat = planted_known x PPA_hat, where planted_known is the March
Prospective Plantings figure for year Y when NASS has it (published before any
cutoff here), else the prior year's final planted area (fallback, flagged).
PPA = production*480 / final planted; model target log(PPA / median past PPA).
Rolling origin: test year Y uses only years < Y. Sealed years 2022-2025 untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TEST_YEARS = range(2013, 2022)
USDA_MONTH = {'aug1': 'AUG', 'sep1': 'SEP', 'oct1': 'OCT', 'nov1': 'NOV'}


def nass_state(nass_dir, name):
    d = pd.DataFrame(json.loads((Path(nass_dir) / f'{name}.json').read_text())['data'])
    d = d[d.prodn_practice_desc == 'ALL PRODUCTION PRACTICES'].copy()
    d['v'] = pd.to_numeric(d.Value.str.replace(',', ''), errors='coerce')
    d['year'] = d.year.astype(int)
    return d


def state_frame(nass_dir, cutoff_key, area='mar'):
    """area='jun' uses the June 30 Acreage survey (public before every cutoff) as planted_known."""
    prod, plant = nass_state(nass_dir, 'state_production'), nass_state(nass_dir, 'state_planted')
    usprod = nass_state(nass_dir, 'us_production')
    fin = prod[prod.reference_period_desc == 'YEAR'].set_index(['state_alpha', 'year']).v.rename('prod')
    pl = plant[plant.reference_period_desc == 'YEAR'].set_index(['state_alpha', 'year']).v.rename('planted')
    mar = plant[plant.reference_period_desc == 'YEAR - MAR ACREAGE'].set_index(['state_alpha', 'year']).v.rename('planted_mar')
    usda = prod[prod.reference_period_desc == f'YEAR - {USDA_MONTH[cutoff_key]} FORECAST'].set_index(['state_alpha', 'year']).v.rename('usda')
    s = pd.concat([fin, pl, mar, usda], axis=1).reset_index().rename(columns={'state_alpha': 'state'})
    s = s.sort_values(['state', 'year'])
    s['ppa'] = s['prod'] * 480 / s.planted
    s['ppa_base'] = s.groupby('state').ppa.transform(lambda x: x.expanding().median().shift(1))
    s['planted_prev'] = s.groupby('state').planted.shift(1)
    known = s.planted_mar
    if area == 'jun':
        jun = json.loads((Path(nass_dir) / 'june_acreage.json').read_text())['planted_june']
        s['planted_jun'] = [jun.get(str(y), {}).get(st) for st, y in zip(s.state, s.year, strict=True)]
        known = s.planted_jun.astype(float).fillna(s.planted_mar)
    s['planted_known'] = known.fillna(s.planted_prev)
    s['planted_fallback'] = known.isna()
    us = pd.DataFrame({'final': usprod[usprod.reference_period_desc == 'YEAR'].set_index('year').v,
                       'usda': usprod[usprod.reference_period_desc == f'YEAR - {USDA_MONTH[cutoff_key]} FORECAST'].set_index('year').v})
    return s, us


def aggregate(county_csv):
    c = pd.read_csv(county_csv, dtype={'fips': str})
    c = c.sort_values(['fips', 'year'])
    c['w'] = c.groupby('fips').y_planted.shift(1)
    cols = [k for k in c.columns if k.endswith(('_z', '_c')) or (k.startswith('n_') and not k.endswith('_a'))]
    g = c.dropna(subset=['w'])
    agg = g.groupby(['state', 'year']).apply(lambda x: pd.Series({k: np.average(x[k].fillna(x[k].mean()) if x[k].notna().any() else 0.0, weights=x.w) for k in cols}),
                                             include_groups=False)
    return agg.reset_index(), cols


def run(county_csv, nass_dir, cutoff_key, feature_filter, alphas=np.logspace(0, 5, 40)):
    s, us = state_frame(nass_dir, cutoff_key)
    a, cols = aggregate(county_csv)
    cols = [k for k in cols if feature_filter(k)]
    d = s.merge(a, on=['state', 'year'], how='inner')
    d['target'] = np.log(d.ppa / d.ppa_base)
    rows = []
    for y in TEST_YEARS:
        tr = d[(d.year < y) & d.target.notna()].dropna(subset=cols)
        te = d[(d.year == y) & d.ppa_base.notna()]
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(tr[cols], tr.target)
        ppa_hat = te.ppa_base * np.exp(m.predict(te[cols].fillna(tr[cols].mean())))
        rows.append(te.assign(pred=ppa_hat * te.planted_known / 480, base=te.ppa_base * te.planted_known / 480))
    r = pd.concat(rows)
    return r, us


def summarize(r, us, label):
    out = {}
    tx = r[r.state == 'TX']
    for name, col in (('model', 'pred'), ('base', 'base'), ('usda', 'usda')):
        e = (tx[col] / tx['prod'] - 1).dropna()
        out[f'TX_{name}_mape'] = float(100 * e.abs().mean())
    tot = r.groupby('year')[['pred', 'base', 'prod']].sum()
    usj = tot.join(us)
    for name, col in (('model', 'pred'), ('base', 'base')):
        out[f'US_{name}_mape'] = float(100 * (usj[col] / usj['final'] - 1).abs().mean())
    out['US_usda_mape'] = float(100 * (usj.usda / usj.final - 1).abs().mean())
    out['TX_by_year'] = {int(y): {'model': round(100 * (p / q - 1), 1), 'usda': None if pd.isna(u) else round(100 * (u / q - 1), 1)}
                         for y, p, q, u in zip(tx.year, tx.pred, tx['prod'], tx.usda, strict=True)}
    print(f"{label:34s} TX MAPE model {out['TX_model_mape']:5.1f} | base {out['TX_base_mape']:5.1f} | USDA {out['TX_usda_mape']:5.1f}"
          f"   US MAPE model {out['US_model_mape']:5.1f} | base {out['US_base_mape']:5.1f} | USDA {out['US_usda_mape']:5.1f}")
    return out


def main(feature_dir, nass_dir, out_json, cutoffs='aug1,oct1'):
    res = {}
    for ck in cutoffs.split(','):
        csv = Path(feature_dir) / f'features_{ck}.csv'
        res[ck] = {}
        for label, flt in (('weather z', lambda k: k.startswith('w_') and k.endswith('_z')),
                           ('weather z+c', lambda k: k.startswith('w_')),
                           ('ndvi', lambda k: k.startswith('n_')),
                           ('weather z + ndvi', lambda k: (k.startswith('w_') and k.endswith('_z')) or k.startswith('n_'))):
            try:
                r, us = run(csv, nass_dir, ck, flt)
            except ValueError:
                continue
            res[ck][label] = summarize(r, us, f'{ck} {label}')
    Path(out_json).write_text(json.dumps(res, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:5])
