"""T31 hybrid: USDA's yield per harvested acre x our harvested-area forecast (dev years only).

USDA is strong on yield per harvested acre (objective yield surveys) and weak on abandonment.
We predict the county harvested/planted ratio from satellite NDVI anomalies and FSA failed-acre
shares, aggregate it to the state with prior-year planted weights, multiply by FSA-scaled planted
area, and use USDA's same-month yield forecast. Production_hat = harvested_hat * USDA yield / 480.
FSA input is always the August file (saved before USDA's report in every test year 2013-2021).
Rolling origin 2013-2021, train on years < Y. Sealed 2022-2025 untouched.
Usage: python hybrid.py feature_dir nass_dir out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import county_up as C
from research.crop_nowcast.state_model import nass_state, state_frame

MONTH = {'aug1': 'AUG', 'sep1': 'SEP', 'oct1': 'OCT'}
LATE = {'nov1': 'NOV', 'dec1': 'DEC'}  # T32


def usda(nass_dir, ck):
    def pick(name, ref):
        d = nass_state(nass_dir, name)
        return d[d.reference_period_desc == ref].set_index(['state_alpha', 'year']).v
    m = {**MONTH, **LATE}[ck]
    f = pd.DataFrame({'final': pick('state_production', 'YEAR'), 'usda': pick('state_production', f'YEAR - {m} FORECAST'),
                      'harv_final': pick('state_harvested', 'YEAR'), 'harv_usda': pick('state_harvested', f'YEAR - {m} FORECAST'),
                      'yield_usda': pick('state_yield', f'YEAR - {m} FORECAST')})
    f.index = f.index.set_names(['state', 'year'])
    us = nass_state(nass_dir, 'us_production')
    usf = pd.DataFrame({'final': us[us.reference_period_desc == 'YEAR'].set_index('year').v,
                        'usda': us[us.reference_period_desc == f'YEAR - {m} FORECAST'].set_index('year').v})
    return f.reset_index(), usf


def harvest_ratio(c, cols):
    """Rolling county HGB on harvested/planted; returns state ratio per test year (prior-year planted weights)."""
    c = c.copy()
    c['hr'] = c.y_harv_ratio.clip(0, 1)
    out = []
    for y in C.TEST_YEARS:
        tr = c[(c.year < y) & c.hr.notna()]
        te = c[(c.year == y) & c.w.notna()].copy()
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30,
                                          l2_regularization=2.0, random_state=0).fit(tr[cols], tr.hr)
        te['hr_hat'] = np.clip(m.predict(te[cols]), 0, 1)
        out.append(te.groupby('state').apply(lambda x: np.average(x.hr_hat, weights=x.w), include_groups=False)
                   .rename('ratio').reset_index().assign(year=y))
    return pd.concat(out)


def run(feature_dir, nass_dir, ck, cols_kind='ndvi+fsa'):
    c = C.add_fsa(C.prepare(Path(feature_dir) / f'features_{ck}.csv'), Path(nass_dir).parent / 'fsa/fsa_cotton.csv', 'aug1')
    cols = {'ndvi+fsa': [k + '_a' for k in C.NDVI] + C.FSA, 'fsa': C.FSA, 'ndvi': [k + '_a' for k in C.NDVI]}[cols_kind]
    r = harvest_ratio(c, cols)
    s, _ = state_frame(nass_dir, 'aug1', 'fsa')  # FSA-scaled planted area from the August file
    f, us = usda(nass_dir, ck)
    f = f.merge(r, on=['state', 'year']).merge(s[['state', 'year', 'planted_known']], on=['state', 'year'])
    f['harv_hat'] = f.planted_known * f.ratio
    f['hybrid'] = f.harv_hat * f.yield_usda / 480
    return f.dropna(subset=['usda', 'final', 'yield_usda']), us


def score(f, us, label):
    tx = f[f.state == 'TX']
    t = f.groupby('year')[['hybrid', 'usda']].sum().join(us, rsuffix='_us')
    t['hyb_us'] = t.usda_us * t.hybrid / t.usda
    e = lambda a, b: 100 * np.mean(np.abs(a / b - 1))
    w = lambda col, ref: 100 * np.average(np.abs(f[col] / f[ref] - 1), weights=f.final)
    res = {'TX': e(tx.hybrid, tx.final), 'TX_usda': e(tx.usda, tx.final), 'US': e(t.hyb_us, t.final), 'US_usda': e(t.usda_us, t.final),
           'state_w': w('hybrid', 'final'), 'state_w_usda': w('usda', 'final'),
           'harv_TX': e(tx.harv_hat, tx.harv_final), 'harv_TX_usda': e(tx.harv_usda, tx.harv_final),
           'years_better_US': int(((t.hyb_us / t.final - 1).abs() < (t.usda_us / t.final - 1).abs()).sum()), 'years': len(t),
           'TX_by_year': {int(y): (round(100 * (a / q - 1), 1), round(100 * (u / q - 1), 1))
                          for y, a, u, q in zip(tx.year, tx.hybrid, tx.usda, tx.final, strict=True)}}
    print(f"{label}: TX {res['TX']:5.1f} vs USDA {res['TX_usda']:5.1f} | US {res['US']:4.1f} vs {res['US_usda']:4.1f} "
          f"(better {res['years_better_US']}/{res['years']}) | prod-wtd state {res['state_w']:4.1f} vs {res['state_w_usda']:4.1f} "
          f"| TX harvested-area err {res['harv_TX']:4.1f} vs USDA {res['harv_TX_usda']:4.1f}")
    return res


def main(feature_dir, nass_dir, out_json):
    res = {}
    for ck in MONTH:
        for kind in ('ndvi+fsa', 'fsa', 'ndvi'):
            f, us = run(feature_dir, nass_dir, ck, kind)
            res[f'{ck}/{kind}'] = score(f, us, f'{ck} T31 {kind:8s}')
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:4])
