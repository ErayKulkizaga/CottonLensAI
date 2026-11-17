"""Where USDA's cotton production error comes from: harvested area vs yield (descriptive, 2012-2025 QuickStats).

Production = harvested area x yield, so the log production error splits exactly:
  log(P_final / P_m) = log(H_final / H_m) + log(Y_final / Y_m)      (+ a rounding residual)
For each report month m (Aug-Dec) and region (TX, US, states pooled) this reports:
  - mean absolute error of production, area and yield parts (percent);
  - the variance share of each part: cov(part, total) / var(total), which sums to 1;
  - whether USDA's latest production revision predicts the area part or the yield part of the remaining error
    (correlation of r = log(P_m / P_prev) with each part), i.e. where the momentum effect lives.
Usage: python error_mechanism.py nass_dir out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.state_model import nass_state

MONTHS = ['AUG', 'SEP', 'OCT', 'NOV', 'DEC']


def wide(nass_dir, lvl, stat):
    d = nass_state(nass_dir, f'{lvl}_{stat}')
    key = ['state_alpha', 'year'] if lvl == 'state' else ['year']
    f = pd.DataFrame({m: d[d.reference_period_desc == f'YEAR - {m} FORECAST'].set_index(key).v for m in MONTHS})
    f['final'] = d[d.reference_period_desc == 'YEAR'].set_index(key).v
    return f


def frame(nass_dir, lvl):
    P, H, Y = (wide(nass_dir, lvl, s) for s in ('production', 'harvested', 'yield'))
    rows = []
    for i, m in enumerate(MONTHS):
        prev = P[MONTHS[i - 1]] if i else np.nan
        for k in range(i - 2, -1, -1):
            prev = prev.fillna(P[MONTHS[k]])
        x = pd.DataFrame({'total': np.log(P.final / P[m]), 'area': np.log(H.final / H[m]), 'yield': np.log(Y.final / Y[m]),
                          'r': np.log(P[m] / prev) if i else np.nan}).dropna(subset=['total', 'area', 'yield'])
        rows.append(x.assign(month=m).reset_index())
    return pd.concat(rows)


def summarise(x):
    v = x.total.var()
    out = {'n': int(len(x)), 'years': [int(x.year.min()), int(x.year.max())],
           'mae_total_pct': float(100 * x.total.abs().mean()), 'mae_area_pct': float(100 * x.area.abs().mean()),
           'mae_yield_pct': float(100 * x['yield'].abs().mean()),
           'var_share_area': float(np.cov(x.area, x.total)[0, 1] / v), 'var_share_yield': float(np.cov(x['yield'], x.total)[0, 1] / v),
           'residual_max_pct': float(100 * (x.total - x.area - x['yield']).abs().max())}
    xr = x.dropna(subset=['r'])
    if len(xr) > 5 and xr.r.std() > 0:
        out['corr_revision_area'] = float(np.corrcoef(xr.r, xr.area)[0, 1])
        out['corr_revision_yield'] = float(np.corrcoef(xr.r, xr['yield'])[0, 1])
        out['corr_revision_total'] = float(np.corrcoef(xr.r, xr.total)[0, 1])
    return out


def main(nass_dir, out_json):
    st, us = frame(nass_dir, 'state'), frame(nass_dir, 'us')
    st = st[st.year >= 2012]
    us = us[us.year >= 2012]
    res = {}
    for m in MONTHS:
        res[m] = {'TX': summarise(st[(st.month == m) & (st.state_alpha == 'TX')]),
                  'US': summarise(us[us.month == m]),
                  'states_pooled': summarise(st[st.month == m])}
    tx = st[st.state_alpha == 'TX'].pivot_table(index='year', columns='month', values=['area', 'yield'])
    res['TX_by_year_pct'] = {int(y): {f'{a}_{m}': round(100 * float(tx.loc[y, (a, m)]), 1) for a in ('area', 'yield')
                                      for m in MONTHS if (a, m) in tx.columns and pd.notna(tx.loc[y, (a, m)])} for y in tx.index}
    for m in MONTHS:
        for reg in ('TX', 'US', 'states_pooled'):
            s = res[m][reg]
            print(f"{m} {reg:13s} n {s['n']:3d} | |err| total {s['mae_total_pct']:5.1f} area {s['mae_area_pct']:5.1f} "
                  f"yield {s['mae_yield_pct']:5.1f} | var share area {s['var_share_area']:+.2f} yield {s['var_share_yield']:+.2f} | "
                  f"corr(r, area) {s.get('corr_revision_area', float('nan')):+.2f} corr(r, yield) {s.get('corr_revision_yield', float('nan')):+.2f} "
                  f"corr(r, total) {s.get('corr_revision_total', float('nan')):+.2f} | resid {s['residual_max_pct']:.2f}")
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
