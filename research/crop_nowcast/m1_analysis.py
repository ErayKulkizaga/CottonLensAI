"""M1: does USDA's in-season Texas harvested area respond to drought as much as the final harvested area does?

Texas, seasons 2000-2025, reports 1 Sep / 1 Oct / 1 Nov / 1 Dec. Signals as of each report's reference date (m1_data.py):
  d2p = cotton-weighted share of Texas cotton area in USDM D2+ drought; pvp = % of Texas cotton rated poor + very poor.
Abandonment rates use the final planted area P: A_m = 1 - H_m / P (USDA, in-season), A_final = 1 - H_final / P.
  1. slope of A_final on the signal vs slope of A_m on it (anchoring: USDA's slope is smaller); season bootstrap CI
     of the ratio slope_m / slope_final;
  2. the area error e = log(H_final / H_m) against the signal: correlation, slope, bootstrap CI;
  3. the same by period (2000-2011 vs 2012-2025), and drought frequency by period (M2c).
Descriptive: these area outcomes have been seen (T48 / error_mechanism). Usage: python m1_analysis.py root out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.state_model import nass_state

MONTHS = ('SEP', 'OCT', 'NOV', 'DEC')


def area_frame(root):
    old = pd.read_csv(ROOT / 'output/crop-nowcast-t48/forecasts_area_yield.csv')
    old = old[(old.state == 'TX') & old.month.isin(MONTHS)][['year', 'month', 'area']]
    h = nass_state(ROOT / 'output/crop-nowcast-t39/nass', 'state_harvested')
    h = h[h.state_alpha == 'TX']
    new = h[h.reference_period_desc.str.endswith('FORECAST')].assign(month=lambda x: x.reference_period_desc.str[7:10])
    new = new[new.month.isin(MONTHS) & (new.year >= 2012)][['year', 'month', 'v']].rename(columns={'v': 'area'})
    fin_old = pd.read_csv(ROOT / 'output/crop-nowcast-t48/finals_area_yield.csv')
    fin_old = fin_old[(fin_old.state == 'TX') & (fin_old.stat == 'AREA HARVESTED')].set_index('year').v
    fin_new = h[h.reference_period_desc == 'YEAR'].set_index('year').v
    fin = pd.concat([fin_old[fin_old.index < 2012], fin_new[fin_new.index >= 2012]])
    planted = pd.read_csv(Path(root) / 'tx_planted_final.csv').set_index('year').planted_final
    a = pd.concat([old, new])
    a = a[a.year >= 2000].assign(final=lambda x: x.year.map(fin), planted=lambda x: x.year.map(planted))
    sig = pd.read_csv(Path(root) / 'tx_signals.csv')
    a = a.merge(sig, on=['year', 'month']).dropna(subset=['area', 'final', 'planted'])
    a['A_m'] = 1 - a.area / a.planted
    a['A_final'] = 1 - a.final / a.planted
    a['e_area'] = np.log(a.final / a.area)
    return a


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.polyfit(x, y, 1)[0])


def boot(g, fn, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    yrs = g.year.unique()
    vals = []
    for _ in range(n):
        s = g.set_index('year').loc[rng.choice(yrs, len(yrs))].reset_index()
        try:
            vals.append(fn(s))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            continue
    v = np.array([x for x in vals if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def block(g, sig):
    out = {'n': int(len(g))}
    sf, sm = slope(g[sig], g.A_final), slope(g[sig], g.A_m)
    out.update(slope_final=sf, slope_usda=sm, ratio_usda_to_final=sm / sf if sf else np.nan,
               ratio_ci95=boot(g, lambda s: slope(s[sig], s.A_m) / slope(s[sig], s.A_final)),
               corr_err=float(np.corrcoef(g[sig], g.e_area)[0, 1]), slope_err=slope(g[sig], g.e_area),
               slope_err_ci95=boot(g, lambda s: slope(s[sig], s.e_area)))
    return out


def main(root, out_json):
    a = area_frame(root)
    res = {'seasons': [int(a.year.min()), int(a.year.max())]}
    for sig in ('d2p_cotton_w', 'pvp'):
        for m in MONTHS:
            g = a[a.month == m]
            res[f'{sig}_{m}_all'] = block(g, sig)
            for lab, lo, hi in (('2000_2011', 2000, 2011), ('2012_2025', 2012, 2025)):
                res[f'{sig}_{m}_{lab}'] = block(g[g.year.between(lo, hi)], sig)
    per = a[a.month == 'SEP'].assign(period=lambda x: np.where(x.year < 2012, '2000-2011', '2012-2025'))
    res['M2c_by_period_sep'] = per.groupby('period').agg(
        d2p_mean=('d2p_cotton_w', 'mean'), pvp_mean=('pvp', 'mean'), A_final_mean=('A_final', 'mean'),
        A_usda_sep_mean=('A_m', 'mean'), area_err_mean=('e_area', 'mean')).round(4).to_dict(orient='index')
    res['by_season_sep'] = a[a.month == 'SEP'].set_index('year')[['d2p_cotton_w', 'pvp', 'A_m', 'A_final', 'e_area']].round(4).to_dict(orient='index')
    for k, v in res.items():
        if isinstance(v, dict) and 'slope_final' in v:
            print(f"{k:28s} n {v['n']:2d} | slope A_final {v['slope_final']:+.3f} vs USDA {v['slope_usda']:+.3f} | "
                  f"ratio {v['ratio_usda_to_final']:+.2f} CI [{v['ratio_ci95'][0]:+.2f},{v['ratio_ci95'][1]:+.2f}] | "
                  f"corr(err) {v['corr_err']:+.2f} slope(err) {v['slope_err']:+.3f} CI [{v['slope_err_ci95'][0]:+.3f},{v['slope_err_ci95'][1]:+.3f}]")
    print('M2c', json.dumps(res['M2c_by_period_sep']))
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
