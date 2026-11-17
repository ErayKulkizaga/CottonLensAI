"""B2 (docs/PLAN_REVIEW5_20261117.md): why the Texas area correction does not carry over to production.
Descriptive only, no new claim. Production error = area part + yield part (logs), so an area correction helps production
only to the extent that the yield part does not move against the area part.
Texas, Sep-Dec; 1995-2011 from the T48 parsed reports, 2012-2025 from the error-mechanism evidence.
Also: the share of the T51 area correction's gain on area that is lost on production.
Usage: python b2_area_yield_offset.py out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MONTHS = ['SEP', 'OCT', 'NOV', 'DEC']


def parts():
    f = pd.read_csv(ROOT / 'output/crop-nowcast-t48/forecasts_area_yield.csv')
    fin = pd.read_csv(ROOT / 'output/crop-nowcast-t48/finals_area_yield.csv').pivot_table(index=['state', 'year'], columns='stat', values='v')
    fin.columns = ['area_f', 'prod_f', 'yield_f']
    x = f[f.state == 'TX'].join(fin, on=['state', 'year']).dropna(subset=['area_f', 'yield_f'])
    old = pd.DataFrame({'year': x.year, 'month': x.month, 'area': np.log(x.area_f / x.area), 'yield': np.log(x.yield_f / x['yield'])})
    by = json.loads((ROOT / 'research/evidence/error-mechanism-2012-2025-20261116.json').read_text())['TX_by_year_pct']
    new = pd.DataFrame([{'year': int(y), 'month': m, 'area': v[f'area_{m}'] / 100, 'yield': v[f'yield_{m}'] / 100}
                        for y, v in by.items() for m in MONTHS if f'area_{m}' in v and f'yield_{m}' in v])
    d = pd.concat([old, new])
    return d[d.month.isin(MONTHS)]


def summary(g):
    b = np.polyfit(g.area, g['yield'], 1)[0]
    return {'n': len(g), 'corr_area_yield': float(np.corrcoef(g.area, g['yield'])[0, 1]), 'slope_yield_on_area': float(b),
            'mae_area_pct': float(100 * g.area.abs().mean()), 'mae_total_pct': float(100 * (g.area + g['yield']).abs().mean())}


def main(out_json):
    d = parts()
    res = {}
    for name, g in (('1995_2011', d[d.year <= 2011]), ('2012_2025', d[d.year >= 2012])):
        res[name] = {'all_months': summary(g), **{m: summary(g[g.month == m]) for m in MONTHS}}
    # T51: gain on area vs gain on production for the same rows (2008-2025)
    t = pd.DataFrame(json.loads((ROOT / 'research/evidence/t51-area-production-20261116.json').read_text())['rows'])
    t = t.dropna(subset=['e', 'eh_anchor', 'prod_err_usda', 'prod_err_area'])  # same rows for both gains
    area_gain = 100 * (t.e.abs().mean() - (t.e - t.eh_anchor).abs().mean())
    prod_gain = t.prod_err_usda.abs().mean() - t.prod_err_area.abs().mean()  # stored signed, in %
    res['t51_gain_points'] = {'area': float(area_gain), 'production': float(prod_gain),
                              'share_of_area_gain_kept_in_production': float(prod_gain / area_gain)}
    print(json.dumps(res, indent=1))
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
