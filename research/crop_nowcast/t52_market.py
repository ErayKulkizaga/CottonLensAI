"""T52 (docs/T52_MARKET_PREREG_20261116.md): report-day cotton futures return vs the predicted direction of USDA's
next US production revision. Prices sealed at 2021-12-31 before any computation.
Usage: python t52_market.py market_history.parquet out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M

SEAL = pd.Timestamp('2021-12-31')
MONTHS = ['AUG', 'SEP', 'OCT', 'NOV', 'DEC']
NUM = {'OCT': 10, 'NOV': 11, 'DEC': 12}


def release_dates():
    d = pd.read_csv(ROOT / 'research/chain/crop_production_dates.csv', parse_dates=['release_date'])
    rel = {(int(r.year), int(r.month)): r.release_date for r in d.itertuples()}
    old = json.loads((ROOT / 'output/crop-nowcast-t45/forecasts/us_forecasts.json').read_text())['data']
    for x in old:
        y, mon = int(x['year']), x['reference_period_desc'][7:10]
        if y in (2010, 2011) and mon in NUM:
            rel.setdefault((y, NUM[mon]), pd.Timestamp(x['release']))
    return rel


def main(parquet, out_json):
    p = pd.read_parquet(parquet)
    p = p[p.series == 'cotton'].assign(day=lambda x: pd.to_datetime(x.observed_on))
    p = p[p.day <= SEAL].sort_values('day').set_index('day')  # seal first
    ret = np.log(p.close).diff().dropna()
    f = M.frame(ROOT / 'output/crop-nowcast-full/nass', 'us').set_index('year')
    rel = release_dates()
    ev = []
    for y in range(2010, 2022):
        for m, mn in NUM.items():
            i = MONTHS.index(m)
            prev1, prev2 = f.loc[y, MONTHS[i - 1]], f.loc[y, MONTHS[i - 2]]
            if np.isnan(prev1) or np.isnan(prev2) or prev1 == prev2 or (y, mn) not in rel:
                continue
            day = rel[(y, mn)]
            if day not in ret.index:
                continue
            sig = np.sign(np.log(prev1 / prev2))
            ev.append({'year': y, 'month': m, 'release': str(day.date()), 'signal_last_revision': float(sig),
                       'predicted_price_dir': float(-sig), 'ret': float(ret.loc[day]),
                       'actual_revision': float(np.sign(np.log(f.loc[y, m] / prev1))) if not np.isnan(f.loc[y, m]) else None})
    e = pd.DataFrame(ev)
    e['pnl'] = e.predicted_price_dir * e.ret
    g = e.groupby('year').pnl.sum()
    rng = np.random.default_rng(42)
    sims = (g.to_numpy() * rng.choice([-1.0, 1.0], size=(20000, len(g)))).sum(1) / len(e)
    obs = float(e.pnl.mean())
    res = {'events': int(len(e)), 'mean_signed_return_pct': 100 * obs, 'hit_rate': float((e.pnl > 0).mean()),
           'season_block_p': float((sims >= obs - 1e-12).mean()),
           'by_era': {k: {'events': int(len(x)), 'mean_signed_return_pct': 100 * float(x.pnl.mean()), 'hit_rate': float((x.pnl > 0).mean())}
                      for k, x in (('2010_2011', e[e.year <= 2011]), ('2012_2021', e[e.year >= 2012]))},
           'signal_hit_on_revision': float((e.signal_last_revision == e.actual_revision).mean()),
           'events_list': ev}
    res['pass'] = bool(obs > 0 and res['season_block_p'] < 0.05)
    print(json.dumps({k: v for k, v in res.items() if k != 'events_list'}, indent=1))
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
