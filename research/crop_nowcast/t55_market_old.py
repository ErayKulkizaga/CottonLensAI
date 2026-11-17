"""T55 (docs/T55_MARKET_2000_2009_PREREG_20261117.md): T52's report-day market test on 2000-2009.
Usage: python t55_market_old.py fetch out_dir | python t55_market_old.py run out_dir out.json [t52.json]
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M

END = pd.Timestamp('2009-12-31')
MONTHS = ['AUG', 'SEP', 'OCT', 'NOV', 'DEC']
NUM = {'OCT': 10, 'NOV': 11, 'DEC': 12}
URL = 'https://query1.finance.yahoo.com/v8/finance/chart/CT=F?period1=946684800&period2=1262304000&interval=1d'


def cmd_fetch(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60).read()
    (out / 'ctf_2000_2009.json').write_bytes(raw)
    (out / 'ctf_2000_2009.sha256').write_text(hashlib.sha256(raw).hexdigest(), encoding='utf-8')
    print('saved', len(raw), 'bytes')


def closes(out_dir):
    d = json.loads((Path(out_dir) / 'ctf_2000_2009.json').read_text(encoding='utf-8'))['chart']['result'][0]
    day = pd.to_datetime(d['timestamp'], unit='s').normalize()
    s = pd.Series(d['indicators']['quote'][0]['close'], index=day, dtype=float).dropna()
    s = s[~s.index.duplicated()].sort_index()
    return s[s.index <= END]  # nothing after 2009


def release_dates():
    old = json.loads((ROOT / 'output/crop-nowcast-t45/forecasts/us_forecasts.json').read_text())['data']
    return {(int(x['year']), NUM[x['reference_period_desc'][7:10]]): pd.Timestamp(x['release'])
            for x in old if x['reference_period_desc'][7:10] in NUM}


def perm(e):
    g = e.groupby('year').pnl.sum()
    rng = np.random.default_rng(42)
    sims = (g.to_numpy() * rng.choice([-1.0, 1.0], size=(20000, len(g)))).sum(1) / len(e)
    obs = float(e.pnl.mean())
    return obs, float((sims >= obs - 1e-12).mean())


def cmd_run(out_dir, out_json, t52_json=None):
    ret = np.log(closes(out_dir)).diff().dropna()
    f = M.frame(ROOT / 'output/crop-nowcast-full/nass', 'us').set_index('year')
    rel = release_dates()
    ev = []
    for y in range(2000, 2010):
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
                       'actual_revision': float(np.sign(np.log(f.loc[y, m] / prev1)))})
    e = pd.DataFrame(ev)
    e['pnl'] = e.predicted_price_dir * e.ret
    obs, p = perm(e)
    res = {'events': len(e), 'seasons': int(e.year.nunique()), 'mean_signed_return_pct': 100 * obs,
           'hit_rate': float((e.pnl > 0).mean()), 'season_block_p': p,
           'signal_hit_on_revision': float((e.signal_last_revision == e.actual_revision).mean()),
           'price_sha256': (Path(out_dir) / 'ctf_2000_2009.sha256').read_text().strip(), 'events_list': ev}
    res['pass'] = bool(obs > 0 and p < 0.05)
    if t52_json:  # secondary view only: add T52's 2010-2011 events
        old = pd.DataFrame(json.loads(Path(t52_json).read_text())['events_list'])
        old = old[old.year <= 2011].assign(pnl=lambda x: x.predicted_price_dir * x.ret)
        both = pd.concat([e, old[e.columns]])
        o2, p2 = perm(both)
        res['secondary_2000_2011'] = {'events': len(both), 'mean_signed_return_pct': 100 * o2,
                                      'hit_rate': float((both.pnl > 0).mean()), 'season_block_p': p2}
    print(json.dumps({k: v for k, v in res.items() if k != 'events_list'}, indent=1))
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    {'fetch': cmd_fetch, 'run': cmd_run}[sys.argv[1]](*sys.argv[2:])
