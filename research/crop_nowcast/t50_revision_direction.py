"""T50: does the sign of USDA's latest revision predict the sign of its next one? (docs/T50_REVISION_DIRECTION_PREREG_20261116.md)

No fitted parameter: predicted sign(next revision) = sign(latest revision). Pairs with a zero revision are not scored.
Usage: python t50_revision_direction.py out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M

O = ROOT / 'output'
MONTHS = ['AUG', 'SEP', 'OCT', 'NOV', 'DEC']


def pairs(nass_dir, level, years):
    """Rows: unit, year, month m, r (latest revision at m), nxt (revision m -> m+1)."""
    f = M.frame(nass_dir, level)
    f = f[f.year.between(*years)]
    unit = f.state if level == 'state' else pd.Series('US', index=f.index)
    out = []
    for i, m in enumerate(MONTHS[1:-1], 1):
        prev = f[MONTHS[i - 1]]
        for k in range(i - 2, -1, -1):
            prev = prev.fillna(f[MONTHS[k]])
        nxt_m = MONTHS[i + 1]
        r = np.log(f[m] / prev)
        nx = np.log(f[nxt_m] / f[m])
        out.append(pd.DataFrame({'unit': unit, 'year': f.year, 'month': m, 'r': r, 'nxt': nx}))
    d = pd.concat(out).dropna()
    zero = int(((d.r == 0) | (d.nxt == 0)).sum())
    d = d[(d.r != 0) & (d.nxt != 0)]
    d['hit'] = (np.sign(d.r) == np.sign(d.nxt)).astype(int)
    return d, zero


def score(d, seed=42, n=20000):
    if d.empty:
        return {'pairs': 0}
    rng = np.random.default_rng(seed)
    years = d.year.to_numpy()
    uy = np.unique(years)
    idx = np.searchsorted(uy, years)
    h = d.hit.to_numpy().astype(float)
    obs = h.mean()
    flips = rng.integers(0, 2, size=(n, len(uy)))
    sims = np.where(flips[:, idx] == 1, 1 - h, h).mean(1)
    from math import comb
    k, N = int(h.sum()), len(h)
    binom = sum(comb(N, j) for j in range(k, N + 1)) / 2 ** N
    return {'pairs': N, 'hits': k, 'hit_rate': float(obs), 'season_block_p': float((sims >= obs - 1e-12).mean()),
            'binomial_p': float(binom), 'seasons': int(len(uy))}


def main(out_json):
    sets = {
        'B_1985_1999': (O / 'crop-nowcast-t47ab/nass', (1985, 1999)),
        'B_2000_2011': (O / 'crop-nowcast-t45/nass', (2000, 2011)),
        'R_2012_2025': (O / 'crop-nowcast-full/nass', (2012, 2025)),
        'C_corn_2000_2025': (O / 'crop-nowcast-t46/corn/nass', (2000, 2025)),
        'C_soy_2000_2025': (O / 'crop-nowcast-t46/soy/nass', (2000, 2025)),
    }
    frames, res = {}, {}
    for name, (nd, yrs) in sets.items():
        us, zu = pairs(nd, 'us', yrs)
        st, zs = pairs(nd, 'state', yrs)
        frames[name] = (us, st)
        res[name] = {'excluded_zero_pairs': zu + zs}
    B_us = pd.concat([frames[k][0] for k in ('B_1985_1999', 'B_2000_2011')])
    B_st = pd.concat([frames[k][1] for k in ('B_1985_1999', 'B_2000_2011')])
    B_tx = B_st[B_st.unit == 'TX']
    res['PRIMARY_B_us_tx'] = score(pd.concat([B_us, B_tx]))
    res['PRIMARY_pass'] = bool(res['PRIMARY_B_us_tx']['hit_rate'] > 0.5 and res['PRIMARY_B_us_tx']['season_block_p'] < 0.05)
    res['B_us'] = score(B_us)
    res['B_tx'] = score(B_tx)
    res['B_states_pooled'] = score(B_st)
    res['B_by_month_us_tx'] = {m: score(pd.concat([B_us, B_tx])[lambda x: x.month == m]) for m in ('SEP', 'OCT', 'NOV')}
    for name in ('R_2012_2025', 'C_corn_2000_2025', 'C_soy_2000_2025'):
        us, st = frames[name]
        res[name].update({'us': score(us), 'tx_or_states': score(st[st.unit == 'TX']) if name.startswith('R') else score(st),
                          'us_plus_tx': score(pd.concat([us, st[st.unit == 'TX']])) if name.startswith('R') else None})
    for k, v in res.items():
        print(k, json.dumps(v))
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
