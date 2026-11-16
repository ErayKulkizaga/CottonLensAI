"""Robustness of the cotton momentum record 2002-2025 and a comparison with a published-style smoothing adjustment.

Descriptive (all years already seen). Nothing here changes the rule or any earlier result.
  1. Multiple comparisons: Holm and Benjamini-Hochberg over the 8 TX/US cells; one pooled year-block test.
  2. Year bootstrap 95% CI of the mean gain per cell (10,000 draws, seed 42).
  3. Leave-one-season-out: the largest single-season influence on each cell's mean gain, and sign flips.
  4. Coefficient path: the rolling b used for each season and month (stability).
  5. Published-style adjustment (Isengildina-Massa et al. 2013, NCCC-134): per series, OLS of the remaining log error
     on the latest log revision WITH an intercept, 10-season rolling window; and the same with an expanding window.
Usage: python robustness.py full_nass_dir out.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('COTTONLENS_MOMENTUM_YEARS', '2002-2025')
os.environ.setdefault('COTTONLENS_MOMENTUM_FIRST_TRAIN', '2000')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M
from research.crop_nowcast.t46_crops import perm_p

MONTHS = ('SEP', 'OCT', 'NOV', 'DEC')


def holm(p):
    o = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def bh(p):
    o = np.argsort(p)[::-1]
    adj, run = np.empty(len(p)), 1.0
    for i, k in enumerate(o):
        run = min(run, p[k] * len(p) / (len(p) - i))
        adj[k] = run
    return adj


def coef_path(nass_dir, m):
    f = M.with_revision(M.frame(nass_dir, 'state'), m)
    out = {}
    for y in M.YEARS:
        tr = f[(f.year < y) & (f.year >= M.FIRST_TRAIN)].dropna(subset=['e', 'r'])
        if len(tr) < M.MIN_ROWS:
            continue
        sd = float(np.sqrt((tr.r ** 2).mean()))
        z = (tr.r / sd).to_numpy()
        out[y] = round(float(z @ tr.e.to_numpy() / (z @ z + M.LAM)), 5)
    return out


def published_style(nass_dir, m, region, window):
    """Series-level OLS e = a + b r on past seasons (window seasons, or all if None); returns {year: (adj err %, usda err %)}."""
    lvl = 'us' if region == 'US' else 'state'
    f = M.with_revision(M.frame(nass_dir, lvl), m)
    if region != 'US':
        f = f[f.state == region]
    f = f.dropna(subset=['usda', 'r', 'final']).sort_values('year')
    res = {}
    for y in M.YEARS:
        tr = f[(f.year < y) & (f.year >= M.FIRST_TRAIN)]
        if window:
            tr = tr[tr.year >= y - window]
        te = f[f.year == y]
        if te.empty or len(tr) < 5:
            continue
        X = np.c_[np.ones(len(tr)), tr.r]
        a, b = np.linalg.lstsq(X, tr.e.to_numpy(), rcond=None)[0]
        t = te.iloc[0]
        adj = t.usda * np.exp(a + b * t.r)
        res[int(y)] = (100 * (adj / t.final - 1), 100 * (t.usda / t.final - 1))
    return res


def main(nass_dir, out_json):
    full = json.loads((ROOT / 'research/evidence/momentum-full-2002-2025-20261116.json').read_text())
    rng = np.random.default_rng(42)
    cells, out = {}, {'cells': {}}
    for m in MONTHS:
        for reg in ('TX', 'US'):
            by = full[m][f'{reg}_by_year']
            yrs = sorted(int(y) for y in by)
            diff = np.array([abs(by[str(y)][1]) - abs(by[str(y)][0]) for y in yrs])
            cells[(m, reg)] = dict(zip(yrs, diff))
            boots = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(1)
            loo = np.array([np.delete(diff, i).mean() for i in range(len(diff))])
            k = int(np.argmax(np.abs(loo - diff.mean())))
            out['cells'][f'{m}_{reg}'] = {
                'n': len(diff), 'mean_gain_pts': float(diff.mean()), 'perm_p': perm_p(diff),
                'boot_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                'loo_most_influential_season': yrs[k], 'loo_mean_without_it': float(loo[k]),
                'loo_min_mean': float(loo.min()), 'loo_sign_flips': int((np.sign(loo) != np.sign(diff.mean())).sum())}
    keys = list(out['cells'])
    p = np.array([out['cells'][k]['perm_p'] for k in keys])
    for k, h, b in zip(keys, holm(p), bh(p)):
        out['cells'][k]['holm_p'], out['cells'][k]['bh_q'] = float(h), float(b)
    years = sorted(set().union(*[set(v) for v in cells.values()]))
    mat = np.array([[cells[c].get(y, np.nan) for c in cells] for y in years])
    obs = float(np.nanmean(mat))
    sims = np.array([np.nanmean(mat * rng.choice([-1.0, 1.0], size=(len(years), 1))) for _ in range(20000)])
    out['pooled'] = {'mean_gain_pts': obs, 'year_block_p': float((sims >= obs - 1e-12).mean())}
    out['coef_path'] = {m: coef_path(nass_dir, m) for m in MONTHS}
    pub = {}
    for m in MONTHS:
        for reg in ('TX', 'US'):
            ours = full[m][f'{reg}_by_year']
            row = {'momentum_rule': float(np.mean([abs(v[0]) for v in ours.values()])),
                   'usda': float(np.mean([abs(v[1]) for v in ours.values()]))}
            for name, w in (('published_style_10y', 10), ('published_style_expanding', None)):
                r = published_style(nass_dir, m, reg, w)
                common = [y for y in r if str(y) in ours]
                row[name] = float(np.mean([abs(r[y][0]) for y in common]))
                row[name + '_usda_same_years'] = float(np.mean([abs(r[y][1]) for y in common]))
                row[name + '_years'] = len(common)
            pub[f'{m}_{reg}'] = row
    out['published_style_comparison'] = pub
    for k, v in out['cells'].items():
        print(f"{k}: gain {v['mean_gain_pts']:+.2f} CI [{v['boot_ci95'][0]:+.2f},{v['boot_ci95'][1]:+.2f}] p {v['perm_p']:.3f} "
              f"Holm {v['holm_p']:.3f} BH {v['bh_q']:.3f} | LOO min {v['loo_min_mean']:+.2f} (drop {v['loo_most_influential_season']}), flips {v['loo_sign_flips']}")
    print('pooled', out['pooled'])
    for m, path in out['coef_path'].items():
        print('b path', m, min(path.values()), '..', max(path.values()), 'first', list(path.items())[:2], 'last', list(path.items())[-1])
    for k, v in pub.items():
        print(f"{k}: momentum {v['momentum_rule']:.2f} | published 10y {v['published_style_10y']:.2f} | expanding "
              f"{v['published_style_expanding']:.2f} | USDA {v['usda']:.2f}")
    Path(out_json).write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
