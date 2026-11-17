"""T51 (docs/prereg/T51_AREA_PRODUCTION_PREREG_20261116.md): Texas area baselines, conformal intervals, production effect,
structural break. Usage: python t51_area_production.py t49_root out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M
from research.crop_nowcast.t49_anchoring import cotton_rows

MONTHS = ('SEP', 'OCT', 'NOV', 'DEC')
FIRST, MIN_TRAIN = 2008, 8


def ols(x, y):
    X = np.c_[np.ones(len(x)), x]
    return np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]


def predict(tr, te_pvp, model):
    if model == 'usda':
        return 0.0
    if model == 'bias':
        return float(tr.e.mean())
    if model == 'bias10':
        return float(tr[tr.year >= tr.year.max() - 9].e.mean())
    sub = tr if model == 'anchor' else tr[tr.year >= tr.year.max() - 9]
    a, b = ols(sub.pvp, sub.e)
    return float(a + b * te_pvp)


def loso_resid(tr):
    out = []
    for y in tr.year.unique():
        a, b = ols(tr[tr.year != y].pvp, tr[tr.year != y].e)
        r = tr[tr.year == y]
        out += list(r.e - (a + b * r.pvp))
    return np.abs(np.array(out))


def block_p(gain_by_year, seed=42, n=20000):
    g = np.asarray(gain_by_year, float)
    rng = np.random.default_rng(seed)
    sims = (g * rng.choice([-1.0, 1.0], size=(n, len(g)))).mean(1)
    return float((sims >= g.mean() - 1e-12).mean())


def main(root, out_json):
    cond = pd.read_csv(Path(root) / 'condition.csv')
    d = cotton_rows(cond, ['TX'])
    d['e'] = np.log(d.final / d.area)
    prod = M.frame(ROOT / 'output/crop-nowcast-full/nass', 'state')
    prod = prod[prod.state == 'TX'].set_index('year')
    mom = json.loads((ROOT / 'research/evidence/momentum-full-2002-2025-20261116.json').read_text())
    rows = []
    for m in MONTHS:
        g = d[d.month == m]
        for y in sorted(g.year.unique()):
            tr = g[g.year < y]
            if y < FIRST or tr.year.nunique() < MIN_TRAIN:
                continue
            te = g[g.year == y].iloc[0]
            rec = {'year': int(y), 'month': m, 'e': float(te.e), 'pvp': float(te.pvp)}
            for mod in ('usda', 'bias', 'bias10', 'anchor', 'anchor10'):
                rec[f'eh_{mod}'] = predict(tr, te.pvp, mod)
            res = loso_resid(tr)
            rec['q80'], rec['q95'] = float(np.quantile(res, 0.8)), float(np.quantile(res, 0.95))
            if y in prod.index and not np.isnan(prod.loc[y, m]) and not np.isnan(prod.loc[y, 'final']):
                P, Pf = float(prod.loc[y, m]), float(prod.loc[y, 'final'])
                rec['prod_err_usda'] = 100 * (P / Pf - 1)
                rec['prod_err_area'] = 100 * (P * np.exp(rec['eh_anchor']) / Pf - 1)
                mb = mom[m]['TX_by_year'].get(str(y))
                rec['prod_err_momentum'] = mb[0] if mb else np.nan
            rows.append(rec)
    r = pd.DataFrame(rows)
    out = {'seasons': [int(r.year.min()), int(r.year.max())], 'A_area_mae_pct': {}, 'B_intervals': {}, 'C_production': {}}
    for mod in ('usda', 'bias', 'bias10', 'anchor', 'anchor10'):
        r[f'ae_{mod}'] = 100 * np.abs(np.expm1(r.e - r[f'eh_{mod}']))
        out['A_area_mae_pct'][mod] = {m: float(r[r.month == m][f'ae_{mod}'].mean()) for m in MONTHS}
        out['A_area_mae_pct'][mod]['all'] = float(r[f'ae_{mod}'].mean())
    out['A_p_anchor_vs'] = {mod: block_p((r[f'ae_{mod}'] - r.ae_anchor).groupby(r.year).mean()) for mod in ('usda', 'bias', 'bias10', 'anchor10')}
    resid = np.abs(r.e - r.eh_anchor)
    for lvl in ('80', '95'):
        cov = resid <= r[f'q{lvl}']
        out['B_intervals'][lvl] = {'coverage_all': float(cov.mean()),
                                   'coverage_by_month': {m: float(cov[r.month == m].mean()) for m in MONTHS},
                                   'mean_halfwidth_pct': float(100 * np.expm1(r[f'q{lvl}']).mean())}
    pr = r.dropna(subset=['prod_err_usda'])
    c = {m: {'usda': float(pr[pr.month == m].prod_err_usda.abs().mean()), 'area_adjusted': float(pr[pr.month == m].prod_err_area.abs().mean()),
             'momentum': float(pr[pr.month == m].prod_err_momentum.abs().mean()), 'seasons': int((pr.month == m).sum())} for m in MONTHS}
    better = sum(c[m]['area_adjusted'] < c[m]['usda'] for m in MONTHS)
    gain = (pr.prod_err_usda.abs() - pr.prod_err_area.abs()).groupby(pr.year).mean()
    out['C_production'] = {'by_month': c, 'months_better_than_usda': int(better), 'year_block_p_vs_usda': block_p(gain),
                           'year_block_p_vs_momentum': block_p((pr.prod_err_momentum.abs() - pr.prod_err_area.abs()).groupby(pr.year).mean())}
    out['C_pass'] = bool(better >= 3 and out['C_production']['year_block_p_vs_usda'] < 0.05)
    s = d[(d.month == 'SEP') & d.year.between(2000, 2025)].sort_values('year')
    x, y, yrs = s.pvp.to_numpy(), s.e.to_numpy(), s.year.to_numpy()

    def ssr(xx, yy):
        a, b = ols(xx, yy)
        return float(((yy - a - b * xx) ** 2).sum())

    def fstat(xx, yy, k):
        full = ssr(xx, yy)
        split = ssr(xx[:k], yy[:k]) + ssr(xx[k:], yy[k:])
        return ((full - split) / 2) / (split / (len(yy) - 4))

    ks = [i for i, yr in enumerate(yrs) if 2005 <= yr <= 2019 and i >= 5 and len(yrs) - i >= 5]
    fs = {int(yrs[k]): float(fstat(x, y, k)) for k in ks}
    k12 = int(np.where(yrs == 2012)[0][0])
    from scipy.stats import f as fdist
    chow = fs[2012]
    rng = np.random.default_rng(42)
    sims = []
    for _ in range(5000):
        o = rng.permutation(len(y))
        sims.append(max(fstat(x[o], y[o], k) for k in ks))
    out['D_break'] = {'chow_F_2012': chow, 'chow_p_2012': float(1 - fdist.cdf(chow, 2, len(y) - 4)),
                      'supF_year': max(fs, key=fs.get), 'supF': max(fs.values()),
                      'supF_perm_p': float((np.array(sims) >= max(fs.values()) - 1e-12).mean()), 'F_by_year': fs, 'n': int(len(y)), 'k2012': k12}
    out['rows'] = r.round(5).to_dict(orient='records')
    for k in ('A_area_mae_pct', 'A_p_anchor_vs', 'B_intervals', 'C_production', 'C_pass', 'D_break'):
        v = out[k]
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != 'F_by_year'} if isinstance(v, dict) else v, default=float)[:900])
    Path(out_json).write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
