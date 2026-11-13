"""Development-years evaluation (2013-2021, rolling origin) of county production per PLANTED acre.

Why this target: yield per harvested acre hides drought because failed fields are
abandoned before harvest (2011 Texas: harvested ratio 0.52). Production per planted
acre (PPA = bales*480/planted) carries both effects and maps directly to the
production number USDA forecasts.

Each test year Y uses a model trained only on years < Y; the baseline is the
county's mean PPA over earlier years. Feature anomalies are relative to the same
county's earlier years only. Sealed years 2022-2025 are untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import features as F  # noqa: E402

TEST_YEARS = range(2013, 2022)


def past_mean(df, col):
    return df.groupby('fips')[col].transform(lambda s: s.expanding().mean().shift(1))


def prepare(df):
    df = df.sort_values(['fips', 'year']).copy()
    ok = df.y_planted.gt(0) & df.y_production.notna()
    df['ppa'] = np.where(ok, df.y_production * 480 / df.y_planted, np.nan)
    df['ppa_base'] = past_mean(df, 'ppa')
    df['target'] = np.log(df.ppa.clip(lower=1) / df.ppa_base)
    raw = [c for c in df.columns if c.startswith(('w_', 'n_')) and not c.endswith('_anom')]
    for c in raw:
        df[c + '_a'] = df[c] - past_mean(df, c)
    return df, raw


def fit_predict(df, cols, kind):
    out = []
    for y in TEST_YEARS:
        tr = df[(df.year < y) & df.target.notna()].dropna(subset=cols) if kind == 'ridge' else df[(df.year < y) & df.target.notna()]
        te = df[(df.year == y) & df.ppa_base.notna()]
        if kind == 'ridge':
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
            m.fit(tr[cols], tr.target)
            pred = m.predict(te[cols].fillna(tr[cols].mean()))
        else:
            m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30,
                                              l2_regularization=2.0, random_state=0)
            m.fit(tr[cols], tr.target)
            pred = m.predict(te[cols])
        out.append(pd.DataFrame({'fips': te.fips, 'state': te.state, 'year': y, 'base': te.ppa_base,
                                 'pred': te.ppa_base * np.exp(pred), 'actual': te.ppa, 'planted': te.y_planted}))
    return pd.concat(out)


def report(name, r):
    r = r.dropna(subset=['actual'])
    em, eb = r.pred - r.actual, r.base - r.actual
    w = r.planted.fillna(0)
    rm, rb = np.sqrt((em ** 2).mean()), np.sqrt((eb ** 2).mean())
    wm, wb = np.average(em.abs(), weights=w), np.average(eb.abs(), weights=w)
    yearly = r.assign(a=em ** 2, b=eb ** 2).groupby('year')[['a', 'b']].mean()
    res = {'n': int(len(r)), 'rmse': float(rm), 'rmse_base': float(rb), 'gain_pct': float(100 * (1 - rm / rb)),
           'planted_wtd_mae_gain_pct': float(100 * (1 - wm / wb)), 'years_better': int((yearly.a < yearly.b).sum()),
           'years': int(len(yearly))}
    print(f"{name:30s} n={res['n']} RMSE {rm:6.1f} vs base {rb:6.1f} -> {res['gain_pct']:5.1f}% | planted-wtd MAE "
          f"{res['planted_wtd_mae_gain_pct']:5.1f}% | years better {res['years_better']}/{res['years']}")
    return res


def main(nass_dir, power_dir, ndvi_dir, panel_json, out_dir, cutoffs='aug1,oct1'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for ck in cutoffs.split(','):
        df = F.build(nass_dir, power_dir, ndvi_dir if ndvi_dir != '-' else None, panel_json, ck)
        df, raw = prepare(df)
        df.to_csv(out / f'features_{ck}.csv', index=False)
        w = [c for c in raw if c.startswith('w_')]
        n = [c for c in raw if c.startswith('n_')]
        wa = [c + '_a' for c in w]
        na = [c + '_a' for c in n]
        sets = {'weather': wa + w + ['mean_harv_ratio']}
        if n:
            sets['ndvi'] = na + n
            sets['weather+ndvi'] = wa + w + na + n + ['mean_harv_ratio']
        res[ck] = {}
        for sname, cols in sets.items():
            for kind in ('ridge', 'hgb'):
                r = fit_predict(df, cols, kind)
                r.to_csv(out / f'pred_{ck}_{sname}_{kind}.csv', index=False)
                res[ck][f'{sname}/{kind}'] = report(f'{ck} {sname} {kind}', r)
    (out / 'dev_results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:7])
