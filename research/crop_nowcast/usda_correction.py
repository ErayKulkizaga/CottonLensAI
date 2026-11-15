"""Correct USDA's monthly cotton production forecast by predicting its own error (dev years only).

Target per state-year: e = log(final / USDA_m). Corrected forecast = USDA_m * exp(e_hat).
e_hat comes from a small pooled linear model fitted on states x years < Y (rolling origin):
  - 'bias'      : intercept only (past mean error of USDA, pooled over states)
  - 'sat'       : intercept + sat_gap, where sat_gap = log(satellite model / USDA_m)
  - 'sat+fsa'   : + fsa_gap, where fsa_gap = FSA failed share - USDA implied abandonment
US = USDA_US * (sum of corrected panel states / sum of USDA panel states).
Everything used for year Y is public before USDA's month-m report, except the assumption
on FSA release timing recorded in the ledger. Sealed years 2022-2025 are not read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import county_up as C
from research.crop_nowcast.state_model import TEST_YEARS, nass_state, state_frame

MONTH = {'aug1': 'AUG', 'sep1': 'SEP', 'oct1': 'OCT'}
FSA_MONTH = {'aug1': 'aug', 'sep1': 'sep', 'oct1': 'oct'}


def usda_frame(nass_dir, ck):
    def pick(name, ref):
        d = nass_state(nass_dir, name)
        return d[d.reference_period_desc == ref].set_index(['state_alpha', 'year']).v
    m = {**MONTH, 'nov1': 'NOV', 'dec1': 'DEC'}[ck]
    f = pd.DataFrame({'final': pick('state_production', 'YEAR'), 'usda': pick('state_production', f'YEAR - {m} FORECAST'),
                      'harv_m': pick('state_harvested', f'YEAR - {m} FORECAST')})
    f.index = f.index.set_names(['state', 'year'])
    us = nass_state(nass_dir, 'us_production')
    usf = pd.DataFrame({'final': us[us.reference_period_desc == 'YEAR'].set_index('year').v,
                        'usda': us[us.reference_period_desc == f'YEAR - {m} FORECAST'].set_index('year').v})
    return f.reset_index(), usf


def fsa_frame(fsa_csv, ck):
    if not fsa_csv or not Path(fsa_csv).exists():
        return None
    d = pd.read_csv(fsa_csv, dtype={'st': str, 'co': str})
    d = d[d.month == FSA_MONTH[ck]]
    g = d.groupby(['year', 'st'])[['planted', 'failed']].sum().reset_index()
    tot = g.planted + g.failed  # FSA 'planted' excludes failed acres
    g['fsa_fail'] = g.failed / tot.where(tot > 0)
    return g


CUTOFF_MD = {'aug1': '08-01', 'sep1': '09-01', 'oct1': '10-01'}


def usdm_frame(usdm_csv, feature_csv, ck):
    """State drought features from county maps released (map date + 2 days) on or before the cutoff,
    averaged with each county's PRIOR-year planted area as weight."""
    u = pd.read_csv(usdm_csv, dtype={'fips': str}, parse_dates=['map_date'])
    u['year'] = u.map_date.dt.year
    u['d2p'] = u.d2 + u.d3 + u.d4
    lim = pd.to_datetime(u.year.astype(str) + '-' + CUTOFF_MD[ck]) - pd.Timedelta(days=2)
    u = u[(u.map_date <= lim) & (u.map_date.dt.month >= 4)]
    last = u.sort_values('map_date').groupby(['fips', 'year']).d2p.last().rename('usdm_last')
    mean = u.groupby(['fips', 'year']).d2p.mean().rename('usdm_mean')
    c = pd.read_csv(feature_csv, dtype={'fips': str})[['fips', 'state', 'year', 'y_planted']].sort_values(['fips', 'year'])
    c['w'] = c.groupby('fips').y_planted.shift(1)
    c = c.join(last, on=['fips', 'year']).join(mean, on=['fips', 'year']).dropna(subset=['w', 'usdm_last'])
    return c.groupby(['state', 'year']).apply(
        lambda x: pd.Series({'usdm_last': np.average(x.usdm_last, weights=x.w) / 100,
                             'usdm_mean': np.average(x.usdm_mean, weights=x.w) / 100}), include_groups=False).reset_index()


def cond_frame(cond_csv, ck):
    """Share of the state's cotton rated poor or very poor in the last weekly report released by the cutoff."""
    d = pd.read_csv(cond_csv, parse_dates=['release'])
    d['year'] = d.release.dt.year
    lim = pd.to_datetime(d.year.astype(str) + '-' + CUTOFF_MD[ck])
    d = d[d.release <= lim].sort_values('release')
    g = d.groupby(['state', 'year']).last().reset_index()
    g['cond_pp'] = (g.vp + g.p) / 100
    return g[['state', 'year', 'cond_pp']]


def build(feature_dir, nass_dir, ck, fsa_csv=None, area='jun', usdm_csv=None, cond_csv=None):
    f, us = usda_frame(nass_dir, ck)
    c = C.prepare(Path(feature_dir) / f'features_{ck}.csv')
    r, _ = C.run(c, nass_dir, ck, [k + '_a' for k in C.NDVI], 'hgb', area)
    f = f.merge(r[['state', 'year', 'pred']], on=['state', 'year'], how='left')
    f['e'] = np.log(f.final / f.usda)
    f['sat_gap'] = np.log(f.pred / f.usda)
    known, _ = state_frame(nass_dir, ck, 'jun')  # June Acreage planted, public before every cutoff
    f = f.merge(known[['state', 'year', 'planted_known']], on=['state', 'year'], how='left')
    f['usda_aband'] = 1 - f.harv_m / f.planted_known
    g = fsa_frame(fsa_csv, ck)
    if g is not None:
        ids = nass_state(nass_dir, 'state_production')[['state_ansi', 'state_alpha']].drop_duplicates()
        g['state'] = g.st.map(dict(zip(ids.state_ansi.str.zfill(2), ids.state_alpha, strict=True)))
        f = f.merge(g[['state', 'year', 'fsa_fail']], on=['state', 'year'], how='left')
        f['fsa_gap'] = f.fsa_fail - f.usda_aband
    if usdm_csv:
        f = f.merge(usdm_frame(usdm_csv, Path(feature_dir) / f'features_{ck}.csv', ck), on=['state', 'year'], how='left')
    if cond_csv:
        f = f.merge(cond_frame(cond_csv, ck), on=['state', 'year'], how='left')
    return f, us


def fit_predict(f, cols, ridge=1.0, rows=None, coefs=None):
    """Rolling pooled ridge on fold-standardised features (intercept unpenalised).
    rows: extra columns that must be present in training rows (to match another variant's rows).
    coefs: optional dict filled with {year: coefficients} for inspection."""
    need = list(dict.fromkeys([*cols, *(rows or [])]))
    out = []
    for y in TEST_YEARS:
        tr = f[(f.year < y)].dropna(subset=['e', *need])
        te = f[(f.year == y)].dropna(subset=['usda'])
        if need and len(tr) < 10:  # too few past rows with this signal: leave USDA unchanged
            eh = np.zeros(len(te))
        elif cols:
            mu, sd = tr[cols].mean(), tr[cols].std(ddof=0).replace(0, 1)
            X = np.c_[np.ones(len(tr)), ((tr[cols] - mu) / sd).to_numpy()]
            A = X.T @ X + ridge * np.diag([0] + [1] * len(cols))
            b = np.linalg.solve(A, X.T @ tr.e.to_numpy())
            xt = np.c_[np.ones(len(te)), ((te[cols].fillna(mu) - mu) / sd).to_numpy()]
            eh = xt @ b
            if coefs is not None:
                coefs[y] = [round(float(v), 4) for v in b]
        else:
            eh = np.full(len(te), tr.e.mean() if len(tr) else 0.0)
        out.append(te.assign(corr=te.usda * np.exp(eh)))
    return pd.concat(out)


def score(r, us):
    tx = r[r.state == 'TX'].dropna(subset=['final'])
    res = {'TX_model': 100 * (tx['corr'] / tx.final - 1).abs().mean(),
           'TX_usda': 100 * (tx.usda / tx.final - 1).abs().mean()}
    s = r.groupby('year')[['corr', 'usda']].sum()
    j = s.join(us, rsuffix='_us').dropna()
    j['us_corr'] = j.usda_us * j['corr'] / j.usda
    res['US_model'] = 100 * (j.us_corr / j.final - 1).abs().mean()
    res['US_usda'] = 100 * (j.usda_us / j.final - 1).abs().mean()
    res['TX_by_year'] = {int(y): (round(100 * (a / q - 1), 1), round(100 * (u / q - 1), 1))
                         for y, a, u, q in zip(tx.year, tx['corr'], tx.usda, tx.final, strict=True)}
    return res


def main(feature_dir, nass_dir, out_json, fsa_csv=None, usdm_csv=None, cond_csv=None):
    out = {}
    for ck in MONTH:
        f, us = build(feature_dir, nass_dir, ck, fsa_csv, usdm_csv=usdm_csv, cond_csv=cond_csv)
        out[ck] = {}
        variants = [('bias', []), ('sat', ['sat_gap'])]
        if 'fsa_gap' in f:
            variants += [('fsa', ['fsa_gap']), ('sat+fsa', ['sat_gap', 'fsa_gap'])]
        if 'usdm_last' in f:
            variants += [('usdm', ['usdm_last', 'usdm_mean'])]
        if 'cond_pp' in f:
            variants += [('cond', ['cond_pp'])]
        if {'fsa_gap', 'usdm_last', 'cond_pp'} <= set(f):
            variants += [('fsa+cond', ['fsa_gap', 'cond_pp']), ('fsa+usdm', ['fsa_gap', 'usdm_last', 'usdm_mean']),
                         ('fsa+usdm+cond', ['fsa_gap', 'usdm_last', 'usdm_mean', 'cond_pp']),
                         ('all', ['sat_gap', 'fsa_gap', 'usdm_last', 'usdm_mean', 'cond_pp'])]
        for name, cols in variants:
            s = score(fit_predict(f, cols), us)
            out[ck][name] = s
            print(f"{ck} {name:8s} TX {s['TX_model']:5.1f} vs USDA {s['TX_usda']:5.1f} | US {s['US_model']:4.1f} vs USDA {s['US_usda']:4.1f}")
    Path(out_json).write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:7])
