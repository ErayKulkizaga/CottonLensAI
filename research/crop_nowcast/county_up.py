"""County-level anomaly model aggregated to state/US production (development years only).

Fit: county target log(PPA / past-mean PPA) on past-only anomalies (NDVI vs the
county's earlier seasons at the same cutoff, weather z-scores vs 1991..Y-1).
Aggregate: state ratio = prior-year-planted-weighted mean of exp(pred) over panel
counties; production_hat = state ppa_base x ratio x planted_known / 480 (same
ppa_base and planted_known as state_model). Rolling origin 2013-2021, train < Y.
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
from research.crop_nowcast.state_model import (
    TEST_YEARS,
    state_frame,
    summarize,
)

NDVI = ['n_max', 'n_last', 'n_mean', 'n_int', 'n_p90max']
LANDSAT = ['l_p10', 'l_p25', 'l_p50', 'l_p75', 'l_p90', 'l_fail25', 'l_fail35', 'l_last', 'l_nobs']


def add_landsat(c, landsat_dir, cutoff_key):
    """Merge Move 3 Landsat county features for one cutoff; anomalies vs the county's earlier years only."""
    parts = [pd.read_csv(f, dtype={'fips': str}) for f in sorted(Path(landsat_dir).rglob('landsat_*_*.csv'))]
    lf = pd.concat(parts)
    lf = lf[lf.cutoff == cutoff_key][['fips', 'year', *LANDSAT]]
    c = c.merge(lf, on=['fips', 'year'], how='left').sort_values(['fips', 'year'])
    for k in LANDSAT:
        c[k + '_a'] = c[k] - c.groupby('fips')[k].transform(lambda s: s.expanding().mean().shift(1))
    return c


def prepare(csv):
    c = pd.read_csv(csv, dtype={'fips': str}).sort_values(['fips', 'year'])
    ok = c.y_planted.gt(0) & c.y_production.notna()
    c['ppa'] = np.where(ok, c.y_production * 480 / c.y_planted, np.nan)
    c['ppa_base'] = c.groupby('fips').ppa.transform(lambda s: s.expanding().mean().shift(1))
    c['target'] = np.log(c.ppa.clip(lower=1) / c.ppa_base)
    for k in NDVI:
        c[k + '_a'] = c[k] - c.groupby('fips')[k].transform(lambda s: s.expanding().mean().shift(1))
    c['w'] = c.groupby('fips').y_planted.shift(1)
    return c


FSA = ['f_fail', 'f_fail_dry', 'f_fail_irr', 'f_irr_share', 'f_fail_a']
FSA_MONTH = {'aug1': 'aug', 'sep1': 'sep', 'oct1': 'oct'}


def add_fsa(c, fsa_csv, cutoff_key):
    """County FSA failed-acre shares from the in-season file of the cutoff month (FSA 'planted' excludes failed).
    2011 files have no irrigation split, so the split features are left empty for 2011."""
    d = pd.read_csv(fsa_csv, dtype={'st': str, 'co': str})
    d = d[d.month == FSA_MONTH[cutoff_key]].assign(fips=lambda x: x.st.str.zfill(2) + x.co.str.zfill(3),
                                                   tot=lambda x: x.planted + x.failed)
    g = d.groupby(['fips', 'year'])[['failed', 'tot']].sum()
    out = pd.DataFrame({'f_fail': g.failed / g.tot.where(g.tot > 0)})
    for code, name in (('N', 'f_fail_dry'), ('I', 'f_fail_irr')):
        h = d[d.irr == code].groupby(['fips', 'year'])[['failed', 'tot']].sum()
        out[name] = h.failed / h.tot.where(h.tot > 0)
    irr = d[d.irr == 'I'].groupby(['fips', 'year']).tot.sum()
    out['f_irr_share'] = (irr / g.tot).fillna(0)
    out.loc[out.index.get_level_values('year') == 2011, ['f_fail_dry', 'f_fail_irr', 'f_irr_share']] = np.nan
    c = c.merge(out.reset_index(), on=['fips', 'year'], how='left').sort_values(['fips', 'year'])
    c['f_fail_a'] = c.f_fail - c.groupby('fips').f_fail.transform(lambda s: s.expanding().mean().shift(1))
    return c

def run(c, nass_dir, cutoff_key, cols, kind, area='mar', nan_ok=()):
    s, us = state_frame(nass_dir, cutoff_key, area)
    rows = []
    for y in TEST_YEARS:
        tr = c[(c.year < y) & c.target.notna()].dropna(subset=[k for k in cols if k not in nan_ok])  # HGB handles NaN
        te = c[(c.year == y) & c.ppa_base.notna() & c.w.notna()].copy()
        if kind == 'external':  # predictions made elsewhere (e.g. the T35 GPU model), column p_ext
            te = te.assign(p=te.p_ext).dropna(subset=['p'])
        elif kind == 'ridge':
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))).fit(tr[cols], tr.target)
            te['p'] = m.predict(te[cols].fillna(tr[cols].mean()))
        else:
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30,
                                              l2_regularization=2.0, random_state=0).fit(tr[cols], tr.target)
            te['p'] = m.predict(te[cols])
        ratio = te.groupby('state').apply(lambda x: np.average(np.exp(x.p), weights=x.w), include_groups=False).rename('ratio')
        st = s[s.year == y].merge(ratio, left_on='state', right_index=True)
        rows.append(st.assign(pred=st.ppa_base * st.ratio * st.planted_known / 480, base=st.ppa_base * st.planted_known / 480))
    return pd.concat(rows), us


def main(feature_dir, nass_dir, out_json, cutoffs='aug1,sep1,oct1', area='mar', landsat_dir=None):
    res = {}
    for ck in cutoffs.split(','):
        c = prepare(Path(feature_dir) / f'features_{ck}.csv')
        if landsat_dir:
            c = add_landsat(c, landsat_dir, ck)
        wz = [k for k in c.columns if k.startswith('w_') and k.endswith('_z')]
        na = [k + '_a' for k in NDVI]
        res[ck] = {}
        for label, cols in (('weather z', wz), ('ndvi anom', na), ('weather z + ndvi anom', wz + na)):
            for kind in ('ridge', 'hgb'):
                r, us = run(c, nass_dir, ck, cols, kind, area)
                res[ck][f'{label}/{kind}'] = summarize(r, us, f'{ck} {area} {label} {kind}')
        if landsat_dir:  # Move 3: primary = ndvi anom + landsat (HGB); landsat alone is secondary
            la = LANDSAT + [k + '_a' for k in LANDSAT]
            for label, cols in (('MOVE3 PRIMARY ndvi anom + landsat', na + la), ('landsat only', la)):
                r, us = run(c, nass_dir, ck, cols, 'hgb', area)
                res[ck][f'{label}/hgb'] = summarize(r, us, f'{ck} {area} {label} hgb')
    Path(out_json).write_text(json.dumps(res, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:7])
