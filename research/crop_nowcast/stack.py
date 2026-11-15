"""T36: rolling percentage ensemble around the USDA correction, as pre-declared in docs/CROP_NOWCAST_TRIALS.md.

Members per state-year: B = T25 bias-corrected USDA, H = T06 HGB (fsa_aug area), N = T35 CNN seed mean
(when a preds dir is given), U = raw USDA. Ensemble = exp(sum w_k log member_k), w on a 0.05 simplex grid,
chosen for year Y on states x years 2013..Y-1 (out-of-sample member predictions only). Fewer than 2 past
years, or a state missing a member -> B. Scored with the T25 scorer. Dev years only.
Usage: python stack.py feature_dir nass_dir fsa_csv out.json [cnn_preds_root]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import county_up as C
from research.crop_nowcast import usda_correction as UC


def grid(k, step=0.05):
    """All weight vectors of length k on the step-simplex (stars and bars)."""
    n = round(1 / step)
    rows = [np.diff([-1, *c, n + k - 1]) - 1 for c in itertools.combinations(range(n + k - 1), k - 1)]
    return np.array(rows) / n


def cnn_state(preds_dir, features_csv, nass_dir, ck):
    p = pd.concat(pd.read_csv(f, dtype={'fips': str}) for f in Path(preds_dir).glob('preds_*_*.csv'))
    p = p.assign(fips=p.fips.str.zfill(5)).groupby(['fips', 'year'], as_index=False).p.mean()
    c = C.prepare(features_csv).merge(p.rename(columns={'p': 'p_ext'}), on=['fips', 'year'], how='left')
    r, _ = C.run(c, nass_dir, ck, [], 'external', area='fsa_aug')
    return r[['state', 'year', 'pred']].rename(columns={'pred': 'N'})


def ensemble(f, members, weights_out):
    """f has columns B and the members; returns f with column 'corr' = ensemble (B where not available)."""
    L = np.log(f[members])
    err = np.log(f.final)
    W = grid(len(members))
    out = f.B.copy()
    for y in UC.TEST_YEARS:
        past = f.year.lt(y) & f.year.ge(2013) & L.notna().all(axis=1) & f.final.notna()
        if f[past].year.nunique() < 2:
            weights_out[y] = None
            continue
        sse = ((L[past].to_numpy() @ W.T - err[past].to_numpy()[:, None]) ** 2).sum(axis=0)
        w = W[sse.argmin()]
        weights_out[y] = dict(zip(members, (round(float(v), 2) for v in w), strict=True))
        cur = f.year.eq(y) & L.notna().all(axis=1)
        out[cur] = np.exp(L[cur].to_numpy() @ w)
    return f.assign(corr=out)


def state_w(r, col):
    r = r.dropna(subset=[col, 'final'])
    return float(100 * np.average((r[col] / r.final - 1).abs(), weights=r.final))


def main(feature_dir, nass_dir, fsa_csv, out_json, cnn_root=None):
    res = {}
    for ck in ('aug1', 'sep1', 'oct1'):
        f, us = UC.build(feature_dir, nass_dir, ck, fsa_csv)
        b = UC.fit_predict(f, [], rows=['fsa_gap'])
        f = f.merge(b[['state', 'year', 'corr']].rename(columns={'corr': 'B'}), on=['state', 'year'])
        csv = Path(feature_dir) / f'features_{ck}.csv'
        h, _ = C.run(C.prepare(csv), nass_dir, ck, [k + '_a' for k in C.NDVI], 'hgb', area='fsa_aug')
        f = f.merge(h[['state', 'year', 'pred']].rename(columns={'pred': 'H'}), on=['state', 'year'], how='left')
        f['U'] = f.usda
        members = ['B', 'H', 'U']
        if cnn_root and (Path(cnn_root) / f'preds_{ck}').exists():
            f = f.merge(cnn_state(Path(cnn_root) / f'preds_{ck}', csv, nass_dir, ck), on=['state', 'year'], how='left')
            members.insert(2, 'N')
        f = f[f.year.isin(UC.TEST_YEARS)]
        weights = {}
        e = ensemble(f, members, weights)
        t25 = UC.score(f.assign(corr=f.B), us)
        ens = UC.score(e, us)
        r = {'members': members, 'weights': weights, 'T36': ens, 'T25_same_rows': t25,
             'state_w': state_w(e, 'corr'), 'state_w_T25': state_w(f, 'B'), 'state_w_usda': state_w(f, 'usda')}
        r['pass'] = {'beats_usda': ens['TX_model'] < ens['TX_usda'] and ens['US_model'] < ens['US_usda'],
                     'beats_t25': ens['TX_model'] < t25['TX_model'] and ens['US_model'] < t25['US_model'],
                     'state_w': r['state_w'] < min(r['state_w_usda'], r['state_w_T25'])}
        r['pass'] = {k: bool(v) for k, v in r['pass'].items()}
        r['pass']['ALL'] = all(r['pass'].values())
        r['TX_years_better_than_T25'] = sum(abs(a) < abs(b) for (a, _), (b, _) in
                                            zip(ens['TX_by_year'].values(), t25['TX_by_year'].values(), strict=True))
        res[ck] = r
        print(f"{ck} T36 {'+'.join(members)}: TX {ens['TX_model']:5.1f} | T25 {t25['TX_model']:5.1f} | USDA {ens['TX_usda']:5.1f}"
              f"   US {ens['US_model']:4.1f} | T25 {t25['US_model']:4.1f} | USDA {ens['US_usda']:4.1f}"
              f"   state_w {r['state_w']:4.1f} | T25 {r['state_w_T25']:4.1f} | USDA {r['state_w_usda']:4.1f}  pass={r['pass']['ALL']}")
        print('   weights', weights)
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:6])
