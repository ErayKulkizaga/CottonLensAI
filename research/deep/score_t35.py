"""Score T35 county predictions with the same local aggregation as T06 (area fsa_aug), per the pre-declared rule.

Seed mean of county predictions -> state/US. The HGB reference (T06, ndvi anom) is scored on identical
test rows (counties without a CNN prediction are dropped from both). Seed spread = min-max of per-seed results.
Usage: python score_t35.py preds_dir features_<ck>.csv nass_dir <ck> out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import county_up as C
from research.crop_nowcast.state_model import summarize


def state_w(r, col):
    """Production-weighted mean absolute state error (%)."""
    r = r.dropna(subset=[col, 'prod'])
    return float(100 * np.average((r[col] / r['prod'] - 1).abs(), weights=r['prod']))


def score(c, nass_dir, ck, cols, kind, label):
    r, us = C.run(c, nass_dir, ck, cols, kind, area='fsa_aug')
    s = summarize(r, us, label)
    return {'TX': s['TX_model_mape'], 'US': s['US_model_mape'], 'TX_usda': s['TX_usda_mape'], 'US_usda': s['US_usda_mape'],
            'state_w': state_w(r, 'pred'), 'state_w_usda': state_w(r, 'usda'), 'TX_by_year': s['TX_by_year']}


def main(preds_dir, features_csv, nass_dir, ck, out_json):
    p = pd.concat(pd.read_csv(f, dtype={'fips': str}).assign(seed=int(f.stem.split('_')[2]))
                  for f in sorted(Path(preds_dir).glob('preds_*_*.csv')))
    p['fips'] = p.fips.str.zfill(5)
    assert p.groupby('seed').size().nunique() == 1 and p.seed.nunique() == 5, 'missing (year, seed) units'
    base = C.prepare(features_csv)
    na = [k + '_a' for k in C.NDVI]

    def with_ext(q):
        c = base.merge(q[['fips', 'year', 'p']].rename(columns={'p': 'p_ext'}), on=['fips', 'year'], how='left')
        c.loc[c.year.isin(C.TEST_YEARS) & c.p_ext.isna(), 'w'] = np.nan  # identical test rows; training unaffected
        return c

    mean = p.groupby(['fips', 'year'], as_index=False).p.mean()
    c = with_ext(mean)
    res = {'cnn_seed_mean': score(c, nass_dir, ck, na, 'external', f'{ck} T35 CNN seed mean'),
           'hgb_reference': score(c, nass_dir, ck, na, 'hgb', f'{ck} T06 HGB same rows'),
           'seeds': {int(s): score(with_ext(q), nass_dir, ck, na, 'external', f'{ck} T35 seed {s}')
                     for s, q in p.groupby('seed')}}
    for k in ('TX', 'US'):
        v = [r[k] for r in res['seeds'].values()]
        res[f'seed_spread_{k}'] = max(v) - min(v)
    m, h = res['cnn_seed_mean'], res['hgb_reference']
    res['pass'] = {'beats_usda': m['TX'] < m['TX_usda'] and m['US'] < m['US_usda'],
                   'beats_t25': m['TX'] < 10.4 and m['US'] < 4.9,
                   'state_w_below_usda': m['state_w'] < m['state_w_usda'],
                   'beats_hgb_by_spread': all(h[k] - m[k] > res[f'seed_spread_{k}'] for k in ('TX', 'US'))}
    res['pass']['ALL'] = all(res['pass'].values())
    print(json.dumps({k: res[k] for k in ('pass', 'seed_spread_TX', 'seed_spread_US')}, indent=1))
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:6])
