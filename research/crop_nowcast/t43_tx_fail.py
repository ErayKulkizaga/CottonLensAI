"""T43: Landsat failed-field share (l_fail35) corrects USDA's Texas forecast (pre-declared in docs/CROP_NOWCAST_TRIALS.md).

x_Y = TX l_fail35 at the cutoff, county values weighted by prior-year planted acres (NASS county, years < Y).
Rolling TX-only ridge of e = log(final / USDA_m) on [1, z(x)], lambda 1, trained on 2010..Y-1 (>= 5 rows, else T25).
T25 = pooled past-mean bias on the FSA variant's rows (as defined), scored on the same TX rows.
Usage: python t43_tx_fail.py nass_dir feature_dir fsa_csv out.json landsat_dir [landsat_dir ...]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault('COTTONLENS_LAST_YEAR', '2024')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import features as F
from research.crop_nowcast import usda_correction as UC

FIRST_TRAIN, MIN_ROWS = 2010, 5
YEARS = range(2013, 2026)


def tx_fail(landsat_dirs, nass_dir, ck):
    lf = pd.concat(pd.read_csv(f, dtype={'fips': str}) for d in landsat_dirs for f in Path(d).rglob('landsat_TX_*.csv'))
    lf = lf[lf.cutoff == ck].drop_duplicates(['fips', 'year'], keep='last')
    pl = F.nass_county(nass_dir, 'county_planted').rename('planted').reset_index().astype({'year': int})
    pl['year'] += 1  # weight for season Y = planted in Y-1
    x = lf.merge(pl, on=['fips', 'year']).dropna(subset=['l_fail35', 'planted'])
    return x.groupby('year').apply(lambda g: np.average(g.l_fail35, weights=g.planted), include_groups=False)


def t25_tx(feature_dir, nass_dir, fsa_csv, ck):
    f, _ = UC.build(feature_dir, nass_dir, ck, fsa_csv)
    r = UC.fit_predict(f, [], rows=['fsa_gap']).set_index(['state', 'year'])['corr']
    out = {y: r[('TX', y)] for y in YEARS if ('TX', y) in r.index}
    tr = f[f.year < 2025].dropna(subset=['e', 'fsa_gap'])  # 2025: T25 exactly as in T42
    u25 = f[(f.state == 'TX') & (f.year == 2025)].usda
    if len(u25) and u25.notna().all():
        out[2025] = float(u25.iloc[0] * np.exp(tr.e.mean()))
    return out


def run(nass_dir, feature_dir, fsa_csv, landsat_dirs, ck):
    x = tx_fail(landsat_dirs, nass_dir, ck)
    f, _ = UC.usda_frame(nass_dir, ck)
    tx = f[f.state == 'TX'].set_index('year').join(x.rename('x'))
    tx['e'] = np.log(tx.final / tx.usda)
    t25 = t25_tx(feature_dir, nass_dir, fsa_csv, ck)
    rows = []
    for y in YEARS:
        if y not in tx.index or pd.isna(tx.loc[y, 'usda']) or pd.isna(tx.loc[y, 'final']) or y not in t25:
            continue
        tr = tx[(tx.index < y) & (tx.index >= FIRST_TRAIN)].dropna(subset=['e', 'x'])
        u, q = tx.loc[y, 'usda'], tx.loc[y, 'final']
        if len(tr) < MIN_ROWS or pd.isna(tx.loc[y, 'x']):
            c, used = t25[y], 'T25'
        else:
            mu, sd = tr.x.mean(), tr.x.std(ddof=0) or 1.0
            X = np.c_[np.ones(len(tr)), (tr.x - mu) / sd]
            b = np.linalg.solve(X.T @ X + np.diag([0.0, 1.0]), X.T @ tr.e.to_numpy())
            c, used = float(u * np.exp(b[0] + b[1] * (tx.loc[y, 'x'] - mu) / sd)), 'model'
        rows.append({'year': y, 'x': None if pd.isna(tx.loc[y, 'x']) else round(float(tx.loc[y, 'x']), 4), 'used': used,
                     'T43': 100 * (c / q - 1), 'T25': 100 * (t25[y] / q - 1), 'USDA': 100 * (u / q - 1)})
    return pd.DataFrame(rows)


def summary(d, lo, hi):
    d = d[(d.year >= lo) & (d.year <= hi)]
    a = d[['T43', 'T25', 'USDA']].abs()
    return {'years': [int(lo), int(hi)], 'n': len(d), 'T43': a.T43.mean(), 'T25': a.T25.mean(), 'USDA': a.USDA.mean(),
            'better_than_usda': int((a.T43 < a.USDA).sum()), 'worse_than_usda': int((a.T43 > a.USDA).sum())}


def main(nass_dir, feature_dir, fsa_csv, out_json, *landsat_dirs):
    res = {}
    for ck in ('aug1', 'sep1', 'oct1'):
        d = run(nass_dir, feature_dir, fsa_csv, landsat_dirs, ck)
        dev, test = summary(d, 2013, 2021), summary(d, 2022, 2025)
        test['pass'] = bool(test['n'] and test['T43'] < min(test['USDA'], test['T25'])
                            and test['better_than_usda'] > test['worse_than_usda'])
        res[ck] = {'dev': dev, 'test': test, 'by_year': d.round(2).to_dict('records')}
        for k, s in (('dev', dev), ('test', test)):
            if s['n']:
                print(f"{ck} {k} {s['years']}: TX T43 {s['T43']:.1f} | T25 {s['T25']:.1f} | USDA {s['USDA']:.1f} "
                      f"| vs USDA better {s['better_than_usda']} worse {s['worse_than_usda']}" + (f" pass={s.get('pass')}" if k == 'test' else ''))
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
