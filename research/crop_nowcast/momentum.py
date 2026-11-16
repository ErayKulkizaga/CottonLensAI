"""T44: USDA revision momentum (pre-declared in docs/CROP_NOWCAST_TRIALS.md).

r = log(USDA_m / USDA_prev), e = log(final / USDA_m); pooled no-intercept ridge e = b * z(r), rolling over years < Y.
Usage: python momentum.py nass_dir out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.stack import state_w
from research.crop_nowcast.state_model import nass_state

MONTHS = ['AUG', 'SEP', 'OCT', 'NOV', 'DEC']
YEARS, FIRST_TRAIN, MIN_ROWS, LAM = range(2014, 2026), 2012, 10, 1.0


def frame(nass_dir, level):
    d = nass_state(nass_dir, f'{level}_production')
    key = ['state_alpha', 'year'] if level == 'state' else ['year']
    f = pd.DataFrame({m: d[d.reference_period_desc == f'YEAR - {m} FORECAST'].set_index(key).v for m in MONTHS})
    f['final'] = d[d.reference_period_desc == 'YEAR'].set_index(key).v
    f = f.reset_index()
    return f.rename(columns={'state_alpha': 'state'}) if level == 'state' else f


def with_revision(f, m):
    """usda = month m forecast; prev = latest earlier month with a forecast (Nov falls back to Sep when Oct is missing)."""
    i = MONTHS.index(m)
    prev = f[MONTHS[i - 1]]
    for k in range(i - 2, -1, -1):
        prev = prev.fillna(f[MONTHS[k]])
    return f.assign(usda=f[m], r=np.log(f[m] / prev), e=np.log(f.final / f[m]))


def predict(f):
    out = []
    for y in YEARS:
        tr = f[(f.year < y) & (f.year >= FIRST_TRAIN)].dropna(subset=['e', 'r'])
        te = f[f.year == y].dropna(subset=['usda', 'r']).copy()
        if te.empty:
            continue
        if len(tr) < MIN_ROWS:
            te['eh'] = 0.0
        else:
            sd = float(np.sqrt((tr.r ** 2).mean())) or 1.0  # scale without centring: no intercept anywhere
            z = (tr.r / sd).to_numpy()
            b = float(z @ tr.e.to_numpy() / (z @ z + LAM))
            te['eh'] = b * te.r / sd
        out.append(te.assign(corr=te.usda * np.exp(te.eh)))
    return pd.concat(out)


def score(nass_dir, m):
    s = predict(with_revision(frame(nass_dir, 'state'), m)).dropna(subset=['final'])
    us = frame(nass_dir, 'us').set_index('year')[[m, 'final']].rename(columns={m: 'usda_us'})
    j = s.groupby('year')[['corr', 'usda']].sum().join(us).dropna()
    j['us_hat'] = j.usda_us * j['corr'] / j.usda
    tx = s[s.state == 'TX'].set_index('year')
    pe = lambda a, b: 100 * (a / b - 1)
    res = {'TX_by_year': {int(y): [round(pe(r['corr'], r.final), 1), round(pe(r.usda, r.final), 1)] for y, r in tx.iterrows()},
           'US_by_year': {int(y): [round(pe(r.us_hat, r.final), 1), round(pe(r.usda_us, r.final), 1)] for y, r in j.iterrows()},
           'state_w': state_w(s, 'corr'), 'state_w_usda': state_w(s, 'usda')}
    for k in ('TX', 'US'):
        a = np.array(list(res[f'{k}_by_year'].values()))
        res[k] = float(np.abs(a[:, 0]).mean())
        res[f'{k}_usda'] = float(np.abs(a[:, 1]).mean())
        res[f'{k}_better'] = int((np.abs(a[:, 0]) < np.abs(a[:, 1])).sum())
        res[f'{k}_worse'] = int((np.abs(a[:, 0]) > np.abs(a[:, 1])).sum())
    res['pass'] = bool(all(res[k] < res[f'{k}_usda'] and res[f'{k}_better'] > res[f'{k}_worse'] for k in ('TX', 'US'))
                       and res['state_w'] < res['state_w_usda'])
    print(f"{m}: TX {res['TX']:5.1f} vs USDA {res['TX_usda']:5.1f} (better {res['TX_better']} worse {res['TX_worse']}) | "
          f"US {res['US']:4.1f} vs {res['US_usda']:4.1f} (better {res['US_better']} worse {res['US_worse']}) | "
          f"state_w {res['state_w']:4.1f} vs {res['state_w_usda']:4.1f}  pass={res['pass']}")
    return res


def main(nass_dir, out_json):
    res = {m: score(nass_dir, m) for m in ('DEC', 'SEP', 'OCT', 'NOV')}
    res['primary_pass'] = bool(res['DEC']['pass'] and res['SEP']['pass'])
    print('PRIMARY (Dec and Sep):', res['primary_pass'])
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
