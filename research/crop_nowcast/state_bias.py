"""T40: state-specific empirical-Bayes correction of USDA's monthly forecast, shrunk toward zero (no pooled bias).

b_s = m_s * tau2 / (tau2 + sigma2 / n_s) from that state's errors e = log(final / USDA_m) over years 2008..Y-1.
Corrected = USDA_m * exp(b_s); US = USDA_US * sum(corrected) / sum(USDA). T25 (pooled bias) on the same rows.
Years come from COTTONLENS_LAST_YEAR (dev 2013-2024 for T40; the 2025 holdout is a separate, single run).
Usage: python state_bias.py feature_dir nass_dir fsa_csv out.json [first_scored_year]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import usda_correction as UC
from research.crop_nowcast.stack import state_w

FIRST_TRAIN = 2008


def eb_bias(past):
    """past: rows with state, e -> {state: shrunk bias}. Shrinkage toward 0 from the fold's own variances."""
    g = past.groupby('state').e
    m, n, v = g.mean(), g.size(), g.var(ddof=1)
    sigma2 = float(v[n >= 2].mean()) if (n >= 2).any() else 0.0
    se2 = sigma2 / n
    tau2 = max(float((m ** 2).mean() - se2.mean()), 0.0)
    return (m * tau2 / (tau2 + se2) if tau2 > 0 else m * 0.0).to_dict()


def predict(f, years):
    out = []
    for y in years:
        past = f[(f.year < y) & (f.year >= FIRST_TRAIN)].dropna(subset=['e'])
        b = eb_bias(past)
        te = f[(f.year == y)].dropna(subset=['usda']).copy()
        te['b'] = te.state.map(b).fillna(0.0)
        out.append(te.assign(corr=te.usda * np.exp(te.b)))
    return pd.concat(out)


def run(feature_dir, nass_dir, fsa_csv, ck, years):
    if ck in UC.MONTH:  # T25 exactly as defined: pooled bias on the FSA variant's rows (needs build's fsa_gap)
        f, us = UC.build(feature_dir, nass_dir, ck, fsa_csv)
        t25 = UC.fit_predict(f, [], rows=['fsa_gap'])
    else:  # Nov/Dec have no FSA files: T25 analogue = pooled past-mean bias (as in T37's B)
        f, us = UC.usda_frame(nass_dir, ck)
        f['e'] = np.log(f.final / f.usda)
        t25 = UC.fit_predict(f, [])
    r = predict(f, years)
    t25 = t25[t25.year.isin(years)][['state', 'year', 'corr']].rename(columns={'corr': 'T25'})
    r = r.merge(t25, on=['state', 'year'], how='left')
    s40, s25 = UC.score(r, us), UC.score(r.assign(corr=r.T25), us)
    res = {'T40': s40, 'T25': s25, 'state_w': state_w(r, 'corr'), 'state_w_T25': state_w(r, 'T25'),
           'state_w_usda': state_w(r, 'usda'), 'TX_bias_by_year': {int(y): round(float(b), 4) for y, b in
                                                                   r[r.state == 'TX'][['year', 'b']].to_numpy()}}
    res['pass'] = bool(s40['TX_model'] < min(s40['TX_usda'], s25['TX_model']) and s40['US_model'] < min(s40['US_usda'], s25['US_model'])
                       and res['state_w'] < min(res['state_w_usda'], res['state_w_T25']))
    print(f"{ck} T40: TX {s40['TX_model']:5.1f} | T25 {s25['TX_model']:5.1f} | USDA {s40['TX_usda']:5.1f}   US {s40['US_model']:4.1f}"
          f" | T25 {s25['US_model']:4.1f} | USDA {s40['US_usda']:4.1f}   state_w {res['state_w']:4.1f} | T25 "
          f"{res['state_w_T25']:4.1f} | USDA {res['state_w_usda']:4.1f}  pass={res['pass']}")
    return res


def main(feature_dir, nass_dir, fsa_csv, out_json, first=2013):
    years = [y for y in UC.TEST_YEARS if y >= int(first)]
    res = {ck: run(feature_dir, nass_dir, fsa_csv, ck, years) for ck in ('aug1', 'sep1', 'oct1', 'nov1', 'dec1')}
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:6])
