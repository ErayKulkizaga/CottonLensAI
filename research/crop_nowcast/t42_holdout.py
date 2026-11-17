"""T42: one-shot 2025 holdout of T25, unchanged (docs/CROP_NOWCAST_TRIALS.md). Run once.

T25 = pooled past-mean of e = log(final / USDA_m) over states x years < 2025 (FSA rows at Aug/Sep, all rows at Nov/Dec).
Usage: python t42_holdout.py t39_root out.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('COTTONLENS_LAST_YEAR', '2024')  # fsa_gap/HGB inputs built for dev years; 2025 is only scored
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import usda_correction as UC
from research.crop_nowcast.stack import state_w

Y = 2025


def cell(t, ck):
    if ck in UC.MONTH:
        f, us = UC.build(t / 'dev-ndvi', t / 'nass', ck, t / 'fsa/fsa_cotton.csv')
        tr = f[f.year < Y].dropna(subset=['e', 'fsa_gap'])
    else:
        f, us = UC.usda_frame(t / 'nass', ck)
        f['e'] = np.log(f.final / f.usda)
        tr = f[f.year < Y].dropna(subset=['e'])
    eh = float(tr.e.mean())
    te = f[(f.year == Y)].dropna(subset=['usda', 'final']).assign(corr=lambda x: x.usda * np.exp(eh))
    tx = te[te.state == 'TX']
    s = te[['corr', 'usda']].sum()
    u = us.loc[Y]
    us_hat = u.usda * s['corr'] / s['usda']
    pct = lambda a, b: float(100 * (a / b - 1))
    r = {'bias_log': eh, 'n_train_rows': len(tr), 'n_states': len(te),
         'TX_T25': pct(tx['corr'].iloc[0], tx.final.iloc[0]), 'TX_usda': pct(tx.usda.iloc[0], tx.final.iloc[0]),
         'US_T25': pct(us_hat, u.final), 'US_usda': pct(u.usda, u.final),
         'state_w_T25': state_w(te, 'corr'), 'state_w_usda': state_w(te, 'usda')}
    r['beats_usda_TX'] = abs(r['TX_T25']) < abs(r['TX_usda'])
    r['beats_usda_US'] = abs(r['US_T25']) < abs(r['US_usda'])
    print(f"{ck} 2025: TX T25 {r['TX_T25']:+.1f}% vs USDA {r['TX_usda']:+.1f}% | US T25 {r['US_T25']:+.1f}% vs USDA "
          f"{r['US_usda']:+.1f}% | state_w {r['state_w_T25']:.1f} vs {r['state_w_usda']:.1f} | bias {eh:+.3f}")
    return r


def main(t39_root, out_json):
    t = Path(t39_root)
    res = {ck: cell(t, ck) for ck in ('sep1', 'aug1', 'nov1', 'dec1')}
    res['primary_pass'] = bool(res['sep1']['beats_usda_TX'] and res['sep1']['beats_usda_US'])
    print('PRIMARY (1 Sep 2025) pass:', res['primary_pass'])
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
