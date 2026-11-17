"""Forward lock: momentum rule (T44 spec, unchanged) applied to USDA's 2026 cotton forecasts.

Trains on seasons 2000-2025 only (rolling rule for Y = 2026). Freezes the Nov and Dec coefficients, writes the
Nov-based 2026 forecast now and the exact formula for the Dec report (10 Dec 2026). The 2026 'YEAR' row in
QuickStats is the current estimate, not a final, and is dropped.
Usage: python lock_2026.py full_nass_dir nass_2026_dir out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M

Y = 2026


def coef(f):
    tr = f[(f.year < Y) & (f.year >= 2000)].dropna(subset=['e', 'r'])
    sd = float(np.sqrt((tr.r ** 2).mean()))
    z = (tr.r / sd).to_numpy()
    return float(z @ tr.e.to_numpy() / (z @ z + M.LAM)), sd, len(tr)


def main(full_dir, dir_2026, out_json):
    res = {'trained_on': [2000, Y - 1], 'rule': 'e_hat = b * log(USDA_m / USDA_prev) / sd; forecast = USDA_m * exp(e_hat)'}
    for lv in ('state', 'us'):
        hist = M.frame(full_dir, lv)
        new = M.frame(dir_2026, lv)
        new = new[new.year == Y].assign(final=np.nan)
        res[f'{lv}_frame'] = pd.concat([hist[hist.year < Y], new])
    for m in ('NOV', 'DEC'):
        b, sd, n = coef(M.with_revision(res['state_frame'], m))
        res[m] = {'b': b, 'sd': sd, 'train_rows': n}
    s = M.with_revision(res['state_frame'], 'NOV')
    s = s[s.year == Y].dropna(subset=['usda', 'r']).copy()
    s['forecast'] = s.usda * np.exp(res['NOV']['b'] * s.r / res['NOV']['sd'])
    us = res['us_frame'].set_index('year').loc[Y]
    lock = {'states': {st: {'usda_nov': float(u), 'locked_forecast': round(float(c))} for st, u, c in zip(s.state, s.usda, s.forecast, strict=True)},
            'US': {'usda_nov': float(us['NOV']), 'locked_forecast': round(float(us['NOV'] * s.forecast.sum() / s.usda.sum()))}}
    tx = lock['states']['TX']
    out = {k: v for k, v in res.items() if not k.endswith('_frame')}
    out['forecast_2026_from_nov'] = lock
    out['dec_procedure'] = (f"On 10 Dec 2026: per state, forecast = USDA_Dec * exp({res['DEC']['b']:.6f} * log(USDA_Dec / USDA_Nov) / "
                            f"{res['DEC']['sd']:.6f}); US = USDA_US_Dec * sum(state forecasts) / sum(state USDA_Dec).")
    Path(out_json).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f"US: USDA Nov {lock['US']['usda_nov']:,.0f} -> locked {lock['US']['locked_forecast']:,} bales | "
          f"TX: USDA Nov {tx['usda_nov']:,.0f} -> locked {tx['locked_forecast']:,}")
    print('NOV b/sd', round(res['NOV']['b'], 5), round(res['NOV']['sd'], 5), '| DEC b/sd', round(res['DEC']['b'], 5), round(res['DEC']['sd'], 5))


if __name__ == '__main__':
    main(*sys.argv[1:4])
