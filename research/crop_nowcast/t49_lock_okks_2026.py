"""B3 forward lock: Oklahoma and Kansas 2026 cotton harvested area, the T49 independent states, from USDA's 1 Nov 2026
forecast and the public poor/very-poor share. Same model as the Texas lock (t49_lock_2026.py), fitted on the pooled
OK + KS seasons 2000-2025 for the report month, as in T49 S1. Point forecasts only: no calibrated interval exists for
these states. Scored against the NASS finals in May 2027, compared with USDA Nov and Dec.
Usage: python t49_lock_okks_2026.py t49_root out.json
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.nass_pull import nass_key
from research.crop_nowcast.state_model import nass_state
from research.crop_nowcast.t49_anchoring import cotton_rows, fit, pvp_at, qs

STATES = ['OK', 'KS']


def main(root, out_json):
    cond = pd.read_csv(Path(root) / 'condition.csv')
    key = nass_key()
    new = []
    for st in STATES:
        for unit in ('PCT POOR', 'PCT VERY POOR'):
            for r in qs(key, commodity_desc='COTTON', statisticcat_desc='CONDITION', unit_desc=unit, agg_level_desc='STATE',
                        state_alpha=st, year='2026'):
                if r.get('week_ending') and r['Value'].strip().replace('.', '').isdigit():
                    new.append({'crop': 'COTTON', 'state': st, 'year': 2026, 'unit': unit, 'week_ending': r['week_ending'],
                                'v': float(r['Value'])})
    cond26 = pd.DataFrame(new)
    hist = cotton_rows(cond, STATES)
    hist = hist[hist.year.between(2000, 2025)].assign(e=lambda d: np.log(d.final / d.area))
    coef = {m: dict(zip(('a', 'b'), map(float, fit(hist[hist.month == m])))) for m in ('NOV', 'DEC')}
    for m, c in coef.items():
        c['train_rows'] = int((hist.month == m).sum())
    h = nass_state(ROOT / 'output/crop-nowcast-2026/nass', 'state_harvested')
    locks = {}
    for st in STATES:
        nov = float(h[(h.state_alpha == st) & (h.year == 2026) & (h.reference_period_desc == 'YEAR - NOV FORECAST')].v.iloc[0])
        pvp = pvp_at(cond26, 'COTTON', st, 2026, 11)
        e_hat = coef['NOV']['a'] + coef['NOV']['b'] * pvp
        locks[st] = {'usda_nov_acres': nov, 'pvp': pvp, 'locked_final_acres': round(nov * float(np.exp(e_hat))),
                     'implied_adjustment_pct': 100 * float(np.expm1(e_hat))}
    res = {
        'locked_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model': 'log(H_final / H_m) = a + b * pvp; OLS on pooled OK + KS seasons 2000-2025',
        'coefficients': coef, 'locks': locks,
        'dec_procedure': 'On the 10 Dec 2026 report: H_final_hat = H_dec * exp(a_DEC + b_DEC * pvp), pvp = poor + very poor '
                         'share for the last week ending on or before 30 Nov 2026.',
        'scoring': 'Absolute % error vs NASS final harvested area (May 2027), each state, compared with USDA Nov and Dec.',
        'condition_2026_sha256': hashlib.sha256(cond26.to_csv(index=False).encode()).hexdigest(),
    }
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('coefficients', 'locks')}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
