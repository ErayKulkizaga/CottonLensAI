"""T49 forward lock: Texas 2026 harvested area from USDA's 1 Nov 2026 forecast and the public poor/very-poor share.

Model (pre-declared in docs/T49_ANCHORING_PREREG_20261116.md): log(H_final / H_m) = a + b * pvp, OLS on Texas seasons
2000-2025 for report month m. Locks the November-based forecast now and the exact December formula (10 Dec 2026 report,
pvp of the last week ending on or before 30 Nov 2026). Scored against the NASS final in May 2027.
Usage: python t49_lock_2026.py t49_root out.json
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


def main(root, out_json):
    cond = pd.read_csv(Path(root) / 'condition.csv')
    key = nass_key()
    new = []
    for unit in ('PCT POOR', 'PCT VERY POOR'):
        for r in qs(key, commodity_desc='COTTON', statisticcat_desc='CONDITION', unit_desc=unit, agg_level_desc='STATE',
                    state_alpha='TX', year='2026'):
            if r.get('week_ending') and r['Value'].strip().replace('.', '').isdigit():
                new.append({'crop': 'COTTON', 'state': 'TX', 'year': 2026, 'unit': unit, 'week_ending': r['week_ending'],
                            'v': float(r['Value'])})
    cond26 = pd.DataFrame(new)
    hist = cotton_rows(cond, ['TX'])
    hist = hist[hist.year.between(2000, 2025)].assign(e=lambda d: np.log(d.final / d.area))
    coef = {m: dict(zip(('a', 'b'), map(float, fit(hist[hist.month == m])))) for m in ('NOV', 'DEC')}
    for m in coef:
        coef[m]['train_seasons'] = int(hist[hist.month == m].year.nunique())
    h = nass_state(ROOT / 'output/crop-nowcast-2026/nass', 'state_harvested')
    h = h[(h.state_alpha == 'TX') & (h.year == 2026)]
    nov = float(h[h.reference_period_desc == 'YEAR - NOV FORECAST'].v.iloc[0])
    pvp = pvp_at(cond26, 'COTTON', 'TX', 2026, 11)
    last_week = str(cond26[pd.to_datetime(cond26.week_ending) <= pd.Timestamp(2026, 10, 31)].week_ending.max())
    e_hat = coef['NOV']['a'] + coef['NOV']['b'] * pvp
    res = {
        'locked_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'model': 'log(H_final / H_m) = a + b * pvp; OLS on Texas 2000-2025',
        'coefficients': coef,
        'inputs_2026': {'usda_tx_harvested_nov_acres': nov, 'pvp_tx_week_ending': last_week, 'pvp_tx': pvp},
        'locked_tx_harvested_final_acres': round(nov * float(np.exp(e_hat))),
        'implied_adjustment_pct': 100 * float(np.expm1(e_hat)),
        'dec_procedure': 'On the 10 Dec 2026 report: H_final_hat = H_dec * exp(a_DEC + b_DEC * pvp), with pvp = poor + very poor '
                         'share of Texas cotton for the last week ending on or before 30 Nov 2026 (NASS Crop Progress).',
        'scoring': 'Absolute % error vs NASS final Texas upland harvested area (released May 2027), compared with USDA Nov and Dec.',
        'condition_2026_sha256': hashlib.sha256(cond26.to_csv(index=False).encode()).hexdigest(),
    }
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: res[k] for k in ('inputs_2026', 'locked_tx_harvested_final_acres', 'implied_adjustment_pct')}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
