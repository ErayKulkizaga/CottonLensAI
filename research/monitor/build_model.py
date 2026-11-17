"""Freeze model_v1.json for the USDA forecast monitor from data through the 2025 season (nothing from 2026).

- momentum: b and sd per report month (SEP..DEC), fitted once on 2000-2025 state data (same formula as lock_2026.coef);
- anchoring: a and b per month for Texas harvested area (log(H_final/H_m) = a + b * pvp), OLS on 2000-2025;
- intervals: empirical 80% / 95% quantiles of the anchoring model's *out-of-sample* absolute log errors from T51
  (rolling, 2008-2025). T51 showed in-sample conformal intervals under-cover, so realised OOS errors are used instead.
Usage: python research/monitor/build_model.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.t47_blind_old import coef
from research.crop_nowcast.t49_anchoring import cotton_rows, fit

MONTHS = ('SEP', 'OCT', 'NOV', 'DEC')


def main():
    model = {'version': 'model_v1', 'trained_through_season': 2025, 'momentum': {}, 'anchoring_tx_area': {}, 'intervals_tx_area': {}}
    for m in MONTHS:
        b, sd, n = coef(ROOT / 'output/crop-nowcast-full/nass', m)
        model['momentum'][m] = {'b': b, 'sd': sd, 'train_rows': n}
    cond = pd.read_csv(ROOT / 'output/t49/condition.csv')
    d = cotton_rows(cond, ['TX'])
    d = d[d.year.between(2000, 2025)].assign(e=lambda x: np.log(x.final / x.area))
    for m in MONTHS:
        a, b = fit(d[d.month == m])
        model['anchoring_tx_area'][m] = {'a': float(a), 'b': float(b), 'train_seasons': int(d[d.month == m].year.nunique())}
    rows = pd.DataFrame(json.loads((ROOT / 'research/evidence/t51-area-production-20261116.json').read_text())['rows'])
    oos = np.abs(rows.e - rows.eh_anchor)
    model['intervals_tx_area'] = {'q80_log': float(np.quantile(oos, 0.8)), 'q95_log': float(np.quantile(oos, 0.95)),
                                  'source': 'T51 rolling out-of-sample residuals 2008-2025', 'n': int(len(oos))}
    text = json.dumps(model, indent=1, sort_keys=True)
    out = ROOT / 'research/monitor/model_v1.json'
    out.write_text(text, encoding='utf-8')
    print(out.name, hashlib.sha256(text.encode()).hexdigest()[:16])
    print(json.dumps(model['intervals_tx_area']))


if __name__ == '__main__':
    main()
