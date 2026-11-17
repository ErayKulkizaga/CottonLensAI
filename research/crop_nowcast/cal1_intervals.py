"""CAL1 (docs/prereg/CAL1_INTERVAL_CALIBRATION_PREREG_20261117.md): real-time calibration of Texas area intervals.
Usage: python cal1_intervals.py out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
T51 = ROOT / 'research/evidence/t51-area-production-20261116.json'
QS = (0.80, 0.95)
WINDOW = 8
MIN_ROWS = 12


def fit_scaled(past):
    """|r| = a + b*pvp by OLS, fitted value floored at 0.01; returns (a, b, k80, k95)."""
    a, b = np.polyfit(past.pvp, past.r, 1)[::-1]
    fitted = np.maximum(a + b * past.pvp, 0.01)
    ratio = past.r / fitted
    return float(a), float(b), *(float(np.quantile(ratio, q)) for q in QS)


def half_widths(past, cur, method):
    if method == 'a_scaled':
        a, b, k80, k95 = fit_scaled(past)
        f = np.maximum(a + b * cur.pvp, 0.01)
        return {0.80: k80 * f, 0.95: k95 * f}
    if method == 'b_window':
        keep = sorted(past.year.unique())[-WINDOW:]
        w = past[past.year.isin(keep)].r
        return {q: np.full(len(cur), np.quantile(w, q)) for q in QS}
    if method == 'c_double':
        return {q: np.full(len(cur), 2 * np.quantile(past.r, q)) for q in QS}
    if method == 'v1_realtime':
        return {q: np.full(len(cur), np.quantile(past.r, q)) for q in QS}
    raise ValueError(method)


def evaluate(d, method):
    out = []
    for y in sorted(d.year.unique()):
        past, cur = d[d.year < y], d[d.year == y]
        if len(past) < MIN_ROWS:
            continue
        h = {q: np.asarray(v) for q, v in half_widths(past, cur, method).items()}
        for i, (_, r) in enumerate(cur.iterrows()):
            out.append({'year': y, 'month': r.month, 'r': r.r, 'h80': float(h[0.80][i]), 'h95': float(h[0.95][i])})
    o = pd.DataFrame(out)
    o['c80'], o['c95'] = o.r <= o.h80, o.r <= o.h95
    # half-width in % of area: exp(h) - 1
    return {'n': len(o), 'seasons': [int(o.year.min()), int(o.year.max())],
            'coverage80': float(o.c80.mean()), 'coverage95': float(o.c95.mean()),
            'coverage80_by_month': o.groupby('month').c80.mean().round(3).to_dict(),
            'coverage95_by_month': o.groupby('month').c95.mean().round(3).to_dict(),
            'mean_halfwidth80_pct': float(100 * np.expm1(o.h80).mean()),
            'mean_halfwidth95_pct': float(100 * np.expm1(o.h95).mean())}


def main(out_json):
    d = pd.DataFrame(json.loads(T51.read_text(encoding='utf-8'))['rows'])
    d['r'] = (d.e - d.eh_anchor).abs()
    res = {m: evaluate(d, m) for m in ('v1_realtime', 'a_scaled', 'b_window', 'c_double')}
    ok = [m for m in ('a_scaled', 'b_window', 'c_double')
          if 0.72 <= res[m]['coverage80'] <= 0.88 and res[m]['coverage95'] >= 0.90]
    chosen = min(ok, key=lambda m: res[m]['mean_halfwidth80_pct']) if ok else None
    res['eligible'], res['chosen'] = ok, chosen
    if chosen:  # parameters fitted on all 2008-2025 rows, for model_v2
        if chosen == 'a_scaled':
            a, b, k80, k95 = fit_scaled(d)
            res['frozen'] = {'method': chosen, 'alpha': a, 'beta': b, 'k80': k80, 'k95': k95, 'floor': 0.01}
        elif chosen == 'b_window':
            w = d[d.year.isin(sorted(d.year.unique())[-WINDOW:])].r
            res['frozen'] = {'method': chosen, 'q80_log': float(np.quantile(w, 0.8)), 'q95_log': float(np.quantile(w, 0.95)),
                             'window_seasons': WINDOW}
        else:
            res['frozen'] = {'method': chosen, 'q80_log': float(2 * np.quantile(d.r, 0.8)), 'q95_log': float(2 * np.quantile(d.r, 0.95))}
        res['frozen']['train_seasons'] = [int(d.year.min()), int(d.year.max())]
    print(json.dumps(res, indent=1))
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:])
