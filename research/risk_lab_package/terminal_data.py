"""Data for the Cotton Risk Terminal: CT time machine, 20-session fan, regime series, counters (zero fit).

Uses the frozen range-EWMA forecast and the risk-lab probabilistic layer on Yahoo CT=F
(continuous proxy; same-day partial bar already dropped by today.py). Multi-horizon
fan quantiles are past-only empirical quantiles of h-session standardized returns;
only h=1 was preregistered elsewhere, longer horizons are labelled as historical checks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import range_ewma as R  # noqa: E402
from research import risk_lab as L  # noqa: E402

QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
HMAX = 20


def ct_frame(raw_csv, cutoff):
    h = pd.read_csv(raw_csv)
    f = pd.DataFrame({'date': pd.to_datetime(h['Date'].str[:10]), 'open': h.Open, 'high': h.High,
                      'low': h.Low, 'close': h.Close})
    return f[f.date < cutoff].reset_index(drop=True)


def main(raw_csv, cutoff, out_path):
    f = ct_frame(raw_csv, pd.Timestamp(cutoff))
    fc = R.forecasts(f)
    pr = L.probabilistic(f, fc.candidate)
    close = f.set_index('date').close
    lr = np.log(close)
    sd1 = np.sqrt(fc.candidate)
    # time machine (next-session view as of each origin)
    tm = pd.DataFrame({'c': close, 's': 100 * sd1, 'p': pr.p_big, 'nx': close.shift(-1), 'ev': pr.event})
    for k in (80, 90):
        tm[f'l{k}'] = close * np.exp(pr[f'lo{k}'])
        tm[f'h{k}'] = close * np.exp(pr[f'hi{k}'])
    tm = tm.dropna(subset=['l80', 'p'])
    cols = ['c', 's', 'p', 'l80', 'h80', 'l90', 'h90', 'nx', 'ev']
    machine = {'dates': [d.strftime('%Y-%m-%d') for d in tm.index],
               **{c: [None if pd.isna(v) else round(float(v), 4) for v in tm[c]] for c in cols}}
    # fan: past-only quantiles of z_h at the last origin, plus historical coverage per horizon
    last = close.index[-1]
    fan, cover = [], {}
    for h in range(1, HMAX + 1):
        zh = (lr.shift(-h) - lr) / (np.sqrt(h) * sd1)
        matured = zh[zh.index <= close.index[-1 - h]].dropna() if len(close) > h else zh.dropna()
        q = np.quantile(matured.to_numpy(), QS)
        fan.append({'h': h, 'q': [round(float(close.iloc[-1] * np.exp(v * np.sqrt(h) * sd1.iloc[-1])), 3) for v in q]})
        if h in (1, 5, 10, 20):
            lag = zh.shift(h).to_numpy()
            qq = L._past_only_quantiles(lag, (0.10, 0.90, 0.05, 0.95))
            ok = ~np.isnan(qq[:, 0]) & np.isfinite(zh.to_numpy()) & (zh.index >= '2005-01-01')
            z = zh.to_numpy()[ok]
            cover[h] = {'n': int(ok.sum()), 'cov80': float(np.mean((z >= qq[ok, 0]) & (z <= qq[ok, 1]))),
                        'cov90': float(np.mean((z >= qq[ok, 2]) & (z <= qq[ok, 3])))}
    # trading-day calendar for the fan x-axis (business days, holidays not modelled)
    future = pd.bdate_range(last + pd.Timedelta(days=1), periods=HMAX)
    regime = tm['s'].resample('W-FRI').last().dropna()
    out = {'cutoff': cutoff, 'last_origin': last.strftime('%Y-%m-%d'), 'last_close': float(close.iloc[-1]),
           'machine': machine, 'fan': fan, 'fan_dates': [d.strftime('%Y-%m-%d') for d in future], 'fan_cover': cover,
           'regime': {'dates': [d.strftime('%Y-%m-%d') for d in regime.index], 's': [round(float(v), 3) for v in regime]},
           'history_close': {'dates': [d.strftime('%Y-%m-%d') for d in close.index[-60:]],
                             'c': [round(float(v), 3) for v in close.iloc[-60:]]}}
    Path(out_path).write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    print(len(machine['dates']), machine['dates'][0], machine['dates'][-1], json.dumps(cover))


if __name__ == '__main__':
    main(*sys.argv[1:4])
