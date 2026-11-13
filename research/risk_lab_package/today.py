"""Today's CT=F risk card from the frozen range-EWMA + risk-lab probabilistic layer (zero fit).

Downloads Yahoo CT=F daily OHLC, keeps only sessions dated strictly before the
run's UTC date (no partial bar), and writes the next-session forecast:
variance, 80/90% filtered-historical-simulation intervals and big-move
probability. Also writes the CT historical calibration of those intervals.
A demonstration card, not forward evidence: nothing is archived externally here.
"""
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research import range_ewma as R  # noqa: E402
from research import risk_lab as L  # noqa: E402


def main(out_dir):
    import yfinance as yf
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = datetime.now(UTC)
    raw = yf.Ticker('CT=F').history(period='max', interval='1d', auto_adjust=False, actions=False)
    raw_path = out / f'CT-{run:%Y%m%dT%H%M%SZ}.csv'
    raw.to_csv(raw_path)
    h = pd.read_csv(raw_path)
    f = pd.DataFrame({'date': pd.to_datetime(h['Date'].str[:10]), 'open': h.Open, 'high': h.High,
                      'low': h.Low, 'close': h.Close})
    dropped = f[f.date.dt.date >= run.date()]
    f = f[f.date.dt.date < run.date()].reset_index(drop=True)
    fc = R.forecasts(f)
    pr = L.probabilistic(f, fc.candidate)
    pe = L.probabilistic(f, fc.ewma)
    last = f.date.iloc[-1]
    close = float(f.close.iloc[-1])
    row = pr.loc[last]
    sd = float(np.sqrt(fc.candidate.loc[last]))
    card = {'run_utc': run.isoformat(), 'source': 'Yahoo Finance CT=F (continuous proxy)',
            'raw_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            'dropped_same_day_bars': [str(d.date()) for d in dropped.date],
            'origin_session': str(last.date()), 'origin_close_cents_lb': close,
            'next_session_sigma_pct': 100 * sd, 'ewma_sigma_pct': 100 * float(np.sqrt(fc.ewma.loc[last])),
            'big_move_threshold_pct': None, 'big_move_probability': float(row.p_big),
            'base_rate_big_move': 1 - L.EVENT_QUANTILE}
    thr = np.log(f.close / f.close.shift()).abs().expanding(L.MIN_Z_HISTORY).quantile(L.EVENT_QUANTILE).iloc[-1]
    card['big_move_threshold_pct'] = 100 * float(thr)
    for c in L.INTERVALS:
        k = int(c * 100)
        card[f'interval{k}_cents_lb'] = [close * float(np.exp(row[f'lo{k}'])), close * float(np.exp(row[f'hi{k}']))]
    # historical calibration on CT (seen data, descriptive)
    hist = pr.dropna(subset=['lo80', 'y'])
    hist = hist[hist.index >= '2011-01-01']
    card['ct_calibration'] = {f'coverage{int(c * 100)}': float(((hist.y >= hist[f'lo{int(c * 100)}'])
                                                                 & (hist.y <= hist[f'hi{int(c * 100)}'])).mean())
                              for c in L.INTERVALS}
    ev = pr.dropna(subset=['p_big', 'event'])
    ev = ev[ev.index >= '2011-01-01']
    card['ct_calibration']['origins'] = int(len(hist))
    card['ct_calibration']['auc_range'] = L._auc(ev.p_big, ev.event)
    pe2 = pe.dropna(subset=['p_big', 'event'])
    card['ct_calibration']['auc_ewma'] = L._auc(pe2.loc[pe2.index >= '2011-01-01'].p_big,
                                                pe2.loc[pe2.index >= '2011-01-01'].event)
    # reliability bins for the big-move probability
    bins = pd.cut(ev.p_big, [0, .05, .1, .15, .2, .3, 1.0])
    card['ct_reliability'] = [{'bin': str(b), 'n': int(g.size), 'mean_p': float(g.p_big.mean()),
                               'observed': float(g.event.mean())} for b, g in ev.groupby(bins, observed=True)]
    tail = pd.DataFrame({'date': f.date.iloc[-120:].dt.strftime('%Y-%m-%d').values,
                         'close': f.close.iloc[-120:].values,
                         'sigma_pct': 100 * np.sqrt(fc.candidate.iloc[-120:].values)})
    card['recent'] = tail.to_dict(orient='records')
    (out / 'today.json').write_text(json.dumps(card, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in card.items() if k not in ('recent', 'ct_reliability')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
