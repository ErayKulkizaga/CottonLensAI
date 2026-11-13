"""Frozen zero-fit range-augmented EWMA risk forecast and its preregistered scoring.

Profile: ``range-ewma-confirmatory-t1-v1``. See
``docs/RANGE_EWMA_PREREGISTRATION_20261113.md``. Nothing here fits an estimator:
the only data-dependent quantity is the past-only Parkinson scale ``k``.
The reference stays the project EWMA (alpha 0.06 on squared close returns).

Stages are separate on purpose: ``admit`` reads only data-quality statistics and
must be committed before ``score`` computes any loss. ``score`` refuses to run
unless the frozen admission file and the preregistration hash are supplied.
This module is not market skill evidence by itself.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE = 'range-ewma-confirmatory-t1-v1'
ALPHA = 0.06            # unchanged project EWMA (lambda 0.94)
BLEND = 0.5             # fixed a priori, never tuned
MIN_K_DAYS = 250        # valid bars required before k exists
TOL = 1e-9
FLOOR = 1e-12
BLOCKS = (20, 60)
RESAMPLES = 10_000
SEED = 42
MIN_YEAR_ORIGINS = 100
END_TARGET = pd.Timestamp('2026-10-30')
ADMISSION = {'min_origins': 2000, 'min_valid_fraction': 0.80, 'max_gap_days': 10}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bars(frame):
    need = ['date', 'open', 'high', 'low', 'close']
    if missing := [c for c in need if c not in frame]:
        raise ValueError(f'Missing columns {missing}')
    f = frame[need].copy()
    f['date'] = pd.to_datetime(f.date)
    if f.date.duplicated().any() or not f.date.is_monotonic_increasing:
        raise ValueError('Unique, sorted dates required')
    if not np.isfinite(f.close).all() or (f.close <= 0).any():
        raise ValueError('Positive finite closes required; no silent drop or fill')
    return f.set_index('date')


def valid_bar(f):
    o, h, lo, c = f.open, f.high, f.low, f.close
    return ((lo > 0) & (h > lo) & (h >= np.maximum(o, c) - TOL) & (lo <= np.minimum(o, c) + TOL)
            & np.isfinite(o) & np.isfinite(h) & np.isfinite(lo))


def forecasts(frame):
    """Per-origin variance forecasts for r2 of the NEXT session; past-only by construction."""
    f = _bars(frame)
    r2 = np.log(f.close / f.close.shift()) ** 2
    valid = valid_bar(f) & r2.notna()
    park = (np.log(f.high / f.low) ** 2 / (4 * math.log(2))).where(valid)
    # k_t uses valid bars strictly before t (shift 1): no same-day scale information.
    count = valid.cumsum().shift(1)
    k = (r2.where(valid).cumsum() / park.cumsum()).ffill().shift(1).where(count >= MIN_K_DAYS)
    proxy = (k * park).where(valid, r2)        # invalid/flat bar -> today's r2, never ffill
    ewma = r2.ewm(alpha=ALPHA, adjust=False).mean()
    range_ewma = proxy.where(k.notna()).ewm(alpha=ALPHA, adjust=False).mean()
    out = pd.DataFrame({'ewma': ewma, 'range_ewma': range_ewma, 'k': k, 'valid_bar': valid})
    out['candidate'] = BLEND * out.ewma + (1 - BLEND) * out.range_ewma
    out['target_1'] = r2.shift(-1)
    out['target_5'] = sum(r2.shift(-i) for i in range(1, 6))
    out['target_date_1'] = pd.Series(f.index, index=f.index).shift(-1)
    out['target_date_5'] = pd.Series(f.index, index=f.index).shift(-5)
    return out


def qlike(actual, predicted):
    p = np.maximum(np.asarray(predicted, float), FLOOR)
    return np.log(p) + np.asarray(actual, float) / p


def origins(fc, horizon=1, end_target=END_TARGET):
    """Eligible origins: candidate defined and target matured on or before end_target."""
    target, tdate = fc[f'target_{horizon}'], fc[f'target_date_{horizon}']
    keep = fc.candidate.notna() & fc.ewma.gt(0) & target.notna() & tdate.le(end_target)
    return fc.loc[keep]


def gains(fc, horizon=1, end_target=END_TARGET):
    o = origins(fc, horizon, end_target)
    a = o[f'target_{horizon}']
    base, cand = horizon * o.ewma, horizon * o.candidate
    return pd.DataFrame({'gain': qlike(a, base) - qlike(a, cand), 'baseline_loss': qlike(a, base)}, index=o.index)


def admit(frame, end_target=END_TARGET):
    """Data-quality admission. Reads no loss and no forecast error."""
    f = _bars(frame)
    fc = forecasts(frame)
    o = origins(fc, 1, end_target)
    gaps = f.index.to_series().diff().dt.days.loc[o.index.min():o.index.max()] if len(o) else pd.Series([np.inf])
    stats = {'origins': int(len(o)), 'valid_fraction': float(valid_bar(f).mean()),
             'max_gap_days': float(gaps.max()), 'first_origin': str(o.index.min().date()) if len(o) else None,
             'last_origin': str(o.index.max().date()) if len(o) else None}
    reasons = []
    if stats['origins'] < ADMISSION['min_origins']:
        reasons.append('too_few_origins')
    if stats['valid_fraction'] < ADMISSION['min_valid_fraction']:
        reasons.append('valid_bar_fraction')
    if stats['max_gap_days'] > ADMISSION['max_gap_days']:
        reasons.append('calendar_gap')
    return {**stats, 'admitted': not reasons, 'reasons': reasons}


def _block_index(n, block, rng):
    starts = rng.integers(0, n - block + 1, math.ceil(n / block))
    return (starts[:, None] + np.arange(block)).ravel()[:n]


def block_ci(values, block, resamples=RESAMPLES, seed=SEED):
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.array([x[_block_index(len(x), block, rng)].mean() for _ in range(resamples)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def pooled_ci(matrix, block, resamples=RESAMPLES, seed=SEED):
    """Equal-market-weight mean of per-market means; common date blocks keep cross-market dependence."""
    m = np.asarray(matrix, float)
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for i in range(resamples):
        s = m[_block_index(len(m), block, rng)]
        out[i] = np.nanmean(np.nanmean(s, axis=0))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def market_summary(g):
    years = g.gain.groupby(g.index.year).agg(['mean', 'size'])
    years = years[years['size'] >= MIN_YEAR_ORIGINS]
    trims = {}
    for q in (0.99, 0.95):
        keep = g.baseline_loss < g.baseline_loss.quantile(q)
        trims[f'trim_top_{round(100 * (1 - q))}pct'] = float(g.gain[keep].mean())
    ci = {f'ci{b}': block_ci(g.gain, b) for b in BLOCKS}
    gate = g.gain.mean() > 0 and all(ci[f'ci{b}'][0] > 0 for b in BLOCKS) and (years['mean'] > 0).mean() >= 0.75
    return {'origins': int(len(g)), 'mean_gain': float(g.gain.mean()), **ci,
            'positive_years': int((years['mean'] > 0).sum()), 'scored_years': int(len(years)),
            'project_three_condition_gate': bool(gate), **trims}


def score(frames, horizon=1, end_target=END_TARGET):
    per = {name: gains(forecasts(f), horizon, end_target) for name, f in sorted(frames.items())}
    matrix = pd.DataFrame({n: g.gain for n, g in per.items()}).sort_index()
    pooled = float(np.nanmean(matrix.mean(axis=0)))
    ci = {f'ci{b}': pooled_ci(matrix.to_numpy(), b) for b in BLOCKS}
    markets = {n: market_summary(g) for n, g in per.items()}
    positive = sum(v['mean_gain'] > 0 for v in markets.values())
    primary = pooled > 0 and all(ci[f'ci{b}'][0] > 0 for b in BLOCKS)
    breadth = positive >= math.ceil(2 * len(markets) / 3)
    return {'profile': PROFILE, 'horizon': horizon, 'pooled_mean_gain': pooled, **ci,
            'positive_markets': int(positive), 'markets_scored': len(markets),
            'primary_pass': bool(primary), 'breadth_pass': bool(breadth),
            'decision': 'CONFIRMED' if primary and breadth else 'NOT_CONFIRMED', 'markets': markets}


def _load(path):
    p = Path(path)
    frame = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    return frame.rename(columns={'observed_on': 'date'})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('admit', help='data-quality admission only; no losses')
    a.add_argument('--market', nargs=2, action='append', metavar=('NAME', 'FILE'), required=True)
    a.add_argument('--out', required=True)
    s = sub.add_parser('score', help='preregistered scoring; needs frozen admission + prereg hash')
    s.add_argument('--admission', required=True)
    s.add_argument('--prereg', required=True)
    s.add_argument('--expected-prereg-sha256', required=True)
    s.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'admit':
        result = {}
        for name, path in args.market:
            result[name] = {'file': str(path), 'sha256': sha256(path), **admit(_load(path))}
        Path(args.out).write_text(json.dumps({'profile': PROFILE, 'stage': 'admission', 'markets': result},
                                             indent=2), encoding='utf-8')
        return
    if sha256(args.prereg) != args.expected_prereg_sha256:
        raise SystemExit('Preregistration hash mismatch; refusing to score')
    adm = json.loads(Path(args.admission).read_text(encoding='utf-8'))
    frames = {}
    for name, rec in adm['markets'].items():
        if rec['admitted']:
            if sha256(rec['file']) != rec['sha256']:
                raise SystemExit(f'{name}: data changed after admission')
            frames[name] = _load(rec['file'])
    result = {'admission_sha256': sha256(args.admission), 'prereg_sha256': args.expected_prereg_sha256,
              'primary_t1': score(frames, 1), 'secondary_t5': score(frames, 5)}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
