"""Risk-lab confirmatory comparisons: range-EWMA vs GARCH/HAR, big-move probability, intervals.

Profile ``risk-lab-confirmatory-t1-v1``; see ``docs/RISK_LAB_PREREGISTRATION_20261113.md``.
Reuses the frozen ``research.range_ewma`` estimator and its pooled scoring helpers.
GARCH(1,1) and HAR are refit once per calendar year on data strictly before
1 January of that year (expanding window); between refits the filters run
forward with fixed parameters. Nothing here scores data until ``score`` is called
with the frozen admission file and the preregistration hash.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from research import range_ewma as R

PROFILE = 'risk-lab-confirmatory-t1-v1'
MIN_FIT_OBS = 1000
MIN_Z_HISTORY = 500
EVENT_QUANTILE = 0.90
INTERVALS = (0.80, 0.90)
BONFERRONI_FAMILY = 4
H3_PRIMARY_EXCLUDE = ('SB=F', 'KC=F')   # explored before preregistration


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- GARCH(1,1)
def _garch_filter(r, omega, alpha, beta, init):
    """s2[0]=init; s2[t+1] = omega + alpha*r[t]^2 + beta*s2[t] (forecast for r[t+1] after r[t])."""
    r = np.asarray(r, float)
    drive = omega + alpha * r * r
    tail, _ = lfilter([1.0], [1.0, -beta], drive, zi=[beta * init])
    return np.r_[init, tail]


def _unpack(theta):
    omega = math.exp(theta[0])
    persistence = 0.999 / (1 + math.exp(-theta[1]))
    alpha = persistence / (1 + math.exp(-theta[2]))
    return omega, alpha, persistence - alpha


def garch_fit(r):
    r = np.asarray(r, float)
    var = float(np.mean(r * r))

    def nll(theta):
        omega, alpha, beta = _unpack(theta)
        s2 = _garch_filter(r[:-1], omega, alpha, beta, var)[: len(r)]
        s2 = np.maximum(s2, 1e-12)
        return 0.5 * float(np.sum(np.log(s2) + r * r / s2))

    start = np.array([math.log(var * 0.05), math.log(0.95 / 0.049), math.log(0.08 / 0.87)])
    res = minimize(nll, start, method='L-BFGS-B')
    if not np.isfinite(res.fun):
        raise ValueError('GARCH fit failed')
    return _unpack(res.x)


def _returns(frame):
    f = R._bars(frame)
    return np.log(f.close / f.close.shift()), f


def _yearly_refits(index):
    years = sorted(set(index.year))
    return [(y, pd.Timestamp(f'{y}-01-01')) for y in years]


def garch_forecasts(frame):
    r, f = _returns(frame)
    out = pd.Series(np.nan, index=f.index)
    rr = r.fillna(0.0).to_numpy()
    for year, cut in _yearly_refits(f.index):
        train = r[(r.index < cut)].dropna()
        if len(train) < MIN_FIT_OBS:
            continue
        omega, alpha, beta = garch_fit(train.to_numpy())
        s2 = _garch_filter(rr, omega, alpha, beta, float(np.mean(train.to_numpy() ** 2)))
        # forecast made at origin t (after r_t) for r_{t+1}: s2[t+1]
        mask = f.index.year == year
        out[mask] = s2[1:][mask]
    return out


# ---------------------------------------------------------------- HAR (OLS on r2)
def har_design(r2):
    return pd.DataFrame({'d': r2, 'w': r2.rolling(5).mean(), 'm': r2.rolling(22).mean()})


def har_forecasts(frame):
    r, f = _returns(frame)
    r2 = r ** 2
    X = har_design(r2)
    y = r2.shift(-1)
    tdate = pd.Series(f.index, index=f.index).shift(-1)
    out = pd.Series(np.nan, index=f.index)
    for year, cut in _yearly_refits(f.index):
        ok = X.notna().all(axis=1) & y.notna() & tdate.lt(cut)
        if ok.sum() < MIN_FIT_OBS:
            continue
        A = np.c_[np.ones(ok.sum()), X[ok].to_numpy()]
        beta, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        floor = 0.1 * float(y[ok].mean())
        mask = (f.index.year == year) & X.notna().all(axis=1).to_numpy()
        pred = np.c_[np.ones(mask.sum()), X[mask].to_numpy()] @ beta
        out[mask] = np.maximum(pred, floor)
    return out


# ---------------------------------------------------------------- probabilistic layer
def _past_only_exceedance(z_hist, thresholds):
    """P(|z| > thr_i) from |z| values matured before origin i (z_hist already lagged)."""
    seen, out = [], np.full(len(thresholds), np.nan)
    for i, (zv, thr) in enumerate(zip(z_hist, thresholds, strict=True)):
        if len(seen) >= MIN_Z_HISTORY and np.isfinite(thr):
            out[i] = (len(seen) - bisect.bisect_right(seen, thr)) / len(seen)
        if np.isfinite(zv):
            bisect.insort(seen, zv)
    return out


def _past_only_quantiles(z_hist, qs):
    seen, out = [], np.full((len(z_hist), len(qs)), np.nan)
    for i, zv in enumerate(z_hist):
        if len(seen) >= MIN_Z_HISTORY:
            n = len(seen)
            out[i] = [seen[min(n - 1, max(0, int(math.floor(q * (n - 1)))))] for q in qs]
        if np.isfinite(zv):
            bisect.insort(seen, zv)
    return out


def probabilistic(frame, variance):
    """Big-move probability and FHS intervals for r_{t+1} given a variance forecast series."""
    r, f = _returns(frame)
    y = r.shift(-1)
    sd = np.sqrt(variance.reindex(f.index))
    absr_hist = r.abs()                                   # |r_t| known at origin t
    thr = absr_hist.expanding(MIN_Z_HISTORY).quantile(EVENT_QUANTILE)
    z = (y / sd)
    z_lag = z.shift(1).to_numpy()                         # z of origin t-1 matures at t
    prob = _past_only_exceedance(np.abs(z_lag), (thr / sd).to_numpy())
    qs = []
    for c in INTERVALS:
        qs += [(1 - c) / 2, 1 - (1 - c) / 2]
    qz = _past_only_quantiles(z_lag, qs)
    out = pd.DataFrame({'y': y, 'event': (y.abs() > thr).astype(float).where(y.notna() & thr.notna()),
                        'p_big': prob}, index=f.index)
    for k, c in enumerate(INTERVALS):
        out[f'lo{int(c * 100)}'] = qz[:, 2 * k] * sd
        out[f'hi{int(c * 100)}'] = qz[:, 2 * k + 1] * sd
    return out


def winkler(lo, hi, y, coverage):
    a = 1 - coverage
    return (hi - lo) + (2 / a) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


# ---------------------------------------------------------------- scoring
def market_losses(frame, end_target=R.END_TARGET):
    fc = R.forecasts(frame)
    g = garch_forecasts(frame).reindex(fc.index)
    h = har_forecasts(frame).reindex(fc.index)
    keep = (fc.candidate.notna() & g.notna() & h.notna() & fc.target_1.notna()
            & fc.target_date_1.le(end_target))
    a = fc.target_1
    df = pd.DataFrame({'q_range': R.qlike(a, fc.candidate), 'q_garch': R.qlike(a, g),
                       'q_har': R.qlike(a, h)}, index=fc.index)[keep]
    pe = probabilistic(frame, fc.ewma).reindex(df.index)
    pr = probabilistic(frame, fc.candidate).reindex(df.index)
    ok = pe.p_big.notna() & pr.p_big.notna() & pe.event.notna() & pe.lo80.notna() & pr.lo80.notna()
    df['brier_ewma'] = (pe.p_big - pe.event) ** 2
    df['brier_range'] = (pr.p_big - pr.event) ** 2
    df['event'] = pe.event
    df['p_ewma'], df['p_range'] = pe.p_big, pr.p_big
    for c in INTERVALS:
        k = int(c * 100)
        df[f'w{k}_ewma'] = winkler(pe[f'lo{k}'], pe[f'hi{k}'], pe.y, c)
        df[f'w{k}_range'] = winkler(pr[f'lo{k}'], pr[f'hi{k}'], pr.y, c)
        df[f'cov{k}_range'] = ((pr.y >= pr[f'lo{k}']) & (pr.y <= pr[f'hi{k}'])).astype(float)
        df[f'cov{k}_ewma'] = ((pe.y >= pe[f'lo{k}']) & (pe.y <= pe[f'hi{k}'])).astype(float)
    return df[ok]


def _ci(matrix, block, level, resamples=R.RESAMPLES, seed=R.SEED):
    m = np.asarray(matrix, float)
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for i in range(resamples):
        s = m[R._block_index(len(m), block, rng)]
        out[i] = np.nanmean(np.nanmean(s, axis=0))
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(out, tail)), float(np.percentile(out, 100 - tail))]


def pooled(gains, markets):
    m = pd.DataFrame({k: gains[k] for k in markets}).sort_index()
    per = m.mean(axis=0)
    level_b = 1 - 0.05 / BONFERRONI_FAMILY
    res = {'pooled_mean_gain': float(per.mean()),
           'per_market_mean_gain': {k: float(v) for k, v in per.items()},
           'positive_markets': int((per > 0).sum()), 'markets': len(markets)}
    for b in R.BLOCKS:
        res[f'ci{b}_95'] = _ci(m.to_numpy(), b, 0.95)
        res[f'ci{b}_bonferroni'] = _ci(m.to_numpy(), b, level_b)
    lower95 = all(res[f'ci{b}_95'][0] > 0 for b in R.BLOCKS)
    lowerb = all(res[f'ci{b}_bonferroni'][0] > 0 for b in R.BLOCKS)
    breadth = res['positive_markets'] >= math.ceil(2 * len(markets) / 3)
    res['pass_95'] = bool(res['pooled_mean_gain'] > 0 and lower95 and breadth)
    res['pass_bonferroni'] = bool(res['pooled_mean_gain'] > 0 and lowerb and breadth)
    return res


def _auc(p, e):
    p, e = np.asarray(p), np.asarray(e)
    pos, neg = p[e == 1], p[e == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    order = np.argsort(np.r_[pos, neg], kind='mergesort')
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.r_[pos, neg]
    for v in np.unique(allv):                          # average ties
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score(frames):
    L = {k: market_losses(f) for k, f in sorted(frames.items())}
    names = list(L)
    h3 = [k for k in names if k not in H3_PRIMARY_EXCLUDE]
    res = {
        'H2a_range_vs_garch_qlike': pooled({k: L[k].q_garch - L[k].q_range for k in names}, names),
        'H2b_range_vs_har_qlike': pooled({k: L[k].q_har - L[k].q_range for k in names}, names),
        'H3_bigmove_brier_range_vs_ewma': pooled({k: L[k].brier_ewma - L[k].brier_range for k in h3}, h3),
        'H4_interval80_winkler_range_vs_ewma': pooled({k: L[k].w80_ewma - L[k].w80_range for k in names}, names),
    }
    res['H3_reported_SB_KC'] = {k: float((L[k].brier_ewma - L[k].brier_range).mean())
                                for k in names if k in H3_PRIMARY_EXCLUDE}
    desc = {}
    for k, d in L.items():
        clim = d.event.expanding().mean().shift(1).fillna(1 - EVENT_QUANTILE)
        bc = float(((clim - d.event) ** 2).mean())
        desc[k] = {'origins': int(len(d)), 'first': str(d.index.min().date()), 'last': str(d.index.max().date()),
                   'event_rate': float(d.event.mean()),
                   'auc_ewma': _auc(d.p_ewma, d.event), 'auc_range': _auc(d.p_range, d.event),
                   'brier_skill_vs_climatology_range': 1 - float(d.brier_range.mean()) / bc,
                   'coverage80_range': float(d.cov80_range.mean()), 'coverage90_range': float(d.cov90_range.mean()),
                   'coverage80_ewma': float(d.cov80_ewma.mean()),
                   'w90_gain_range_vs_ewma': float((d.w90_ewma - d.w90_range).mean()),
                   'mean_qlike': {c: float(d[c].mean()) for c in ('q_range', 'q_garch', 'q_har')}}
    res['descriptive'] = desc
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--admission', required=True)
    ap.add_argument('--prereg', required=True)
    ap.add_argument('--expected-prereg-sha256', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if sha256(a.prereg) != a.expected_prereg_sha256:
        raise SystemExit('Preregistration hash mismatch; refusing to score')
    adm = json.loads(Path(a.admission).read_text(encoding='utf-8'))
    frames = {}
    for name, rec in adm['markets'].items():
        if rec['admitted']:
            if sha256(rec['file']) != rec['sha256']:
                raise SystemExit(f'{name}: data changed after admission')
            frames[name] = R._load(rec['file'])
    out = {'profile': PROFILE, 'prereg_sha256': a.expected_prereg_sha256,
           'admission_sha256': sha256(a.admission), **score(frames)}
    Path(a.out).write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
