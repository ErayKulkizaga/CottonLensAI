"""Link 1: cotton price reaction on USDA Crop Production report days, 2012-2021 (frozen prereg).

Prereg: docs/LINK1_EVENT_STUDY_PREREG_20261115.md (LF sha256 checked below).
Seal: prices are filtered to <= 2021-12-31 before any computation; NASS data stop at 2021.
Usage: python link1_event_study.py market_history.parquet nass_dir out.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.state_model import nass_state

PREREG = ROOT / 'docs/LINK1_EVENT_STUDY_PREREG_20261115.md'
PREREG_SHA = '16e2b4403c46e9ab2e0b17494d21a30d3f0bdc7281d7146290b5bc6bf1433e0c'
SEAL = pd.Timestamp('2021-12-31')
MON = {9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
PREV = {9: 'AUG', 10: 'SEP', 11: 'OCT', 12: 'NOV'}
N_PERM, SEED, ALPHA, LIMIT = 10_000, 42, 0.025, 0.04


def sha_lf(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def events(nass_dir, dates_csv):
    d = nass_state(nass_dir, 'us_production')
    f = {r: d[d.reference_period_desc == f'YEAR - {r} FORECAST'].set_index('year').v for r in [*MON.values(), 'AUG']}
    rel = pd.read_csv(dates_csv, parse_dates=['release_date']).groupby(['year', 'month']).release_date.min()
    rows = []
    for (y, m), date in rel.items():
        if m not in MON:
            continue
        cur, prev = f[MON[m]].get(y), f[PREV[m]].get(y)
        rows.append({'year': y, 'month': m, 'release': date, 'cur': cur, 'prev': prev,
                     'r': np.log(cur / prev) if cur and prev else np.nan})
    return pd.DataFrame(rows)


def prices(parquet):
    p = pd.read_parquet(parquet)
    p = p[p.series == 'cotton'].assign(day=lambda x: pd.to_datetime(x.observed_on))
    p = p[p.day <= SEAL].sort_values('day').set_index('day')  # seal first, before anything else
    return np.log(p.close).diff().dropna()


def h1(ev_abs, pool_abs, rng):
    obs = ev_abs.mean() / pool_abs.mean()
    draws = np.array([rng.choice(pool_abs, len(ev_abs), replace=False).mean() for _ in range(N_PERM)]) / pool_abs.mean()
    return float(obs), float((1 + (draws >= obs).sum()) / (1 + N_PERM))


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def h2(r, R, rng):
    obs = spearman(r, R)
    perm = np.array([spearman(r, rng.permutation(R)) for _ in range(N_PERM)])
    return obs, float((1 + (perm <= obs).sum()) / (1 + N_PERM))


def oracle_gate(e):
    """R_hat = beta*r, beta fitted (OLS through origin) on events of earlier years; first two years warm-up."""
    first = e.year.min()
    rows = []
    for _, x in e[e.year >= first + 2].iterrows():
        tr = e[e.year < x.year]
        beta = (tr.r * tr.R).sum() / (tr.r ** 2).sum()
        rows.append((abs(x.R - beta * x.r), abs(x.R)))
    m, n = np.array(rows).T
    return {'events': len(rows), 'mae_oracle': float(m.mean()), 'mae_naive': float(n.mean()),
            'gain_pct': float(100 * (1 - m.mean() / n.mean()))}


def analyse(e, ret, pool_days, label):
    rng = np.random.RandomState(SEED)
    pool = ret.loc[ret.index.isin(pool_days)].abs().to_numpy()
    ratio, p1 = h1(e.R.abs().to_numpy(), pool, rng)
    rho, p2 = h2(e.r.to_numpy(), e.R.to_numpy(), np.random.RandomState(SEED))
    g = oracle_gate(e)
    out = {'label': label, 'events': len(e), 'pool_days': len(pool), 'H1_ratio': ratio, 'H1_p': p1, 'H1_pass': p1 < ALPHA,
           'H2_spearman': rho, 'H2_p': p2, 'H2_pass': p2 < ALPHA, 'oracle': g, 'gate_pass': g['gain_pct'] >= 5.0}
    print(f"{label}: n={len(e)} | H1 ratio {ratio:.2f} p={p1:.4f} | H2 rho {rho:+.3f} p={p2:.4f} | "
          f"oracle gain {g['gain_pct']:+.1f}% on {g['events']} events")
    return out


def main(parquet, nass_dir, out_json):
    if sha_lf(PREREG) != PREREG_SHA:
        raise SystemExit('preregistration changed after freezing')
    ret = prices(parquet)
    e = events(nass_dir, ROOT / 'research/chain/crop_production_dates.csv')
    excluded = e[e.r.isna()][['year', 'month']].to_dict('records')
    e = e.dropna(subset=['r']).copy()
    days = ret.index
    e['day'] = [days[days.searchsorted(d)] for d in e.release]  # first trading day on/after release
    e['R'] = ret.loc[e.day].to_numpy()
    after = {days[days.get_loc(d) + 1] for d in e.day if days.get_loc(d) + 1 < len(days)}
    in_season = days[(days.month.isin([8, 9, 10, 11, 12, 1])) & (days >= pd.Timestamp('2012-08-01'))]
    pool_days = in_season.difference(pd.DatetimeIndex(sorted(set(e.day) | after)))
    res = {'prereg_sha256_lf': PREREG_SHA, 'seal': str(SEAL.date()), 'excluded_no_previous_forecast': excluded,
           'deviation': 'NASS series is UPLAND cotton production (about 97% of US all-cotton); prereg said all-cotton',
           'primary': analyse(e, ret, pool_days, 'primary (all events)')}
    flagged = e.R.abs() >= LIMIT
    res['flagged_abs_R_ge_4pct'] = e[flagged][['year', 'month']].to_dict('records')
    res['sensitivity_without_flagged'] = analyse(e[~flagged], ret, pool_days, 'without |R|>=4%')
    p = res['primary']
    if p['H1_pass'] and p['H2_pass'] and p['gate_pass']:
        res['decision'] = 'CHAIN_HAS_PRICE_PAYOFF -> Link 2'
    elif p['H1_pass'] and not p['H2_pass']:
        res['decision'] = 'PRICE_REACTS_BUT_REVISION_IS_WRONG_SURPRISE -> Link 3 (consensus) needed'
    elif not p['H1_pass']:
        res['decision'] = 'NO_SPECIAL_REACTION -> tell user; return to production accuracy'
    else:
        res['decision'] = 'H1 and H2 pass but oracle gain below 5% -> sign real, size too small for the price gate'
    res['events_table'] = [{'year': int(x.year), 'month': int(x.month), 'day': str(x.day.date()),
                            'revision_pct': round(100 * x.r, 2), 'return_pct': round(100 * x.R, 2)} for _, x in e.iterrows()]
    print('DECISION:', res['decision'])
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main(*sys.argv[1:4])
