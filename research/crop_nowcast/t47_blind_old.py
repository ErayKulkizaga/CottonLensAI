"""T47: blind test of the cotton momentum rule on never-seen USDA forecasts, 1995-1999 (stage A) and 1985-1994 (stage B).

Pre-declared in docs/T47_BLIND_1985_1999_PREREG_20261116.md. The coefficients b and sd per report month are fitted
once on 2000-2025 (the same formula as lock_2026.coef) and applied unchanged to the blind seasons.
Subcommands:
  finals  root first last         QuickStats upland state/US production finals ('YEAR' rows only)
  gate    root                    parser checks on the blind forecasts (no finals used, no values printed)
  score   root full_nass_dir out  apply frozen coefficients, write the result
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import momentum as M
from research.crop_nowcast.stack import state_w
from research.crop_nowcast.t46_crops import perm_p

SCORED = ('SEP', 'OCT', 'NOV', 'DEC')


def clean(text):
    """Amendment 1: 1990s layouts. 'Production, by Type' -> 'Production by Type'; footnoted rows 'FL 2/  :' -> 'FL  :';
    'Am-Pima' -> 'Amer-Pima' so the Upland block ends there."""
    import re
    text = text.replace('Production, by Type', 'Production by Type')
    text = re.sub(r'^Am-Pima', 'Amer-Pima', text, flags=re.M)  # Aug 1996: Pima block header, else Pima rows overwrite Upland
    return re.sub(r'^(\s*[A-Za-z][A-Za-z .]*?)\s+\d+/(\s*:)', r'\1\2', text, flags=re.M)


def cmd_parse(raw_dir, root):
    """Cached reports -> forecasts/{state,us}_forecasts.json via the unchanged T45 parser on cleaned text."""
    import datetime as dt

    from research.crop_nowcast.cp_pull import MON, parse
    rows = {'state': [], 'us': []}
    for p in sorted(Path(raw_dir).glob('CropProd-*.txt')):
        res = parse(clean(p.read_bytes().decode('latin-1')))
        if not res or res[0].month not in MON:
            print('unparsed', p.name)
            continue
        rel, vals = res
        assert isinstance(rel, dt.date)
        for st, v in vals.items():
            rec = {'year': str(rel.year), 'reference_period_desc': f'YEAR - {MON[rel.month]} FORECAST', 'Value': f'{v:,}',
                   'prodn_practice_desc': 'ALL PRODUCTION PRACTICES', 'release': str(rel), 'state_alpha': st, 'source': p.name}
            rows['us' if st == 'US' else 'state'].append(rec)
    out = Path(root) / 'forecasts'
    out.mkdir(parents=True, exist_ok=True)
    for k, r in rows.items():
        (out / f'{k}_forecasts.json').write_text(json.dumps({'data': r}), encoding='utf-8')
    print('state rows', len(rows['state']), 'us rows', len(rows['us']))


def cmd_finals(root, first, last):
    import hashlib
    import time

    from research.crop_nowcast.nass_pull import fetch
    key = os.environ['NASS_API_KEY']
    out = Path(root) / 'finals'
    out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for lvl, agg in (('state', 'STATE'), ('us', 'NATIONAL')):
        rows = []
        for y in range(int(first), int(last) + 1):
            raw = fetch(dict(agg_level_desc=agg, statisticcat_desc='PRODUCTION', unit_desc='480 LB BALES',
                             reference_period_desc='YEAR'), key, y)
            part = [r for r in json.loads(raw)['data'] if r['reference_period_desc'] == 'YEAR']
            assert all(int(r['year']) == y for r in part)
            rows += part
        body = json.dumps({'data': rows}).encode()
        (out / f'{lvl}_production.json').write_bytes(body)
        rec[lvl] = {'rows': len(rows), 'sha256': hashlib.sha256(body).hexdigest(),
                    'retrieved_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        print(lvl, len(rows))
    text = json.dumps(rec, indent=1)
    assert key not in text
    (out / 'receipt.json').write_text(text, encoding='utf-8')


def _fc(root, lvl):
    d = pd.DataFrame(json.loads((Path(root) / 'forecasts' / f'{lvl}_forecasts.json').read_text())['data'])
    d['v'] = pd.to_numeric(d.Value.str.replace(',', ''))
    d['year'] = d.year.astype(int)
    return d


def cmd_gate(root):
    """Per report: TX and US present, sum of states within 2% of the US line. Reports failing are listed and excluded."""
    s, u = _fc(root, 'state'), _fc(root, 'us')
    bad, ok = [], []
    for (y, ref), g in s.groupby(['year', 'reference_period_desc']):
        us = u[(u.year == y) & (u.reference_period_desc == ref)].v
        good = 'TX' in set(g.state_alpha) and len(us) == 1 and abs(g.v.sum() / us.iloc[0] - 1) < 0.02
        (ok if good else bad).append(f'{y} {ref[7:10]}')
    print('reports passing gate', len(ok), '| failing', bad)
    (Path(root) / 'gate.json').write_text(json.dumps({'pass': ok, 'fail': bad}, indent=1), encoding='utf-8')
    return bad


def coef(full_dir, m):
    f = M.with_revision(M.frame(full_dir, 'state'), m)
    tr = f[f.year.between(2000, 2025)].dropna(subset=['e', 'r'])
    sd = float(np.sqrt((tr.r ** 2).mean()))
    z = (tr.r / sd).to_numpy()
    return float(z @ tr.e.to_numpy() / (z @ z + M.LAM)), sd, len(tr)


def blind_dir(root):
    """nass-style dir: blind forecasts (gate-passing reports only) + QuickStats finals."""
    gate = set(json.loads((Path(root) / 'gate.json').read_text())['pass'])
    d = Path(root) / 'nass'
    d.mkdir(exist_ok=True)
    for lvl in ('state', 'us'):
        fc = [r for r in json.loads((Path(root) / 'forecasts' / f'{lvl}_forecasts.json').read_text())['data']
              if f"{r['year']} {r['reference_period_desc'][7:10]}" in gate]
        fin = json.loads((Path(root) / 'finals' / f'{lvl}_production.json').read_text())['data']
        (d / f'{lvl}_production.json').write_text(json.dumps({'data': fc + fin}), encoding='utf-8')
    return d


def cmd_score(root, full_dir, out_json):
    d = blind_dir(root)
    res, cells = {'coef_trained_on': [2000, 2025]}, {}
    pe = lambda a, b: 100 * (a / b - 1)
    for m in SCORED:
        b, sd, n = coef(full_dir, m)
        s = M.with_revision(M.frame(d, 'state'), m).dropna(subset=['usda', 'r', 'final']).copy()
        s['corr'] = s.usda * np.exp(b * s.r / sd)
        us = M.frame(d, 'us').set_index('year')[[m, 'final']].rename(columns={m: 'usda_us'})
        j = s.groupby('year')[['corr', 'usda']].sum().join(us).dropna()
        j['us_hat'] = j.usda_us * j['corr'] / j.usda
        tx = s[s.state == 'TX'].set_index('year')
        r = {'b': b, 'sd': sd, 'train_rows': n,
             'TX_by_year': {int(y): [round(pe(x['corr'], x.final), 2), round(pe(x.usda, x.final), 2)] for y, x in tx.iterrows()},
             'US_by_year': {int(y): [round(pe(x.us_hat, x.final), 2), round(pe(x.usda_us, x.final), 2)] for y, x in j.iterrows()},
             'state_w': state_w(s, 'corr'), 'state_w_usda': state_w(s, 'usda')}
        for k in ('TX', 'US'):
            a = np.array(list(r[f'{k}_by_year'].values()))
            diff = np.abs(a[:, 1]) - np.abs(a[:, 0])
            r[k], r[f'{k}_usda'] = float(np.abs(a[:, 0]).mean()), float(np.abs(a[:, 1]).mean())
            r[f'{k}_better'], r[f'{k}_worse'] = int((diff > 0).sum()), int((diff < 0).sum())
            r[f'{k}_perm_p'] = perm_p(diff)
            cells[(m, k)] = dict(zip(r[f'{k}_by_year'], diff))
        res[m] = r
        print(f"{m}: b {b:+.4f} | TX {r['TX']:.2f} vs {r['TX_usda']:.2f} ({r['TX_better']}/{r['TX_worse']}) | "
              f"US {r['US']:.2f} vs {r['US_usda']:.2f} ({r['US_better']}/{r['US_worse']}) | "
              f"states {r['state_w']:.2f} vs {r['state_w_usda']:.2f}")
    lower = sum(res[m][k] < res[m][f'{k}_usda'] for m in SCORED for k in ('TX', 'US'))
    st = sum(res[m]['state_w'] < res[m]['state_w_usda'] for m in SCORED)
    years = sorted(set().union(*[set(v) for v in cells.values()]))
    mat = np.array([[cells[c].get(y, np.nan) for c in cells] for y in years])
    obs = float(np.nanmean(mat))
    rng = np.random.default_rng(42)
    sims = np.array([np.nanmean(mat * rng.choice([-1.0, 1.0], size=(len(years), 1))) for _ in range(20000)])
    res.update(years=years, tx_us_cells_lower=int(lower), state_months_lower=int(st), pooled_mean_gain_pts=obs,
               pooled_year_block_p=float((sims >= obs - 1e-12).mean()))
    res['primary_pass'] = bool(lower >= 6 and st >= 3)
    print(f"years {years[0]}-{years[-1]} ({len(years)}) | TX/US cells lower {lower}/8 | state months lower {st}/4 | "
          f"pooled gain {obs:+.3f} pts, year-block p {res['pooled_year_block_p']:.3f} | PRIMARY {res['primary_pass']}")
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    {'parse': cmd_parse, 'finals': cmd_finals, 'gate': cmd_gate, 'score': cmd_score}[cmd](*a)
