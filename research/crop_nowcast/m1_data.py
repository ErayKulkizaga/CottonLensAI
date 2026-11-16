"""M1 inputs for Texas, 2000-2025 (public, point-in-time at each report's survey reference date = 1st of the month).

  usdm      Drought Monitor, Texas counties, weekly, Aug-Dec (UNL USDM data services; categorical % area)
  qs        QuickStats: Texas upland planted-area finals 1995-2025, Texas cotton condition (% poor, % very poor) weekly
  build     one row per season x report month: drought D2+ share (cotton-weighted) and % poor+very poor as of the 1st
Usage: python m1_data.py usdm|qs|build root
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MONTHS = {'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
USDM = 'https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent'


def get(url, accept='application/json'):
    for k in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research-client', 'Accept': accept}),
                                          timeout=180).read()
        except urllib.error.HTTPError as exc:
            if exc.code == 400:  # QuickStats answers 400 when a query has no rows; 403/429 are throttling: retry
                raise SystemExit(f'http {exc.code}')
            time.sleep(10 * (k + 1))
        except OSError:
            time.sleep(10 * (k + 1))
    raise SystemExit('failed ' + url.split('?')[0])


def cmd_usdm(root):
    out = Path(root) / 'usdm'
    out.mkdir(parents=True, exist_ok=True)
    for y in range(2000, 2026):
        f = out / f'tx_{y}.csv'
        if f.exists():
            continue
        d = json.loads(get(f'{USDM}?aoi=TX&startdate=7/15/{y}&enddate=12/5/{y}&statisticsType=1'))
        pd.DataFrame(d)[['mapDate', 'fips', 'none', 'd0', 'd1', 'd2', 'd3', 'd4']].to_csv(f, index=False)
        print(y, len(d), flush=True)


def cmd_qs(root):
    from research.crop_nowcast.nass_pull import fetch
    key = __import__('research.crop_nowcast.nass_pull', fromlist=['nass_key']).nass_key()
    out = Path(root)
    rows = []
    for y in range(1995, 2026):
        d = json.loads(fetch(dict(agg_level_desc='STATE', state_alpha='TX', statisticcat_desc='AREA PLANTED', unit_desc='ACRES',
                                  reference_period_desc='YEAR'), key, y))['data']
        rows += [{'year': y, 'planted_final': float(r['Value'].replace(',', ''))} for r in d
                 if r.get('prodn_practice_desc') == 'ALL PRODUCTION PRACTICES' and r['reference_period_desc'] == 'YEAR']
    pd.DataFrame(rows).to_csv(out / 'tx_planted_final.csv', index=False)
    cond = []
    for y in range(2000, 2026):
        for unit in ('PCT POOR', 'PCT VERY POOR'):
            q = dict(source_desc='SURVEY', commodity_desc='COTTON', statisticcat_desc='CONDITION', unit_desc=unit,
                     agg_level_desc='STATE', state_alpha='TX', year=str(y), format='JSON', key=key)
            import urllib.parse
            try:
                d = json.loads(get('https://quickstats.nass.usda.gov/api/api_GET/?' + urllib.parse.urlencode(q)))['data']
            except SystemExit:
                d = []
            cond += [{'year': y, 'unit': unit, 'week_ending': r['week_ending'], 'v': float(r['Value'])} for r in d
                     if r.get('week_ending') and r['Value'].strip().replace('.', '').isdigit()]
    pd.DataFrame(cond).to_csv(out / 'tx_condition.csv', index=False)
    print('planted rows', len(rows), 'condition rows', len(cond))


def cmd_build(root):
    """Weights: mean county planted 2008-2025 (QuickStats county_planted), fixed over time (declared)."""
    from research.crop_nowcast.state_model import nass_state
    cp = nass_state(ROOT / 'output/crop-nowcast-t39/nass', 'county_planted')
    cp = cp[(cp.state_alpha == 'TX') & (cp.reference_period_desc == 'YEAR') & (cp.county_ansi.astype(str).str.len() > 0)]
    cp['fips'] = '48' + cp.county_ansi.astype(str).str.zfill(3)
    w = cp.groupby('fips').v.mean()
    rows = []
    for f in sorted((Path(root) / 'usdm').glob('tx_*.csv')):
        u = pd.read_csv(f, dtype={'fips': str}, parse_dates=['mapDate'])
        u['d2p'] = u.d2 + u.d3 + u.d4
        y = int(f.stem[3:])
        for m, mon in MONTHS.items():
            ref = pd.Timestamp(y, mon, 1)
            last = u[u.mapDate <= ref - pd.Timedelta(days=2)]  # USDM map dated Tuesday, published Thursday
            last = last[last.mapDate == last.mapDate.max()]
            j = last.set_index('fips').d2p.to_frame().join(w.rename('w'), how='inner')
            rows.append({'year': y, 'month': m, 'usdm_map': str(last.mapDate.max().date()),
                         'd2p_cotton_w': float(np.average(j.d2p, weights=j.w)) / 100,
                         'd2p_state': float(last.d2p.mean()) / 100})
    d = pd.DataFrame(rows)
    c = pd.read_csv(Path(root) / 'tx_condition.csv', parse_dates=['week_ending'])
    pv = c.pivot_table(index=['year', 'week_ending'], columns='unit', values='v').reset_index()
    pv['pvp'] = pv['PCT POOR'] + pv['PCT VERY POOR']
    pr = []
    for _, r in d.iterrows():
        ref = pd.Timestamp(int(r.year), MONTHS[r.month], 1)
        g = pv[(pv.year == r.year) & (pv.week_ending <= ref - pd.Timedelta(days=1))]  # week ends Sunday, released Monday
        pr.append(float(g.sort_values('week_ending').pvp.iloc[-1]) / 100 if len(g) else np.nan)
    d['pvp'] = pr
    d.to_csv(Path(root) / 'tx_signals.csv', index=False)
    print(d.groupby('month')[['d2p_cotton_w', 'pvp']].count().to_string())


if __name__ == '__main__':
    {'usdm': cmd_usdm, 'qs': cmd_qs, 'build': cmd_build}[sys.argv[1]](*sys.argv[2:])
