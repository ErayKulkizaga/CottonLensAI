"""County-year features as of a nowcast cutoff (development years only).

Availability rules (frozen here, documented in the data contract):
- weather: daily values dated <= cutoff - WEATHER_LAG_DAYS;
- NDVI: a composite counts only if end_date + NDVI_LAG_DAYS <= cutoff;
- yields/areas: only years < Y (prior-year county estimates are assumed released
  before 1 August of Y; recorded as an assumption);
- the CDL mask used inside the NDVI series is already year Y-1.
No value from year Y's NASS estimates enters a feature.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

WEATHER_LAG_DAYS = 3
NDVI_LAG_DAYS = 8
SEASON_START = '04-01'
CUTOFFS = {'aug1': '08-01', 'sep1': '09-01', 'oct1': '10-01', 'nov1': '11-01', 'dec1': '12-01'}


def nass_county(nass_dir, name):
    d = pd.DataFrame(json.loads((Path(nass_dir) / f'{name}.json').read_text())['data'])
    d = d[(d.reference_period_desc == 'YEAR') & (d.prodn_practice_desc == 'ALL PRODUCTION PRACTICES')
          & d.county_ansi.ne('')].copy()
    d['value'] = pd.to_numeric(d.Value.str.replace(',', ''), errors='coerce')
    d['fips'] = d.state_ansi.str.zfill(2) + d.county_ansi.str.zfill(3)
    return d.set_index(['fips', 'year']).value.astype(float).groupby(level=[0, 1]).first()


def targets(nass_dir):
    t = pd.DataFrame({'yield': nass_county(nass_dir, 'county_yield'),
                      'production': nass_county(nass_dir, 'county_production'),
                      'harvested': nass_county(nass_dir, 'county_harvested'),
                      'planted': nass_county(nass_dir, 'county_planted')})
    t.index = t.index.set_names(['fips', 'year'])
    t['harv_ratio'] = (t.harvested / t.planted).clip(0, 1.05)
    return t.reset_index().astype({'year': int})


def weather(power_dir, fips, clim_dir=None):
    parts = []
    for d, span in ((clim_dir, '1991_2007'), (power_dir, '2008_2021')):
        if d is None:
            continue
        p = json.loads((Path(d) / f'{fips}_{span}.json').read_text())['properties']['parameter']
        w = pd.DataFrame(p)
        w.index = pd.to_datetime(w.index, format='%Y%m%d')
        parts.append(w)
    return pd.concat(parts).sort_index().replace(-999.0, np.nan)


def weather_feats(w, year, cutoff):
    end = pd.Timestamp(f'{year}-{cutoff}') - pd.Timedelta(days=WEATHER_LAG_DAYS)
    s = w.loc[pd.Timestamp(f'{year}-{SEASON_START}'):end]
    tmean = (s.T2M_MAX + s.T2M_MIN) / 2
    gdd = (tmean - 15.6).clip(lower=0)
    out = {'w_prcp': s.PRECTOTCORR.sum(), 'w_tmax': s.T2M_MAX.mean(), 'w_hot35': (s.T2M_MAX > 35).sum(),
           'w_hot38': (s.T2M_MAX > 38).sum(), 'w_gdd': gdd.sum(), 'w_rh': s.RH2M.mean(), 'w_srad': s.ALLSKY_SFC_SW_DWN.mean(),
           'w_dry_days': (s.PRECTOTCORR < 1).sum()}
    for m in (5, 6, 7, 8, 9, 10):
        sm = s[s.index.month == m]
        if len(sm):
            out[f'w_prcp_m{m}'] = sm.PRECTOTCORR.sum()
            out[f'w_tmax_m{m}'] = sm.T2M_MAX.mean()
    # pre-season soil-moisture proxy: Jan-Mar precipitation
    pre = w.loc[pd.Timestamp(f'{year}-01-01'):pd.Timestamp(f'{year}-03-31')]
    out['w_prcp_preseason'] = pre.PRECTOTCORR.sum()
    return out


def ndvi_table(ndvi_dir):
    parts = [pd.read_csv(p, dtype={'fips': str}) for p in sorted(Path(ndvi_dir).glob('*.csv'))]
    d = pd.concat(parts)
    d['fips'] = d.fips.str.zfill(5)
    d['end'] = pd.to_datetime(d.end)
    # counties split across tiles: pixel-weighted mean, other stats from the larger part
    d['w'] = d.n_valid.clip(lower=0)
    g = d.sort_values('w').groupby(['fips', 'year', 'end'])
    agg = g.apply(lambda x: pd.Series({'mean': np.average(x['mean'], weights=x.w) if x.w.sum() else np.nan,
                                        'median': x['median'].iloc[-1], 'p90': x['p90'].iloc[-1], 'n': x.w.sum()}),
                  include_groups=False)
    return agg.reset_index()


def ndvi_feats(nd, fips, year, cutoff):
    lim = pd.Timestamp(f'{year}-{cutoff}') - pd.Timedelta(days=NDVI_LAG_DAYS)
    s = nd[(nd.fips == fips) & (nd.year == year) & (nd.end <= lim) & (nd.n >= 20)].sort_values('end')
    if s.empty:
        return {}
    v = s['median'].to_numpy()
    out = {'n_max': np.nanmax(v), 'n_last': v[-1], 'n_mean': np.nanmean(v), 'n_p90max': np.nanmax(s.p90),
           'n_int': np.nansum(np.clip(v - 0.2, 0, None)), 'n_count': len(v)}
    for m in (6, 7, 8, 9):
        sm = s[s.end.dt.month == m]['median']
        if len(sm):
            out[f'n_m{m}'] = sm.mean()
    return out


def build(nass_dir, power_dir, ndvi_dir, panel_json, cutoff_key, clim_dir=None):
    cutoff = CUTOFFS[cutoff_key]
    panel = json.loads(Path(panel_json).read_text())['counties']
    t = targets(nass_dir)
    nd = ndvi_table(ndvi_dir) if ndvi_dir and any(Path(ndvi_dir).glob('*.csv')) else None
    rows = []
    for c in panel:
        f = c['fips']
        w = weather(power_dir, f, clim_dir)
        hist = t[t.fips == f].set_index('year')
        clim = {}
        if clim_dir is not None:
            for y0 in range(1991, 2022):
                clim[y0] = weather_feats(w, y0, cutoff)
        for year in range(2009, 2022):
            r = {'fips': f, 'state': c['state'], 'year': year}
            past = hist[hist.index < year]
            ys = past['yield'].dropna()
            if len(ys) >= 3:
                k = np.polyfit(ys.index, ys.values, 1) if len(ys) >= 5 else (0.0, ys.mean())
                r['trend_yield'] = float(np.polyval(k, year)) if len(ys) >= 5 else float(ys.mean())
                r['prev_yield'] = float(ys.iloc[-1])
                r['mean_yield'] = float(ys.mean())
            hr = past['harv_ratio'].dropna()
            if len(hr):
                r['mean_harv_ratio'] = float(hr.mean())
                r['prev_harv_ratio'] = float(hr.iloc[-1])
            wf = weather_feats(w, year, cutoff)
            r.update(wf)
            if clim:
                # long climatology anomaly: same cutoff, years 1991..Y-1 only
                for k, v in wf.items():
                    past = [clim[y0].get(k) for y0 in range(1991, year)]
                    past = [x for x in past if x is not None and np.isfinite(x)]
                    if past:
                        r[k + '_c'] = v - float(np.mean(past))
                        sd = float(np.std(past))
                        r[k + '_z'] = (v - float(np.mean(past))) / sd if sd > 0 else 0.0
            if nd is not None:
                cur = ndvi_feats(nd, f, year, cutoff)
                r.update(cur)
                # past-only anomaly against the county's earlier seasons at the same cutoff
                prev = [ndvi_feats(nd, f, y, cutoff).get('n_max') for y in range(2009, year)]
                prev = [p for p in prev if p is not None]
                if cur and prev:
                    r['n_max_anom'] = cur['n_max'] - float(np.mean(prev))
            if year in hist.index:
                for k in ('yield', 'production', 'harvested', 'planted', 'harv_ratio'):
                    r[f'y_{k}'] = hist.loc[year, k]
            rows.append(r)
    return pd.DataFrame(rows)
