"""Small weather hypothesis on the existing source-control runner, Tier A only."""
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.sources.power_weather import POINTS

PROFILE='weather-exploration-v1'
LAGS=information.LAGS
FIELDS=('rain_7','rain_30','temperature_7','heat_excess_7')


def read_weather(table):
    table=Path(table);source=read_record(table.with_suffix('.manifest.json'))
    if (digest(table)!=source['table_sha256'] or source['model_eligible'] or source['release_allowed']
            or source['publication_timestamp_verified'] or source['first_version_verified']):
        raise ValueError('Pinned Tier-A weather required')
    rows=pd.read_csv(table,parse_dates=['date'])
    if len(rows)!=source['rows']:raise ValueError('Weather coverage changed')
    regional(rows)
    return rows,source


def regional(rows):
    if (set(rows.point)!=set(POINTS) or rows.date.isna().any() or rows.duplicated(['date','point']).any()
            or rows.date.min()<pd.Timestamp('2010-01-01') or rows.date.max()>=pd.Timestamp('2024-01-01')):
        raise ValueError('Unique pre-audit fixed weather samples required')
    calendars=[]
    for _,part in rows.groupby('point'):
        dates=pd.date_range(part.date.min(),part.date.max())
        if not part.date.is_monotonic_increasing or list(part.date)!=list(dates):
            raise ValueError('Retained complete daily weather calendar required')
        calendars.append(list(part.date))
    if any(c!=calendars[0] for c in calendars):raise ValueError('Same-day three-point footprint required')
    values=rows[['PRECTOTCORR','T2M_MAX']].to_numpy(dtype=float)
    if np.isinf(values).any() or rows.PRECTOTCORR.lt(0).any() or rows.T2M_MAX.lt(-90).any() or rows.T2M_MAX.gt(65).any():
        raise ValueError('Weather units/domain invalid')
    daily=rows.groupby('date')[['PRECTOTCORR','T2M_MAX']].agg(lambda s:s.mean() if s.notna().all() else np.nan)
    # Daily UTC rolling sums, not a compressed sequence of observed valid days.
    return pd.DataFrame({'date':daily.index,'rain_7':daily.PRECTOTCORR.rolling(7,min_periods=7).sum(),
        'rain_30':daily.PRECTOTCORR.rolling(30,min_periods=30).sum(),
        'temperature_7':daily.T2M_MAX.rolling(7,min_periods=7).mean(),
        'heat_excess_7':(daily.T2M_MAX-35).clip(lower=0).rolling(7,min_periods=7).sum()}).reset_index(drop=True)


def add_weather(history,rows):
    daily=regional(rows)
    if history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique sorted Cotton observations required')
    result=history.copy();calendar=history.date.to_numpy(dtype='datetime64[ns]')
    assumed=(daily.date+pd.Timedelta(days=3)).to_numpy(dtype='datetime64[ns]')
    starts=np.searchsorted(calendar,assumed,side='right');raw=daily[list(FIELDS)].to_numpy()
    season=history.date.dt.month.between(4,11).astype(float)
    for lag in LAGS:
        available=starts+lag-1;values=np.full((len(history),len(FIELDS)),np.nan)
        age=np.full(len(history),np.nan);cursor=0;selected=None
        for index in range(len(history)):
            while cursor<len(available) and available[cursor]<=index:
                if np.isfinite(raw[cursor]).all():selected=cursor
                cursor+=1
            if selected is not None:
                age[index]=index-starts[selected]
                if index-available[selected]<=3:values[index]=raw[selected]
        for field,column in zip(FIELDS,values.T,strict=True):
            result[f'weather_{field}_L{lag}']=column
            result[f'weather_growing_{field}_L{lag}']=column*season
        result[f'weather_age_L{lag}']=age
        result[f'weather_missing_L{lag}']=(~np.isfinite(values)).any(axis=1).astype(float)
        result[f'weather_growing_season_L{lag}']=season
    return result


def group_names():
    groups={'base':list(FEATURE_NAMES)}
    for lag in LAGS:
        controls=[f'weather_{f}_L{lag}' for f in ('age','missing','growing_season')]
        groups[f'missing_L{lag}']=[*FEATURE_NAMES,*controls]
        groups[f'weather_L{lag}']=[*FEATURE_NAMES,*controls,
            *[f'weather_{f}_L{lag}' for f in FIELDS],*[f'weather_growing_{f}_L{lag}' for f in FIELDS]]
    return groups


def prepare(repo,folder,reference,table):
    rows,source=read_weather(table)
    information.prepare_information(repo,folder,Path(reference),profile=PROFILE,source=source,
        add_features=lambda history:add_weather(history,rows),first_source_day=rows.date.min()+pd.Timedelta(days=3),
        groups=group_names(),controls={f'weather_L{l}':f'missing_L{l}' for l in LAGS},prefix='weather',
        hypothesis='Fixed Southern High Plains rainfall/heat adds T+5 information beyond calendar/age/missingness',
        policy={'assumed_day':'UTC weather day +3 calendar days, not certified publication',
            'cotton_lags':list(LAGS),'max_extra_age':3,'calendar':'7/30 complete UTC days, no missing-row compression',
            'season':'April-November fixed calendar proxy; 35C exploratory excess-heat threshold',
            'geography':'three fixed equal-weight samples, no future production weights',
            'publication_clock_verified':False,'first_version_verified':False})


def dispatch(args):
    def prepare_source(folder):
        if not all((args.reference_root,args.weather_table)):raise ValueError('Pinned parent/weather table required')
        prepare(args.repo,folder,args.reference_root,args.weather_table)
    information.dispatch_information(args,prepare_source)
