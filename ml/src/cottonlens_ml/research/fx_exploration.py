"""Same-date ECB currency crosses, causal Cotton windows; Tier A only."""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.ecb_fx import CURRENCIES, parse_ecb_csv

PROFILE='fx-exploration-v1'
LAGS=information.LAGS
PAIRS=('BRL','CNY','INR')
FIELDS=('return_1','return_5','return_21','volatility_21')


def crosses(rows):
    if (rows.date.isna().any() or rows.date.duplicated().any() or not rows.date.is_monotonic_increasing
            or rows.date.max()>=pd.Timestamp('2024-01-01')):
        raise ValueError('Unique sorted pre-audit reference dates required')
    result=rows.copy()
    values=result[[f'eur_{c}' for c in CURRENCIES]].to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)]<=0).any():
        raise ValueError('Positive finite or explicitly missing reference rates required')
    for currency in PAIRS:
        with np.errstate(over='ignore',under='ignore'):
            cross=result[f'eur_{currency}']/result.eur_USD
        if np.isinf(cross).any() or cross.le(0).any():raise ValueError('Invalid cross-rate numeric domain')
        result[f'{currency.lower()}_per_usd']=cross
    return result


def annual_rates(raw,year):
    summary=parse_ecb_csv(raw,year)
    if summary['repeated_currency_dates']:raise ValueError('Ambiguous historical FX revisions; cannot choose a vintage')
    frame=pd.read_csv(io.BytesIO(raw),parse_dates=['TIME_PERIOD'])
    if not frame.groupby('TIME_PERIOD').size().eq(4).all():
        raise ValueError('Same-date four-currency coverage required; never cross asynchronous quotes')
    pivot=frame.pivot(index='TIME_PERIOD',columns='CURRENCY',values='OBS_VALUE').sort_index()
    pivot.columns=[f'eur_{c}' for c in pivot.columns]
    return crosses(pivot.reset_index().rename(columns={'TIME_PERIOD':'date'})),summary


def compile_rates(root,output):
    root,output=Path(root),Path(output)
    audit=json.loads((root/'archive-audit-2010-2023.json').read_text(encoding='utf-8'))
    years=audit['years']
    if sorted(y['year'] for y in years)!=list(range(2010,2024)):
        raise ValueError('Fourteen distinct pinned annual FX sources required')
    frames=[];sources=[]
    for item in years:
        year=item['year'];folder=root/'ecb_fx'/str(year)/item['source_sha256']
        source,receipt=folder/'source.csv',folder/'retrieval.json'
        if digest(source)!=item['source_sha256'] or digest(receipt)!=item['receipt_sha256']:
            raise ValueError('Pinned annual FX source/receipt changed')
        meta=json.loads(receipt.read_text(encoding='utf-8'))
        if (meta['sha256']!=item['source_sha256'] or meta['year']!=year
                or meta['publication_clock_verified'] or meta['vintage_verified'] or meta['model_eligible']):
            raise ValueError('Annual FX scope/availability changed')
        frame,summary=annual_rates(source.read_bytes(),year)
        if summary['rows']!=item['rows'] or summary['missing_value_rows']!=item['missing_value_rows']:
            raise ValueError('Annual FX coverage differs from pinned audit')
        frames.append(frame);sources.append({**item,'source_url':meta['source_url'],'query':meta['query']})
    frame=crosses(pd.concat(frames,ignore_index=True).sort_values('date').reset_index(drop=True))
    output.parent.mkdir(parents=True,exist_ok=True);raw=frame.to_csv(index=False,date_format='%Y-%m-%d').encode()
    if output.exists():
        if output.read_bytes()!=raw:raise ValueError('Conflicting FX table preserved')
    else:output.write_bytes(raw)
    manifest={'table_sha256':digest(output),'rows':len(frame),'source_versions':sources,
        'audit_sha256':digest(root/'archive-audit-2010-2023.json'),'source_tier':'A_exploration_only',
        'raw_currency_rows':audit['total_rows'],'missing_raw_values':audit['missing_value_rows'],
        'unit':'same-date ECB currency/EUR divided by USD/EUR = currency units per USD',
        'scope':'BRL/CNY/INR reference rates; PKR absent; not intraday traded prices',
        'source_url':'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html',
        'calendar_policy':'first Cotton observation strictly after reference day; lag1/2/6, not actual publication clock',
        'model_eligible':False,'release_allowed':False,'publication_timestamp_verified':False,
        'first_version_verified':False,'cost_tl':0,'redistribution_reviewed':False}
    freeze_record(output.with_suffix('.manifest.json'),manifest)
    return manifest


def read_rates(table):
    table=Path(table);manifest=read_record(table.with_suffix('.manifest.json'))
    if (digest(table)!=manifest['table_sha256'] or manifest['model_eligible'] or manifest['release_allowed']
            or manifest['publication_timestamp_verified'] or manifest['first_version_verified']):
        raise ValueError('Pinned Tier-A FX table required')
    frame=pd.read_csv(table,parse_dates=['date']);expected=crosses(frame)
    if len(frame)!=manifest['rows']:raise ValueError('Frozen FX coverage changed')
    for c in PAIRS:
        name=f'{c.lower()}_per_usd'
        if not np.allclose(frame[name],expected[name],rtol=1e-12,atol=1e-12,equal_nan=True):
            raise ValueError('Same-date cross-rate identity differs')
    return frame,manifest


def add_rates(history,rates):
    if (history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing):
        raise ValueError('Unique sorted Cotton observations required')
    rates=crosses(rates);result=history.copy();calendar=history.date.to_numpy(dtype='datetime64[ns]')
    starts=np.searchsorted(calendar,rates.date.to_numpy(dtype='datetime64[ns]'),side='right')
    for lag in LAGS:
        for currency in PAIRS:
            pair=currency.lower();raw=rates[f'{pair}_per_usd'].to_numpy()
            level=np.full(len(history),np.nan);age=np.full(len(history),np.nan);cursor=0;selected=None
            available=starts+lag-1
            for index in range(len(history)):
                while cursor<len(available) and available[cursor]<=index:
                    # Holidays/empty observations do not refresh quote age.
                    if np.isfinite(raw[cursor]):selected=cursor
                    cursor+=1
                if selected is not None:
                    age[index]=index-starts[selected]
                    if index-available[selected]<=3:level[index]=raw[selected]
            log=pd.Series(np.log(level),index=history.index)
            returns=log.diff()
            values={'return_1':returns,'return_5':log.diff(5),'return_21':log.diff(21),
                    'volatility_21':returns.rolling(21,min_periods=21).std(ddof=0)}
            for field,value in values.items():result[f'fx_{pair}_{field}_L{lag}']=value
            result[f'fx_{pair}_age_L{lag}']=age
            result[f'fx_{pair}_missing_L{lag}']=(~np.isfinite(np.column_stack(list(values.values())))).any(axis=1).astype(float)
    return result


def group_names():
    groups={'base':list(FEATURE_NAMES)}
    for lag in LAGS:
        control=[f'fx_{c.lower()}_{f}_L{lag}' for c in PAIRS for f in ('age','missing')]
        groups[f'missing_L{lag}']=[*FEATURE_NAMES,*control]
        groups[f'currency_L{lag}']=[*FEATURE_NAMES,*control,
            *[f'fx_{c.lower()}_{f}_L{lag}' for c in PAIRS for f in FIELDS]]
    return groups


def prepare(repo,folder,reference,table):
    rows,source=read_rates(table)
    information.prepare_information(repo,folder,Path(reference),profile=PROFILE,source=source,
        add_features=lambda history:add_rates(history,rows),first_source_day=rows.date.min(),
        groups=group_names(),controls={f'currency_L{l}':f'missing_L{l}' for l in LAGS},prefix='fx-currency',
        hypothesis='INR/CNY/BRL currency return/risk context adds T+5 information beyond age/missingness',
        policy={'assumed_day':'ECB observation calendar day, not certified first publication','cotton_lags':list(LAGS),
            'max_extra_age':3,'windows':'1/5/21 retained Cotton observations; no dropped dates',
            'crosses':'same-source-day pairs only','publication_clock_verified':False})


def dispatch(args):
    def prepare_source(folder):
        if not all((args.reference_root,args.fx_table)):raise ValueError('Pinned parent and FX table required')
        prepare(args.repo,folder,args.reference_root,args.fx_table)
    information.dispatch_information(args,prepare_source)
