"""Pinned disaggregated futures-only positions; calendar assumptions = Tier A.

Annual archives contain today's historical values, not certified first vintages.
Known uncertain/delayed report windows are masked as inputs, never as origins.
"""
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import freeze_record, read_record

PROFILE='cftc-exploration-v1'
LAGS=information.LAGS
FIELDS=('managed_net_share','producer_net_share','managed_share_change')
MARKET='033661'
# Conservative report-date windows, not a reconstruction of actual publication.
# Catch-up schedules were intentions, not certified first-version release clocks.
MASK_WINDOWS=(('2013-09-24','2013-11-12','shutdown_2013'),
              ('2018-12-24','2019-03-12','shutdown_2018_2019'),
              ('2023-01-31','2023-03-14','ion_2023'))
MASK_DAYS={'2012-11-27':'classification_revision', '2019-03-26':'cotton_position_revision'}


def positions(raw,year):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names=[n for n in archive.namelist() if n.lower().endswith(('.txt','.csv'))]
        if len(names)!=1 or archive.getinfo(names[0]).file_size>50*1024*1024:
            raise ValueError('One bounded annual CFTC text member required')
        frame=pd.read_csv(archive.open(names[0]),low_memory=False,dtype=str)
    frame.columns=[re.sub(r'[^a-z0-9]+','_',c.strip().lower()).strip('_') for c in frame.columns]
    if frame.columns.duplicated().any():raise ValueError('Ambiguous CFTC columns')
    names={'cftc_contract_market_code':'market_code','open_interest_all':'open_interest',
        'm_money_positions_long_all':'managed_long','m_money_positions_short_all':'managed_short',
        'prod_merc_positions_long_all':'producer_long','prod_merc_positions_short_all':'producer_short',
        'futonly_or_combined':'report_type','market_and_exchange_names':'market_name'}
    date=[c for c in frame if c.startswith('report_date_as_') and 'yyyy' in c]
    if len(date)!=1 or not set(names).issubset(frame):raise ValueError('Disaggregated futures-only schema required')
    code=frame.cftc_contract_market_code.str.strip().str.replace('.0','',regex=False).str.zfill(6)
    frame=frame.loc[code.eq(MARKET),[date[0],*names]].rename(columns={date[0]:'report_date',**names})
    frame['market_code']=MARKET
    frame['report_date']=pd.to_datetime(frame.report_date,format='mixed',errors='raise')
    if (frame.empty or frame.report_date.dt.year.ne(year).any() or frame.report_date.duplicated().any()
            or not frame.report_type.str.strip().eq('FutOnly').all()
            or not frame.market_name.str.upper().str.contains('COTTON').all()):
        raise ValueError('Unique Cotton futures-only dates in requested year required')
    for field in ('open_interest','managed_long','managed_short','producer_long','producer_short'):
        frame[field]=pd.to_numeric(frame[field],errors='raise')
    values=frame[['open_interest','managed_long','managed_short','producer_long','producer_short']].to_numpy()
    if (not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any()
            or (frame.open_interest<=0).any() or (values[:,1:]>values[:,0,None]).any()):
        raise ValueError('Nonnegative integer positions bounded by positive open interest required')
    return frame.sort_values('report_date').reset_index(drop=True)


def derive(rows):
    rows=rows.sort_values('report_date').reset_index(drop=True).copy()
    if rows.report_date.isna().any() or rows.report_date.duplicated().any():raise ValueError('Unique report dates required')
    values=rows[['open_interest','managed_long','managed_short','producer_long','producer_short']].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any()
            or (values[:,0]<=0).any() or (values[:,1:]>values[:,0,None]).any()):
        raise ValueError('Invalid position counts/open interest')
    rows['assumed_day']=rows.report_date+pd.Timedelta(days=7)
    rows['input_mask_reason']=''
    for start,end,reason in MASK_WINDOWS:
        rows.loc[rows.report_date.between(start,end),'input_mask_reason']=reason
    for day,reason in MASK_DAYS.items():rows.loc[rows.report_date.eq(pd.Timestamp(day)),'input_mask_reason']=reason
    rows['managed_net_share']=(rows.managed_long-rows.managed_short)/rows.open_interest
    rows['producer_net_share']=(rows.producer_long-rows.producer_short)/rows.open_interest
    valid=rows.report_date.diff().dt.days.eq(7)&rows.input_mask_reason.eq('')&rows.input_mask_reason.shift().eq('')
    rows['managed_share_change']=rows.managed_net_share.diff().where(valid)
    return rows


def compile_positions(root,output):
    root,output=Path(root),Path(output)
    inventory=read_record(root/'archive-inventory.json');sources=[];frames=[]
    for item in inventory['files']:
        path=root/item['folder'];record=read_record(path/'retrieval.json')
        if digest(path/'source.bin')!=item['sha256'] or record['sha256']!=item['sha256'] or record['source_url']!=item['source_url']:
            raise ValueError('Raw CFTC archive identity changed')
        match=re.fullmatch(r'https://www.cftc.gov/files/dea/history/fut_disagg_txt_(\d{4})\.zip',item['source_url'])
        sources.append(item)
        if match:
            year=int(match[1])
            if not 2010<=year<=2023:raise ValueError('Audit period cannot enter pilot source')
            frames.append(positions((path/'source.bin').read_bytes(),year))
    if len(frames)!=14 or {int(f.report_date.dt.year.iloc[0]) for f in frames}!=set(range(2010,2024)):
        raise ValueError('Exactly fourteen distinct pinned annual archives required')
    frame=derive(pd.concat(frames,ignore_index=True))
    output.parent.mkdir(parents=True,exist_ok=True);raw=frame.to_csv(index=False,date_format='%Y-%m-%d').encode()
    if output.exists():
        if output.read_bytes()!=raw:raise ValueError('Conflicting source table preserved')
    else:output.write_bytes(raw)
    manifest={'table_sha256':digest(output),'rows':len(frame),'source_versions':sources,
        'inventory_sha256':digest(root/'archive-inventory.json'),'scope':'Cotton No.2 033661, disaggregated futures-only, all maturities',
        'unit':'contracts; signed net positions divided by same-report all-maturity open interest',
        'source_url':'https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm',
        'mask_windows':MASK_WINDOWS,'mask_days':MASK_DAYS,'masked_reports':int(frame.input_mask_reason.ne('').sum()),
        'calendar_policy':'report date plus seven days, then next Cotton observation and lag1/2/6; not certified availability',
        'source_tier':'A_exploration_only','model_eligible':False,'release_allowed':False,
        'publication_timestamp_verified':False,'first_version_verified':False,'cost_tl':0,'redistribution_reviewed':False}
    freeze_record(output.with_suffix('.manifest.json'),manifest)
    return manifest


def read_positions(table):
    table=Path(table);manifest=read_record(table.with_suffix('.manifest.json'))
    if (digest(table)!=manifest['table_sha256'] or manifest['model_eligible'] or manifest['release_allowed']
            or manifest['publication_timestamp_verified'] or manifest['first_version_verified']):
        raise ValueError('Pinned Tier-A positions required')
    rows=pd.read_csv(table,parse_dates=['report_date','assumed_day'],dtype={'market_code':str},keep_default_na=False)
    rows['managed_share_change']=pd.to_numeric(rows.managed_share_change.replace('',np.nan))
    if (len(rows)!=manifest['rows'] or not rows.report_date.is_monotonic_increasing
            or rows.report_date.max()>=pd.Timestamp('2024-01-01') or not rows.market_code.eq(MARKET).all()
            or not rows.report_type.str.strip().eq('FutOnly').all()):raise ValueError('Frozen Cotton futures-only coverage required')
    expected=derive(rows)
    if not rows.assumed_day.equals(expected.assumed_day) or not rows.input_mask_reason.equals(expected.input_mask_reason):
        raise ValueError('Frozen calendar/masking policy changed')
    for field in FIELDS:
        if not np.allclose(rows[field],expected[field],rtol=1e-12,atol=1e-12,equal_nan=True):
            raise ValueError('Causal CFTC ratios/change differ')
    return rows,manifest


def add_positions(history,rows):
    if (history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing
            or rows.report_date.isna().any() or rows.report_date.duplicated().any()
            or not rows.report_date.is_monotonic_increasing or rows.assumed_day.isna().any()):
        raise ValueError('Unique sorted chronological observations required')
    result=history.copy();calendar=history.date.to_numpy(dtype='datetime64[ns]')
    after=np.searchsorted(calendar,rows.assumed_day.to_numpy(dtype='datetime64[ns]'),side='right')
    reported=np.searchsorted(calendar,rows.report_date.to_numpy(dtype='datetime64[ns]'),side='right')-1
    values=rows[list(FIELDS)].to_numpy(dtype=float);masked=rows.input_mask_reason.ne('').to_numpy()
    for lag in LAGS:
        aligned=np.full((len(history),3),np.nan);age=np.full(len(history),np.nan);cursor=0;selected=None
        starts=after+lag-1
        for index in range(len(history)):
            while cursor<len(starts) and starts[cursor]<=index:
                if not masked[cursor]:selected=cursor
                cursor+=1
            if selected is not None:
                age[index]=index-reported[selected]
                if index-starts[selected]<=10:aligned[index]=values[selected]
        for j,field in enumerate(FIELDS):result[f'cftc_{field}_L{lag}']=aligned[:,j]
        result[f'cftc_age_L{lag}']=age
        result[f'cftc_missing_L{lag}']=(~np.isfinite(aligned)).any(axis=1).astype(float)
    return result


def group_names():
    groups={'base':list(FEATURE_NAMES)}
    for lag in LAGS:
        control=[f'cftc_age_L{lag}',f'cftc_missing_L{lag}']
        groups[f'missing_L{lag}']=[*FEATURE_NAMES,*control]
        groups[f'position_L{lag}']=[*FEATURE_NAMES,*control,*[f'cftc_{f}_L{lag}' for f in FIELDS]]
    return groups


def prepare(repo,folder,reference,table):
    rows,source=read_positions(table)
    information.prepare_information(repo,folder,Path(reference),profile=PROFILE,source=source,
        add_features=lambda history:add_positions(history,rows),first_source_day=rows.assumed_day.min(),
        groups=group_names(),controls={f'position_L{l}':f'missing_L{l}' for l in LAGS},prefix='cftc-position',
        hypothesis='Managed-money and producer positioning adds T+5 information beyond age/missingness',
        policy={'assumed_day':'report date plus seven days, not first publication','cotton_lags':list(LAGS),
                'max_extra_age':10,'known_uncertain_inputs_masked':True,'publication_clock_verified':False})


def dispatch(args):
    def prepare_source(folder):
        if not all((args.reference_root,args.cftc_table)):raise ValueError('Pinned parent and CFTC table required')
        prepare(args.repo,folder,args.reference_root,args.cftc_table)
    information.dispatch_information(args,prepare_source)
