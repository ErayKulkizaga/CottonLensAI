"""One bounded weekly-sales hypothesis; latest API vintages are Tier-A only."""
import argparse
import importlib.metadata
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer

PROFILE = 'fas-exploration-v1'
LAGS = information.LAGS
FIELDS = ('currentMYNetSales','weeklyExports')


def compile_weekly(raw_root, acquisition, catalog, recovery, output):
    """Use pinned acquired bytes, never sum overlapping marketing-year snapshots."""
    catalog, acquisition, recovery = map(Path,(catalog,acquisition,recovery))
    commodities = json.loads(catalog.read_text(encoding='utf-8'))
    cotton = [c for c in commodities if c.get('commodityCode')==1404]
    if len(cotton)!=1 or cotton[0].get('unitId')!=2 or cotton[0].get('commodityName')!='All Upland Cotton':
        raise ValueError('Pinned All Upland Cotton running-bale catalog required')
    evidence = read_record(recovery)
    reconciled = [r.get('cotton_content_comparison',{}) for r in evidence['responses']]
    if len(reconciled)<2 or not all(r.get('content_matches_at_printed_precision') for r in reconciled):
        raise ValueError('Existing sampled FAS content/unit reconciliation required')
    if evidence['numeric_inputs']['commodity_catalog_sha256'] != digest(catalog):
        raise ValueError('Reconciliation/catalog identity mismatch')
    audit = json.loads(acquisition.read_text(encoding='utf-8'))
    pinned = [r for r in audit['archives'] if r['provider']=='fas']
    if sorted(r['year'] for r in pinned)!=list(range(2010,2025)):
        raise ValueError('All fifteen pinned acquisition years required')
    if next(r['source_sha256'] for r in pinned if r['year']==2020)!=evidence['numeric_inputs']['api_snapshot_sha256']:
        raise ValueError('Sample reconciliation must match pinned 2020 API bytes')
    payloads = {p.parent.name:p for p in Path(raw_root).glob('*/*/source.json')}
    parts, sources = [], {}
    for record in pinned:
        sha, year = record['source_sha256'], record['year']
        path = payloads.get(sha)
        if path is None or digest(path)!=sha:
            raise ValueError(f'Pinned annual FAS bytes missing/corrupt: {year}')
        data = pd.DataFrame(json.loads(path.read_text(encoding='utf-8')))
        required = ['commodityCode','unitId','countryCode','weekEndingDate',*FIELDS]
        if len(data)!=record['rows'] or not set(required).issubset(data):
            raise ValueError('FAS schema/count changed')
        if not data.commodityCode.eq(1404).all() or not data.unitId.eq(2).all():
            raise ValueError('Mixed commodity or units')
        data['week_date'] = pd.to_datetime(data.weekEndingDate,errors='raise')
        if data.week_date.dt.tz is not None or not data.week_date.eq(data.week_date.dt.normalize()).all():
            raise ValueError('Unambiguous calendar week dates required')
        if data.duplicated(['week_date','countryCode']).any():
            raise ValueError('Duplicate country/week would double count exports')
        for field in FIELDS:
            data[field] = pd.to_numeric(data[field],errors='raise')
            if not np.isfinite(data[field]).all():
                raise ValueError('Nonfinite sales/exports')
        if (data.weeklyExports<0).any():
            raise ValueError('Negative shipment volume; source review required')
        # API marketYear identifies the year ending July31. A boundary week can
        # appear in both annual files; select one year, never add both together.
        marketing = data.week_date.dt.year+(data.week_date.dt.month>=8).astype(int)
        use = data.loc[(marketing==year)&(data.week_date<'2024-01-01')]
        weekly = use.groupby('week_date',sort=True)[list(FIELDS)].sum().reset_index()
        weekly['market_year'] = year
        weekly['country_rows'] = use.groupby('week_date').size().to_numpy()
        parts.append(weekly);sources[str(year)] = sha
    weekly = pd.concat(parts,ignore_index=True).sort_values('week_date').reset_index(drop=True)
    if weekly.week_date.duplicated().any():
        raise ValueError('Overlapping marketing-year totals')
    contiguous = weekly.week_date.diff().dt.days.eq(7)&weekly.market_year.eq(weekly.market_year.shift())
    for field in FIELDS:
        weekly[field+'_change'] = weekly[field].diff().where(contiguous)
    # Hypothesis assumption, not actual release: report week-end +7 calendar days,
    # then first recorded Cotton observation, with +1/+5 extra lag stresses.
    weekly['assumed_day'] = weekly.week_date+pd.Timedelta(days=7)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    raw=weekly.to_csv(index=False,date_format='%Y-%m-%d').encode()
    if output.exists():
        if output.read_bytes()!=raw:raise ValueError('Preserve conflicting compiled table')
    else:output.write_bytes(raw)
    manifest={'table_sha256':digest(output),'rows':len(weekly),'annual_sources':sources,
        'catalog_sha256':digest(catalog),'acquisition_sha256':digest(acquisition),'recovery_sha256':digest(recovery),
        'historical_publication_verified':False,'first_version_verified':False,'model_eligible':False,
        'unit':'running_bales','commodity':'All Upland Cotton','source_tier':'A_exploration_only',
        'release_allowed':False,'assumed_day':'weekEndingDate + 7 calendar days; schedule assumption only',
        'boundary_policy':'week-ending date chooses marketing year ending next July31; no cross-year changes',
        'reconciliation_scope':'two stock-total examples only, not full history or weekly flows',
        'source_url':'https://apps.fas.usda.gov/opendatawebV2/','cost_tl':0,'redistribution_reviewed':False}
    freeze_record(output.with_suffix('.manifest.json'),manifest)
    return manifest


def add_weekly(history, weekly):
    result=history.copy()
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing or weekly.week_date.duplicated().any():
        raise ValueError('Unique sorted Cotton and weekly source dates required')
    calendar=history.date.to_numpy(dtype='datetime64[ns]')
    dates=weekly.week_date.to_numpy(dtype='datetime64[ns]')
    values=weekly[[*FIELDS,*[f+'_change' for f in FIELDS]]].to_numpy(dtype=float)
    after=np.searchsorted(calendar,weekly.assumed_day.to_numpy(dtype='datetime64[ns]'),side='right')
    report_index=np.searchsorted(calendar,dates,side='right')-1
    for lag in LAGS:
        events=sorted(zip(after+lag-1,dates,report_index,values,strict=True),key=lambda e:(e[0],e[1]))
        aligned=np.full((len(history),4),np.nan);age=np.full(len(history),np.nan)
        selected=None;cursor=0
        for index in range(len(history)):
            while cursor<len(events) and events[cursor][0]<=index:
                if selected is None or events[cursor][1]>selected[1]:selected=events[cursor]
                cursor+=1
            if selected is not None:
                age[index]=index-selected[2]
                if index-selected[0]<=10:aligned[index]=selected[3]
        # Fixed unit conversion, not estimated from full-series future values.
        for j,name in enumerate(('net_sales','exports','net_sales_change','exports_change')):
            result[f'fas_{name}_L{lag}']=aligned[:,j]/1_000_000
        result[f'fas_age_L{lag}']=age
        result[f'fas_missing_L{lag}']= (~np.isfinite(aligned)).any(axis=1).astype(float)
    return result


def group_names():
    groups={'base':list(FEATURE_NAMES)}
    for lag in LAGS:
        control=[f'fas_age_L{lag}',f'fas_missing_L{lag}']
        groups[f'missing_L{lag}']=[*FEATURE_NAMES,*control]
        groups[f'sales_L{lag}']=[*FEATURE_NAMES,*control,*[f'fas_{f}_L{lag}' for f in
            ('net_sales','exports','net_sales_change','exports_change')]]
    return groups


def prepare(repo,folder,reference,table):
    parent=read_record(Path(reference)/'ready.json');table=Path(table)
    source=read_record(table.with_suffix('.manifest.json'))
    if digest(Path(reference)/'history.parquet')!=parent['history_sha256'] or digest(table)!=source['table_sha256']:
        raise ValueError('Pinned parent/FAS table identity changed')
    if source['model_eligible'] or source['release_allowed'] or source['source_tier']!='A_exploration_only':
        raise ValueError('Only separate exploratory source policy allowed')
    weekly=pd.read_csv(table,parse_dates=['week_date','assumed_day'])
    if len(weekly)!=source['rows'] or weekly.week_date.max()>=pd.Timestamp('2024-01-01'):
        raise ValueError('Seen audit cannot enter FAS exploration')
    history=pd.read_parquet(Path(reference)/'history.parquet')
    if history.date.max()>=pd.Timestamp('2024-01-01'):raise ValueError('Seen audit excluded')
    history=add_weekly(history,weekly)
    start=pd.Timestamp(max(history.date.min(),weekly.assumed_day.min()+pd.Timedelta(days=1)))
    split=information.short_split(history,start,profile=PROFILE,first_year=2016)
    identity={'profile':PROFILE,'source_id':research_source_identity(repo)['source_id'],
        'reference_ready_sha256':digest(Path(reference)/'ready.json'),'source_evidence':source,
        'research_data_id':frame_identity(history,list(history)),'split':split,'groups':group_names(),
        'comparison_controls':{f'sales_L{lag}':f'missing_L{lag}' for lag in LAGS},'report_prefix':'fas-sales',
        'include_reference_diagnostics':True,'bootstrap_lengths':[10,20,40],
        'python':sys.version.split()[0],'versions':{p:importlib.metadata.version(p) for p in
            ('numpy','pandas','scipy','scikit-learn','xgboost')},'lock_sha256':digest(Path(repo)/'ml/uv.lock'),
        'execution':{'device':'cpu','threads':2},'publication_sources':[],
        'release_allowed':False,'gate_evaluation_allowed':False}
    with writer(folder):
        if (folder/'ready.json').exists():
            old=read_record(folder/'ready.json')
            if old['identity']!=identity or digest(folder/'history.parquet')!=old['history_sha256']:
                raise ValueError('Frozen FAS exploration changed; use new namespace')
            return
        if (folder/'history.parquet').exists():raise ValueError('Incomplete prepare preserved')
        history.to_parquet(folder/'history.parquet',index=False)
        freeze_record(folder/'ready.json',{'identity':identity,'history_sha256':digest(folder/'history.parquet')})
        freeze_record(folder/'preregistered.json',{'created_at':datetime.now(UTC).isoformat(),
            'hypothesis':'Weekly net sales/shipments add T+5 information beyond age and missingness',
            'lags':list(LAGS),'models':['Ridge alpha1','XGBoost depth2 fixed recipe'],'seed':42,
            'selection':'three63 past inner blocks; shrinkage grid; no outer tuning',
            'source_tier':'A_exploration_only','release_allowed':False,'gate_evaluated':False,
            'maximum_fit_jobs':information.budget(identity,history)})


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder=root_path(args.drive_root,args.experiment)
    if args.stage not in {'prepare','status','pilot-plan','pilot','compare','report'}:
        raise ValueError('Tier-A source cannot lock/export or enter wider search')
    if args.mirror_root:
        from cottonlens_ml.research.mirror import Mirror
        metadata_only=args.stage in ('status','pilot-plan','compare','report')
        if not ((folder/'ready.json').exists() and metadata_only and len(list((folder/'information-outputs').glob('*.json')))>=56):
            Mirror(folder,root_path(args.mirror_root,args.experiment)).hydrate(metadata_only=metadata_only)
    if args.stage=='prepare':
        if not args.reference_root or not args.fas_table:raise ValueError('Pinned parent and FAS table required')
        prepare(args.repo,folder,args.reference_root,args.fas_table)
    elif args.stage=='status':
        print(json.dumps({'prepared':(folder/'ready.json').exists(),'saved_outputs':len(list((folder/'information-outputs').glob('*.json'))),
            'ledger':Ledger(folder/'ledger',{}).summary(),'source_tier':'A','release_allowed':False}));return
    elif args.stage in ('compare','report'):print(json.dumps(information.compare(folder),indent=2))
    else:
        experiment=Experiment(folder,repo=args.repo)
        if args.stage=='pilot-plan':
            print(json.dumps({'maximum_fit_jobs':information.budget(experiment.identity,experiment.history),
                'folds':len(experiment.identity['split']['folds']),'groups':list(experiment.identity['groups']),'release_allowed':False}));return
        with writer(folder):
            if args.mirror_root:
                experiment.mirror=Mirror(folder,root_path(args.mirror_root,args.experiment));experiment.after_fit=experiment.mirror.fit
            print(json.dumps(information.run(experiment,args.max_minutes)))
    if args.mirror_root:
        mirror=Mirror(folder,root_path(args.mirror_root,args.experiment))
        paths=['ready.json','history.parquet','preregistered.json']+[p.relative_to(folder).as_posix() for p in (folder/'reports').glob('*.json')]
        mirror.enqueue(paths);print(json.dumps(mirror.flush()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root','acquisition','catalog','recovery','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(compile_weekly(args.raw_root,args.acquisition,args.catalog,args.recovery,args.output),indent=2))


if __name__=='__main__':main()
