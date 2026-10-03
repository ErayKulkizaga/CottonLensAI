"""One World cotton balance hypothesis from archived, unverified-time vintages."""
import re
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.sources.wasde import ReleasePage, cotton_rows

PROFILE='wasde-exploration-v1'
LAGS=information.LAGS
FIELDS=('stock_use','production_use','stock_use_change')
EVIDENCE=('numeric-evidence-2016-2019.json','numeric-evidence-2020-2023-r2.json')


def balance(rows, release_day):
    """Select latest crop year/current forecast month, never last-year actuals."""
    day=pd.Timestamp(release_day)
    world=[r for r in rows if r['region']=='World']
    if not world or {r['report_month'] for r in world}!={day.strftime('%B %Y')}:
        raise ValueError('Report month and dated release page disagree')
    seasons={r['marketing_year']:re.fullmatch(r'(\d{4})/(\d{2})(?: Est\.| Proj\.)?',r['marketing_year'])
             for r in world}
    if any(m is None or int(m[2])!=(int(m[1])+1)%100 for m in seasons.values()):
        raise ValueError('Unreviewed cotton marketing year')
    season=max(seasons,key=lambda key:int(seasons[key][1]))
    selected=[r for r in world if r['marketing_year']==season and r['forecast_month']==day.strftime('%b')]
    values={}
    for field in ('Production','Domestic Use','Ending Stocks'):
        matches=[r for r in selected if r['attribute']==field]
        if len(matches)!=1 or matches[0]['value'] is None:
            raise ValueError('One finite current forecast per World balance field required')
        if matches[0]['units']!='(Million 480-Pound Bales)':raise ValueError('Wrong balance units')
        values[field]=float(matches[0]['value'])
    if not all(np.isfinite(v) and v>=0 for v in values.values()) or values['Domestic Use']<=0:
        raise ValueError('Positive consumption and nonnegative finite balance required')
    return {'crop_year':int(seasons[season][1]),'stock_use':values['Ending Stocks']/values['Domestic Use'],
            'production_use':values['Production']/values['Domestic Use']}


def compile_balance(archive_root, output):
    """Reuse immutable archives and numeric reviews offline; no clock admission."""
    root,output=Path(archive_root),Path(output)
    cached={}
    for p in (root/'wasde').glob('*/retrieval.json'):
        receipt=read_record(p)
        if receipt['kind']!='wasde' or digest(p.parent/'source.bin')!=receipt['sha256']:
            raise ValueError('Corrupt archived WASDE bytes')
        cached.setdefault(receipt['source_url'],set()).add(p.parent)
    for p in (root/'wasde_url_aliases').glob('*/alias.json'):
        alias=read_record(p);folder=root/'wasde'/alias['sha256'];receipt=read_record(folder/'retrieval.json')
        if receipt['source_url']!=alias['canonical_source_url'] or digest(folder/'source.bin')!=alias['sha256']:
            raise ValueError('Corrupt source alias')
        cached.setdefault(alias['source_url'],set()).add(folder)
    def one(url):
        paths=cached.get(url,set())
        if len(paths)!=1:raise ValueError('Missing/ambiguous pinned WASDE version')
        return next(iter(paths))/'source.bin'
    versions=[];proofs={};sources=[]
    for name in EVIDENCE:
        proof=read_record(root/name);proofs[name]=digest(root/name)
        if proof['model_eligible'] or proof['publication_timing_verified']:
            raise ValueError('This route requires explicit Tier-A diagnostic evidence')
        for item in proof['releases']:
            if item['numeric_status']!='passed':raise ValueError('Unreconciled WASDE release')
            url=item['url']
            dated=re.search(r'/(\d{4}-\d{2}-\d{2})(?:-\d+)?$',url)
            if not dated or not url.startswith('https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates/'):
                raise ValueError('Unknown dated official release URL')
            day=pd.Timestamp(dated[1])
            if day>=pd.Timestamp('2024-01-01'):raise ValueError('Seen audit cannot enter compiler')
            page_path=one(url);page=ReleasePage();page.feed(page_path.read_text(encoding='utf-8-sig'))
            links={urljoin(url,l) for l in page.links if l.endswith('.xml')}
            if len(links)!=1:raise ValueError('One XML source required')
            xml=one(next(iter(links)))
            if item.get('xml_sha256',digest(xml))!=digest(xml):raise ValueError('Review/XML identity differs')
            values=balance(cotton_rows(xml.read_bytes()),day)
            sources.append({'release_url':url,'page_sha256':digest(page_path),'xml_sha256':digest(xml),
                            'numeric_evidence_file':name})
            versions.append({'assumed_day':day,**values})
    frame=pd.DataFrame(versions).sort_values('assumed_day')
    # Same-day duplicate pages are collapsed only for identical hypothesis inputs.
    for _,group in frame.groupby('assumed_day'):
        if len(group[['crop_year','stock_use','production_use']].drop_duplicates())!=1:
            raise ValueError('Conflicting same-day balance versions; explicit review required')
    frame=frame.drop_duplicates('assumed_day').reset_index(drop=True)
    # A later duplicate page with unchanged inputs is not a new revision/age reset.
    unchanged=frame[['crop_year','stock_use','production_use']].eq(frame[['crop_year','stock_use','production_use']].shift()).all(axis=1)
    same_month=frame.assumed_day.dt.to_period('M').eq(frame.assumed_day.shift().dt.to_period('M'))
    frame=frame.loc[~(unchanged&same_month)].reset_index(drop=True)
    same_year=frame.crop_year.eq(frame.crop_year.shift())
    recent=frame.assumed_day.diff().dt.days.le(62)
    frame['stock_use_change']=frame.stock_use.diff().where(same_year&recent)
    output.parent.mkdir(parents=True,exist_ok=True)
    raw=frame.to_csv(index=False,date_format='%Y-%m-%d').encode()
    if output.exists():
        if output.read_bytes()!=raw:raise ValueError('Conflicting table preserved')
    else:output.write_bytes(raw)
    manifest={'table_sha256':digest(output),'rows':len(frame),'source_versions':sources,
        'numeric_evidence_sha256':proofs,'unit':'ratios from million 480-pound bales',
        'scope':'World, latest forecast crop year, current report-month forecast',
        'source_url':'https://esmis.nal.usda.gov/publication/world-agricultural-supply-and-demand-estimates',
        'source_tier':'A_exploration_only','model_eligible':False,'release_allowed':False,
        'publication_timestamp_verified':False,'first_version_verified':False,'cost_tl':0,
        'redistribution_reviewed':False,'missing_report_months':['2019-01'],
        'version_policy':'identical same-day or consecutive same-month inputs collapse; changed updates preserved',
        'delta_policy':'only same crop year and <=62 calendar days; never across crop-year reset'}
    freeze_record(output.with_suffix('.manifest.json'),manifest)
    return manifest


def read_balance(table):
    table=Path(table);manifest=read_record(table.with_suffix('.manifest.json'))
    if (digest(table)!=manifest['table_sha256'] or manifest['model_eligible']
            or manifest['release_allowed'] or manifest['publication_timestamp_verified']):
        raise ValueError('Pinned Tier-A balance required')
    rows=pd.read_csv(table,parse_dates=['assumed_day'])
    if (len(rows)!=manifest['rows'] or rows.assumed_day.isna().any() or rows.assumed_day.duplicated().any()
            or not rows.assumed_day.is_monotonic_increasing or rows.assumed_day.max()>=pd.Timestamp('2024-01-01')):
        raise ValueError('Unique sorted pre-audit release days required')
    values=rows[['stock_use','production_use']].to_numpy()
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid balance ratios')
    valid=rows.crop_year.eq(rows.crop_year.shift())&rows.assumed_day.diff().dt.days.le(62)
    expected=rows.stock_use.diff().where(valid)
    if not np.allclose(rows.stock_use_change,expected,equal_nan=True,rtol=1e-12,atol=1e-12):
        raise ValueError('Revision differs from causal same-crop-year change')
    return rows,manifest


def add_balance(history, reports):
    if (history.date.isna().any() or history.date.duplicated().any() or not history.date.is_monotonic_increasing
            or reports.assumed_day.isna().any() or reports.assumed_day.duplicated().any()
            or not reports.assumed_day.is_monotonic_increasing):
        raise ValueError('Unique finite chronological observations required')
    result=history.copy();calendar=history.date.to_numpy(dtype='datetime64[ns]')
    after=np.searchsorted(calendar,reports.assumed_day.to_numpy(dtype='datetime64[ns]'),side='right')
    values=reports[list(FIELDS)].to_numpy(dtype=float)
    for lag in LAGS:
        aligned=np.full((len(history),len(FIELDS)),np.nan);age=np.full(len(history),np.nan)
        starts=after+lag-1;cursor=0;selected=None
        for index in range(len(history)):
            while cursor<len(starts) and starts[cursor]<=index:
                selected=cursor;cursor+=1
            if selected is not None:
                age[index]=index-after[selected]
                if index-starts[selected]<=40:aligned[index]=values[selected]
        for j,field in enumerate(FIELDS):result[f'wasde_{field}_L{lag}']=aligned[:,j]
        result[f'wasde_age_L{lag}']=age
        result[f'wasde_missing_L{lag}']=(~np.isfinite(aligned)).any(axis=1).astype(float)
    return result


def group_names():
    groups={'base':list(FEATURE_NAMES)}
    for lag in LAGS:
        control=[f'wasde_age_L{lag}',f'wasde_missing_L{lag}']
        groups[f'missing_L{lag}']=[*FEATURE_NAMES,*control]
        groups[f'balance_L{lag}']=[*FEATURE_NAMES,*control,*[f'wasde_{f}_L{lag}' for f in FIELDS]]
    return groups


def prepare(repo,folder,reference,table):
    rows,source=read_balance(table)
    information.prepare_information(repo,folder,Path(reference),profile=PROFILE,source=source,
        add_features=lambda history:add_balance(history,rows),first_source_day=rows.assumed_day.min(),
        groups=group_names(),controls={f'balance_L{l}':f'missing_L{l}' for l in LAGS},prefix='wasde-balance',
        hypothesis='World cotton balance ratios/revisions add T+5 information beyond age/missingness',
        policy={'assumed_day':'dated archive page, not verified first UTC publication','cotton_lags':list(LAGS),
                'max_extra_age':40,'publication_clock_verified':False},priority_fraction=5/8)


def dispatch(args):
    def prepare_source(folder):
        if not all((args.reference_root,args.wasde_table)):raise ValueError('Pinned parent and WASDE table required')
        prepare(args.repo,folder,args.reference_root,args.wasde_table)
    information.dispatch_information(args,prepare_source)
