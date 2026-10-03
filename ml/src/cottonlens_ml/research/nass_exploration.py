"""Season-bounded national cotton condition control; unverified vintage = Tier A."""
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.ledger import read_record

PROFILE='nass-exploration-v1'
LAGS=information.LAGS
PERCENT_COLUMNS=['very_poor_pct','poor_pct','fair_pct','good_pct','excellent_pct']


def read_condition(table, audits):
    table,audits=Path(table),Path(audits)
    manifest=read_record(table.with_suffix('.manifest.json'))
    if digest(table)!=manifest['table_sha256'] or manifest['model_eligible'] or manifest['publication_clock_verified']:
        raise ValueError('Pinned diagnostic NASS table and Tier-A policy required')
    rows=pd.read_csv(table,parse_dates=['week_ending','release_page_date'])
    if (len(rows)!=manifest['row_count'] or rows.week_ending.duplicated().any()
            or rows[['week_ending','release_page_date']].isna().any().any()):
        raise ValueError('Unique condition weeks and frozen count required')
    values=rows[PERCENT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>100)).any() or not np.all(values.sum(axis=1)==100):
        raise ValueError('Five bounded percentages must sum to100')
    if rows.release_page_date.lt(rows.week_ending).any() or rows.week_ending.max()>=pd.Timestamp('2024-01-01'):
        raise ValueError('Past source date and post-week assumed publication day required')
    seen=set()
    for year,sha in manifest['source_year_audit_sha256'].items():
        p=audits/f'year-audit-{year}.json'
        if digest(p)!=sha:raise ValueError('NASS annual content audit changed')
        annual=read_record(p)
        if annual['year']!=int(year):raise ValueError('Audit year mismatch')
        for result in annual['results']:
            day=pd.Timestamp(result['week_ending'])
            if day in seen:raise ValueError('Duplicate audited source week')
            seen.add(day)
            matching_rows=rows.loc[rows.week_ending==day]
            if len(matching_rows)!=1:raise ValueError('CSV/audit coverage differs')
            row=matching_rows.iloc[0]
            if (result['status'] not in ('numeric_match','reference_missing_zero')
                    or row.quickstats_check!=result['status'] or row.report_sha256!=result['report_sha256']
                    or row.release_page_url!=result['release_page']
                    or row.release_page_date!=pd.Timestamp(result['release_page'].rsplit('/',1)[1])):
                raise ValueError('Condition/content/page identity differs')
    if seen!=set(rows.week_ending):raise ValueError('Unreviewed NASS week')
    rows=rows.sort_values('week_ending').reset_index(drop=True)
    rows['good_excellent']=(rows.good_pct+rows.excellent_pct)/100
    rows['poor_very_poor']=(rows.poor_pct+rows.very_poor_pct)/100
    contiguous=rows.week_ending.diff().dt.days.eq(7)&rows.week_ending.dt.year.eq(rows.week_ending.shift().dt.year)
    rows['good_excellent_change']=rows.good_excellent.diff().where(contiguous)
    rows['assumed_day']=rows.release_page_date
    source={'table_sha256':digest(table),'manifest_sha256':digest(table.with_suffix('.manifest.json')),
        'annual_audit_sha256':manifest['source_year_audit_sha256'],'rows':len(rows),
        'content_counts':manifest['content_check_counts'],'unit':'fraction from published percentage',
        'scope':'US national Upland Cotton condition; seasonal observations',
        'source_url':'https://www.nass.usda.gov/Publications/Todays_Reports/',
        'model_eligible':False,'historical_publication_verified':False,'first_version_verified':False,
        'source_tier':'A_exploration_only','release_allowed':False,'cost_tl':0,'redistribution_reviewed':False}
    return rows,source


def add_condition(history, condition):
    if (history.date.duplicated().any() or not history.date.is_monotonic_increasing
            or history.date.isna().any() or condition.week_ending.duplicated().any()
            or condition[['week_ending','assumed_day']].isna().any().any()):
        raise ValueError('Unique sorted source and Cotton observations required')
    result=history.copy();calendar=history.date.to_numpy(dtype='datetime64[ns]')
    days=condition.week_ending.to_numpy(dtype='datetime64[ns]')
    after=np.searchsorted(calendar,condition.assumed_day.to_numpy(dtype='datetime64[ns]'),side='right')
    reported=np.searchsorted(calendar,days,side='right')-1
    values=condition[['good_excellent','poor_very_poor','good_excellent_change']].to_numpy(dtype=float)
    for lag in LAGS:
        events=sorted(zip(after+lag-1,days,reported,values,strict=True),key=lambda x:(x[0],x[1]))
        aligned=np.full((len(history),3),np.nan);age=np.full(len(history),np.nan)
        selected=None;cursor=0
        for index in range(len(history)):
            while cursor<len(events) and events[cursor][0]<=index:
                if selected is None or events[cursor][1]>selected[1]:selected=events[cursor]
                cursor+=1
            if selected is not None:
                age[index]=index-selected[2]
                # Bound by publication age AND same crop year: no winter carry.
                if index-selected[0]<=10 and history.date.iloc[index].year==pd.Timestamp(selected[1]).year:
                    aligned[index]=selected[3]
        for j,name in enumerate(('good_excellent','poor_very_poor','good_excellent_change')):
            result[f'nass_{name}_L{lag}']=aligned[:,j]
        result[f'nass_age_L{lag}']=age
        result[f'nass_missing_L{lag}']=(~np.isfinite(aligned)).any(axis=1).astype(float)
    return result


def group_names():
    groups={'base':list(FEATURE_NAMES)}
    for lag in LAGS:
        control=[f'nass_age_L{lag}',f'nass_missing_L{lag}']
        groups[f'missing_L{lag}']=[*FEATURE_NAMES,*control]
        groups[f'condition_L{lag}']=[*FEATURE_NAMES,*control,*[f'nass_{f}_L{lag}' for f in
            ('good_excellent','poor_very_poor','good_excellent_change')]]
    return groups


def prepare(repo,folder,reference,table,audits):
    condition,source=read_condition(table,audits)
    information.prepare_information(repo,folder,Path(reference),profile=PROFILE,source=source,
        add_features=lambda history:add_condition(history,condition),first_source_day=condition.assumed_day.min(),
        groups=group_names(),controls={f'condition_L{lag}':f'missing_L{lag}' for lag in LAGS},prefix='nass-condition',
        hypothesis='National Cotton condition adds T+5 information beyond seasonality/age/missingness',
        policy={'assumed_day':'dated release page, not verified UTC availability','cotton_lags':list(LAGS),
            'max_extra_age':10,'winter_carry':False,'published_at_verified':False})


def dispatch(args):
    def prepare_source(folder):
        if not all((args.reference_root,args.nass_table,args.nass_audit_root)):
            raise ValueError('Pinned parent, NASS table and annual audit root required')
        prepare(args.repo,folder,args.reference_root,args.nass_table,args.nass_audit_root)
    information.dispatch_information(args,prepare_source)
