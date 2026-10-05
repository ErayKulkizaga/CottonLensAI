"""Bounded Tier-A quotation-spread control, on Experiment/Ledger's CPU fit path.

Displayed publication dates are assumptions, not certified release clocks. This
profile deliberately cannot lock/export or claim any price acceptance gate.
"""
import importlib.metadata
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.full_year import (
    chunks,
    inner_price,
    json_predictions,
    predict_chunks,
)
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import eligible, mature
from cottonlens_ml.research.statistics import paired_bootstrap

PROFILE = 'ams-exploration-v1'
LAGS = (1, 2, 6)  # baseline one observation after assumed day; additional +1/+5.


def read_quotes(table, publications):
    table, publications = Path(table), Path(publications)
    manifest = read_record(table.with_suffix('.manifest.json'))
    review = read_record(publications)
    if digest(table) != manifest['table_sha256'] or review['table_sha256'] != manifest['table_sha256']:
        raise ValueError('AMS diagnostic table identity mismatch')
    if manifest['model_eligible'] or review['model_eligible']:
        raise ValueError('This exploratory route cannot replace an admitted source compiler')
    rows = pd.read_csv(table, parse_dates=['report_date'])
    if len(rows) != manifest['row_count'] or rows.report_date.duplicated().any():
        raise ValueError('Unique AMS quotation dates required')
    value = rows.spot_41_4_34_cents_per_lb
    if not np.isfinite(value).all() or (value <= 0).any():
        raise ValueError('Positive cents/lb quotation required')
    clocks = pd.DataFrame(review['version_records'])
    clocks['report_date'] = pd.to_datetime(clocks.report_date)
    if clocks.report_date.duplicated().any() or not clocks.matches_esmis_bytes.eq(True).all():
        raise ValueError('Ambiguous or unmatched AMS publication content')
    joined = rows.merge(clocks, on='report_date', how='inner', validate='one_to_one')
    if not joined.report_sha256.eq(joined.document_sha256).all():
        raise ValueError('AMS quotation/publication bytes differ')
    # Only a calendar day assumption. Do not fabricate UTC or published_at.
    displayed = pd.to_datetime(joined.published_local_naive)
    if displayed.dt.tz is not None or displayed.isna().any():
        raise ValueError('Unverified displayed calendar dates required')
    joined['assumed_day'] = pd.concat([joined.report_date, displayed.dt.normalize()], axis=1).max(axis=1)
    evidence = {'table_sha256': digest(table), 'manifest_sha256': digest(table.with_suffix('.manifest.json')),
                'publication_review_sha256': digest(publications), 'raw_rows': len(rows),
                'byte_matched_rows': len(joined),
                'excluded_no_publication_row': rows.loc[~rows.report_date.isin(joined.report_date), 'report_date'].dt.strftime('%Y-%m-%d').tolist(),
                'historical_publication_verified': False, 'first_version_verified': False,
                'unit': 'cents/lb', 'quality': '41-4/34 seven-market quotation, not transaction VWAP',
                'source_url': 'https://esmis.nal.usda.gov/publication/daily-spot-quotations-excerpts',
                'tier': 'A_exploration_only', 'cost_tl': 0,
                'release_allowed': False, 'redistribution_reviewed': False}
    return joined.sort_values('report_date').reset_index(drop=True), evidence


def add_quotes(history, quotes):
    """As-of calendar assumptions; no bfill, scaling, outcome use, or row removal."""
    result = history.copy()
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError('Unique sorted Cotton dates required')
    calendar = history.date.to_numpy(dtype='datetime64[ns]')
    report_dates = quotes.report_date.to_numpy(dtype='datetime64[ns]')
    after_day = np.searchsorted(calendar, quotes.assumed_day.to_numpy(dtype='datetime64[ns]'), side='right')
    report_indexes = np.searchsorted(calendar, report_dates, side='right')-1
    for lag in LAGS:
        events = sorted(zip(after_day+lag-1, report_dates, report_indexes,
                            quotes.spot_41_4_34_cents_per_lb.to_numpy(), strict=True))
        spot, age = np.full(len(history), np.nan), np.full(len(history), np.nan)
        cursor, latest, selected = 0, None, None
        for index in range(len(history)):
            while cursor < len(events) and events[cursor][0] <= index:
                event = events[cursor]
                # A late record for an older report must not replace newer information.
                if latest is None or event[1] > latest:
                    latest, selected = event[1], event
                cursor += 1
            if selected is not None:
                age[index] = index-selected[2]
                # Three extra recorded observations after assumed availability.
                if index-selected[0] <= 3:
                    spot[index] = selected[3]
        result[f'ams_spread_L{lag}'] = spot/result.cotton_close.to_numpy()-1
        result[f'ams_age_L{lag}'] = age
        result[f'ams_missing_L{lag}'] = (~np.isfinite(spot)).astype(float)
    return result


def short_split(history, start, *, profile=PROFILE, first_year=2020):
    source_history = history.loc[history.date >= start]
    rows = eligible(source_history)
    rows = rows.loc[(rows.date < '2024-01-01') & (rows.target_date_5 < '2024-01-01')]
    folds, excluded = [], []
    for year in range(first_year, 2024):
        outer = rows.loc[rows.date.dt.year == year]
        if outer.empty:
            excluded.append({'year': year, 'reason': 'no_common_origin'})
            continue
        preceding = mature(history, outer.date.min()).loc[lambda f: f.date >= start].tail(189)
        inner = []
        for i in range(3):
            block = preceding.iloc[63*i:63*(i+1)]
            if len(block) != 63:
                break
            train = mature(history, block.date.min()).loc[lambda f: f.date >= start]
            if len(train) < 500:
                break
            inner.append({'origins': block.date.dt.strftime('%Y-%m-%d').tolist(), 'cutoff': block.date.min().isoformat()})
        if len(inner) != 3:
            excluded.append({'year': year, 'reason': 'three_63_inner_blocks_with_500_past_source_rows_required'})
            continue
        folds.append({'fold': len(folds)+1, 'year': year, 'inner': inner,
                      'origins': outer.date.dt.strftime('%Y-%m-%d').tolist()})
    if not folds:
        raise ValueError('No chronological source cohort with sufficient prior history')
    body = {'version': profile, 'folds': folds, 'coverage_start': str(start.date()), 'excluded_years': excluded,
            'purge_observations': 5, 'refit_cadence': 21, 'audit_used': False,
            'role': 'short_reused_historical_exploration', 'gate_evaluation_allowed': False}
    return {**body, 'split_id': content_id(body)}


def group_names():
    groups = {'base': list(FEATURE_NAMES)}
    for lag in LAGS:
        control = [f'ams_age_L{lag}', f'ams_missing_L{lag}']
        groups[f'missing_L{lag}'] = [*FEATURE_NAMES, *control]
        groups[f'quote_L{lag}'] = [*FEATURE_NAMES, *control, f'ams_spread_L{lag}']
    return groups


def specs(names):
    from cottonlens_ml.research.full_year import price_recipes
    recipes = price_recipes(5)
    return [{**r, 'features': names} for r in (recipes[1], recipes[-1])]


def prepare(repo, folder, reference, table, publications):
    parent = read_record(reference/'ready.json')
    if digest(reference/'history.parquet') != parent['history_sha256']:
        raise ValueError('Frozen parent history changed')
    history = pd.read_parquet(reference/'history.parquet')
    if history.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('Seen audit cannot enter exploration')
    quotes, source = read_quotes(table, publications)
    history = add_quotes(history, quotes)
    start_index = np.searchsorted(history.date.to_numpy(), quotes.assumed_day.min().to_datetime64(), side='right')+5
    split = short_split(history, history.date.iloc[start_index])
    identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'],
        'reference_ready_sha256': digest(reference/'ready.json'), 'source_evidence': source,
        'research_data_id': frame_identity(history, list(history)), 'split': split, 'groups': group_names(),
        'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in
        ('numpy','pandas','scipy','scikit-learn','xgboost')}, 'lock_sha256': digest(repo/'ml/uv.lock'),
        'execution': {'device': 'cpu', 'threads': 2}, 'publication_sources': [],
        'availability_policy': {'assumed_day': 'max(report date, unverified displayed publication day)',
        'cotton_lags': list(LAGS), 'max_extra_age': 3, 'published_at_verified': False},
        'release_allowed': False, 'gate_evaluation_allowed': False}
    with writer(folder):
        if (folder/'ready.json').exists():
            old = read_record(folder/'ready.json')
            if old['identity'] != identity or digest(folder/'history.parquet') != old['history_sha256']:
                raise ValueError('Frozen exploration changed; use a new namespace')
            return
        if (folder/'history.parquet').exists():
            raise ValueError('Incomplete prepare preserved; inspect before continuing')
        history.to_parquet(folder/'history.parquet', index=False)
        freeze_record(folder/'ready.json', {'identity': identity, 'history_sha256': digest(folder/'history.parquet')})
        freeze_record(folder/'preregistered.json', {'created_at': datetime.now(UTC).isoformat(),
            'hypothesis': 'AMS quoted spot/proxy spread adds T+5 information beyond timing/missingness',
            'lags': list(LAGS), 'models': ['Ridge alpha1','XGBoost depth2 fixed recipe'], 'seeds': [42],
            'selection': 'past three63 inner blocks, fixed shrinkage grid; no outer selection',
            'source_tier': 'A', 'release_allowed': False, 'no_eight_fold_gate': True,
            'baseline_primary': 'Naive', 'maximum_fit_jobs': budget(identity, history)})


def budget(identity, history):
    total = 0
    for fold in identity['split']['folds']:
        inner_chunks = sum(len(chunks(history,b['origins'])) for b in fold['inner'])
        total += len(identity['groups'])*(2*inner_chunks+3+2*len(chunks(history,fold['origins'])))
    return total


def prepare_information(repo, folder, reference, *, profile, source, add_features,
                        first_source_day, groups, controls, prefix, hypothesis, policy,
                        priority_fraction=None):
    """Freeze a new Tier-A source on the existing information fit/evaluation path."""
    parent=read_record(reference/'ready.json')
    if digest(reference/'history.parquet')!=parent['history_sha256']:
        raise ValueError('Pinned parent changed')
    history=pd.read_parquet(reference/'history.parquet')
    if history.date.max()>=pd.Timestamp('2024-01-01'):
        raise ValueError('Seen audit cannot enter exploration')
    history=add_features(history)
    start=max(history.date.min(),pd.Timestamp(first_source_day)+pd.Timedelta(days=1))
    split=short_split(history,start,profile=profile,first_year=2016)
    if priority_fraction is not None and not 0 < priority_fraction <= 1:
        raise ValueError('Research priority fraction must be in (0,1]')
    wins=5 if priority_fraction is None else math.ceil(len(split['folds'])*priority_fraction)
    priority={'inner_gain_vs_base_and_matching_control_pct':.5,
        'inner_wins_vs_base_and_matching_control':wins,'outer_positive_gain':True,'outer_fold_wins':wins,
        'all_lag_stresses_required':True,'role':'research budget, not release gate',
        'cohort_folds':len(split['folds']),'minimum_fraction':priority_fraction}
    identity={'profile':profile,'source_id':research_source_identity(repo)['source_id'],
        'reference_ready_sha256':digest(reference/'ready.json'),'source_evidence':source,
        'research_data_id':frame_identity(history,list(history)),'split':split,'groups':groups,
        'comparison_controls':controls,'report_prefix':prefix,'include_reference_diagnostics':True,
        'bootstrap_lengths':[10,20,40],'python':sys.version.split()[0],
        'versions':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','xgboost')},
        'lock_sha256':digest(repo/'ml/uv.lock'),'execution':{'device':'cpu','threads':2},
        'publication_sources':[],'availability_policy':policy,'research_priority_rule':priority,
        'release_allowed':False,'gate_evaluation_allowed':False}
    with writer(folder):
        if (folder/'ready.json').exists():
            old=read_record(folder/'ready.json')
            if old['identity']!=identity or digest(folder/'history.parquet')!=old['history_sha256']:
                raise ValueError('Frozen information experiment changed; use new namespace')
            return
        if (folder/'history.parquet').exists():raise ValueError('Incomplete prepare preserved')
        history.to_parquet(folder/'history.parquet',index=False)
        freeze_record(folder/'ready.json',{'identity':identity,'history_sha256':digest(folder/'history.parquet')})
        freeze_record(folder/'preregistered.json',{'created_at':datetime.now(UTC).isoformat(),
            'hypothesis':hypothesis,'lags':list(LAGS),'models':['Ridge alpha1','XGBoost depth2 fixed recipe'],
            'seed':42,'selection':'three63 past inner blocks; shrinkage grid; no outer tuning',
            'source_tier':'A_exploration_only','release_allowed':False,'gate_evaluated':False,
            'priority_rule':priority,
            'maximum_fit_jobs':budget(identity,history)})


def dispatch_information(args, prepare_source):
    """Shared orchestration, same Experiment/Ledger, no parallel trainer."""
    from cottonlens_ml.research.engine import Experiment, root_path
    folder=root_path(args.drive_root,args.experiment)
    if args.stage not in {'prepare','status','pilot-plan','pilot','compare','report'}:
        raise ValueError('Tier-A source cannot lock/export or enter wider search')
    if args.mirror_root:
        from cottonlens_ml.research.mirror import Mirror
        metadata_only=args.stage in ('status','pilot-plan','compare','report')
        ready=read_record(folder/'ready.json') if (folder/'ready.json').exists() else None
        expected=len(ready['identity']['groups'])*len(ready['identity']['split']['folds']) if ready else None
        if not (ready and metadata_only and len(list((folder/'information-outputs').glob('*.json')))>=expected):
            Mirror(folder,root_path(args.mirror_root,args.experiment)).hydrate(metadata_only=metadata_only)
    if args.stage=='prepare':prepare_source(folder)
    elif args.stage=='status':
        print(json.dumps({'prepared':(folder/'ready.json').exists(),
            'saved_outputs':len(list((folder/'information-outputs').glob('*.json'))),
            'ledger':Ledger(folder/'ledger',{}).summary(),'source_tier':'A','release_allowed':False}));return
    elif args.stage in ('compare','report'):print(json.dumps(compare(folder),indent=2))
    else:
        experiment=Experiment(folder,repo=args.repo)
        if args.stage=='pilot-plan':
            print(json.dumps({'maximum_fit_jobs':budget(experiment.identity,experiment.history),
                'folds':len(experiment.identity['split']['folds']),'groups':list(experiment.identity['groups']),
                'origins':sum(len(f['origins']) for f in experiment.identity['split']['folds']),'release_allowed':False}));return
        with writer(folder):
            if args.mirror_root:
                experiment.mirror=Mirror(folder,root_path(args.mirror_root,args.experiment))
                experiment.after_fit=experiment.mirror.fit
            print(json.dumps(run(experiment,args.max_minutes)))
    if args.mirror_root:
        mirror=Mirror(folder,root_path(args.mirror_root,args.experiment))
        names=['ready.json','history.parquet','preregistered.json']+[p.relative_to(folder).as_posix() for p in (folder/'reports').glob('*.json')]
        mirror.enqueue(names);print(json.dumps(mirror.flush()))


def run(experiment, max_minutes):
    if not 0 < max_minutes <= 240:
        raise ValueError('CPU session must be within four hours')
    deadline = time.monotonic()+max_minutes*60
    def before():
        if time.monotonic() >= deadline:
            raise FitBudgetReached('Planned pause; unchanged identity resumes saved fits')
    experiment.before_compute = before
    try:
        for fold in experiment.identity['split']['folds']:
            for group, names in experiment.identity['groups'].items():
                path = experiment.root/'information-outputs'/f'{group}-{fold["year"]}.json'
                if path.exists():
                    read_record(path)
                    print(f'CACHE {group} year={fold["year"]}', flush=True)
                    continue
                decision = experiment.root/'information-decisions'/path.name
                if decision.exists():
                    chosen = read_record(decision)
                else:
                    recipes = experiment.identity.get('fixed_recipes', {}).get(group)
                    choices = [inner_price(experiment, r, fold) for r in (recipes if recipes is not None else specs(names))]
                    # Strictly past inner outcomes, then simpler family/weight.
                    best = min(enumerate(choices), key=lambda x:(x[1]['inner_score'],x[0],x[1]['weight']))[1]
                    chosen = {'selected':best,'all_candidates':choices,'selection_used_outer':False}
                    freeze_record(decision,chosen)
                best = chosen['selected']
                test = experiment.history.loc[experiment.history.date.isin(pd.to_datetime(fold['origins']))].copy()
                role=f'{experiment.identity.get("profile","information")}-{group}-{fold["year"]}'
                test['raw_predicted_return'] = predict_chunks(experiment,best['recipe'],fold['origins'],role,best['iterations'])
                test['predicted_return'] = best['weight']*test.raw_predicted_return
                references=[]
                if experiment.identity.get('include_reference_diagnostics'):
                    references=['median_return','past_majority_sign']
                    test['median_return'],test['past_majority_sign']=np.nan,np.nan
                    for block in chunks(experiment.history,fold['origins']):
                        past=experiment.train_rows(block.date.min(),best['recipe'])
                        signs,counts=np.unique(np.sign(past.target_return_5),return_counts=True)
                        mask=test.date.isin(block.date)
                        test.loc[mask,'median_return']=float(past.target_return_5.median())
                        test.loc[mask,'past_majority_sign']=float(signs[np.argmax(counts)])
                test['date'] = test.date.dt.strftime('%Y-%m-%d')
                freeze_record(path,{'year':fold['year'],'group':group,'decision_sha256':digest(decision),
                    'records':json_predictions(test[['date','cotton_close','target_return_5','predicted_return','raw_predicted_return',*references]]),
                    'inner_score':best['inner_score'],'recipe':best['recipe'],'weight':best['weight'],
                    'tier':'A_exploration_only','gate_evaluated':False})
                if getattr(experiment,'mirror',None):
                    experiment.mirror.enqueue([path.relative_to(experiment.root).as_posix(),decision.relative_to(experiment.root).as_posix()])
                    experiment.mirror.flush()
                print(f'SAVED INFORMATION {group} year={fold["year"]}',flush=True)
    except FitBudgetReached:
        return {'status':'planned_pause','saved_outputs':len(list((experiment.root/'information-outputs').glob('*.json')))}
    return {'status':'complete','gate_evaluated':False,'release_allowed':False}


def compare(folder):
    ready = read_record(folder/'ready.json')
    identity = ready['identity']
    outputs = [read_record(p) for p in sorted((folder/'information-outputs').glob('*.json'))]
    expected = {(g,f['year']) for g in identity['groups'] for f in identity['split']['folds']}
    if {(r['group'],r['year']) for r in outputs} != expected:
        return {'status':'pending','complete_outputs':len(outputs),'required_outputs':len(expected)}
    groups, matrices = {}, {}
    for group in identity['groups']:
        parts, scores, reference_rows = [], [], []
        for fold in identity['split']['folds']:
            r = next(r for r in outputs if r['group']==group and r['year']==fold['year'])
            frame = pd.DataFrame(r['records'])
            if frame.date.tolist()!=fold['origins']:
                raise ValueError('Frozen common origins changed')
            price, a, p = (frame[c].to_numpy() for c in ('cotton_close','target_return_5','predicted_return'))
            naive, error = price*np.abs(np.expm1(a)), price*np.abs(np.exp(a)-np.exp(p))
            parts.append(np.column_stack([naive,error]));scores.append(r['inner_score'])
            reference_rows.append(frame)
        matrices[group] = parts
        values = np.concatenate(parts)
        groups[group] = {'naive_mae_gain_pct':float(100*(1-values[:,1].mean()/values[:,0].mean())),
            'inner_score':float(np.mean(scores)),'price_mae':float(values[:,1].mean()),
            'versus_naive':paired_bootstrap(parts,repetitions=10000,normalization=float(values[:,0].mean()))}
        if identity.get('include_reference_diagnostics'):
            frame=pd.concat(reference_rows,ignore_index=True)
            c,a,p=(frame[n].to_numpy() for n in ('cotton_close','target_return_5','predicted_return'))
            groups[group]['median_price_mae']=float(np.mean(c*np.abs(np.exp(a)-np.exp(frame.median_return))))
            groups[group]['direction_pct']=float(100*np.mean(np.sign(a)==np.sign(p)))
            groups[group]['majority_direction_pct']=float(100*np.mean(np.sign(a)==frame.past_majority_sign))
            groups[group]['flat_count']=int(np.sum(p==0))
            groups[group]['fold_wins']=sum(float(b[:,1].mean())<float(b[:,0].mean()) for b in parts)
            directions=[np.column_stack([np.sign(f.target_return_5)==np.sign(f.predicted_return),
                np.sign(f.target_return_5)==f.past_majority_sign]).astype(float) for f in reference_rows]
            groups[group]['direction_vs_majority']={str(b):paired_bootstrap(directions,block=b,repetitions=10000)
                for b in identity['bootstrap_lengths']}
            groups[group]['versus_naive_sensitivity']={str(b):paired_bootstrap(parts,block=b,repetitions=10000,
                normalization=float(values[:,0].mean())) for b in identity['bootstrap_lengths'] if b!=20}
    controls = identity.get('comparison_controls', {f'quote_L{lag}':f'missing_L{lag}' for lag in LAGS})
    for quote, control in controls.items():
        base = identity.get('comparison_bases', {}).get(quote, 'base')
        for benchmark in (base,control):
            parts = [np.column_stack([b[:,1],q[:,1]]) for b,q in zip(matrices[benchmark],matrices[quote],strict=True)]
            groups[quote]['vs_'+benchmark] = paired_bootstrap(parts,repetitions=10000,normalization=groups[benchmark]['price_mae'])
            if identity.get('bootstrap_lengths'):
                groups[quote]['vs_'+benchmark+'_sensitivity']={str(b):paired_bootstrap(parts,block=b,repetitions=10000,
                    normalization=groups[benchmark]['price_mae']) for b in identity['bootstrap_lengths'] if b!=20}
    body={'status':'complete','evidence':'Tier A, unverified vintage/timing; historical exploratory only',
        'ready_sha256':digest(folder/'ready.json'),'output_sha256':{p.name:digest(p) for p in sorted((folder/'information-outputs').glob('*.json'))},
        'groups':groups,'folds':len(identity['split']['folds']),'origin_count':sum(len(f['origins']) for f in identity['split']['folds']),
        'lag_stresses_complete':True,'gate_evaluated':False,'release_allowed':False,'primary':'Naive',
        'next_action':'Inspect source contribution versus matching missingness/age control and all lag stresses; no automatic search expansion'}
    prefix = identity.get('report_prefix','ams-quotation')
    freeze_record(folder/'reports'/(prefix+'-'+content_id(body)[:16]+'.json'),body)
    return body


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder=root_path(args.drive_root,args.experiment)
    allowed={'prepare','status','pilot-plan','pilot','compare','report'}
    if args.stage not in allowed:
        raise ValueError('Tier-A source cannot lock/export/enter a release or wider search')
    if args.mirror_root:
        from cottonlens_ml.research.mirror import Mirror
        metadata_only = args.stage in ('status','pilot-plan','compare','report')
        expected = len(group_names())
        if not ((folder/'ready.json').exists() and metadata_only
                and len(list((folder/'information-outputs').glob('*.json'))) >= expected):
            Mirror(folder,root_path(args.mirror_root,args.experiment)).hydrate(metadata_only=metadata_only)
    if args.stage=='prepare':
        if not all((args.reference_root,args.ams_table,args.ams_publications)):
            raise ValueError('Frozen parent, AMS table and checksum-bound publication review required')
        prepare(args.repo,folder,args.reference_root,args.ams_table,args.ams_publications)
    elif args.stage=='status':
        print(json.dumps({'prepared':(folder/'ready.json').exists(),'saved_outputs':len(list((folder/'information-outputs').glob('*.json'))),
            'ledger':Ledger(folder/'ledger',{}).summary(),'source_tier':'A','release_allowed':False}))
        return
    elif args.stage in ('compare','report'):
        print(json.dumps(compare(folder),indent=2))
    else:
        experiment=Experiment(folder,repo=args.repo)
        if args.stage=='pilot-plan':
            print(json.dumps({'maximum_fit_jobs':budget(experiment.identity,experiment.history),
                'split':experiment.identity['split'],'groups':list(experiment.identity['groups']),'release_allowed':False}))
            return
        with writer(folder):
            if args.mirror_root:
                from cottonlens_ml.research.mirror import Mirror
                experiment.mirror=Mirror(folder,root_path(args.mirror_root,args.experiment))
                experiment.after_fit=experiment.mirror.fit
            print(json.dumps(run(experiment,args.max_minutes)))
    if args.mirror_root:
        from cottonlens_ml.research.mirror import Mirror
        mirror=Mirror(folder,root_path(args.mirror_root,args.experiment))
        names=['ready.json','history.parquet','preregistered.json']
        names += [p.relative_to(folder).as_posix() for p in (folder/'reports').glob('*.json')]
        mirror.enqueue(names);print(json.dumps(mirror.flush()))
