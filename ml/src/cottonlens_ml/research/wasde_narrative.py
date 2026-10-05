"""One fixed T+5 incremental-text hypothesis on the preserved WASDE cohort."""
import copy
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.full_year import SHRINKAGE, chunks, price_recipes
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.protocol import mature, rows_at
from cottonlens_ml.sources.wasde_text import align_corpus

PROFILE = 'wasde-text-pilot-v1'
LAGS = (1, 2, 6)


def groups_and_recipes(parent):
    groups, recipes = {}, {}
    for lag in LAGS:
        timing = [f'wasde_text_age_L{lag}', f'wasde_text_missing_L{lag}']
        numeric = list(parent['identity']['groups'][f'balance_L{lag}'])+timing
        latent = [f'wasde_text_svd{i}_L{lag}' for i in range(8)]
        for arm, names in [('base', [*FEATURE_NAMES, *timing]), ('numeric', numeric), ('text', numeric+latent)]:
            name = f'{arm}_L{lag}'
            spec = {**price_recipes(5)[1], 'features': names}
            if arm == 'text':
                spec['text_adapter'] = {'column': f'wasde_narrative_L{lag}',
                    'id_column': f'wasde_narrative_id_L{lag}', 'latent_names': latent,
                    'max_features': 128, 'min_df': 2, 'dimensions': 8, 'minimum_documents': 12, 'seed': 42}
            groups[name], recipes[name] = names, [spec]
    return groups, recipes


def prepare(repo, folder, reference, corpus):
    repo, folder, reference, corpus = map(Path, (repo, folder, reference, corpus))
    parent = read_record(reference/'ready.json')
    if (parent['identity']['profile'] != 'wasde-exploration-v1'
            or parent['identity']['groups']['base'] != list(FEATURE_NAMES)
            or digest(reference/'history.parquet') != parent['history_sha256']):
        raise ValueError('Original frozen WASDE cohort/snapshot required')
    history = pd.read_parquet(reference/'history.parquet')
    if history.date.max() >= pd.Timestamp('2024-01-01'):
        raise ValueError('Seen audit cannot enter text research')
    split = copy.deepcopy(parent['identity']['split'])
    if ([fold['year'] for fold in split['folds']] != list(range(2019, 2024))
            or sum(len(fold['origins']) for fold in split['folds']) != 1254):
        raise ValueError('Preserve the exact 1254-origin five-year source cohort')
    for lag in LAGS:
        aligned = align_corpus(history, corpus, lag=lag)
        for column in ('wasde_narrative','wasde_narrative_id','wasde_text_age','wasde_text_missing'):
            history[f'{column}_L{lag}'] = aligned[column]
    for h in (1, 5):
        mask = history[f'target_date_{h}'].ge(pd.Timestamp('2024-01-01'))
        history.loc[mask, f'target_return_{h}'] = np.nan
        history.loc[mask, f'target_date_{h}'] = pd.NaT
    groups, recipes = groups_and_recipes(parent)
    # Availability preflight only: no vocabulary, SVD, scaler, or model is fitted.
    counts = []
    for fold in split['folds']:
        for origins in [*[b['origins'] for b in fold['inner']], fold['origins']]:
            for chunk in chunks(history, origins):
                past = mature(history, chunk.date.min())
                start = max(pd.Timestamp(split['coverage_start']), history.date.iloc[119])
                past = past.loc[past.date >= start]
                for lag in LAGS:
                    count = past[f'wasde_narrative_id_L{lag}'].nunique()
                    if count < 12:
                        raise ValueError('Insufficient mature training narratives; pilot blocked')
                    counts.append(int(count))
    inner = sum(sum(len(chunks(history, b['origins'])) for b in f['inner']) for f in split['folds'])
    outer = sum(len(chunks(history, f['origins'])) for f in split['folds'])
    budget = {'annual_outputs':45,'inner_refits':inner*9,'outer_refits':outer*9,'total_refits':(inner+outer)*9}
    if budget != {'annual_outputs':45,'inner_refits':405,'outer_refits':549,'total_refits':954}:
        raise ValueError('Frozen job budget changed')
    identity = {'profile':PROFILE,'source_id':research_source_identity(repo)['source_id'],
        'reference_ready_sha256':digest(reference/'ready.json'),'corpus_sha256':digest(corpus),
        'research_data_id':frame_identity(history,list(history.columns)), 'split':split,
        'groups':groups,'fixed_recipes':recipes,'fit_budget':budget,
        'comparison_controls':{f'text_L{lag}':f'numeric_L{lag}' for lag in LAGS},
        'comparison_bases':{f'text_L{lag}':f'base_L{lag}' for lag in LAGS},
        'report_prefix':'wasde-text','include_reference_diagnostics':True,'bootstrap_lengths':[10,20,40],
        'python':sys.version.split()[0],
        'versions':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','xgboost')},
        'lock_sha256':digest(repo/'ml/uv.lock'),'execution':{'device':'cpu','threads':2,'parallel_jobs':1},
        'source_tier':'A_exploration_only','release_allowed':False,'gate_evaluation_allowed':False,
        'availability_policy':'Unverified report-date assumption; lag1 and additional1/5 recorded observations; max age45',
        'text_preflight':{'minimum_unique_mature_documents':min(counts),'learned_transform_fitted':False},
        'priority_rule':{'inner_gain_vs_both_controls_pct':.5,'inner_year_wins':4,
            'positive_outer_gain_vs_both_controls':True,'outer_year_wins':4,'all_lag_stresses_required':True}}
    with writer(folder):
        if (folder/'ready.json').exists():
            prior = read_record(folder/'ready.json')
            if prior['identity'] != identity or digest(folder/'history.parquet') != prior['history_sha256']:
                raise ValueError('Frozen text source/data/environment changed; use a new namespace')
            return prior
        if (folder/'history.parquet').exists():
            raise ValueError('Incomplete prepare preserved; inspect before continuing')
        history.to_parquet(folder/'history.parquet',index=False)
        ready = {'identity':identity,'history_sha256':digest(folder/'history.parquet')}
        freeze_record(folder/'ready.json',ready)
        freeze_record(folder/'preregistered.json',{'hypothesis':'T+5 cotton narrative adds information beyond timing and numeric balance',
            'primary_comparison':'text_L1 vs numeric_L1; base and lag2/6 are mandatory controls/stresses',
            'recipes':recipes,'budget':budget,'shrinkage':list(SHRINKAGE),'corpus_sha256':digest(corpus),
            'source_tier':'A_exploration_only','release_allowed':False,'gate_evaluated':False,
            'checkpoint_date':'2026-10-22T20:06:59.863911+00:00','price_gate_changed':False})
    return ready


def verify_cached(folder):
    ready = read_record(folder/'ready.json')
    if digest(folder/'history.parquet') != ready['history_sha256']:
        raise ValueError('Text snapshot corrupted')
    identity, history = ready['identity'], pd.read_parquet(folder/'history.parquet')
    parent = {'identity':{'groups':{f'balance_L{lag}':identity['groups'][f'numeric_L{lag}'][:-2] for lag in LAGS}}}
    groups, recipes = groups_and_recipes(parent)
    if (identity['profile'] != PROFILE or identity['groups'] != groups
            or identity['fixed_recipes'] != recipes or identity['release_allowed']
            or identity['gate_evaluation_allowed'] or identity['source_tier'] != 'A_exploration_only'):
        raise ValueError('Registered text profile/recipe/eligibility changed')
    for group in identity['groups']:
        recipe = identity['fixed_recipes'][group][0]
        for fold in identity['split']['folds']:
            name = f'{group}-{fold["year"]}.json'
            decision = folder/'information-decisions'/name
            marker = folder/'information-outputs'/name
            if decision.exists():
                choice = read_record(decision)
                if (choice['selection_used_outer'] or choice['selected']['recipe'] != recipe
                        or choice['selected']['weight'] not in SHRINKAGE
                        or choice['selected']['iterations'] != 1
                        or not np.isfinite(choice['selected']['inner_score'])):
                    raise ValueError('Frozen text decision changed')
            if not marker.exists():
                continue
            if not decision.exists():
                raise ValueError('Text output missing its frozen selection decision')
            record = read_record(marker)
            actual = rows_at(history, fold['origins'])
            frame = pd.DataFrame(record['records'])
            if (record['recipe'] != recipe or record['group'] != group or record['year'] != fold['year']
                    or record['gate_evaluated'] or record['decision_sha256'] != digest(decision)
                    or frame.date.tolist() != fold['origins']
                    or record['inner_score'] != choice['selected']['inner_score']
                    or record['weight'] != choice['selected']['weight']
                    or not np.isfinite(frame[['cotton_close','target_return_5','predicted_return',
                                             'raw_predicted_return','median_return','past_majority_sign']]).all().all()
                    or not np.array_equal(frame.cotton_close, actual.cotton_close)
                    or not np.array_equal(frame.target_return_5, actual.target_return_5)
                    or not np.array_equal(frame.predicted_return, record['weight']*frame.raw_predicted_return)):
                raise ValueError('Frozen text origins/output/decision changed; no silent retraining')
    return ready


def compare(folder):
    ready = verify_cached(folder)
    result = information.compare(folder)
    if result['status'] != 'complete':
        return result
    identity, signals = ready['identity'], {}
    for lag in LAGS:
        text = f'text_L{lag}'
        checks = {}
        for control in (f'base_L{lag}',f'numeric_L{lag}'):
            choices = [read_record(folder/'information-outputs'/f'{group}-{fold["year"]}.json')
                       for fold in identity['split']['folds'] for group in (control,text)]
            past = [(choices[i]['inner_score'],choices[i+1]['inner_score']) for i in range(0,len(choices),2)]
            gain = 100*(1-np.mean([b for _,b in past])/np.mean([a for a,_ in past]))
            inner_wins = sum(b < a for a,b in past)
            outer_wins = 0
            for i in range(0,len(choices),2):
                errors = []
                for row in choices[i:i+2]:
                    frame = pd.DataFrame(row['records'])
                    errors.append(np.mean(frame.cotton_close*np.abs(np.expm1(frame.target_return_5)-np.expm1(frame.predicted_return))))
                outer_wins += errors[1] < errors[0]
            outer_gain = 100*(1-result['groups'][text]['price_mae']/result['groups'][control]['price_mae'])
            checks[control] = {'inner_gain_pct':float(gain),'inner_year_wins':int(inner_wins),
                'outer_gain_pct':float(outer_gain),'outer_year_wins':int(outer_wins),
                'priority_passed':bool(gain>=.5 and inner_wins>=4 and outer_gain>0 and outer_wins>=4)}
        signals[str(lag)] = checks
    decision = {**result,'priority_checks':signals,
        'research_priority_passed':all(check['priority_passed'] for controls in signals.values() for check in controls.values()),
        'primary_hypothesis':'T+5 incremental narrative; reused historical Tier-A research',
        'next_action':'Review fixed hypothesis; no automatic wider search or release'}
    freeze_record(folder/'reports'/'wasde-text-decision.json',decision)
    return decision


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    from cottonlens_ml.research.mirror import Mirror
    if args.stage not in ('prepare','status','pilot-plan','pilot','compare','report'):
        raise ValueError('Tier-A narrative cannot lock/export/search')
    folder = root_path(args.drive_root,args.experiment)
    mirror = Mirror(folder,root_path(args.mirror_root,args.experiment)) if args.mirror_root else None
    if mirror:
        mirror.hydrate_metadata()
        if args.stage == 'pilot' and len(list((folder/'information-outputs').glob('*.json'))) != 45:
            mirror.hydrate()
    if args.stage == 'prepare':
        if not args.reference_root or not args.wasde_corpus:
            raise ValueError('Frozen WASDE reference and narrative corpus required')
        ready = prepare(args.repo,folder,args.reference_root,args.wasde_corpus)
        result = {'status':'prepared','fit_budget':ready['identity']['fit_budget']}
        if mirror:
            mirror.publish_metadata(['ready.json','history.parquet','preregistered.json'])
    elif args.stage == 'status':
        result = {'prepared':(folder/'ready.json').exists(),'writer_lock_present':(folder/'.writer-lock').exists(),
                  'saved_outputs':len(list((folder/'information-outputs').glob('*.json'))),'required_outputs':45,
                  'ledger':Ledger(folder/'ledger',{}).summary(),'source_tier':'A','release_allowed':False}
    elif args.stage == 'pilot-plan':
        result = verify_cached(folder)['identity']['fit_budget']
    elif args.stage == 'pilot':
        verify_cached(folder)
        with writer(folder):
            experiment = Experiment(folder,repo=args.repo)
            if mirror:
                experiment.mirror, experiment.after_fit = mirror, mirror.fit
            result = information.run(experiment,args.max_minutes)
            if mirror:
                members = [p.relative_to(folder).as_posix() for directory in ('information-decisions','information-outputs')
                           for p in (folder/directory).glob('*.json')]
                if members:
                    mirror.publish_metadata(members)
    else:
        result = compare(folder)
        if mirror and result['status'] == 'complete':
            mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder/'reports').glob('*.json')])
    # Final terminal JSON follows all uploads; notebook continuation reads this.
    print(json.dumps(result,indent=2))
