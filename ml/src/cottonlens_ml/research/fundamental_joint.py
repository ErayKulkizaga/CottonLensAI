"""One joint demand/supply-state hypothesis; no new vintage/source admission."""
import copy
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.fixed_information import (
    dispatch_fixed,
    incremental_compare,
    verify_outputs,
)
from cottonlens_ml.research.full_year import SHRINKAGE, chunks, price_recipes
from cottonlens_ml.research.ledger import freeze_record, read_record, writer

PROFILE = 'fundamental-joint-pilot-v1'
LAGS = (1,2,6)
KINDS = {'fas':'sales','nass':'condition','weather':'weather'}
INTERACTIONS = {
    'sales_under_poor_condition':['fas_net_sales_change','nass_poor_very_poor'],
    'heat_under_poor_condition':['weather_growing_heat_excess_7','nass_poor_very_poor'],
    'rain_under_poor_condition':['weather_growing_rain_30','nass_poor_very_poor'],
}


CONTROLS = {'fas':('age','missing'),'nass':('age','missing'),
            'weather':('age','missing','growing_season')}
VALUES = {'fas':('net_sales','exports','net_sales_change','exports_change'),
          'nass':('good_excellent','poor_very_poor','good_excellent_change'),
          'weather':('rain_7','rain_30','temperature_7','heat_excess_7',
                     'growing_rain_7','growing_rain_30','growing_temperature_7','growing_heat_excess_7')}


def groups_and_recipes(parents=None):
    groups, recipes = {}, {}
    for lag in LAGS:
        timing, numeric = list(FEATURE_NAMES), []
        for kind, arm in KINDS.items():
            control = [*FEATURE_NAMES,*[f'{kind}_{v}_L{lag}' for v in CONTROLS[kind]]]
            full = [*control,*[f'{kind}_{v}_L{lag}' for v in VALUES[kind]]]
            if parents and (parents[kind]['identity']['groups'][f'missing_L{lag}'] != control
                    or parents[kind]['identity']['groups'][f'{arm}_L{lag}'] != full):
                raise ValueError('Frozen source feature order/control mismatch')
            timing.extend(control[24:])
            numeric.extend(full[len(control):])
        joint = timing+numeric
        crosses = [f'joint_{name}_L{lag}' for name in INTERACTIONS]
        for arm, names in (('timing',timing),('joint',joint),('interaction',joint+crosses)):
            group = f'{arm}_L{lag}'
            groups[group], recipes[group] = names, [{**price_recipes(5)[1],'features':names}]
    return groups,recipes


def combine(parents, histories):
    """Validate common rows/targets before adding existing point-in-time assumptions."""
    result = histories['fas'].copy()
    shared = ['date','cotton_session_index','cotton_close',*FEATURE_NAMES,
              'target_date_1','target_date_5','target_return_1','target_return_5']
    split = copy.deepcopy(parents['fas']['identity']['split'])
    for kind in KINDS:
        identity = parents[kind]['identity']
        if (identity['profile'] != f'{kind}-exploration-v1' or identity['release_allowed']
                or identity['gate_evaluation_allowed'] or identity['split']['purge_observations'] != 5
                or identity['split']['refit_cadence'] != 21):
            raise ValueError('Explicit unadmitted Tier-A parents required')
        history = histories[kind]
        try:
            pd.testing.assert_frame_equal(result[shared],history[shared],check_dtype=False,check_exact=True)
        except AssertionError as exc:
            raise ValueError('Source calendar/base/prices/labels differ; no inner join or row dropping') from exc
        other = identity['split']['folds']
        if (len(other) != len(split['folds']) or any(a['year'] != b['year'] or a['origins'] != b['origins']
                or a['inner'] != b['inner'] for a,b in zip(split['folds'],other,strict=True))):
            raise ValueError('Sources must share frozen inner and outer origins')
        for lag in LAGS:
            names = identity['groups'][f'{KINDS[kind]}_L{lag}'][24:]
            result[names] = history[names]
    if (result.date.max() >= pd.Timestamp('2024-01-01') or result.date.duplicated().any()
            or not result.date.is_monotonic_increasing
            or [f['year'] for f in split['folds']] != list(range(2016,2024))
            or sum(len(f['origins']) for f in split['folds']) != 2006):
        raise ValueError('Exact pre-audit 2006-origin eight-year cohort required')
    for lag in LAGS:
        poor = result[f'nass_poor_very_poor_L{lag}']
        if not poor.dropna().between(0,1).all():
            raise ValueError('NASS poor-condition fraction outside [0,1]')
        for name,(a,b) in INTERACTIONS.items():
            # Missing inputs stay missing; no joint-coverage filtering or zero filling.
            result[f'joint_{name}_L{lag}'] = result[f'{a}_L{lag}']*result[f'{b}_L{lag}']
    split['coverage_start'] = max(p['identity']['split']['coverage_start'] for p in parents.values())
    split['version'], split['role'] = PROFILE, 'seen_historical_Tier_A_research'
    split.pop('split_id',None)
    split['split_id'] = content_id(split)
    return result,split


def load_packet(packet):
    packet = Path(packet)
    manifest = read_record(packet/'input-manifest.json')
    expected = {f'{kind}/{name}' for kind in KINDS for name in ('ready.json','history.parquet')}
    if set(manifest['files']) != expected or manifest['profile'] != PROFILE:
        raise ValueError('Exact frozen three-source packet required')
    parents,histories = {},{}
    for name,sha in manifest['files'].items():
        if digest(packet/name) != sha:
            raise ValueError('Frozen joint input checksum mismatch')
    for kind in KINDS:
        parents[kind] = read_record(packet/kind/'ready.json')
        if digest(packet/kind/'history.parquet') != parents[kind]['history_sha256']:
            raise ValueError('Frozen source snapshot checksum mismatch')
        histories[kind] = pd.read_parquet(packet/kind/'history.parquet')
    return parents,histories


def prepare(repo, folder, packet):
    repo,folder = Path(repo),Path(folder)
    parents,histories = load_packet(packet)
    history,split = combine(parents,histories)
    groups,recipes = groups_and_recipes(parents)
    # No new origins or models are selected by missingness or outside results.
    for h in (1,5):
        mask = history[f'target_date_{h}'].ge(pd.Timestamp('2024-01-01'))
        history.loc[mask,f'target_date_{h}'], history.loc[mask,f'target_return_{h}'] = pd.NaT,np.nan
    inner = sum(sum(len(chunks(history,b['origins'])) for b in f['inner']) for f in split['folds'])
    outer = sum(len(chunks(history,f['origins'])) for f in split['folds'])
    budget = {'annual_outputs':72,'inner_refits':inner*9,'outer_refits':outer*9,'total_refits':(inner+outer)*9}
    if budget != {'annual_outputs':72,'inner_refits':648,'outer_refits':873,'total_refits':1521}:
        raise ValueError('Registered joint refit budget changed')
    identity = {'profile':PROFILE,'source_id':research_source_identity(repo)['source_id'],
        'parents':{k:{'ready_sha256':digest(Path(packet)/k/'ready.json'),'history_sha256':p['history_sha256']} for k,p in parents.items()},
        'research_data_id':frame_identity(history,list(history.columns)),'split':split,'groups':groups,'fixed_recipes':recipes,
        'fit_budget':budget,'interactions':INTERACTIONS,
        'comparison_controls':{f'interaction_L{l}':f'joint_L{l}' for l in LAGS},
        'comparison_bases':{f'interaction_L{l}':f'timing_L{l}' for l in LAGS},
        'report_prefix':'fundamental-joint','include_reference_diagnostics':True,'bootstrap_lengths':[10,20,40],
        'python':sys.version.split()[0], 'versions':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','xgboost')},
        'lock_sha256':digest(repo/'ml/uv.lock'),'execution':{'device':'cpu','threads':2,'parallel_jobs':1},
        'availability_policy':'Inherit each frozen source lag/max-age/season and unverified vintage assumption; no new admission',
        'source_tier':'A_exploration_only','release_allowed':False,'gate_evaluation_allowed':False,
        'priority_rule':{'inner_gain_pct':.5,'inner_year_wins':5,'positive_outer_gain':True,'outer_year_wins':5,
                         'both_controls_all_lag_stresses':True},
        'checkpoint_date':'2026-10-22T20:06:59.863911+00:00'}
    with writer(folder):
        if (folder/'ready.json').exists():
            prior = read_record(folder/'ready.json')
            if prior['identity'] != identity or digest(folder/'history.parquet') != prior['history_sha256']:
                raise ValueError('Frozen joint inputs/recipe/environment changed; use new namespace')
            return prior
        if (folder/'history.parquet').exists():
            raise ValueError('Incomplete joint prepare preserved')
        history.to_parquet(folder/'history.parquet',index=False)
        ready = {'identity':identity,'history_sha256':digest(folder/'history.parquet')}
        freeze_record(folder/'ready.json',ready)
        freeze_record(folder/'preregistered.json',{'hypothesis':'Demand/weather signal depends on crop-condition state',
            'primary_comparison':'interaction_L1 vs joint_L1 and timing_L1; lag2/6 mandatory stresses',
            'source_category':'Combined already-tested sources; distinct conditional representation, not a newly downloaded source',
            'recipes':recipes,'interactions':INTERACTIONS,'shrinkage':list(SHRINKAGE),'budget':budget,
            'source_tier':'A_exploration_only','release_allowed':False,'gate_evaluated':False,
            'rule':identity['priority_rule'],'checkpoint_date':identity['checkpoint_date'],
            'on_negative':'Close fixed joint-state hypothesis; no automatic interaction/model/window grid'})
    return ready


def validate(folder):
    groups, recipes = groups_and_recipes()
    return verify_outputs(folder,PROFILE,groups,recipes)


def compare(folder):
    return incremental_compare(folder,validate,lags=LAGS,main_arm='interaction',
                               controls=('timing','joint'),wins=5,prefix='fundamental-joint')


def dispatch(args):
    if args.stage == 'prepare' and not args.information_inputs:
        raise ValueError('Frozen three-source input packet required')
    return dispatch_fixed(args,prepare=lambda folder:prepare(args.repo,folder,args.information_inputs),
                          validate=validate,compare=compare)
