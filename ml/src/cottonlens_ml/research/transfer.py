"""Bounded agricultural shared-training adapter for the existing matched runner."""
import importlib.metadata
import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.engine import Experiment, root_path
from cottonlens_ml.research.full_year import chunks
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.research.panel import FEATURES, SERIES, mature_auxiliary
from cottonlens_ml.research.protocol import Preprocessor, Target, rows_at
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import verify_history

PROFILE = 'agri-transfer-pilot-v1'
GROUPS = ('cotton_only', 'cotton_corn_soybean')
POLICY = 'cotton_transform_fixed_asset_weights_v1'


def validate_design(registration, *, profile=PROFILE, model=None, fit_budget=None):
    d = registration['design']
    model = model or {'family':'ridge','alpha':10,'target':'scaled_log','seed':42}
    fit_budget = fit_budget or {'total_maximum':676,'annual_outputs':32}
    if (registration['design_id'] != content_id(d) or d['profile'] != profile
            or d['series'] != list(SERIES) or d['features'] != FEATURES
            or d['arms'] != list(GROUPS) or d['feature_lag_own_observations'] != 1
            or d['model'] != model
            or d['cadence'] != 21 or d['purge_cotton_observations'] != 5
            or d['split']['purge_observations'] != 5 or d['split']['refit_cadence'] != 21
            or [f['year'] for f in d['split']['folds']] != list(range(2016,2024))
            or d['shrinkage_weights'] != [0,.25,.5,.75,1]
            or d['fit_budget'] != fit_budget
            or d['price_gate_unchanged'] != {'gain_pct':5,'direction_pct':{'1':53,'5':55},'year_wins':6}
            or d['audit_2024_used'] is not False or d['automatic_release'] is not False):
        raise ValueError('Frozen transfer design changed')
    return d


def recipe(design, group, h):
    if group not in GROUPS or h not in (1,5):
        raise ValueError('Only frozen transfer arms and horizons are allowed')
    return {'family':'ridge','params':{'alpha':10.},'horizon':h,'task':'price','device':'cpu',
        'seed':42,'target':'scaled_log','years':None,'cadence':21,'window':1,
        'features':FEATURES,'training_policy':POLICY,'arm':group}


def training_state(history, train, test, spec):
    """Cotton-only transforms; fixed total weights keep Ridge penalty comparable."""
    recipe_fn = recipe
    if spec.get('training_policy') == 'cotton_transform_fixed_asset_weights_xgb_v1':
        from cottonlens_ml.research.nonlinear_transfer import recipe as recipe_fn
    if spec != recipe_fn({}, spec.get('arm'), spec.get('horizon')):
        raise ValueError('Transfer fit recipe changed')
    assets = set(train.asset)
    expected = {'cotton'} if spec['arm']==GROUPS[0] else set(SERIES)
    if assets != expected or train.duplicated(['asset','date']).any():
        raise ValueError('Complete distinct transfer asset cohorts required')
    cutoff = test.date.min()
    indexes = np.flatnonzero(history.date.eq(cutoff).to_numpy())
    if len(indexes)!=1 or indexes[0]<5:
        raise ValueError('Cotton observation purge boundary unavailable')
    boundary = history.date.iloc[indexes[0]-5]
    if (not np.isfinite(train[['target_return_1','target_return_5']]).all().all()
            or not train.date.lt(boundary).all() or not train.target_date_5.lt(cutoff).all()
            or not train.feature_asof.lt(train.date).all() or not test.asset.eq('cotton').all()):
        raise ValueError('Immature label, future feature or non-Cotton evaluation row')
    cotton = train.loc[train.asset.eq('cotton')]
    processor = Preprocessor.fit(cotton, FEATURES)
    target = Target.fit(cotton, spec['horizon'], 'scaled_log')
    proportions = {'cotton':1.} if len(assets)==1 else {'cotton':.5,'corn':.25,'soybean':.25}
    weights = np.empty(len(train))
    totals = {}
    for asset, share in proportions.items():
        mask = train.asset.eq(asset).to_numpy()
        if mask.sum()<126:
            raise ValueError('Insufficient mature history for a pinned asset')
        weights[mask] = len(cotton)*share/mask.sum()
        totals[asset] = float(weights[mask].sum())
    if not np.isclose(weights.sum(),len(cotton)):
        raise ValueError('Transfer objective normalization changed')
    return processor, target, weights, {'training_policy':spec['training_policy'],'asset_loss_weight_totals':totals,
        'cotton_transform_rows':len(cotton),'asset_rows':train.asset.value_counts().to_dict(),
        'source_tier':'A_exploration_only','release_allowed':False}


class TransferExperiment(Experiment):
    recipe_fn = staticmethod(recipe)

    def __init__(self, root, repo=None):
        super().__init__(root,repo=repo)
        if digest(self.root/'panel.parquet') != self.ready['panel_sha256']:
            raise ValueError('Frozen auxiliary snapshot corrupted')
        self.panel = pd.read_parquet(self.root/'panel.parquet')

    def train_rows(self, cutoff, spec):
        if spec != self.recipe_fn(self.identity['design'],spec.get('arm'),spec.get('horizon')):
            raise ValueError('Transfer training recipe changed')
        cotton = super().train_rows(cutoff,spec)
        if spec['arm']==GROUPS[0]:
            return cotton
        index = np.flatnonzero(self.history.date.eq(cutoff).to_numpy())
        if len(index)!=1 or index[0]<5:
            raise ValueError('Cotton cutoff not on the frozen calendar')
        boundary = self.history.date.iloc[index[0]-5]
        clock = pd.Timestamp(cutoff).tz_localize('UTC')+pd.Timedelta(days=1,minutes=15)
        parts = [cotton]
        for asset in ('corn','soybean'):
            rows = mature_auxiliary(self.panel.loc[self.panel.asset.eq(asset)],clock,boundary)
            rows = rows.loc[rows.date.ge(self.history.date.iloc[119])].copy()
            rows = rows.rename(columns={'close':'cotton_close'})
            rows['cotton_session_index'] = np.nan
            parts.append(rows[cotton.columns])
        return pd.concat(parts,ignore_index=True)


def prepare(repo, folder, packet, *, validate_fn=validate_design):
    registration = read_record(packet/'preregistered.json')
    design = validate_fn(registration)
    reference = packet/'reference'
    old, original = verify_history(reference)
    hashes = design['data_hashes']
    if (digest(reference/'ready.json')!=hashes['Cotton_ready']
            or digest(reference/'history.parquet')!=hashes['Cotton_history']
            or digest(packet/'panel-history.parquet')!=hashes['panel_table']
            or digest(repo/'ml/src/cottonlens_ml/research/panel.py')!=hashes['feature_helper']
            or old['identity']['split']!=design['split']):
        raise ValueError('Frozen transfer inputs or feature helper changed')
    panel = pd.read_parquet(packet/'panel-history.parquet')
    if set(panel.asset)!=set(SERIES) or panel.duplicated(['asset','date']).any() or panel.date.ge('2024-01-01').any():
        raise ValueError('Invalid auxiliary panel identity')
    history = panel.loc[panel.asset.eq('cotton')].rename(columns={'close':'cotton_close'}).reset_index(drop=True)
    if not history.date.equals(original.date) or not np.array_equal(history.cotton_close,original.cotton_close):
        raise ValueError('Cotton price cohort changed')
    history['cotton_session_index'] = original.cotton_session_index.to_numpy()
    for h in (1,5):
        # The preserved reference may contain terminal labels from the seen audit.
        # Match only pre-2024 targets; never import those audit outcomes into research.
        dates = original[f'target_date_{h}'].where(original[f'target_date_{h}'].lt('2024-01-01'))
        labels = original[f'target_return_{h}'].where(dates.notna())
        if (not history[f'target_date_{h}'].equals(dates)
                or not np.allclose(history[f'target_return_{h}'],labels,atol=1e-12,rtol=0,equal_nan=True)):
            raise ValueError('Cotton labels changed')
        history[f'target_return_{h}'] = labels.to_numpy()
    for fold in design['split']['folds']:
        rows_at(history,fold['origins'])
    budget = 4*(sum(len(chunks(history,b['origins'])) for f in design['split']['folds'] for b in f['inner'])
        +sum(len(chunks(history,f['origins'])) for f in design['split']['folds']))
    if design['model']['family']=='xgboost':
        budget += 32*3
    if budget>design['fit_budget']['total_maximum']:
        raise ValueError('Frozen transfer work budget exceeded')
    identity = {'profile':design['profile'],'source_id':research_source_identity(repo)['source_id'],
        'design_id':registration['design_id'],'design':design,'split':design['split'],
        'python':sys.version.split()[0],'versions':{p:importlib.metadata.version(p) for p in
            ('numpy','pandas','scipy','scikit-learn','xgboost')},'lock_sha256':digest(repo/'ml/uv.lock'),
        'execution':{'device':design['model'].get('device','cpu'),'threads':2},'exact_fit_budget':budget}
    with writer(folder):
        if (folder/'ready.json').exists():
            ready = read_record(folder/'ready.json')
            if (ready['identity']!=identity or digest(folder/'history.parquet')!=ready['history_sha256']
                    or digest(folder/'panel.parquet')!=ready['panel_sha256']):
                raise ValueError('Frozen transfer code, environment or data changed')
            return ready
        copy_immutable(reference/'ready.json',folder/'reference/ready.json')
        copy_immutable(reference/'history.parquet',folder/'reference/history.parquet')
        copy_immutable(packet/'panel-history.parquet',folder/'panel.parquet')
        pending = folder/'history.pending.parquet'
        history.to_parquet(pending,index=False)
        copy_immutable(pending,folder/'history.parquet')
        pending.unlink()
        freeze_record(folder/'preregistered.json',registration)
        ready = {'identity':identity,'history_sha256':digest(folder/'history.parquet'),
            'panel_sha256':digest(folder/'panel.parquet'),'created_at':datetime.now(UTC).isoformat()}
        freeze_record(folder/'ready.json',ready)
        return ready


def options():
    return {'group_names':GROUPS,'namespace':'transfer','recipe_fn':recipe,'validate_fn':validate_design}


def dispatch(args):
    folder = root_path(args.drive_root,args.experiment)
    mirror = Mirror(folder,root_path(args.mirror_root,args.experiment)) if args.mirror_root else None
    if mirror:
        mirror.hydrate_metadata()
        if args.stage=='pilot' and len(list((folder/'transfer-outputs').glob('*.json')))!=32:
            mirror.hydrate()
    if args.stage=='prepare':
        if not args.reference_root:
            raise ValueError('Frozen transfer input packet required')
        ready = prepare(args.repo,folder,args.reference_root)
        if mirror:
            mirror.publish_metadata(['ready.json','history.parquet','panel.parquet','preregistered.json',
                'reference/ready.json','reference/history.parquet'])
        result = {'status':'prepared','exact_fit_budget':ready['identity']['exact_fit_budget']}
    elif args.stage=='status':
        result = {'prepared':(folder/'ready.json').exists(),'writer_lock_present':(folder/'.writer-lock').exists(),
            'saved_outputs':len(list((folder/'transfer-outputs').glob('*.json'))),'required_outputs':32,
            'ledger':Ledger(folder/'ledger',{}).summary(),'model_payloads_verified_by_status':False}
    elif args.stage=='pilot-plan':
        result = {'fit_budget':read_record(folder/'ready.json')['identity']['exact_fit_budget'],'required_outputs':32}
    elif args.stage=='pilot':
        with writer(folder):
            experiment = TransferExperiment(folder,repo=args.repo)
            if mirror:
                experiment.mirror,experiment.after_fit = mirror,mirror.fit
            result = path_pilot.run(experiment,args.max_minutes,**options())
    elif args.stage in ('compare','report'):
        result = path_pilot.compare(folder,**options(),hypothesis_prefix='P',
            scope='Tier-A exploratory transfer; reused history; no independent or release claim')
        if mirror and result['status']=='complete':
            mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder/'reports').glob('transfer-*.json')])
    else:
        raise ValueError('Transfer supports bounded pilot/report stages; no automatic release')
    print(json.dumps(result,indent=2))
