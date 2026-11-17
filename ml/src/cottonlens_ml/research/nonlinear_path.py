"""One fixed nonlinear temporal-representation hypothesis on the matched runner."""
import json

from cottonlens_ml.cohort import content_id
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.engine import Experiment, root_path
from cottonlens_ml.research.full_year import SHRINKAGE
from cottonlens_ml.research.ledger import Ledger, read_record, writer
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.research.return_path import PATH_COLUMNS, PATH_FEATURES
from cottonlens_ml.runtime_guard import require_colab_training

PROFILE = 'nonlinear-path-pilot-v1'
GROUPS = ('base', 'base_plus_return_path')
MODEL = {'family':'xgboost','device':'cuda','target':'scaled_log','loss':'reg:squarederror',
         'params':{'max_depth':2,'eta':.03,'min_child_weight':20,'alpha':0,'lambda':1,
                   'subsample':1.,'colsample_bytree':1.},
         'max_iterations':600,'patience':50,'years':None,'seed':42,'cadence':21,'window':1}
BUDGET = {'total':772,'early_stop_fits':96,'refit_fits':676,'annual_outputs':32}


def validate_design(registration):
    d = registration['design']
    if (registration['design_id'] != content_id(d) or d['profile'] != PROFILE
            or d['groups'] != {'base':list(FEATURE_NAMES),'base_plus_return_path':PATH_FEATURES}
            or d['new_columns'] != PATH_COLUMNS or d['recipes'] != MODEL
            or d['fit_budget'] != BUDGET or d['shrinkage_weights'] != list(SHRINKAGE)
            or d['split']['purge_observations'] != 5 or d['split']['refit_cadence'] != 21
            or [f['year'] for f in d['split']['folds']] != list(range(2016,2024))
            or any(len(f['inner']) != 3 for f in d['split']['folds'])
            or sum(len(f['origins']) for f in d['split']['folds']) != 2006
            or d['price_gate'] != {'gain_pct':5.,'direction_pct':{'1':53.,'5':55.},'year_wins':6}
            or d['automatic_release'] is not False or d['audit_2024_used'] is not False):
        raise ValueError('Frozen nonlinear path design changed')
    return d


def recipe(design, group, h):
    if group not in GROUPS or h not in (1,5):
        raise ValueError('Frozen nonlinear path arm/horizon required')
    return {**MODEL,'params':dict(MODEL['params']),'horizon':h,'task':'price',
            'features':list(design['groups'][group])}


def options():
    return {'group_names':GROUPS,'namespace':'nonlinear-path','recipe_fn':recipe,'validate_fn':validate_design}


def prepare(repo, folder, packet):
    require_colab_training()
    return path_pilot.prepare(repo,folder,packet/'reference',profile=PROFILE,validate_fn=validate_design,
                              registration_name='preregistered.json',device='cuda')


def dispatch(args):
    if args.stage not in ('status','prepare','pilot-plan','pilot','compare','report'):
        raise ValueError('Nonlinear path supports bounded research only; no automatic release')
    folder = root_path(args.drive_root,args.experiment)
    mirror = Mirror(folder,root_path(args.mirror_root,args.experiment)) if args.mirror_root else None
    if mirror:
        mirror.hydrate_metadata()
        if args.stage=='pilot' and len(list((folder/'nonlinear-path-outputs').glob('*.json')))!=32:
            mirror.hydrate()
    if args.stage=='prepare':
        if not args.reference_root:
            raise ValueError('Frozen nonlinear path input packet required')
        ready = prepare(args.repo,folder,args.reference_root)
        if mirror:
            mirror.publish_metadata(['ready.json','history.parquet','preregistered.json',
                                     'reference/ready.json','reference/history.parquet'])
        result = {'status':'prepared','fit_budget':ready['identity']['design']['fit_budget']}
    elif args.stage=='status':
        result = {'prepared':(folder/'ready.json').exists(),'writer_lock_present':(folder/'.writer-lock').exists(),
                  'saved_outputs':len(list((folder/'nonlinear-path-outputs').glob('*.json'))),'required_outputs':32,
                  'ledger':Ledger(folder/'ledger',{}).summary(),'model_payloads_verified_by_status':False}
    elif args.stage=='pilot-plan':
        result = read_record(folder/'ready.json')['identity']['design']['fit_budget']
    elif args.stage=='pilot':
        require_colab_training()
        with writer(folder):
            experiment = Experiment(folder,repo=args.repo)
            if mirror:
                experiment.mirror,experiment.after_fit = mirror,mirror.fit
            result = path_pilot.run(experiment,args.max_minutes,**options())
    else:
        result = path_pilot.compare(folder,**options(),hypothesis_prefix='P',
            scope='Fixed nonlinear return-path hypothesis; reused historical research; no independent/release claim')
        if mirror and result['status']=='complete':
            mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder/'reports').glob('nonlinear-path-*.json')])
    print(json.dumps(result,indent=2))
