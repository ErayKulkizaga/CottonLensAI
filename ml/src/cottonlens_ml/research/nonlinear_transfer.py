"""One fixed nonlinear shared-training comparison, using the existing ledger/runner."""
import json

from cottonlens_ml.research import path_pilot, transfer
from cottonlens_ml.research.engine import root_path
from cottonlens_ml.research.ledger import Ledger, read_record, writer
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.runtime_guard import require_colab_training

PROFILE = 'agri-nonlinear-pilot-v1'
POLICY = 'cotton_transform_fixed_asset_weights_xgb_v1'
MODEL = {'family':'xgboost','target':'scaled_log','loss':'reg:squarederror','seed':42,'device':'cuda',
    'tree_method':'hist','params':{'max_depth':2,'eta':.03,'min_child_weight':20,'alpha':0,'lambda':1,
        'subsample':1.,'colsample_bytree':1.},'max_iterations':600,'patience':50}
BUDGET = {'total_maximum':772,'annual_outputs':32,'early_stop_fits':96,'refit_fits':676}


def validate_design(registration):
    return transfer.validate_design(registration,profile=PROFILE,model=MODEL,fit_budget=BUDGET)


def recipe(design,group,h):
    spec = transfer.recipe({},group,h)
    spec.update({k:v for k,v in MODEL.items() if k!='tree_method'})
    spec['params'] = dict(MODEL['params'])
    spec['training_policy'] = POLICY
    return spec


class NonlinearExperiment(transfer.TransferExperiment):
    recipe_fn = staticmethod(recipe)


def options():
    return {'group_names':transfer.GROUPS,'namespace':'nonlinear','recipe_fn':recipe,'validate_fn':validate_design}


def prepare(repo,folder,packet):
    require_colab_training()
    return transfer.prepare(repo,folder,packet,validate_fn=validate_design)


def dispatch(args):
    if args.stage not in ('status','prepare','pilot-plan','pilot','compare','report'):
        raise ValueError('Nonlinear transfer supports bounded pilot/report only; release is blocked')
    folder = root_path(args.drive_root,args.experiment)
    mirror = Mirror(folder,root_path(args.mirror_root,args.experiment)) if args.mirror_root else None
    if mirror:
        mirror.hydrate_metadata()
        if args.stage=='pilot' and len(list((folder/'nonlinear-outputs').glob('*.json')))!=32:
            mirror.hydrate()
    if args.stage=='prepare':
        if not args.reference_root:
            raise ValueError('Frozen nonlinear transfer packet required')
        ready = prepare(args.repo,folder,args.reference_root)
        if mirror:
            mirror.publish_metadata(['ready.json','history.parquet','panel.parquet','preregistered.json',
                'reference/ready.json','reference/history.parquet'])
        result = {'status':'prepared','exact_fit_budget':ready['identity']['exact_fit_budget']}
    elif args.stage=='status':
        result = {'prepared':(folder/'ready.json').exists(),'writer_lock_present':(folder/'.writer-lock').exists(),
            'saved_outputs':len(list((folder/'nonlinear-outputs').glob('*.json'))),'required_outputs':32,
            'ledger':Ledger(folder/'ledger',{}).summary(),'model_payloads_verified_by_status':False}
    elif args.stage=='pilot-plan':
        result = {'fit_budget':read_record(folder/'ready.json')['identity']['exact_fit_budget'],'required_outputs':32}
    elif args.stage=='pilot':
        require_colab_training()
        with writer(folder):
            experiment = NonlinearExperiment(folder,repo=args.repo)
            if mirror:
                experiment.mirror,experiment.after_fit = mirror,mirror.fit
            result = path_pilot.run(experiment,args.max_minutes,**options())
    else:
        result = path_pilot.compare(folder,**options(),hypothesis_prefix='N',
            scope='Tier-A exploratory fixed nonlinear transfer; reused history; no independent or release claim')
        if mirror and result['status']=='complete':
            mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder/'reports').glob('nonlinear-*.json')])
    print(json.dumps(result,indent=2))
