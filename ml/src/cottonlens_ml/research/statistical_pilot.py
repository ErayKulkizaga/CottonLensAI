"""Frozen small statistical comparison on the existing matched runner/ledger."""
import importlib.metadata
import json
import sys
from datetime import UTC, datetime

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research import path_pilot
from cottonlens_ml.research.engine import Experiment, root_path
from cottonlens_ml.research.full_year import SHRINKAGE, chunks, inner_price
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.mirror import Mirror
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.review import verify_history
from cottonlens_ml.research.statistical_models import ORDERS, InvalidStatisticalFit
from cottonlens_ml.runtime_guard import require_colab_training

PROFILE = 'statistical-pilot-v1'
GROUPS = ('drift_reference','selected_statistical')
POLICY = {'series':'Cotton log prices, no time compression','coefficient_cutoff':'last mature purged training origin',
    'filter_cutoff':'observed close through each forecast origin','refit_parameters_during_state_updates':False,
    'orders':[list(o) for o in ORDERS],'trend':'n','optimizer':'statespace','maxiter':100,
    'nonconvergence':'exclude incomplete inner candidate; outer failure blocks, no silent replacement',
    'drift':'h times mean daily log return of contiguous mature training price series',
    'local_cpu_training_allowed':False,'gpu_required':False}


def recipes(design,group,h):
    base = {'task':'price','horizon':h,'device':'cpu','seed':42,'years':None,'cadence':21,
        'target':'log_price_difference','window':1,'features':[]}
    drift = {**base,'family':'drift','params':{}}
    if group==GROUPS[0]: return [drift]
    if group!=GROUPS[1]: raise ValueError('Unknown statistical group')
    return [{**base,'family':'naive','params':{}},drift,
        *[{**base,'family':'arima','params':{'order':list(o),'trend':'n','maxiter':100}} for o in ORDERS]]


def validate_design(registration):
    d = registration['design']
    if (registration['design_id']!=content_id(d) or d['profile']!=PROFILE or d['policy']!=POLICY
            or d['groups']!=list(GROUPS) or d['shrinkage_weights']!=list(SHRINKAGE)
            or d['split']['purge_observations']!=5 or d['split']['refit_cadence']!=21
            or [f['year'] for f in d['split']['folds']]!=list(range(2016,2024))
            or d['price_gate']!={'gain_pct':5,'direction_pct':{'1':53,'5':55},'year_wins':6}
            or not 0<d['fit_budget']['maximum_ledger_jobs']<=964
            or not 0<d['fit_budget']['maximum_arima_estimations']<=482
            or d['fit_budget']['annual_outputs']!=32
            or d['automatic_release'] is not False or d['audit_2024_used'] is not False):
        raise ValueError('Frozen statistical design changed')
    return d


def select(experiment,options,fold):
    candidates = []
    reference = len(options)==1
    for spec in options:
        invalid = experiment.root/'statistical-invalid'/(
            content_id({'identity':experiment.identity,'recipe':spec,'year':fold['year']})+'.json')
        if invalid.exists():
            body = read_record(invalid)
            if (body['recipe']!=spec or body['year']!=fold['year']
                    or body['identity_id']!=content_id(experiment.identity)):
                raise ValueError('Frozen numerical failure identity changed')
            candidates.append({'recipe':spec,'status':'numerical_failure'})
            continue
        try:
            choice = inner_price(experiment,spec,fold,weights=(1.,) if reference else SHRINKAGE)
        except InvalidStatisticalFit as exc:
            if spec['family']!='arima':
                raise
            freeze_record(invalid,{'recipe':spec,'year':fold['year'],'reason':str(exc),
                'status':'numerical_failure','identity_id':content_id(experiment.identity)})
            if getattr(experiment,'mirror',None):
                experiment.mirror.publish_metadata([invalid.relative_to(experiment.root).as_posix()])
            candidates.append({'recipe':spec,'status':'numerical_failure'})
            continue
        candidates.append({'recipe':spec,'status':'complete','selected':choice})
    completed = [c['selected'] for c in candidates if c['status']=='complete']
    if not completed: raise InvalidStatisticalFit('No completed statistical candidate')
    return {'selected':min(completed,key=lambda c:c['inner_score']),'candidates':candidates}


def prepare(repo,folder,packet):
    require_colab_training()
    registration = read_record(packet/'preregistered.json'); design = validate_design(registration)
    old,history = verify_history(packet/'reference')
    if (old['identity']['split']!=design['split'] or digest(packet/'reference/ready.json')!=design['ready_sha256']
            or digest(packet/'reference/history.parquet')!=design['history_sha256']
            or digest(repo/'ml/src/cottonlens_ml/research/statistical_models.py')!=design['helper_sha256']):
        raise ValueError('Frozen statistical origins or data changed')
    inner = sum(len(chunks(history,b['origins'])) for f in design['split']['folds'] for b in f['inner'])
    outer = sum(len(chunks(history,f['origins'])) for f in design['split']['folds'])
    budget = {'maximum_ledger_jobs':8*inner+4*outer,'maximum_arima_estimations':4*inner+2*outer,
        'annual_outputs':32}
    if budget!=design['fit_budget']: raise ValueError('Frozen statistical budget changed')
    identity = {'profile':PROFILE,'source_id':research_source_identity(repo)['source_id'],
        'design':design,'design_id':registration['design_id'],'split':design['split'],
        'python':sys.version.split()[0],'versions':{p:importlib.metadata.version(p) for p in
            ('numpy','pandas','scipy','scikit-learn','xgboost','statsmodels')},
        'lock_sha256':digest(repo/'ml/uv.lock'),'execution':{'device':'cpu','threads':2}}
    with writer(folder):
        if (folder/'ready.json').exists():
            ready = read_record(folder/'ready.json')
            if ready['identity']!=identity or digest(folder/'history.parquet')!=ready['history_sha256']:
                raise ValueError('Frozen statistical source/environment/data changed')
            return ready
        copy_immutable(packet/'reference/history.parquet',folder/'history.parquet')
        copy_immutable(packet/'preregistered.json',folder/'preregistered.json')
        ready = {'identity':identity,'history_sha256':digest(folder/'history.parquet'),
            'created_at':datetime.now(UTC).isoformat()}
        freeze_record(folder/'ready.json',ready)
        return ready


def options():
    return {'group_names':GROUPS,'namespace':'statistical','recipe_fn':recipes,'validate_fn':validate_design}


def dispatch(args):
    if args.stage not in ('status','prepare','pilot-plan','pilot','compare','report'):
        raise ValueError('Statistical pilot supports bounded research only; no release')
    folder = root_path(args.drive_root,args.experiment)
    mirror = Mirror(folder,root_path(args.mirror_root,args.experiment)) if args.mirror_root else None
    if mirror:
        mirror.hydrate_metadata()
        if args.stage=='pilot' and len(list((folder/'statistical-outputs').glob('*.json')))!=32: mirror.hydrate()
    if args.stage=='prepare':
        if not args.reference_root: raise ValueError('Frozen statistical packet required')
        ready = prepare(args.repo,folder,args.reference_root)
        if mirror: mirror.publish_metadata(['ready.json','history.parquet','preregistered.json'])
        result = {'status':'prepared','fit_budget':ready['identity']['design']['fit_budget']}
    elif args.stage=='status':
        result = {'prepared':(folder/'ready.json').exists(),'saved_outputs':len(list((folder/'statistical-outputs').glob('*.json'))),
            'required_outputs':32,'writer_lock_present':(folder/'.writer-lock').exists(),
            'ledger':Ledger(folder/'ledger',{}).summary(),'model_payloads_verified_by_status':False}
    elif args.stage=='pilot-plan': result = read_record(folder/'ready.json')['identity']['design']['fit_budget']
    elif args.stage=='pilot':
        require_colab_training()
        with writer(folder):
            experiment = Experiment(folder,repo=args.repo)
            if mirror: experiment.mirror,experiment.after_fit = mirror,mirror.fit
            result = path_pilot.run(experiment,args.max_minutes,**options(),selection_fn=select)
            if mirror and list((folder/'statistical-invalid').glob('*.json')):
                mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder/'statistical-invalid').glob('*.json')])
    else:
        result = path_pilot.compare(folder,**options(),hypothesis_prefix='S',
            scope='Reused historical statistical references; no independent/release claim; baseline arm is unshrunk drift')
        if mirror and result['status']=='complete':
            mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder/'reports').glob('statistical-*.json')])
    print(json.dumps(result,indent=2))
