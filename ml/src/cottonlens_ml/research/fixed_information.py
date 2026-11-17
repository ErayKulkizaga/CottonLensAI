"""Bounded information profiles on the existing runner, ledger and mirror."""
import json

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest
from cottonlens_ml.research import ams_exploration as information
from cottonlens_ml.research.full_year import SHRINKAGE
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.protocol import rows_at


def verify_outputs(folder, profile, groups, recipes):
    ready = read_record(folder/'ready.json')
    identity = ready['identity']
    if (identity['profile'] != profile or identity['groups'] != groups
            or identity['fixed_recipes'] != recipes or identity['release_allowed']
            or identity['gate_evaluation_allowed'] or identity['source_tier'] != 'A_exploration_only'
            or digest(folder/'history.parquet') != ready['history_sha256']):
        raise ValueError('Frozen information profile/recipe/snapshot changed')
    history = pd.read_parquet(folder/'history.parquet')
    expected = {f'{g}-{f["year"]}.json' for g in groups for f in identity['split']['folds']}
    if not {p.name for p in (folder/'information-outputs').glob('*.json')}.issubset(expected):
        raise ValueError('Unexpected information output; no duplicate/extra origins')
    for group, recipe in recipes.items():
        for fold in identity['split']['folds']:
            name = f'{group}-{fold["year"]}.json'
            decision, output = folder/'information-decisions'/name, folder/'information-outputs'/name
            if decision.exists():
                choice = read_record(decision)
                best = choice['selected']
                if (choice['selection_used_outer'] or best['recipe'] != recipe[0]
                        or best['weight'] not in SHRINKAGE or best['iterations'] != 1
                        or not np.isfinite(best['inner_score'])):
                    raise ValueError('Frozen past information decision changed')
            if not output.exists():
                continue
            if not decision.exists():
                raise ValueError('Information output missing frozen decision')
            row = read_record(output)
            frame = pd.DataFrame(row['records'])
            actual = rows_at(history, fold['origins'])
            if (row['group'] != group or row['year'] != fold['year'] or row['recipe'] != recipe[0]
                    or row['gate_evaluated'] or row['decision_sha256'] != digest(decision)
                    or row['inner_score'] != best['inner_score'] or row['weight'] != best['weight']
                    or frame.date.tolist() != fold['origins']
                    or not np.isfinite(frame[['cotton_close','target_return_5','predicted_return',
                                             'raw_predicted_return','median_return','past_majority_sign']]).all().all()
                    or not np.array_equal(frame.cotton_close, actual.cotton_close)
                    or not np.array_equal(frame.target_return_5, actual.target_return_5)
                    or not np.array_equal(frame.predicted_return, row['weight']*frame.raw_predicted_return)):
                raise ValueError('Frozen information output/origins changed; no silent retraining')
    return ready


def incremental_compare(folder, validate, *, lags, main_arm, controls, wins, prefix):
    ready = validate(folder)
    result = information.compare(folder)
    if result['status'] != 'complete':
        return result
    checks = {}
    for lag in lags:
        checks[str(lag)] = {}
        main = f'{main_arm}_L{lag}'
        for control_arm in controls:
            control = f'{control_arm}_L{lag}'
            pairs = [(read_record(folder/'information-outputs'/f'{control}-{f["year"]}.json'),
                      read_record(folder/'information-outputs'/f'{main}-{f["year"]}.json'))
                     for f in ready['identity']['split']['folds']]
            inner_gain = 100*(1-np.mean([b['inner_score'] for _,b in pairs])/np.mean([a['inner_score'] for a,_ in pairs]))
            outer_wins = 0
            for a,b in pairs:
                losses = []
                for row in (a,b):
                    frame = pd.DataFrame(row['records'])
                    losses.append(float(np.mean(frame.cotton_close*np.abs(np.exp(frame.target_return_5)-np.exp(frame.predicted_return)))))
                outer_wins += losses[1] < losses[0]
            inner_wins = sum(b['inner_score'] < a['inner_score'] for a,b in pairs)
            outer_gain = 100*(1-result['groups'][main]['price_mae']/result['groups'][control]['price_mae'])
            checks[str(lag)][control] = {'inner_gain_pct':float(inner_gain),'inner_year_wins':int(inner_wins),
                'outer_gain_pct':float(outer_gain),'outer_year_wins':int(outer_wins),
                'priority_passed':bool(inner_gain >= .5 and inner_wins >= wins and outer_gain > 0 and outer_wins >= wins)}
    result = {**result,'priority_checks':checks,
              'research_priority_passed':all(c['priority_passed'] for cs in checks.values() for c in cs.values()),
              'next_action':'Review registered hypothesis; no automatic larger search, export or source admission'}
    freeze_record(folder/'reports'/f'{prefix}-decision.json',result)
    return result


def dispatch_fixed(args, *, prepare, validate, compare):
    from cottonlens_ml.research.engine import Experiment, root_path
    from cottonlens_ml.research.mirror import Mirror
    if args.stage not in ('prepare','status','pilot-plan','pilot','compare','report'):
        raise ValueError('Tier-A fixed profile cannot lock/export/search')
    folder = root_path(args.drive_root,args.experiment)
    mirror = Mirror(folder,root_path(args.mirror_root,args.experiment)) if args.mirror_root else None
    if mirror:
        mirror.hydrate_metadata()
    if args.stage == 'prepare':
        ready = prepare(folder)
        result = {'status':'prepared','fit_budget':ready['identity']['fit_budget']}
        if mirror:
            mirror.publish_metadata(['ready.json','history.parquet','preregistered.json'])
    elif args.stage == 'status':
        result = {'prepared':(folder/'ready.json').exists(),
                  'saved_outputs':len(list((folder/'information-outputs').glob('*.json'))),
                  'local_ledger_inventory':Ledger(folder/'ledger',{}).summary(),
                  'inventory_scope':'local cache; metadata-only restore omits fitted payloads',
                  'writer_lock_present':(folder/'.writer-lock').exists(),'release_allowed':False}
    elif args.stage == 'pilot-plan':
        result = validate(folder)['identity']['fit_budget']
    elif args.stage == 'pilot':
        ready = validate(folder)
        if mirror and len(list((folder/'information-outputs').glob('*.json'))) != ready['identity']['fit_budget']['annual_outputs']:
            mirror.hydrate()
            validate(folder)  # Restored payload/decision packages cannot bypass frozen output checks.
        with writer(folder):
            experiment = Experiment(folder,repo=args.repo)
            if mirror:
                experiment.mirror, experiment.after_fit = mirror, mirror.fit
            result = information.run(experiment,args.max_minutes)
            if mirror:
                members = [p.relative_to(folder).as_posix() for d in ('information-decisions','information-outputs')
                           for p in (folder/d).glob('*.json')]
                if members:
                    mirror.publish_metadata(members)
    else:
        result = compare(folder)
        if mirror and result['status'] == 'complete':
            mirror.publish_metadata([p.relative_to(folder).as_posix() for p in (folder/'reports').glob('*.json')])
    print(json.dumps(result,indent=2))
