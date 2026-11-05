"""Fixed, blinded cross-era replication of Cotton T+1 risk versus EWMA."""

import argparse
import importlib.metadata
import json
import math
import shutil
import time
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.volatility import FLOOR, qlike

from research import ct_manual_forward as forward
from research import ct_pre2010_risk_source as source
from research import ct_shock_control as learner
from research.verify_ct_prospective_readiness import decimal_difference
from research.verify_risk_weak_teacher import difference_interval

PROFILE = 'ct-pre2010-same-contract-risk-t1-r2'
ARM = 'own_level'
FIRST = '2008-11-24'
LAST_TARGET = '2009-11-24'
FIT_BUDGET = 31
WEEKLY = ('https://apps.ams.usda.gov/Cotton/Historical%20Weekly%20Reports/'
          'MY%202008-2009/mp_cn812%20Weekly%20Cot%20Mkt%20Rev/')
LATE = {
    '2009-04-09': {'published': '2009-04-10', 'witness_day': '2009-04-13',
                   'report': 'CN20090410WWWCMR.pdf'},
    '2009-05-22': {'published': '2009-05-29', 'witness_day': '2009-05-26',
                   'report': 'CN20090529WWCMR.pdf'},
}
HELPERS = ('research/ct_pre2010_risk.py', 'research/ct_pre2010_risk_source.py',
           'research/ams_full_daily_source.py', 'research/verify_ct_prospective_readiness.py',
           'research/tests/test_ct_pre2010_risk.py')


def job_plan(dates, origins, releases):
    train = [d for d in dates[119:] if not any(x <= d <= end for x, end in releases.items())]
    next_day = dict(pairwise(dates))
    maturity = dict(zip(dates[:-5], dates[5:], strict=True))

    def eligible(cutoff):
        result = [d for d in train if maturity.get(d, cutoff) < cutoff
                  and max(releases.get(d, d), releases.get(next_day[d], next_day[d])) < cutoff]
        if len(result) < 500:
            raise ValueError(f'Fewer than 500 mature feature origins at {cutoff}')
        return result

    folds, jobs = [], []
    for year in sorted({int(d[:4]) for d in origins}):
        outer = [d for d in origins if int(d[:4]) == year]
        validation = eligible(outer[0])[-189:]
        folds.append({'year': year, 'origins': outer})
        for start in range(0, 189, 21):
            days = validation[start:start+21]
            jobs.append({'year': year, 'role': f'inner_{start//63}', 'bucket': start//21 % 3,
                         'cutoff': days[0], 'train_dates': eligible(days[0]), 'origins': days})
        for start in range(0, len(outer), 21):
            days = outer[start:start+21]
            jobs.append({'year': year, 'role': 'outer', 'bucket': start//21,
                         'cutoff': days[0], 'train_dates': eligible(days[0]), 'origins': days})
    return {'split': {'folds': folds}, 'jobs': jobs}


def required_quotes(dates):
    out = {}
    for i, day in enumerate(dates):
        months = {source.delivery(dates[i-1])} if i else set()
        if i < len(dates)-1:
            months.add(source.delivery(day))
        out[day] = sorted(months)
    return out


def history(dates, rows):
    if set(rows) != set(dates) or dates != sorted(set(dates)):
        raise ValueError('Require exact ordered session coverage')
    levels, realized = [100.], [np.nan]
    checked_ewma, state = [np.nan], None
    with localcontext() as context:
        context.prec = 40
        for origin, target in pairwise(dates):
            month = source.delivery(origin)
            start = Decimal(rows[origin]['quotes'][month])
            finish = Decimal(rows[target]['quotes'][month])
            if not all(v.is_finite() and v > 0 for v in (start, finish)):
                raise ValueError('A named settlement must be finite and positive')
            factor = finish/start
            levels.append(levels[-1]*float(factor))
            squared = factor.ln()**2
            realized.append(float(squared))
            state = squared if state is None else Decimal('.94')*state+Decimal('.06')*squared
            checked_ewma.append(max(float(state), FLOOR))
    h = forward.history(pd.DataFrame({'date': pd.to_datetime(dates), 'cotton_close': levels}))
    np.testing.assert_allclose(h.daily_variance_proxy, realized, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(h.ct_ewma, checked_ewma, rtol=1e-10, atol=1e-14)
    return h, pd.Series(realized[1:]+[np.nan], index=dates), pd.Series(checked_ewma, index=dates)


def environment():
    return {**forward.environment(), 'source_versions': {
        name: importlib.metadata.version(name) for name in ('pypdf', 'xlrd', 'requests')}}


def prepare(repo, audit, root, parent):
    if root.exists():
        raise ValueError('Use an unused research namespace')
    validate(repo/'research')
    registry, trials = load_registry(repo/'research')
    lookup = check(registry, trials, query=PROFILE, horizon=1)
    if lookup['status'] != 'NO_REGISTERED_MATCH':
        raise ValueError('Inspect registered comparable study before execution')
    proof = json.loads((repo/'research/evidence/ct-recent-risk-audit-result-20261104.json').read_bytes())
    anchor = proof['pre2010_training_readiness']
    ext = audit/'pre2010-timing-extension-restored/pre2010-timing-prefix-extension-v1'
    core = audit/'pre2010-training-restored/pre2010-training-only-v1'
    if digest(ext/'timing-feasibility.json') != anchor['timing_feasibility_sha256']:
        raise ValueError('Verified timing evidence differs')
    timing = json.loads((ext/'timing-feasibility.json').read_bytes())
    if digest(core/'screen-v2/training-prices.json') != timing['training_readiness_prices_hash']:
        raise ValueError('Verified training evidence differs')
    # The initial prefix and extension are both read from the previously verified archive.
    extra_manifest = json.loads((audit/'pre2010-timing-extension-fresh'
                                 /'pre2010-timing-prefix-extension-v1-manifest.json').read_bytes())
    extra_path = 'pre2010-timing-prefix-extension-v1/training-quote-extension.json'
    expected = next(f['sha256'] for f in extra_manifest['files'] if f['path'] == extra_path)
    if digest(ext/'training-quote-extension.json') != expected:
        raise ValueError('Verified timing extension prices differ')
    training = json.loads((core/'screen-v2/training-prices.json').read_bytes())
    extension = json.loads((ext/'training-quote-extension.json').read_bytes())
    if len(training) != 818 or len(extension) != 14 or set(training)&set(extension):
        raise ValueError('Training origin coverage differs')
    training.update({d: {'quotes': {m: v['today'] for m, v in row.items()}}
                     for d, row in extension.items()})
    origins = timing['proposed_exact_origins']
    if len(origins) != 252 or origins[0] != FIRST or timing['nominal_252nd_target'] != LAST_TARGET:
        raise ValueError('Fixed evaluation schedule differs')
    prefix = sorted(training)
    if prefix[-1] >= FIRST:
        raise ValueError('An evaluation body is in the preparation input')
    dates = prefix+origins+[LAST_TARGET]
    required = required_quotes(dates)
    for day, row in training.items():
        if not set(required[day]) <= set(row['quotes']):
            raise ValueError(f'Required training price missing at {day}')
    releases = {**timing['weekly_repairs_printed_date'], **{d: x['published'] for d,x in LATE.items()}}
    design = job_plan(dates, origins, releases)
    if len(design['jobs']) != FIT_BUDGET:
        raise ValueError('Changed fixed 31-fit budget')
    metadata = audit/'pre2010-restored/pre2010-central-directory-probe-v1'
    plan = source.source_plan(metadata, origins+[LAST_TARGET])
    control = forward.native_learning(parent, repo)
    snapshot = forward.source(repo)
    names = set(snapshot['ml']['files']) | set(snapshot['helpers']) | set(HELPERS)
    manifest = {name: digest(repo/name) for name in sorted(names)}
    contract = {
        'profile': PROFILE, 'recipe': learner.recipe(ARM, learner.STATE_PROFILE),
        'origins': origins, 'sessions': dates, 'required': required, 'archives': plan,
        'release_proxy': releases, 'late_evaluation_sources': LATE, 'weekly_base': WEEKLY,
        'fits': FIT_BUDGET, 'synthetic_fits': 0,
        'baseline': 'EWMA lambda .94 using same held-contract log returns',
        'target': 'r[t+1]^2 for the nearest scheduled delivery month strictly after the origin month; hold it across both prices',
        'source_policy': 'Dated PDF then same-date workbook. Two known misdated pairs require official dated weekly quote and exact match to following dated daily Previous; all 252 slots retained.',
        'timing': 'Treat normal daily archived settlements as known after that session. Weekly repairs enter only after their printed week issue date. Exclude affected fit feature rows; both forecasts use the last complete EWMA on affected evaluation origins. Require target maturity strictly before cutoff.',
        'selection': 'Fixed annual 3x63 past-only selection on .00/.25/.50/.75/1.00; ties use the smaller weight; refit every 21 origins',
        'statistic': 'Mean paired EWMA QLIKE minus selected candidate QLIKE. Both year-stratified moving-block 20/60 95% lower bounds and mean must be >0. 10000 draws seed42. Trim is descriptive.',
        'amendment': 'R1 structurally preflighted archived quote coverage before any fit or score. Two evaluation documents are misdated; no R1 outcome admission. R2 keeps the same target, 252 origins, learner, EWMA and inference gate.',
        'fits_or_scores_before_amendment': 0,
        'parser_amendment': 'Select named futures table block, then verify both weekly values against subsequent dated daily Previous; no model or inference change.', 'current_market_skill': False, 'first_vintage_verified': False,
        'forward_publication': False,
    }
    with writer(root):
        for name in manifest:
            path = root/'source-snapshot'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo/name, path)
        for name, body in [('training.json',training), ('source-manifest.json',manifest),
                           ('design.json',design), ('contract.json',contract), ('native-control.json',control)]:
            freeze_record(root/name,body)
        identity = {
            'profile':PROFILE, 'source_id':content_id(manifest), 'data_id':content_id(training),
            'design_id':content_id(design), 'contract_id':content_id(contract),
            'control_id':content_id(control), **environment(),
        }
        freeze_record(root/'ready.json',identity)
    return identity


def inputs(root):
    manifest = read_record(root/'source-manifest.json')
    design = read_record(root/'design.json')
    contract = read_record(root/'contract.json')
    identity = read_record(root/'ready.json')
    training = read_record(root/'training.json')
    control = read_record(root/'native-control.json')
    snapshot = root/'source-snapshot'
    if (Path(__file__).resolve() != (snapshot/'research/ct_pre2010_risk.py').resolve()
            or any(digest(snapshot/name) != sha for name, sha in manifest.items())
            or any(identity[k] != v for k,v in environment().items())
            or identity['source_id'] != content_id(manifest) or identity['data_id'] != content_id(training)
            or identity['design_id'] != content_id(design) or identity['contract_id'] != content_id(contract)
            or identity['control_id'] != content_id(control)
            or len(design['jobs']) != FIT_BUDGET or contract['fits'] != FIT_BUDGET):
        raise ValueError('Frozen execution identity changed')
    if job_plan(contract['sessions'],contract['origins'],contract['release_proxy']) != design:
        raise ValueError('Frozen job dates changed')
    return identity, contract, design, training


def outcomes(root, contract, capture):
    folder = root/'outcomes'
    if capture:
        folder.mkdir(exist_ok=True)
        source.capture(folder, contract['archives'])
    data = {}
    hashes = {}
    previous = {}
    for plan in contract['archives']:
        path = folder/(plan['season']+'.bin')
        receipt = read_record(folder/(plan['season']+'-receipt.json'))
        if (receipt['source'] != plan['url'] or receipt['sha256'] != digest(path)
                or receipt['range'] != [plan['start'], plan['stop']]):
            raise ValueError('Outcome capture receipt differs')
        hashes[path.name] = receipt['sha256']
        raw = path.read_bytes()
        data.update(source.normalize(plan,raw,contract['required'],contract['late_evaluation_sources']))
        for day,witness in contract['late_evaluation_sources'].items():
            after = witness['witness_day']
            if after in plan['members']:
                pdf = source.archive_member(raw,plan,plan['members'][after]['pdf'])
                previous[day] = source.pdf_quotes(pdf,after,previous=True)
    for day,witness in contract['late_evaluation_sources'].items():
        path = folder/witness['report']
        raw = source.weekly_payload(path,contract['weekly_base']+path.name,capture)
        hashes[path.name] = digest(path)
        quotes = source.weekly_quotes(raw,day,witness['published'],contract['required'][day])
        if {m:previous[day][m] for m in quotes} != quotes:
            raise ValueError(f'Weekly named quotes differ from subsequent explicit Previous on {day}')
        data[day] = {'quotes':quotes,'member':path.name,'late_release':witness['published']}
    if sorted(data) != contract['origins']+[LAST_TARGET]:
        raise ValueError('Exact paired evaluation dates required')
    return data, hashes


def stale_origins(sessions, origins, late):
    prior = dict(zip(sessions[1:],sessions[:-1],strict=True))
    return {t: prior[day] for day,x in late.items() for t in origins if day<=t<=x['published']}


def paired(h, contract, forecasts, weights, actual, ewma):
    days = contract['origins']
    if list(forecasts) != days:
        raise ValueError('Forecast coverage differs')
    values = np.array([forecasts[d] for d in days])
    y, b = actual.loc[days].to_numpy(), ewma.loc[days].to_numpy()
    g = [decimal_difference(yy, bb, mm) for yy, bb, mm in zip(y,b,values[:,1],strict=True)]
    np.testing.assert_allclose(g, qlike(y,b)-qlike(y,values[:,1]), atol=1e-12)
    frame = pd.DataFrame({
        'origin': days, 'target_date': h.set_index(h.date.dt.strftime('%Y-%m-%d')).loc[days]
        .target_date_1.dt.strftime('%Y-%m-%d').to_list(),
        'contract': [source.delivery(d) for d in days], 'year': [int(d[:4]) for d in days],
        'actual': y, 'ewma': b, 'raw': values[:,0], 'candidate': values[:,1], 'weight': values[:,2],
        'gain': g, 'ewma_loss': qlike(y,b),
    })
    groups = frame.groupby('year')
    parts = [np.column_stack([group.gain,np.ones(len(group))]) for _,group in groups]
    intervals = {str(block):difference_interval(parts,block) for block in (20,60)}
    mean = float(frame.gain.mean())
    trim = frame.drop(frame.nlargest(math.ceil(len(frame)*.01),'ewma_loss').index)
    result = {
        'origins':len(frame), 'fits':FIT_BUDGET, 'mean_gain':mean, 'intervals':intervals,
        'years':{str(k):{'origins':len(v),'gain':float(v.gain.mean())} for k,v in groups},
        'weights':weights, 'raw_mean_gain':float((frame.ewma_loss-qlike(y,values[:,0])).mean()),
        'trimmed_gain':float(trim.gain.mean()), 'fallback_origins':int(frame.weight.eq(0).sum()),
        'stale_evaluation_origins':stale_origins(contract['sessions'], days, contract['late_evaluation_sources']),
        'archival_replication_gate':bool(mean > 0 and all(v[0]>0 for v in intervals.values())),
        'first_vintage_verified':False,'forward_publication':False,'current_market_skill':False,
    }
    return frame, result


def run(root, phase):
    identity, contract, design, training = inputs(root)
    with writer(root):
        data, hashes = outcomes(root,contract,phase=='capture')
        receipt = {'input_id':content_id(data), 'archive_hashes':hashes, 'verified_dates':len(data)}
        if phase == 'capture':
            freeze_record(root/'outcome-admission.json',receipt)
            return receipt
        if read_record(root/'outcome-admission.json') != receipt:
            raise ValueError('Outcome identity changed after admission')
        h, actual, ewma = history(contract['sessions'], {**training, **data})
        stale = stale_origins(contract['sessions'],contract['origins'],contract['late_evaluation_sources'])
        by_date = h.set_index(h.date.dt.strftime('%Y-%m-%d')).copy()
        columns = ['ct_ewma', *forward.NAMES]
        for day, past in stale.items():
            by_date.loc[day,columns] = by_date.loc[past,columns].to_numpy()
            ewma.loc[day] = ewma.loc[past]
        h = by_date.reset_index(drop=True)
        ledger = Ledger(root/'ledger',{**identity, 'outcome_id':receipt['input_id']})
        deadline = time.monotonic()+30*60

        def predict(job):
            train = forward.projection(h,job['train_dates'])
            test = forward.projection(h,job['origins'])
            test['variance_target_1'] = np.nan
            spec = learner.specification(job,train,ARM,learner.STATE_PROFILE)
            if phase == 'fit':
                if time.monotonic() >= deadline:
                    raise TimeoutError('30-minute CPU budget')
                ledger.run(spec,lambda p:learner.fit(train,test,p,ARM,learner.STATE_PROFILE))
            return learner.native(ledger,job,train,test,ARM,learner.STATE_PROFILE)

        forecasts, weights = learner.folded_output(h,design,predict)
        for day in stale:
            forecasts[day] = (float(ewma.loc[day]),float(ewma.loc[day]),0.)
        if ledger.summary()['durably_saved'] != FIT_BUDGET:
            raise ValueError('Require exactly 31 saved native fits')
        frame, result = paired(h,contract,forecasts,weights,actual,ewma)
        path = root/'paired-losses.csv'
        if phase == 'replay':
            pd.testing.assert_frame_equal(pd.read_csv(path,float_precision='round_trip'),frame,check_exact=True)
            if read_record(root/'result.json') != result:
                raise ValueError('Archived result differs')
        else:
            with path.open('x', encoding='utf-8', newline='\n') as file:
                frame.to_csv(file,index=False)
            freeze_record(root/'result.json',result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'capture', 'fit', 'replay'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    result = prepare(Path.cwd(),args.audit,args.root,args.parent) if args.phase == 'prepare' else run(args.root,args.phase)
    print(json.dumps(result, indent=2))
