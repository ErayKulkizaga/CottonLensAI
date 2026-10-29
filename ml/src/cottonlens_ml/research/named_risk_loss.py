"""One fixed-capacity QLIKE-versus-OLS control; old OLS fits are references."""
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import named_risk_history as parent
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.volatility import FLOOR, qlike

PROFILE = 'named-risk-loss-control-t1-v1'
RECIPE = {**parent.RECIPE, 'feature_arm': 'named', 'training_loss': 'qlike', 'loss_control': PROFILE}
UNCHANGED = ('ml/src/cottonlens_ml/research/models.py', 'ml/src/cottonlens_ml/research/named_risk_history.py',
             'ml/src/cottonlens_ml/research/named_risk_control.py', 'ml/src/cottonlens_ml/research/protocol.py',
             'ml/src/cottonlens_ml/runtime_guard.py')


def ols_body(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_predict')
    test = ast.dump(ast.parse("spec['family'] == 'har'", mode='eval').body)
    nodes = [n for n in ast.walk(func) if isinstance(n, ast.If) and ast.dump(n.test) == test]
    if len(nodes) != 1:
        raise ValueError('Unambiguous unchanged OLS branch required')
    floor = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'FLOOR' for t in n.targets)]
    if len(floor) != 1:
        raise ValueError('Explicit unchanged variance floor required')
    return {'body': [ast.dump(n, include_attributes=False) for n in nodes[0].body],
            'prefix': [ast.dump(n, include_attributes=False) for n in func.body[:2]],
            'floor': ast.dump(floor[0], include_attributes=False)}


def verify_reference(folder, repo):
    """No fake current-source cache identity: validate old source and native fits."""
    folder, repo = Path(folder), Path(repo)
    original = read_record(folder/'inputs/history-ready.json')
    proof = read_record(folder/'inputs/history-proof.json')
    source = read_record(folder/'inputs/history-source.json')
    if (source['source_id'] != proof['source_id'] or original['identity']['source_id'] != source['source_id']
            or source['source_id'] != content_id(source['files'])
            or digest(folder/'inputs/prior-volatility.py') != source['files']['ml/src/cottonlens_ml/research/volatility.py']
            or ols_body(folder/'inputs/prior-volatility.py') != ols_body(repo/'ml/src/cottonlens_ml/research/volatility.py')):
        raise ValueError('Prior OLS implementation/source binding differs')
    for name in UNCHANGED:
        if digest(repo/name) != source['files'][name]:
            raise ValueError('Prior control path changed; do not relabel archived fits')
    old = original['identity']
    if (old['python'] != sys.version.split()[0] or old['lock_sha256'] != digest(repo/'ml/uv.lock')
            or old['versions'] != {p: importlib.metadata.version(p) for p in old['versions']}):
        raise ValueError('Reference environment differs')
    h, prior = parent.load_inputs(folder/'inputs')
    expected = parent.plan(h, prior['identity']['split'])
    if (expected != old['design'] or frame_identity(h, list(h)) != old['research_data_id']
            or digest(folder/'inputs/history-report.json') != proof['result_sha256']):
        raise ValueError('Reference cohort/data/result mismatch')
    reference = Ledger(folder/'reference/ledger', old)
    validated = {}
    for fold in expected['folds']:
        records = []
        for job in fold['jobs']:
            train = parent.projection(h, job['train_dates'], 'named', training=True)
            test = parent.projection(h, job['origins'], 'named', training=False)
            spec = {'recipe': {**parent.RECIPE, 'feature_arm': 'named'},
                    'role': f'history-{fold["year"]}-{job["role"]}-{job["bucket"]}', 'iterations': 1,
                    'train_dates': job['train_dates'], 'test_dates': job['origins'], 'validation_dates': [],
                    'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}
            key = reference.key(spec)
            record = read_record(reference.root/'completed'/f'{key}.json')
            if record['specification'] != spec or record['identity'] != old or record['experiment_id'] != key or not record['files']:
                raise ValueError('Reference native identity mismatch')
            for name, sha in record['files'].items():
                if not safe_member(name) or digest(reference.root/name) != sha:
                    raise ValueError('Reference native payload hash mismatch')
            model = json.loads((reference.root/record['payload_root']/'variance-model.json').read_bytes())
            if model['kind'] != 'log_har' or model['features'] != RECIPE['features']:
                raise ValueError('Wrong reference family')
            pred = np.maximum(np.exp(np.column_stack([np.ones(len(test)), test[RECIPE['features']]])
                                     @ model['coef']) * model['smearing'], FLOOR)
            np.testing.assert_allclose(pred, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
            records.append(record)
        result = parent.output(h, fold, 'named', records)
        if result != read_record(folder/'reference/risk-outputs'/f'named-{fold["year"]}.json'):
            raise ValueError('Prior forecasts/selection differ')
        validated[str(fold['year'])] = result
    return h, original, validated


def design(original):
    prior = original['identity']['design']
    return {'profile': PROFILE, 'recipe': RECIPE, 'folds': prior['folds'], 'comparison': 'raw OLS minus raw QLIKE on same named risk',
            'fit_budget': {'market': sum(len(f['jobs']) for f in prior['folds']), 'synthetic': 1,
                           'threads': 2, 'processes': 1, 'session_minutes': 30},
            'ols_reference_fits': sum(len(f['jobs']) for f in prior['folds']),
            'unchanged': ['named labels', 'three own-history features', 'four effective forecast parameters',
                          'mature train/validation dates', 'refit cadence', 'named EWMA', 'past-only weights', '246 ordinal slots'],
            'optimizer': {'method': 'BFGS', 'maxiter': 500, 'gtol': 1e-6, 'failed_fit': 'no fallback/no automatic retry'},
            'clock_vintage_verified': False, 'independent_holdout': False, 'eight_year_gate_allowed': False,
            'release_allowed': False, 'sparse_calendar_limit_preserved': True,
            'deliberate_repeat_reason': 'Reuse exact completed OLS references; isolate supervised loss without new capacity/data'}


def prepare(repo, folder, evidence):
    repo, folder, evidence = map(Path, (repo, folder, evidence))
    if folder.exists():
        raise ValueError('New immutable namespace required')
    original = read_record(evidence/'execution/ready.json')
    with writer(folder):
        for name in original['identity']['inputs_sha256']:
            copy_immutable(evidence/'execution'/name, folder/name)
        extra = {'history-ready.json': evidence/'execution/ready.json',
                 'history-report.json': evidence/'execution/reports/result.json',
                 'history-source.json': evidence/'execution/source-manifest.json',
                 'history-proof.json': repo/'research/evidence/named-risk-history-result-20261012.json',
                 'prior-volatility.py': evidence/'source-snapshot/ml/src/cottonlens_ml/research/volatility.py'}
        for name, path in extra.items():
            copy_immutable(path, folder/'inputs'/name)
        for fold in original['identity']['design']['folds']:
            output = read_record(evidence/'execution/risk-outputs'/f'named-{fold["year"]}.json')
            copy_immutable(evidence/'execution/risk-outputs'/f'named-{fold["year"]}.json',
                           folder/'reference/risk-outputs'/f'named-{fold["year"]}.json')
            for key in output['fit_ids'].values():
                path = evidence/'execution/ledger/completed'/f'{key}.json'
                record = read_record(path)
                copy_immutable(path, folder/'reference/ledger/completed'/path.name)
                for name in record['files']:
                    copy_immutable(evidence/'execution/ledger'/name, folder/'reference/ledger'/name)
        h, old, _ = verify_reference(folder, repo)
        d = design(old)
        validate(repo/'research')
        reg, trials = load_registry(repo/'research')
        related = check(reg, trials, family='har', horizon=1, query='qlike')
        identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'], 'design': d,
                    'design_id': content_id(d), 'split': old['identity']['split'], 'research_data_id': frame_identity(h, list(h)),
                    'python': sys.version.split()[0], 'versions': old['identity']['versions'],
                    'lock_sha256': digest(repo/'ml/uv.lock'),
                    'inputs_sha256': {p.relative_to(folder).as_posix(): digest(p)
                                     for section in ('inputs', 'reference', 'learning-reference')
                                     for p in (folder/section).rglob('*') if p.is_file()}}
        h.to_parquet(folder/'history.parquet', index=False)
        freeze_record(folder/'source-manifest.json', research_source_identity(repo))
        freeze_record(folder/'history-check.json', {'related_records': [r['id'] for r in related['experiments']], 'new_fits': 0})
        freeze_record(folder/'decision-contract.json', d)
        freeze_record(folder/'ready.json', {'identity': identity, 'history_sha256': digest(folder/'history.parquet')})
    return d


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder/'ready.json');i = ready['identity']
    if (i['profile'] != PROFILE or i['source_id'] != research_source_identity(repo)['source_id']
            or i['python'] != sys.version.split()[0] or i['lock_sha256'] != digest(repo/'ml/uv.lock')
            or i['versions'] != {p: importlib.metadata.version(p) for p in i['versions']}):
        raise ValueError('Source/environment mismatch')
    for name, sha in i['inputs_sha256'].items():
        if digest(folder/name) != sha:
            raise ValueError('Frozen input/reference hash mismatch')
    h, old, controls = verify_reference(folder, repo)
    d = design(old)
    if (d != i['design'] or content_id(d) != i['design_id'] or d != read_record(folder/'decision-contract.json')
            or digest(folder/'history.parquet') != ready['history_sha256'] or frame_identity(h, list(h)) != i['research_data_id']):
        raise ValueError('Cohort/data/contract mismatch')
    pd.testing.assert_frame_equal(h, pd.read_parquet(folder/'history.parquet'), check_exact=True)
    return ready, h, controls


def learning_control(experiment):
    from cottonlens_ml.research.models import fit_predict
    ledger = Ledger(experiment.root/'learning/ledger', {**experiment.identity, 'role': 'synthetic_not_market'})
    n, keys = fit_consumption(ledger.root)
    if n != len(keys) or n > 1:
        raise ValueError('Synthetic failed/excess fit; no silent retry')
    x = np.random.default_rng(42).uniform(-10, -4, (1200, 3))
    a = np.exp(-1 + x @ np.array([.2, .4, .1]))
    frame = pd.DataFrame(x, columns=RECIPE['features'])
    frame['date'] = pd.bdate_range('2000-01-03', periods=1200);frame['variance_target_1'] = a
    train, test = frame.iloc[:1000], frame.iloc[1000:]
    r = ledger.run({'recipe': RECIPE, 'role': 'synthetic_known_qlike_har', 'kind': 'known_signal'},
                   lambda w: fit_predict(frame, train, None, test, RECIPE, w, iterations=1))
    pred = np.asarray(r['result']['variance_predictions'])
    gain = 1-np.abs(pred-a[1000:]).mean()/np.abs(train.variance_target_1.mean()-a[1000:]).mean()
    if gain < .5 or r['result']['origins'] != test.date.dt.strftime('%Y-%m-%d').tolist():
        raise ValueError('QLIKE known-signal control failed')
    freeze_record(experiment.root/'learning-control.json', {'status': 'passed', 'relative_mae_gain': float(gain),
                  'fit_id': r['experiment_id'], 'new_synthetic_fits': 1, 'market_skill': False})


def run(folder, repo, authorization, minutes):
    from cottonlens_ml.research.engine import Experiment
    from cottonlens_ml.runtime_guard import require_training
    require_training('har', 'cpu')
    folder = Path(folder);ready, h, _ = verify(folder, repo);d = ready['identity']['design']
    a = read_record(Path(authorization))
    if (a.get('ready_sha256') != digest(folder/'ready.json') or a.get('source_id') != ready['identity']['source_id']
            or a.get('design_id') != ready['identity']['design_id'] or a.get('fresh_restore_verified') is not True
            or a.get('fit_budget') != d['fit_budget'] or not 0 < minutes <= 30):
        raise ValueError('Frozen authorization/recovery/budget required')
    deadline = time.monotonic()+minutes*60
    with writer(folder):
        e = Experiment(folder, repo=repo);learning_control(e)
        used, keys = fit_consumption(e.ledger.root)
        if used != len(keys) or used > d['fit_budget']['market']:
            raise ValueError('Failed/excess market fits; no automatic retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('Checkpoint pause')
            if count[0] >= d['fit_budget']['market']:
                raise ValueError('Fit budget exhausted')
            count[0] += 1

        e.before_compute = before_compute
        try:
            for fold in d['folds']:
                records = []
                for job in fold['jobs']:
                    train = parent.projection(h, job['train_dates'], 'named', training=True)
                    test = parent.projection(h, job['origins'], 'named', training=False)
                    records.append(e.fit(RECIPE, train, None, test,
                        f'loss-{fold["year"]}-{job["role"]}-{job["bucket"]}', iterations=1))
                result = parent.output(h, fold, 'named', records)
                freeze_record(folder/'risk-outputs'/f'qlike-{fold["year"]}.json', {**result, 'training_loss': 'qlike'})
        except FitBudgetReached:
            return {'status': 'planned_pause', 'market_fits': count[0], 'synthetic_fits': 1}
    return {'status': 'completed', 'market_fits': count[0], 'synthetic_fits': 1, 'reused_ols_fits': d['ols_reference_fits']}


def compare(folder, repo):
    folder = Path(folder);ready, h, controls = verify(folder, repo)
    ledger = Ledger(folder/'ledger', ready['identity'])
    paired, keys, optimizers = [], set(), []
    for fold in ready['identity']['design']['folds']:
        records = []
        for job in fold['jobs']:
            train = parent.projection(h, job['train_dates'], 'named', training=True)
            test = parent.projection(h, job['origins'], 'named', training=False)
            spec = {'recipe': RECIPE, 'role': f'loss-{fold["year"]}-{job["role"]}-{job["bucket"]}', 'iterations': 1,
                    'train_dates': job['train_dates'], 'test_dates': job['origins'], 'validation_dates': [],
                    'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}
            key = ledger.key(spec);r = read_record(ledger.root/'completed'/f'{key}.json')
            if r['identity'] != ready['identity'] or r['specification'] != spec or r['experiment_id'] != key or not r['files']:
                raise ValueError('QLIKE native identity mismatch')
            for name, sha in r['files'].items():
                if not safe_member(name) or digest(ledger.root/name) != sha:
                    raise ValueError('QLIKE native payload mismatch')
            model = json.loads((ledger.root/r['payload_root']/'variance-model.json').read_bytes())
            if model['kind'] != 'log_har' or model['training_loss'] != 'qlike' or model['smearing'] != 1.:
                raise ValueError('Wrong learner/loss/output scaling')
            pred = np.maximum(np.exp(np.column_stack([np.ones(len(test)), test[RECIPE['features']]]) @ model['coef']), FLOOR)
            np.testing.assert_allclose(pred, r['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
            eta = np.column_stack([np.ones(len(train)), train[RECIPE['features']]]) @ model['coef']
            ratio = train.named_variance.to_numpy()*np.exp(-eta)
            z = (train[RECIPE['features']].to_numpy()-model['optimization']['training_feature_mean'])/model['optimization']['training_feature_scale']
            gradient = np.column_stack([np.ones(len(train)), z]).T @ (1-ratio)/len(train)
            if (not model['optimization']['success'] or np.max(np.abs(gradient)) > 1.001e-6
                    or not math.isclose(float(np.mean(eta+ratio)),model['optimization']['final_training_qlike'],abs_tol=1e-10)):
                raise ValueError('Optimizer native reconstruction fails')
            optimizers.append({'fit_id':key,'gradient_inf_norm':float(np.max(np.abs(gradient))),
                               'iterations':model['optimization']['iterations']})
            records.append(r);keys.add(key)
        result = {**parent.output(h, fold, 'named', records), 'training_loss':'qlike'}
        if result != read_record(folder/'risk-outputs'/f'qlike-{fold["year"]}.json'):
            raise ValueError('QLIKE predictions/selection differ')
        old = controls[str(fold['year'])]
        for a, b in zip(old['records'], result['records'], strict=True):
            for name in ('date','target_date','eligible','forecast_available','contract','named_actual','baseline'):
                if a[name] != b[name]:raise ValueError('Exact paired identity required')
            row = {'year':fold['year'],'date':a['date'],'target_date':a['target_date'],'contract':a['contract'],
                   'eligible':a['eligible'],'forecast_available':a['forecast_available'],'actual':a['named_actual'],
                   'baseline_variance':a['baseline'],'ols_raw':a['raw'],'qlike_raw':b['raw'],
                   'ols_selected':a['selected'],'qlike_selected':b['selected'],
                   'own_22_session_span':a['own_22_session_span'],'own_latest_pair_age':a['own_latest_pair_age']}
            if a['eligible']:
                for name, p in [('baseline',a['baseline']),('ols_raw',a['raw']),('qlike_raw',b['raw']),
                                ('ols_selected',a['selected']),('qlike_selected',b['selected'])]:
                    row[name+'_loss'] = float(qlike(np.array([a['named_actual']]), np.array([p]))[0])
            paired.append(row)
    frame = pd.DataFrame(paired);valid = frame.loc[frame.eligible]
    spec = importlib.util.spec_from_file_location('frozen_loss_blocks',folder/'inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
    comparisons = {}
    for name,a,b in [('loss_mechanism','ols_raw','qlike_raw'),('qlike_raw_vs_ewma','baseline','qlike_raw'),
                     ('ols_raw_vs_ewma','baseline','ols_raw'),('qlike_selected_vs_ewma','baseline','qlike_selected')]:
        a,b = a+'_loss',b+'_loss';offset = max(0.,1-min(valid[a].min(),valid[b].min()))
        sample = frame[['year']].copy()
        sample['a'] = np.where(frame.eligible,frame[a].fillna(0)+offset,0)
        sample['b'] = np.where(frame.eligible,frame[b].fillna(0)+offset,0)
        sample['weight'] = frame.eligible.astype(int)
        parts = [g[['a','b','weight']].to_numpy() for _,g in sample.groupby('year')]
        intervals = {}
        for block in (20,60):
            v = bootstrap.weighted_bootstrap(parts,block=block)
            intervals[str(block)] = {k:v[k] for k in ('eligible','ordinal_slots','block','replicates','paired_difference','difference_ci_95')}
        trimmed = valid.drop(valid.nlargest(max(1,math.ceil(len(valid)*.01)),'baseline_loss').index)
        comparisons[name] = {'delta':float((valid[a]-valid[b]).mean()),'bootstrap':intervals,
                            'trim_top1pct_ewma_loss_delta':float((trimmed[a]-trimmed[b]).mean())}
    mechanism = comparisons['loss_mechanism']['bootstrap']
    positive = all(v['difference_ci_95'][0]>0 for v in mechanism.values())
    negative = all(v['difference_ci_95'][1]<=0 for v in mechanism.values())
    report = {'profile':PROFILE,'new_fits_during_report':0,'new_market_fits_verified':len(keys),
              'reused_ols_reference_fits':ready['identity']['design']['ols_reference_fits'],
              'retained_origins':len(frame),'forecast_available':int(frame.forecast_available.sum()),'eligible_origins':len(valid),
              'comparisons':comparisons,'optimizer_certificates':optimizers,
              'classification':'DECISIVE_POSITIVE' if positive else 'DECISIVE_NEGATIVE' if negative else 'INCONCLUSIVE',
              'classification_scope':'Conditional one-year fixed-capacity loss mechanism only',
              'independent_holdout':False,'eight_year_gate_allowed':False,'release_allowed':False,
              'clock_vintage_verified':False,'sparse_calendar_limit_preserved':True}
    dest = folder/'reports';dest.mkdir(exist_ok=True)
    if (dest/'paired-losses.csv').exists():raise ValueError('Immutable report exists')
    frame.to_csv(dest/'paired-losses.csv',index=False,lineterminator='\n')
    report['paired_losses_sha256'] = digest(dest/'paired-losses.csv')
    freeze_record(dest/'result.json',report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    for name in ('root', 'repo'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--evidence', type=Path);p.add_argument('--authorization', type=Path)
    p.add_argument('--max-minutes', type=float, default=30)
    a = p.parse_args()
    if a.stage == 'compare':
        print(compare(a.root,a.repo));return
    if a.stage == 'pilot':
        if a.authorization is None:p.error('authorization required')
        print(run(a.root, a.repo, a.authorization, a.max_minutes));return
    if a.stage == 'prepare':
        if a.evidence is None:p.error('evidence required')
        d = prepare(a.repo, a.root, a.evidence)
    else:
        r, _, _ = verify(a.root, a.repo);d = r['identity']['design']
    print({'new_fits': 0, 'budget': d['fit_budget'], 'reused_ols_fits': d['ols_reference_fits']})


if __name__ == '__main__':
    main()
