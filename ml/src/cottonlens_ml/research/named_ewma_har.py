"""Fixed four-parameter QLIKE HAR residual over the available named EWMA state."""
import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import named_risk_history as parent
from cottonlens_ml.research import named_risk_loss as loss
from cottonlens_ml.research.contract_curve_execution import fit_consumption
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer
from cottonlens_ml.research.recency import copy_immutable
from cottonlens_ml.research.volatility import FLOOR, qlike
from cottonlens_ml.runtime_guard import require_training

PROFILE = 'named-ewma-har-control-t1-v1'
RECIPE = {'family': 'har', 'horizon': 1, 'device': 'cpu', 'window': 1,
          'features': ['har_1', 'har_5', 'har_22'], 'training_loss': 'qlike',
          'feature_formula': 'own_har_k-log(named_ewma)',
          'target_formula': 'mature_named_actual/origin_named_ewma',
          'forecast_formula': 'origin_named_ewma*exp(intercept+relative_har@coef)',
          'parameters': 4, 'seed': 42, 'profile': PROFILE}


def fit_offset(train, test, workspace):
    from cottonlens_ml.research.volatility import fit_predict
    require_training('har', 'cpu')
    transformed = []
    for original, training in [(train, True), (test, False)]:
        rows = original.copy()
        baseline = rows.named_ewma.to_numpy(float)
        if not np.isfinite(baseline).all() or (baseline < FLOOR).any():
            raise ValueError('Finite positive available named EWMA required')
        for k in (1, 5, 22):
            rows[f'har_{k}'] = rows[f'har_{k}']-np.log(baseline)
        if training:
            rows['variance_target_1'] = rows.variance_target_1/baseline
        transformed.append(rows)
    ratio_root = workspace/'ratio-learner'; ratio_root.mkdir()
    r = fit_predict(transformed[0], transformed[0], transformed[1], RECIPE, ratio_root)
    model = json.loads((ratio_root/'variance-model.json').read_bytes())
    x = np.column_stack([np.ones(len(test)), transformed[1][RECIPE['features']]])
    ratio = np.exp(x@model['coef'])
    if (ratio < FLOOR).any() or not np.isfinite(ratio).all():
        raise ValueError('No silent ratio output clipping')
    np.testing.assert_array_equal(ratio, r['variance_predictions'])
    predicted = test.named_ewma.to_numpy()*ratio
    if (predicted < FLOOR).any() or not np.isfinite(predicted).all():
        raise ValueError('No silent absolute variance clipping')
    model = {**model, 'kind': 'ewma_offset_har', 'recipe': RECIPE, 'ratio_smearing': model['smearing'],
             'baseline_in_model': 'named_ewma at each origin', 'effective_parameters': 4}
    (workspace/'variance-model.json').write_text(json.dumps(model), encoding='utf-8')
    return {**r, 'variance_predictions': predicted.tolist()}


def design(old):
    folds = old['identity']['design']['folds']
    return {'profile': PROFILE, 'recipe': RECIPE, 'folds': folds,
            'fit_budget': {'market': sum(len(f['jobs']) for f in folds), 'synthetic': 1,
                           'processes': 1, 'threads': 2, 'session_minutes': 30},
            'primary': 'raw EWMA-offset HAR versus original named EWMA; same four parameter capacity as prior standalone QLIKE HAR',
            'secondary': 'same past-only 0/.25/.5/.75/1 shrinkage and top1pct loss sensitivity',
            'unchanged': ['origin/labels', 'mature train/validation', 'reference EWMA',
                          '21-session refit', '500 minimum train', '246 ordinal slots'],
            'no_search': True, 'deliberate_repeat_reason': 'Prior fixed-HAR replaces EWMA with incomplete state; intercept-only EWMA scale already failed. Test conditional relative-HAR correction at same capacity, no new source', 'clock_vintage_verified': False, 'independent_holdout': False,
            'eight_year_gate_allowed': False, 'release_allowed': False,
            'sparse_calendar_limit_preserved': True}


def prepare(repo, folder, reference):
    repo, folder, reference = map(Path, (repo, folder, reference))
    if folder.exists():
        raise ValueError('New immutable namespace required')
    with writer(folder):
        old = read_record(reference/'ready.json')
        for name in old['identity']['inputs_sha256']:
            copy_immutable(reference/name, folder/name)
        h, previous, _ = loss.verify_reference(folder, repo)
        d = design(previous)
        identity = {'profile': PROFILE, 'source_id': research_source_identity(repo)['source_id'],
                    'design': d, 'design_id': content_id(d), 'split': previous['identity']['split'],
                    'research_data_id': frame_identity(h, list(h)), 'python': sys.version.split()[0],
                    'versions': previous['identity']['versions'], 'lock_sha256': digest(repo/'ml/uv.lock'),
                    'inputs_sha256': old['identity']['inputs_sha256']}
        h.to_parquet(folder/'history.parquet', index=False)
        freeze_record(folder/'source-manifest.json', research_source_identity(repo))
        freeze_record(folder/'decision-contract.json', d)
        freeze_record(folder/'ready.json', {'identity': identity, 'history_sha256': digest(folder/'history.parquet')})
    return d


def verify(folder, repo):
    folder, repo = Path(folder), Path(repo)
    ready = read_record(folder/'ready.json'); i = ready['identity']
    if (i['profile'] != PROFILE or i['source_id'] != research_source_identity(repo)['source_id']
            or i['python'] != sys.version.split()[0] or i['lock_sha256'] != digest(repo/'ml/uv.lock')
            or i['versions'] != {p: importlib.metadata.version(p) for p in i['versions']}):
        raise ValueError('Source/environment mismatch')
    for name, sha in i['inputs_sha256'].items():
        if digest(folder/name) != sha:
            raise ValueError('Frozen input/reference changed')
    h, old, _ = loss.verify_reference(folder, repo)
    d = design(old)
    if (d != i['design'] or content_id(d) != i['design_id'] or d != read_record(folder/'decision-contract.json')
            or frame_identity(h, list(h)) != i['research_data_id'] or digest(folder/'history.parquet') != ready['history_sha256']):
        raise ValueError('Data/design mismatch')
    pd.testing.assert_frame_equal(h, pd.read_parquet(folder/'history.parquet'), check_exact=True)
    return ready, h


def spec(job, train):
    return {'recipe': RECIPE, 'role': f'offset-{job["role"]}-{job["bucket"]}', 'iterations': 1,
            'train_dates': job['train_dates'], 'test_dates': job['origins'], 'validation_dates': [],
            'train_identity': frame_identity(train, list(train)), 'repeat_reason': None}


def run(folder, repo, authorization, minutes=30):
    require_training('har', 'cpu')
    folder = Path(folder); ready, h = verify(folder, repo); d = ready['identity']['design']
    a = read_record(authorization)
    if (a.get('ready_sha256') != digest(folder/'ready.json') or a.get('source_id') != ready['identity']['source_id']
            or a.get('design_id') != ready['identity']['design_id'] or a.get('fresh_restore_verified') is not True
            or a.get('fit_budget') != d['fit_budget'] or not 0 < minutes <= 30):
        raise ValueError('Frozen authorization/fresh recovery/budget required')
    deadline = time.monotonic()+minutes*60
    with writer(folder):
        learning = Ledger(folder/'learning/ledger', {**ready['identity'], 'role': 'synthetic_not_market'})
        n, keys = fit_consumption(learning.root)
        if n != len(keys) or n > 1:
            raise ValueError('Failed/excess synthetic fit; no retry')
        rng = np.random.default_rng(42)
        baseline = np.exp(rng.uniform(-12, -4, 1200)); x = rng.uniform(-3, 3, (1200, 3))
        f = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1200), 'named_ewma': baseline})
        for i, k in enumerate((1, 5, 22)):
            f[f'har_{k}'] = np.log(baseline)+x[:, i]
        f['variance_target_1'] = baseline*np.exp(.1+x@np.array([.2, .1, -.1]))
        r = learning.run({'recipe': RECIPE, 'role': 'synthetic_ewma_offset_har', 'kind': 'known_signal'},
                         lambda w: fit_offset(f.iloc[:1000], f.iloc[1000:], w))
        np.testing.assert_allclose(r['result']['variance_predictions'], f.variance_target_1.iloc[1000:], rtol=1e-5)
        freeze_record(folder/'learning-control.json', {'status': 'passed', 'fit_id': r['experiment_id'], 'market_skill': False})
        ledger = Ledger(folder/'ledger', ready['identity'])
        used, keys = fit_consumption(ledger.root)
        if used != len(keys) or used > d['fit_budget']['market']:
            raise ValueError('Failed/excess market fit; no retry')
        for fold in d['folds']:
            records = []
            for job in fold['jobs']:
                if time.monotonic() >= deadline:
                    return {'status': 'planned_pause'}
                train = parent.projection(h, job['train_dates'], 'named', training=True)
                test = parent.projection(h, job['origins'], 'named', training=False)
                records.append(ledger.run(spec(job, train), lambda w, tr=train, te=test: fit_offset(tr, te, w)))
            freeze_record(folder/'risk-outputs'/f'offset-{fold["year"]}.json', parent.output(h, fold, 'named', records))
    return {'status': 'completed', 'market_fits': d['fit_budget']['market'], 'synthetic_fits': 1}


def compare(folder, repo):
    folder = Path(folder); ready, h = verify(folder, repo); ledger = Ledger(folder/'ledger', ready['identity'])
    rows, factors = [], []
    for fold in ready['identity']['design']['folds']:
        records = []
        for job in fold['jobs']:
            train = parent.projection(h, job['train_dates'], 'named', training=True)
            test = parent.projection(h, job['origins'], 'named', training=False)
            s = spec(job, train); key = ledger.key(s); r = read_record(ledger.root/'completed'/f'{key}.json')
            if r['identity'] != ready['identity'] or r['specification'] != s or r['experiment_id'] != key or not r['files']:
                raise ValueError('Completed fit binding mismatch')
            for name, sha in r['files'].items():
                if digest(ledger.root/name) != sha:
                    raise ValueError('Native scale hash mismatch')
            model = json.loads((ledger.root/r['payload_root']/'variance-model.json').read_bytes())
            if model['kind'] != 'ewma_offset_har' or model['recipe'] != RECIPE or model['effective_parameters'] != 4:
                raise ValueError('Native anchored learner identity differs')
            x = test[RECIPE['features']].to_numpy()-np.log(test.named_ewma.to_numpy())[:, None]
            expected = test.named_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), x])@model['coef'])
            np.testing.assert_array_equal(r['result']['variance_predictions'], expected)
            raw = train[RECIPE['features']].to_numpy()-np.log(train.named_ewma.to_numpy())[:, None]
            z = (raw-model['optimization']['training_feature_mean'])/model['optimization']['training_feature_scale']
            eta = np.column_stack([np.ones(len(train)), raw])@model['coef']
            ratio = (train.named_variance/train.named_ewma).to_numpy()*np.exp(-eta)
            gradient = np.column_stack([np.ones(len(train)), z]).T@(1-ratio)/len(train)
            if not model['optimization']['success'] or np.max(np.abs(gradient)) > 1.001e-6:
                raise ValueError('Anchored optimizer gradient certificate fails')
            records.append(r); factors.append({'cutoff': job['cutoff'], 'fit_id': key,
                                               'gradient_inf_norm': float(np.max(np.abs(gradient))),
                                               'iterations': model['optimization']['iterations']})
        out = parent.output(h, fold, 'named', records)
        if out != read_record(folder/'risk-outputs'/f'offset-{fold["year"]}.json'):
            raise ValueError('Forecast/selection mismatch')
        for r in out['records']:
            row = {**r, 'year': fold['year']}
            if row['eligible']:
                for key in ('baseline', 'raw', 'selected'):
                    row[key+'_loss'] = float(qlike(np.array([row['named_actual']]), np.array([row[key]]))[0])
            rows.append(row)
    frame = pd.DataFrame(rows); valid = frame.loc[frame.eligible]
    loader = importlib.util.spec_from_file_location('frozen_scale_blocks', folder/'inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(loader); loader.loader.exec_module(bootstrap)
    comparisons = {}
    for arm in ('raw', 'selected'):
        a, b = 'baseline_loss', arm+'_loss'; offset = max(0., 1-min(valid[a].min(), valid[b].min()))
        sample = frame[['year']].copy()
        sample['a'] = np.where(frame.eligible, frame[a].fillna(0)+offset, 0)
        sample['b'] = np.where(frame.eligible, frame[b].fillna(0)+offset, 0)
        sample['weight'] = frame.eligible.astype(int)
        parts = [g[['a', 'b', 'weight']].to_numpy() for _, g in sample.groupby('year')]
        intervals = {str(block): bootstrap.weighted_bootstrap(parts, block=block)['difference_ci_95'] for block in (20, 60)}
        trimmed = valid.drop(valid.nlargest(max(1, math.ceil(len(valid)*.01)), 'baseline_loss').index)
        comparisons[arm] = {'delta': float((valid[a]-valid[b]).mean()), 'ci': intervals,
                            'trim_top1pct': float((trimmed[a]-trimmed[b]).mean())}
    report = {'profile': PROFILE, 'new_fits_during_report': 0, 'optimizer_certificates': factors, 'comparisons': comparisons,
              'retained_origins': len(frame), 'eligible_origins': len(valid), 'independent_holdout': False,
              'release_allowed': False, 'eight_year_gate_allowed': False, 'clock_vintage_verified': False}
    dest = folder/'reports'; dest.mkdir(exist_ok=True)
    if (dest/'paired-losses.csv').exists():
        raise ValueError('Immutable report exists')
    frame.to_csv(dest/'paired-losses.csv', index=False, lineterminator='\n')
    report['paired_losses_sha256'] = digest(dest/'paired-losses.csv')
    freeze_record(dest/'result.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    for name in ('repo', 'root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reference', type=Path); p.add_argument('--authorization', type=Path)
    a = p.parse_args()
    if a.stage == 'prepare':
        if a.reference is None:
            p.error('reference required')
        out = prepare(a.repo, a.root, a.reference)
    elif a.stage == 'verify':
        out = verify(a.root, a.repo)[0]['identity']['design']
    elif a.stage == 'pilot':
        if a.authorization is None:
            p.error('authorization required')
        out = run(a.root, a.repo, a.authorization)
    else:
        out = compare(a.root, a.repo)
    print(out)


if __name__ == '__main__':
    main()
