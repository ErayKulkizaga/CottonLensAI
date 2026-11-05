"""Fixed-design noisy risk recovery; existing learner/ledger, no market fits."""

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
from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
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
from cottonlens_ml.runtime_guard import require_training

from research import ams_named_ewma_har as parent

PROFILE = 'risk-weak-teacher-t1-v1'
SEEDS = (42, 1701, 2703)
CONDITIONS = {'null': 0., 'weak': .15}
HELPERS = ('research/risk_weak_teacher.py', 'research/tests/test_risk_weak_teacher.py',
           'research/ams_named_ewma_har.py', 'research/ams_named_readiness.py',
           'research/ams_upland_futures.py')
VERSIONS = ('numpy', 'pandas', 'scipy', 'pyarrow')


def source(repo):
    body = {'ml': research_source_identity(repo), 'helpers': {p: digest(repo/p) for p in HELPERS}}
    return {**body, 'source_id': content_id(body)}


def calibrate(history, dates):
    if history.date.duplicated().any():
        raise ValueError('Unique ordinal origins required')
    rows = parent.projection(history, dates, training=True)
    x = rows.own_har_5.to_numpy()-np.log(rows.named_ewma.to_numpy())
    noise = (rows.named_variance/rows.named_ewma).to_numpy()
    if (not np.isfinite([x, noise]).all() or (noise < 0).any()
            or noise.mean() <= 0 or x.std() <= 1e-12):
        raise ValueError('Finite nondegenerate past calibration required')
    scale, mean = float(x.std()), float(x.mean())
    beta = CONDITIONS['weak']
    offset = float(np.log(np.exp(beta*(x-mean)/scale).mean()))
    return {'dates': dates, 'feature_mean': mean, 'feature_scale': scale,
            'weak_log_normalizer': offset, 'innovation_pool': (noise/noise.mean()).tolist(),
            'noise_source': 'earliest mature training labels only; IID empirical resampling',
            'feature': 'own_har_5-log(named_ewma)', 'beta': beta}


def simulate(history, calibration, condition, seed):
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError('Unregistered teacher condition or seed')
    h = history.copy()
    x = h.own_har_5.to_numpy()-np.log(h.named_ewma.to_numpy())
    available = h.own_feature_ready & h.named_ewma_eligible
    s = (x-calibration['feature_mean'])/calibration['feature_scale']
    eta = (CONDITIONS[condition]*s-calibration['weak_log_normalizer']
           if condition == 'weak' else np.zeros(len(h)))
    mean = h.named_ewma.to_numpy()*np.exp(eta)
    innovation = np.random.default_rng(seed).choice(calibration['innovation_pool'], len(h))
    if not np.isfinite(mean[available]).all() or (mean[available] <= 0).any():
        raise ValueError('Teacher left finite variance domain')
    h['teacher_mean'] = np.where(available, mean, np.nan)
    h['teacher_innovation'] = innovation
    valid = h.named_eligible & available
    h.loc[valid, 'named_variance'] = mean[valid]*innovation[valid]
    return h


def model_rows(history, dates, training):
    # The learner never receives the latent mean or innovation as an input.
    return parent.projection(history, dates, training=training).drop(
        columns=['teacher_mean', 'teacher_innovation'])


def expected_gain(mean, baseline, predicted):
    mean, baseline, predicted = [np.asarray(a, float) for a in (mean, baseline, predicted)]
    if (mean.shape != baseline.shape or mean.shape != predicted.shape
            or not np.isfinite([mean, baseline, predicted]).all()
            or (mean <= 0).any() or (baseline <= 0).any() or (predicted <= 0).any()):
        raise ValueError('Finite positive aligned means and variances required')
    return np.log(baseline/predicted)+mean/baseline-mean/predicted


def spec(job, train, condition, seed):
    recipe = {**parent.learner.RECIPE, 'synthetic_condition': condition, 'synthetic_seed': seed}
    return {'recipe': recipe, 'kind': 'synthetic_risk_weak_teacher',
            'role': f'synthetic-{condition}-{seed}-{job["role"]}-{job["bucket"]}', 'iterations': 1,
            'train_dates': job['train_dates'], 'test_dates': job['origins'],
            'validation_dates': [], 'train_identity': frame_identity(train, list(train)),
            'repeat_reason': None}


def frozen_inputs(root):
    files = [p for section in ('inputs', 'cases') for p in (root/section).rglob('*') if p.is_file()]
    files += [root/name for name in ('calibration.json', 'decision-contract.json',
                                     'history-check.json', 'source-manifest.json')]
    return {p.relative_to(root).as_posix(): digest(p) for p in files}


def prepare(repo, root, reference):
    repo, root, reference = map(Path, (repo, root, reference))
    if root.exists():
        raise ValueError('Preserve existing namespace')
    validate(repo/'research')
    old, history = parent.verify(reference, repo)
    related = check(*load_registry(repo/'research'), family='har', horizon=1, query='synthetic')
    folds = old['identity']['design']['folds']
    first = min((j for f in folds for j in f['jobs']), key=lambda j: j['cutoff'])
    calibration = calibrate(history, first['train_dates'])
    cutoff = pd.Timestamp(first['cutoff'])
    if not (history.loc[history.date.isin(pd.to_datetime(calibration['dates'])), 'target_date_5'] < cutoff).all():
        raise ValueError('Calibration labels not mature before every job')
    design = {'profile': PROFILE, 'folds': folds, 'recipe': parent.learner.RECIPE,
              'seeds': list(SEEDS), 'conditions': CONDITIONS, 'calibration_cutoff': first['cutoff'],
              'fit_budget': {'market': 0, 'synthetic': len(SEEDS)*len(CONDITIONS)*sum(len(f['jobs']) for f in folds),
                             'processes': 1, 'threads': 2, 'session_minutes': 30},
              'primary': 'positive noise-averaged raw gain for all3weak seeds and mean oracle capture>=0.5',
              'secondary': 'past-selected expected gain, realized losses and null false apparent wins',
              'fixed_design': True, 'baseline': 'frozen origin named EWMA state, not recomputed from synthetic returns',
              'noise': 'IID resampling normalized earliest619mature empirical squared-return/EWMA ratios',
              'teacher': 'EWMA*exp(.15*standardized relativeHAR5-past lognormalizer); null=EWMA',
              'limitations': 'conditional fixed-design recovery; no calibrated real-market power or endogenous price process',
              'raw_price_data_changed': False, 'market_skill': False, 'release_allowed': False,
              'historical_availability_verified': False, 'independent_holdout': False}
    with writer(root):
        for name, origin in {'history.parquet': reference/'inputs/history.parquet',
                             'parent-ready.json': reference/'ready.json',
                             'parent-report.json': reference/'reports/result.json',
                             'parent-source.json': reference/'source-manifest.json',
                             'bootstrap.py': reference/'inputs/bootstrap.py'}.items():
            copy_immutable(origin, root/'inputs'/name)
        freeze_record(root/'calibration.json', calibration)
        freeze_record(root/'decision-contract.json', design)
        freeze_record(root/'history-check.json', {'profiles': [r['profile'] for r in related['experiments']],
                                                 'related_recipes': len(related['fit_recipes']), 'new_fits': 0})
        ids = {}
        for condition in CONDITIONS:
            for seed in SEEDS:
                h = simulate(history, calibration, condition, seed)
                name = f'{condition}-{seed}'
                path = root/'cases'/f'{name}.parquet'; path.parent.mkdir(exist_ok=True)
                h.to_parquet(path, index=False)
                ids[name] = frame_identity(h, list(h))
        manifest = source(repo)
        freeze_record(root/'source-manifest.json', manifest)
        identity = {'profile': PROFILE, 'source_id': manifest['source_id'], 'design_id': content_id(design),
                    'case_data_ids': ids, 'research_data_id': content_id(ids),
                    'split': {'design_id': content_id(design)}, 'calibration_id': content_id(calibration),
                    'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in VERSIONS},
                    'lock_sha256': digest(repo/'ml/uv.lock')}
        hashes = frozen_inputs(root)
        freeze_record(root/'ready.json', {'identity': identity, 'frozen_hashes': hashes})
    return {'status': 'prepared', 'new_fits': 0, 'fit_budget': design['fit_budget'],
            'source_id': manifest['source_id'], 'design_id': identity['design_id']}


def verify(root, repo):
    root, repo = Path(root), Path(repo)
    ready = read_record(root/'ready.json'); identity = ready['identity']
    if (identity['source_id'] != source(repo)['source_id']
            or identity['python'] != sys.version.split()[0]
            or identity['versions'] != {p: importlib.metadata.version(p) for p in VERSIONS}
            or identity['lock_sha256'] != digest(repo/'ml/uv.lock')):
        raise ValueError('Frozen source/environment changed')
    for name, checksum in ready['frozen_hashes'].items():
        if not safe_member(name) or digest(root/name) != checksum:
            raise ValueError('Frozen input/teacher changed')
    design = read_record(root/'decision-contract.json'); calibration = read_record(root/'calibration.json')
    history = pd.read_parquet(root/'inputs/history.parquet')
    if content_id(design) != identity['design_id'] or calibrate(history, calibration['dates']) != calibration:
        raise ValueError('Design/calibration changed')
    for name, data_id in identity['case_data_ids'].items():
        condition, seed = name.split('-')
        actual = pd.read_parquet(root/'cases'/f'{name}.parquet')
        if frame_identity(actual, list(actual)) != data_id:
            raise ValueError('Teacher identity changed')
        pd.testing.assert_frame_equal(simulate(history, calibration, condition, int(seed)), actual, check_exact=True)
    return ready, design


def run(root, repo, authorization, minutes):
    require_training('har', 'cpu')
    ready, design = verify(root, repo)
    auth = read_record(authorization)
    if (auth.get('ready_sha256') != digest(root/'ready.json')
            or auth.get('source_id') != ready['identity']['source_id']
            or auth.get('design_id') != ready['identity']['design_id']
            or auth.get('fresh_restore_verified') is not True
            or auth.get('fit_budget') != design['fit_budget'] or not 0 < minutes <= 30):
        raise ValueError('Frozen recovery/budget authorization required')
    deadline = time.monotonic()+minutes*60
    with writer(root):
        ledger = Ledger(root/'ledger', ready['identity'])
        used, keys = fit_consumption(ledger.root)
        if used != len(keys) or used > design['fit_budget']['synthetic']:
            raise ValueError('Failed/excess attempt; no silent retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('Checkpoint session boundary')
            if count[0] >= design['fit_budget']['synthetic']:
                raise ValueError('Frozen fit budget exhausted')
            count[0] += 1

        try:
            for condition in CONDITIONS:
                for seed in SEEDS:
                    h = pd.read_parquet(root/'cases'/f'{condition}-{seed}.parquet')
                    for fold in design['folds']:
                        for job in fold['jobs']:
                            train = model_rows(h, job['train_dates'], True)
                            test = model_rows(h, job['origins'], False)
                            ledger.run(spec(job, train, condition, seed),
                                       lambda p, tr=train, te=test: parent.learner.fit_offset(tr, te, p),
                                       before_compute=before_compute)
        except FitBudgetReached:
            return {'status': 'planned_pause', 'synthetic_fits': count[0], 'market_fits': 0}
    return {'status': 'completed', 'synthetic_fits': count[0], 'market_fits': 0}


def native_record(ledger, job, train, test, condition, seed):
    key = ledger.key(spec(job, train, condition, seed))
    record = read_record(ledger.root/'completed'/f'{key}.json')
    if record['identity'] != ledger.identity or record['specification'] != spec(job, train, condition, seed):
        raise ValueError('Native identity/specification mismatch')
    for name, checksum in record['files'].items():
        if not safe_member(name) or digest(ledger.root/name) != checksum:
            raise ValueError('Native payload hash mismatch')
    model = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_text())
    if model['kind'] != 'ewma_offset_har' or model['recipe'] != parent.learner.RECIPE:
        raise ValueError('Native model differs')
    cols = parent.learner.RECIPE['features']
    x = train[cols].to_numpy()-np.log(train.named_ewma.to_numpy())[:, None]
    eta = np.column_stack([np.ones(len(train)), x])@model['coef']
    ratio = (train.named_variance/train.named_ewma).to_numpy()*np.exp(-eta)
    optimization = model['optimization']
    z = (x-optimization['training_feature_mean'])/optimization['training_feature_scale']
    gradient = np.column_stack([np.ones(len(train)), z]).T@(1-ratio)/len(train)
    if (not optimization['success'] or np.max(np.abs(gradient)) > 1.001e-6
            or not math.isclose(float(np.mean(eta+ratio)), optimization['final_training_qlike'], abs_tol=1e-10)):
        raise ValueError('Native optimizer certificate failed')
    x = test[cols].to_numpy()-np.log(test.named_ewma.to_numpy())[:, None]
    predicted = test.named_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), x])@model['coef'])
    np.testing.assert_allclose(predicted, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
    if record['result']['origins'] != job['origins']:
        raise ValueError('Native test origin mismatch')
    return record


def compare(root, repo):
    ready, design = verify(root, repo)
    ledger = Ledger(root/'ledger', ready['identity'])
    used, complete = fit_consumption(ledger.root)
    if used != len(complete) or used != design['fit_budget']['synthetic']:
        raise ValueError('Incomplete/failed control is not a result')
    loader = importlib.util.spec_from_file_location('fixed_risk_bootstrap', root/'inputs/bootstrap.py')
    helper = importlib.util.module_from_spec(loader); loader.loader.exec_module(helper)
    entries, summaries, ids = [], [], set()
    for condition in CONDITIONS:
        for seed in SEEDS:
            h = pd.read_parquet(root/'cases'/f'{condition}-{seed}.parquet')
            rows, weights = [], {}
            for fold in design['folds']:
                records = []
                for job in fold['jobs']:
                    train, test = model_rows(h, job['train_dates'], True), model_rows(h, job['origins'], False)
                    r = native_record(ledger, job, train, test, condition, seed)
                    records.append(r); ids.add(r['experiment_id'])
                out = parent.parent.output(h, fold, 'named', records)
                weights[str(fold['year'])] = out['selection']['weight']
                rows.extend({**r, 'year': fold['year']} for r in out['records'])
            frame = pd.DataFrame(rows)
            means = h.set_index(h.date.dt.strftime('%Y-%m-%d')).teacher_mean
            frame['teacher_mean'] = means.loc[frame.date].to_numpy()
            frame['condition'], frame['seed'] = condition, seed
            if len(frame) != 2006 or int(frame.eligible.sum()) != 1989:
                raise ValueError('Original ordinal cohort changed')
            valid = frame.eligible
            summary = {'condition': condition, 'seed': seed, 'yearly_weights': weights, 'metrics': {}}
            for arm in ('raw', 'selected', 'teacher_mean'):
                mean, baseline, predicted = (frame.loc[valid, c].to_numpy() for c in ('teacher_mean', 'baseline', arm))
                actual = frame.loc[valid, 'named_actual'].to_numpy()
                expected = expected_gain(mean, baseline, predicted)
                realized = np.log(baseline/predicted)+actual/baseline-actual/predicted
                frame.loc[valid, arm+'_expected_gain'] = expected
                frame.loc[valid, arm+'_realized_gain'] = realized
                if condition == 'null' and (expected > 1e-10).any():
                    raise ValueError('Null structural-mean gain cannot be positive')
                limits = {}
                for kind, values in [('expected', expected), ('realized', realized)]:
                    offset = max(1., 1-float(values.min()))
                    tmp = frame[['year']].copy()
                    tmp['a'] = np.where(valid, frame[arm+'_'+kind+'_gain'].fillna(0)+offset, 0)
                    tmp['b'] = np.where(valid, offset, 0); tmp['weight'] = valid.astype(int)
                    parts = [g[['a', 'b', 'weight']].to_numpy() for _, g in tmp.groupby('year')]
                    limits[kind] = {str(b): helper.weighted_bootstrap(parts, block=b)['difference_ci_95'] for b in (20, 60)}
                summary['metrics'][arm] = {'expected_gain': float(expected.mean()), 'realized_gain': float(realized.mean()),
                                            'gain_ci': limits,
                                            'positive_expected_years': int(sum(g[arm+'_expected_gain'].mean() > 0
                                                                             for _, g in frame.loc[valid].groupby('year')))}
            oracle = summary['metrics']['teacher_mean']['expected_gain']
            summary['raw_oracle_capture'] = summary['metrics']['raw']['expected_gain']/oracle if oracle > 0 else None
            summaries.append(summary); entries.append(frame)
    if ids != set(complete):
        raise ValueError('Control coverage differs from ledger')
    weak = [s for s in summaries if s['condition'] == 'weak']
    passed = (all(s['metrics']['raw']['expected_gain'] > 0 for s in weak)
              and np.mean([s['raw_oracle_capture'] for s in weak]) >= .5)
    result = {'profile': PROFILE, 'status': 'completed', 'new_market_fits': 0,
              'native_synthetic_fits': len(ids), 'recovery_passed': bool(passed),
              'cases': summaries, 'market_skill': False, 'release_allowed': False,
              'scope': 'fixed-design/IID earliest-training empirical noise; not real-market power calibration',
              'source_id': ready['identity']['source_id'], 'design_id': ready['identity']['design_id']}
    dest = root/'reports'; dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists() or (dest/'paired-gains.csv').exists():
        raise ValueError('Preserve prior report')
    pd.concat(entries, ignore_index=True).to_csv(dest/'paired-gains.csv', index=False, lineterminator='\n')
    freeze_record(dest/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    for name in ('repo', 'root', 'reference', 'authorization'):
        parser.add_argument('--'+name, type=Path, required=name in ('repo', 'root'))
    parser.add_argument('--minutes', type=float, default=30)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.repo, args.root, args.reference)
    elif args.stage == 'verify':
        ready, design = verify(args.root, args.repo)
        result = {'status': 'verified', 'new_fits': 0, 'fit_budget': design['fit_budget']}
    elif args.stage == 'pilot':
        result = run(args.root, args.repo, args.authorization, args.minutes)
    else:
        result = compare(args.root, args.repo)
    print(json.dumps(result))
