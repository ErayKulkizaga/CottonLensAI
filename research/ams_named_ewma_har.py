"""One fixed historical named-contract EWMA-offset HAR control on AMS data."""

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
from cottonlens_ml.code_identity import safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research import named_ewma_har as learner
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
from cottonlens_ml.research.volatility import qlike
from cottonlens_ml.runtime_guard import require_training

from research.ams_upland_futures import sha

PROFILE = 'ams-named-ewma-har-full-year-t1-v1'
SOURCE_FILES = ('research/ams_upland_futures.py', 'research/ams_named_readiness.py',
                'research/ams_named_ewma_har.py',
                'ml/src/cottonlens_ml/research/named_ewma_har.py',
                'ml/src/cottonlens_ml/research/named_risk_history.py',
                'ml/src/cottonlens_ml/research/named_risk_control.py',
                'ml/src/cottonlens_ml/research/volatility.py',
                'ml/src/cottonlens_ml/research/ledger.py')
VERSIONS = ('numpy', 'pandas', 'scipy', 'pyarrow')


def source_manifest(repo):
    files = {name: sha(repo/name) for name in SOURCE_FILES}
    return {'files': files, 'source_id': content_id(files)}


def ledger_identity(identity):
    return {k: v for k, v in identity.items() if k != 'design'}


def specification(job, train):
    return {'recipe': learner.RECIPE, 'role': f'ams-offset-{job["role"]}-{job["bucket"]}',
            'iterations': 1, 'train_dates': job['train_dates'], 'test_dates': job['origins'],
            'validation_dates': [], 'train_identity': frame_identity(train, list(train)),
            'repeat_reason': None}


def prepare(repo, root, readiness, bootstrap, split):
    repo, root, readiness = map(Path, (repo, root, readiness))
    if root.exists():
        raise ValueError('Preserve existing research namespace')
    validate(repo/'research')
    registry, trials = load_registry(repo/'research')
    related = check(registry, trials, family='har', horizon=1, query='qlike')
    with writer(root):
        inputs = {'history.parquet': readiness/'history.parquet',
                  'plan.json': readiness/'plan.json',
                  'readiness-result.json': readiness/'result.json',
                  'source-result.json': readiness.parent/'full-source-v4/result.json',
                  'observations.csv': readiness.parent/'full-source-v4/observations.csv',
                  'bootstrap.py': Path(bootstrap), 'split.json': Path(split)}
        for name, path in inputs.items():
            copy_immutable(path, root/'inputs'/name)
        ready = json.loads((root/'inputs/readiness-result.json').read_text())
        source = json.loads((root/'inputs/source-result.json').read_text())
        if (ready['new_fits'] != 0 or ready['supported_years'] != 8
                or ready['source_clock_verified'] or ready['training_allowed']
                or source['source_clock_verified'] or source['training_allowed']
                or ready['source_result_sha256'] != sha(root/'inputs/source-result.json')
                or ready['history_with_named_sha256'] != sha(root/'inputs/history.parquet')
                or ready['plan_sha256'] != sha(root/'inputs/plan.json')
                or source['observations_sha256'] != sha(root/'inputs/observations.csv')):
            raise ValueError('Exact unadmitted historical source/readiness required')
        h = pd.read_parquet(root/'inputs/history.parquet')
        split_data = json.loads((root/'inputs/split.json').read_text())
        plan = json.loads((root/'inputs/plan.json').read_text())
        if (parent.plan(h, split_data) != plan or len(plan['folds']) != 8
                or [f['year'] for f in plan['folds']] != list(range(2016, 2024))
                or any(not r['supported'] for r in plan['readiness'])):
            raise ValueError('Frozen source and full-year temporal plan differ')
        fit_count = sum(len(f['jobs']) for f in plan['folds'])
        design = {'profile': PROFILE, 'recipe': learner.RECIPE, 'folds': plan['folds'],
                  'fit_budget': {'market': fit_count, 'synthetic': 1,
                                 'processes': 1, 'threads': 2, 'session_minutes': 30},
                  'primary': 'past-selected named EWMA-offset HAR versus same-contract EWMA',
                  'secondary': 'raw fixed candidate versus EWMA',
                  'selection': 'past189/3x63 named QLIKE; weights0,.25,.5,.75,1; smallest tie',
                  'gate': 'both20/60 gain CI lower>0;>=6/8positive years;top1pct baseline-loss trim>0',
                  'clock': 'daily archive source date+1day00UTC assumed; origin+1day00:15UTC',
                  'conditional_historical_fit': True, 'source_clock_verified': False,
                  'first_vintage_verified': False, 'independent_holdout': False,
                  'release_allowed': False, '2024_plus_used_for_selection': False,
                  'deliberate_repeat_reason': 'The earlier 2023 named HAR had sparse same-contract history;'
                  ' this official 2012-2023 daily archive supplies continuous past history and a new eight-year'
                  ' named cohort. The fixed learner, EWMA lambda, maturity, shrinkage and risk gate remain unchanged.'}
        manifest = source_manifest(repo)
        identity = {'profile': PROFILE, 'source_id': manifest['source_id'],
                    'design_id': content_id(design), 'design': design,
                    'research_data_id': frame_identity(h, list(h)),
                    'python': sys.version.split()[0],
                    'versions': {name: importlib.metadata.version(name) for name in VERSIONS},
                    'lock_sha256': sha(repo/'ml/uv.lock'),
                    'inputs_sha256': {name: sha(root/'inputs'/name) for name in inputs}}
        for name in SOURCE_FILES:
            copy_immutable(repo/name, root/'source-snapshot'/name)
        freeze_record(root/'source-manifest.json', manifest)
        freeze_record(root/'history-check.json',
                      {'status': related['status'],
                       'related_experiment_ids': [r['id'] for r in related['experiments']],
                       'related_fit_recipe_ids': [r['id'] for r in related['fit_recipes']],
                       'no_match_is_not_novelty': True})
        freeze_record(root/'decision-contract.json', design)
        freeze_record(root/'ready.json', {'identity': identity})
    return {'status': 'prepared', 'fit_budget': design['fit_budget'],
            'source_id': manifest['source_id'], 'design_id': identity['design_id']}


def verify(root, repo):
    root, repo = Path(root), Path(repo)
    ready = read_record(root/'ready.json'); identity = ready['identity']
    if (identity['profile'] != PROFILE or identity['source_id'] != source_manifest(repo)['source_id']
            or identity['python'] != sys.version.split()[0]
            or identity['versions'] != {name: importlib.metadata.version(name) for name in VERSIONS}
            or identity['lock_sha256'] != sha(repo/'ml/uv.lock')
            or content_id(identity['design']) != identity['design_id']
            or read_record(root/'decision-contract.json') != identity['design']):
        raise ValueError('Frozen source/design/environment differs')
    for name, expected in identity['inputs_sha256'].items():
        if sha(root/'inputs'/name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    h = pd.read_parquet(root/'inputs/history.parquet')
    if frame_identity(h, list(h)) != identity['research_data_id']:
        raise ValueError('Frozen named research data changed')
    return ready, h


def projection(h, dates, *, training):
    rows = parent.projection(h, dates, 'named', training=training)
    if not rows.named_ewma_eligible.all():
        raise ValueError('Origin named EWMA is not mature')
    return rows


def run(root, repo, authorization, minutes=30):
    require_training('har', 'cpu')
    root = Path(root); ready, h = verify(root, repo); identity = ready['identity']
    design = identity['design']; permit = read_record(Path(authorization))
    if (permit.get('ready_sha256') != sha(root/'ready.json')
            or permit.get('source_id') != identity['source_id']
            or permit.get('design_id') != identity['design_id']
            or permit.get('fit_budget') != design['fit_budget']
            or permit.get('fresh_restore_verified') is not True
            or not 0 < minutes <= 30):
        raise ValueError('Exact preregistration recovery/source/budget authorization required')
    deadline = time.monotonic()+60*minutes
    market_identity = ledger_identity(identity)
    with writer(root):
        learning = Ledger(root/'learning/ledger', {**market_identity, 'role': 'synthetic_not_market'})
        used, keys = fit_consumption(learning.root)
        if used != len(keys) or used > 1:
            raise ValueError('Failed/excess synthetic fit; no retry')
        rng = np.random.default_rng(42)
        baseline = np.exp(rng.uniform(-12, -4, 1200))
        x = rng.uniform(-3, 3, (1200, 3))
        fixture = pd.DataFrame({'date': pd.bdate_range('2000-01-03', periods=1200),
                                'named_ewma': baseline})
        for column, name in enumerate(learner.RECIPE['features']):
            fixture[name] = np.log(baseline)+x[:, column]
        fixture['variance_target_1'] = baseline*np.exp(.1+x@np.array([.2, .1, -.1]))
        smoke = learning.run({'recipe': learner.RECIPE, 'role': 'synthetic_named_offset_har',
                              'kind': 'known_signal'},
                             lambda path: learner.fit_offset(fixture.iloc[:1000], fixture.iloc[1000:], path))
        np.testing.assert_allclose(smoke['result']['variance_predictions'],
                                   fixture.variance_target_1.iloc[1000:], rtol=1e-5)
        control = {'status': 'passed', 'fit_id': smoke['experiment_id'], 'market_skill': False}
        if (root/'learning-control.json').exists():
            if read_record(root/'learning-control.json') != control:
                raise ValueError('Synthetic control changed on resume')
        else:
            freeze_record(root/'learning-control.json', control)
        ledger = Ledger(root/'ledger', market_identity)
        used, keys = fit_consumption(ledger.root)
        if used != len(keys) or used > design['fit_budget']['market']:
            raise ValueError('Failed/excess market fit; no retry')
        count = [used]

        def before_compute():
            if time.monotonic() >= deadline:
                raise FitBudgetReached('Checkpoint session boundary')
            if count[0] >= design['fit_budget']['market']:
                raise ValueError('Frozen fit budget exhausted')
            count[0] += 1

        try:
            for fold in design['folds']:
                for job in fold['jobs']:
                    train = projection(h, job['train_dates'], training=True)
                    test = projection(h, job['origins'], training=False)
                    ledger.run(specification(job, train),
                               lambda path, tr=train, te=test: learner.fit_offset(tr, te, path),
                               before_compute=before_compute)
        except FitBudgetReached:
            return {'status': 'planned_pause', 'market_fits': count[0], 'synthetic_fits': 1}
    return {'status': 'completed', 'market_fits': design['fit_budget']['market'], 'synthetic_fits': 1}


def replay_native(ledger, job, train, test):
    spec = specification(job, train); key = ledger.key(spec)
    record = read_record(ledger.root/'completed'/f'{key}.json')
    if (record['identity'] != ledger.identity or record['specification'] != spec
            or record['experiment_id'] != key):
        raise ValueError('Native fit identity mismatch')
    for name, expected in record['files'].items():
        if not safe_member(name) or sha(ledger.root/name) != expected:
            raise ValueError('Native fit payload changed')
    model = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_text())
    if (model['kind'] != 'ewma_offset_har' or model['recipe'] != learner.RECIPE
            or model['effective_parameters'] != 4 or model['ratio_smearing'] != 1.):
        raise ValueError('Native model contract differs')
    x = train[learner.RECIPE['features']].to_numpy()-np.log(train.named_ewma.to_numpy())[:, None]
    eta = np.column_stack([np.ones(len(train)), x])@model['coef']
    ratio = (train.named_variance/train.named_ewma).to_numpy()*np.exp(-eta)
    optimization = model['optimization']
    z = (x-optimization['training_feature_mean'])/optimization['training_feature_scale']
    gradient = np.column_stack([np.ones(len(train)), z]).T@(1-ratio)/len(train)
    if (not optimization['success'] or np.max(np.abs(gradient)) > 1.001e-6
            or not math.isclose(float(np.mean(eta+ratio)),
                                optimization['final_training_qlike'], abs_tol=1e-10)):
        raise ValueError('Native optimizer certificate failed')
    xt = test[learner.RECIPE['features']].to_numpy()-np.log(test.named_ewma.to_numpy())[:, None]
    predicted = test.named_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), xt])@model['coef'])
    np.testing.assert_allclose(predicted, record['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
    if record['result']['origins'] != job['origins']:
        raise ValueError('Native prediction origins changed')
    return record, float(np.max(np.abs(gradient)))


def compare(root, repo):
    root = Path(root); ready, h = verify(root, repo); design = ready['identity']['design']
    ledger = Ledger(root/'ledger', ledger_identity(ready['identity']))
    loader = importlib.util.spec_from_file_location('frozen_named_bootstrap', root/'inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(loader); loader.loader.exec_module(bootstrap)
    entries, keys, gradients = [], set(), []
    for fold in design['folds']:
        records = []
        for job in fold['jobs']:
            train = projection(h, job['train_dates'], training=True)
            test = projection(h, job['origins'], training=False)
            record, gradient = replay_native(ledger, job, train, test)
            records.append(record); keys.add(record['experiment_id']); gradients.append(gradient)
        out = parent.output(h, fold, 'named', records)
        for item in out['records']:
            entries.append({**item, 'year': fold['year'],
                            'selected_weight': out['selection']['weight']})
    frame = pd.DataFrame(entries)
    if len(keys) != design['fit_budget']['market'] or len(frame) != 2006:
        raise ValueError('Native fits or ordinal cohort incomplete')
    for arm in ('baseline', 'raw', 'selected'):
        frame[arm+'_loss'] = np.nan
    valid = frame.eligible
    for arm in ('baseline', 'raw', 'selected'):
        frame.loc[valid, arm+'_loss'] = qlike(frame.loc[valid, 'named_actual'].to_numpy(),
                                              frame.loc[valid, arm].to_numpy())
    comparisons = {}
    for arm in ('raw', 'selected'):
        a, b = 'baseline_loss', arm+'_loss'
        clean = frame.loc[valid]
        offset = max(0., 1-min(clean[a].min(), clean[b].min()))
        sample = frame[['year']].copy()
        sample['a'] = np.where(valid, frame[a].fillna(0)+offset, 0)
        sample['b'] = np.where(valid, frame[b].fillna(0)+offset, 0)
        sample['weight'] = valid.astype(int)
        parts = [g[['a', 'b', 'weight']].to_numpy() for _, g in sample.groupby('year')]
        intervals = {str(block): bootstrap.weighted_bootstrap(parts, block=block)['difference_ci_95']
                     for block in (20, 60)}
        yearly = {str(year): float((g[a]-g[b]).mean()) for year, g in clean.groupby('year')}
        trimmed = clean.drop(clean.nlargest(math.ceil(len(clean)*.01), a).index)
        gain = float((clean[a]-clean[b]).mean())
        trim_gain = float((trimmed[a]-trimmed[b]).mean())
        passed = (all(ci[0] > 0 for ci in intervals.values())
                  and sum(value > 0 for value in yearly.values()) >= 6 and trim_gain > 0)
        comparisons[arm] = {'baseline_minus_candidate_gain': gain, 'gain_ci': intervals,
                            'yearly_gain': yearly,
                            'positive_years': sum(value > 0 for value in yearly.values()),
                            'trim_top1pct_gain': trim_gain, 'fixed_risk_gate_passed': bool(passed)}
    report = {'profile': PROFILE, 'new_fits_during_report': 0, 'native_market_fits': len(keys),
              'optimizer_gradient_inf_max': max(gradients), 'retained_origins': len(frame),
              'eligible_origins': int(valid.sum()), 'comparisons': comparisons,
              'selected_risk_gate_passed': comparisons['selected']['fixed_risk_gate_passed'],
              'price_gates_passed': False, 'independent_holdout': False,
              'source_clock_verified': False, 'first_vintage_verified': False,
              'release_allowed': False}
    dest = root/'reports'; dest.mkdir(exist_ok=True)
    if (dest/'paired-losses.csv').exists():
        raise ValueError('Preserve prior comparison')
    frame.to_csv(dest/'paired-losses.csv', index=False, lineterminator='\n')
    report['paired_losses_sha256'] = sha(dest/'paired-losses.csv')
    freeze_record(dest/'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'pilot', 'compare'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    for name in ('readiness', 'bootstrap', 'split', 'authorization'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.repo, args.root, args.readiness, args.bootstrap, args.split)
    elif args.stage == 'verify':
        ready, _ = verify(args.root, args.repo)
        result = {'status': 'verified', 'new_fits': 0,
                  'fit_budget': ready['identity']['design']['fit_budget']}
    elif args.stage == 'pilot':
        result = run(args.root, args.repo, args.authorization)
    else:
        result = compare(args.root, args.repo)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
