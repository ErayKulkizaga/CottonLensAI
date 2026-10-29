"""One frozen QLIKE curvature direction on existing named risk predictions."""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record, writer

from research import ams_named_ewma_har as parent

PROFILE = 'ams-risk-curvature-score-t1-v1'
HELPERS = ('research/risk_curvature_score.py', 'research/tests/test_risk_curvature_score.py',
           'research/ams_named_ewma_har.py', 'research/ams_named_readiness.py',
           'research/ams_upland_futures.py')
VERSIONS = ('numpy', 'pandas', 'scipy', 'pyarrow')


def source(repo):
    body = {'ml': research_source_identity(repo), 'helpers': {p: digest(repo/p) for p in HELPERS}}
    return {**body, 'source_id': content_id(body)}


def curvature_basis(train, test, mean, scale):
    train, test, mean, scale = [np.asarray(a, float) for a in (train, test, mean, scale)]
    if (train.ndim != 2 or test.ndim != 2 or train.shape[1] != 3 or test.shape[1] != 3
            or mean.shape != (3,) or scale.shape != (3,) or (scale <= 0).any()
            or not all(np.isfinite(a).all() for a in (train, test, mean, scale))):
        raise ValueError('Finite three-feature past normalization required')
    np.testing.assert_allclose(train.mean(axis=0), mean, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(np.where(train.std(axis=0) > 1e-12, train.std(axis=0), 1.), scale,
                               rtol=1e-12, atol=1e-13)
    z = np.column_stack([np.ones(len(train)), (train-mean)/scale])
    future = np.column_stack([np.ones(len(test)), (test-mean)/scale])
    target = z[:, 2]**2-1  # one prespecified relative-HAR5 square
    coefficients, _, rank, _ = np.linalg.lstsq(z, target, rcond=None)
    residual = target-z@coefficients
    rms = float(np.sqrt(np.mean(residual**2)))
    if rank != 4 or rms <= 1e-12:
        raise ValueError('Curvature not identifiable separately from affine features')
    orthogonal = float(np.max(np.abs(z.T@residual/len(train))))
    if orthogonal > 1e-9:
        raise ValueError('Past feature projection not orthogonal')
    return (future[:, 2]**2-1-future@coefficients)/rms, {
        'projection_coefficients': coefficients.tolist(), 'residual_rms': rms,
        'training_mean': mean.tolist(), 'training_scale': scale.tolist(),
        'projection_orthogonality_inf': orthogonal, 'feature_only_projection': True}


def improvement_score(actual, predicted, basis):
    actual, predicted, basis = [np.asarray(a, float) for a in (actual, predicted, basis)]
    if (actual.shape != predicted.shape or basis.shape != actual.shape
            or not np.isfinite([actual, predicted, basis]).all()
            or (actual < 0).any() or (predicted <= 0).any()):
        raise ValueError('Aligned finite actual, positive predictions and basis required')
    return basis*(actual/predicted-1)


def supported(point, intervals, yearly, trimmed):
    positive = all(ci[0] > 0 for ci in intervals.values())
    negative = all(ci[1] < 0 for ci in intervals.values())
    direction = 1 if positive else (-1 if negative else 0)
    wins = sum(direction*x > 0 for x in yearly.values()) if direction else 0
    return bool(direction and wins >= 6 and direction*trimmed > 0), direction, wins


def prepare(repo, root, reference):
    repo, root, reference = map(Path, (repo, root, reference))
    if root.exists():
        raise ValueError('Preserve prior namespace')
    validate(repo/'research')
    ready, _ = parent.verify(reference, repo)
    proof = read_record(repo/'research/evidence/ams-named-ewma-har-result-20261014.json')
    if (digest(reference/'reports/result.json') != proof['result_sha256']
            or digest(reference/'reports/paired-losses.csv') != proof['paired_losses_sha256']
            or ready['identity']['source_id'] != proof['source_id']):
        raise ValueError('Completed native reference differs from registered evidence')
    related = check(*load_registry(repo/'research'), horizon=1, query='curvature')
    design = {'profile': PROFILE, 'new_forecast_model_fits': 0,
              'raw_baseline_and_model_predictions_changed': False,
              'feature': 'one relative-HAR5 squared term, affine-orthogonalized and RMS-scaled on each past train',
              'local_model': 'p_gamma = existing_raw_prediction*exp(gamma*phi)',
              'improvement_score': 'phi*(actual/existing_raw_prediction-1) = -dQLIKE/dgamma at0',
              'decision': 'both20/60 two-sided CIs wholly same side of0;>=6/8annual same sign;top1pctEWMA-loss trim same sign',
              'signs': 'both signs allowed before analysis; no ex-post direction/term/horizon search',
              'cohort': 'seen2016-2023/2006ordinal slots/1989paired from exact registered AMS study',
              'bootstrap': {'blocks': [20, 60], 'replicates': 10000, 'seed': 42, 'year_boundaries_preserved': True},
              'not_claimed': ['QLIKE gain', 'learnable gamma', 'market skill', 'calibrated real-market power'],
              'feature_projection_scope': '97 label-free four-column linear algebra projections, not forecast model fits',
              'independent_holdout': False, 'historical_availability_verified': False,
              'market_skill': False, 'release_allowed': False, 'new_model_training_allowed': False}
    with writer(root):
        shutil.copytree(reference, root/'reference')
        freeze_record(root/'parent-proof.json', proof)
        freeze_record(root/'decision-contract.json', design)
        freeze_record(root/'history-check.json', {'status': related['status'],
                                                 'related': [r['id'] for r in related['experiments']], 'new_fits': 0})
        manifest = source(repo); freeze_record(root/'source-manifest.json', manifest)
        identity = {'source_id': manifest['source_id'], 'design_id': content_id(design),
                    'python': sys.version.split()[0], 'versions': {p: importlib.metadata.version(p) for p in VERSIONS},
                    'inputs_sha256': {name: digest(root/name) for name in ('parent-proof.json',
                                         'reference/ready.json', 'reference/reports/result.json',
                                         'reference/reports/paired-losses.csv', 'decision-contract.json')}}
        freeze_record(root/'ready.json', identity)
    return {'status': 'prepared', 'new_forecast_model_fits': 0, **identity}


def verify(root, repo):
    ready = read_record(root/'ready.json'); design = read_record(root/'decision-contract.json')
    if (ready['source_id'] != source(repo)['source_id'] or ready['design_id'] != content_id(design)
            or ready['python'] != sys.version.split()[0]
            or ready['versions'] != {p: importlib.metadata.version(p) for p in VERSIONS}):
        raise ValueError('Frozen analysis/environment changed')
    for name, checksum in ready['inputs_sha256'].items():
        if digest(root/name) != checksum:
            raise ValueError('Frozen reference/contract changed')
    parent_ready, h = parent.verify(root/'reference', repo)
    return ready, design, parent_ready, h


def run(root, repo):
    root, repo = Path(root), Path(repo)
    ready, design, old, h = verify(root, repo)
    if design['new_forecast_model_fits'] != 0 or design['new_model_training_allowed']:
        raise ValueError('Only the frozen zero-forecast-fit analysis is allowed')
    if (root/'result.json').exists() or (root/'scores.csv').exists():
        raise ValueError('Preserve previous score output')
    reference = root/'reference'
    ledger = Ledger(reference/'ledger', parent.ledger_identity(old['identity']))
    outputs, transformations = {}, {}
    for fold in old['identity']['design']['folds']:
        for job in fold['jobs']:
            if job['role'] != 'outer':
                continue
            train = parent.projection(h, job['train_dates'], training=True)
            test = parent.projection(h, job['origins'], training=False)
            if not (train.target_date_5 < pd.Timestamp(job['cutoff'])).all():
                raise ValueError('Training label maturity differs')
            record, _ = parent.replay_native(ledger, job, train, test)
            model = json.loads((ledger.root/record['payload_root']/'variance-model.json').read_text())
            x = train[parent.learner.RECIPE['features']].to_numpy()-np.log(train.named_ewma.to_numpy())[:, None]
            xt = test[parent.learner.RECIPE['features']].to_numpy()-np.log(test.named_ewma.to_numpy())[:, None]
            optimization = model['optimization']
            phi, transform = curvature_basis(x, xt, optimization['training_feature_mean'],
                                             optimization['training_feature_scale'])
            transform.update(cutoff=job['cutoff'], train_count=len(train), train_dates_sha256=content_id(job['train_dates']))
            transformations[record['experiment_id']] = transform
            for i, day in enumerate(job['origins']):
                if day in outputs:
                    raise ValueError('Duplicate outer prediction origin')
                outputs[day] = {'phi': float(phi[i]), 'native_fit_id': record['experiment_id'],
                                'native_raw': record['result']['variance_predictions'][i]}
    frame = pd.read_csv(reference/'reports/paired-losses.csv', float_precision='round_trip',
                        keep_default_na=False, na_values=[''])
    origins = [d for f in old['identity']['design']['folds'] for d in f['origins']]
    if frame.date.tolist() != origins or len(origins) != 2006 or int(frame.eligible.sum()) != 1989:
        raise ValueError('Exact registered cohort required')
    if len(transformations) != 97 or set(outputs) != set(frame.loc[frame.forecast_available, 'date']):
        raise ValueError('Native outer prediction coverage differs')
    frame['phi'], frame['score'], frame['native_fit_id'] = 0., 0., ''
    for row in frame.itertuples():
        if not row.forecast_available:
            if row.eligible:
                raise ValueError('Eligible target without native prediction')
            continue
        item = outputs[row.date]
        if not math.isclose(row.raw, item['native_raw'], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Native/stored forecast differs')
        frame.loc[row.Index, ['phi', 'native_fit_id']] = [item['phi'], item['native_fit_id']]
    frame['weight'] = frame.eligible.astype(int)
    valid = frame.eligible
    source_rows = h.set_index(h.date.dt.strftime('%Y-%m-%d')).loc[frame.date]
    if (not np.array_equal(frame.target_date, source_rows.target_date_1.dt.strftime('%Y-%m-%d'))
            or not np.array_equal(valid.to_numpy(), (source_rows.named_eligible
                                   & source_rows.named_ewma_eligible & source_rows.own_feature_ready).to_numpy())):
        raise ValueError('Target date or eligible population differs from source')
    np.testing.assert_allclose(frame.loc[valid, 'named_actual'], source_rows.loc[valid.to_numpy(), 'named_variance'],
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(frame.loc[valid, 'baseline'], source_rows.loc[valid.to_numpy(), 'named_ewma'],
                               rtol=1e-12, atol=1e-14)
    frame.loc[valid, 'score'] = improvement_score(frame.loc[valid, 'named_actual'],
                                                 frame.loc[valid, 'raw'], frame.loc[valid, 'phi'])
    loader = importlib.util.spec_from_file_location('frozen_score_bootstrap', reference/'inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(loader); loader.loader.exec_module(bootstrap)
    offset = max(1., 1-float(frame.loc[valid, 'score'].min()))
    tmp = frame[['year', 'weight']].copy()
    tmp['a'] = np.where(valid, frame.score+offset, 0); tmp['b'] = np.where(valid, offset, 0)
    parts = [g[['a', 'b', 'weight']].to_numpy() for _, g in tmp.groupby('year')]
    intervals = {str(b): bootstrap.weighted_bootstrap(parts, block=b)['difference_ci_95'] for b in (20, 60)}
    yearly = {str(y): float(g.loc[g.eligible, 'score'].mean()) for y, g in frame.groupby('year')}
    point = float(frame.loc[valid, 'score'].mean())
    trimmed = frame.loc[valid].drop(frame.loc[valid].nlargest(math.ceil(valid.sum()*.01), 'baseline_loss').index)
    trim_mean = float(trimmed.score.mean())
    passed, direction, wins = supported(point, intervals, yearly, trim_mean)
    result = {'profile': PROFILE, 'classification': 'DECISIVE_POSITIVE' if passed else 'INCONCLUSIVE',
              'classification_scope': 'one conditional curvature score only; not forecast gain or all-nonlinearity test',
              'new_forecast_model_fits': 0, 'feature_only_projections': len(transformations),
              'retained_origins': len(frame), 'eligible_origins': int(valid.sum()),
              'mean_improvement_score': point, 'score_ci_95': intervals, 'yearly_score': yearly,
              'supported_direction': direction, 'same_direction_years': wins,
              'trim_top1pct_mean_score': trim_mean, 'score_gate_passed': passed,
              'market_skill': False, 'new_model_training_allowed': False,
              'release_allowed': False, 'independent_holdout': False,
              'historical_availability_verified': False,
              'source_id': ready['source_id'], 'design_id': ready['design_id']}
    frame.to_csv(root/'scores.csv', index=False, lineterminator='\n')
    freeze_record(root/'transformations.json', transformations)
    result['scores_sha256'] = digest(root/'scores.csv')
    result['transformations_sha256'] = digest(root/'transformations.json')
    freeze_record(root/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'run'))
    for name in ('repo', 'root', 'reference'):
        parser.add_argument('--'+name, type=Path, required=name != 'reference')
    args = parser.parse_args()
    if args.stage == 'prepare':
        result = prepare(args.repo, args.root, args.reference)
    elif args.stage == 'run':
        result = run(args.root, args.repo)
    else:
        verify(args.root, args.repo); result = {'status': 'verified', 'new_forecast_model_fits': 0}
    print(json.dumps(result))
