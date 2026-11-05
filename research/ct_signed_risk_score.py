"""One frozen, past-projected signed-shock direction at existing CT forecasts.

This computes a local QLIKE derivative, not a fitted forecast or market skill.
The previously observed negative association motivates this development test;
neither sign is chosen from this test's results.
"""

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
from cottonlens_ml.research import ct_ewma_har as parent
from cottonlens_ml.research.history import check, load_registry, validate
from cottonlens_ml.research.ledger import Ledger, freeze_record, read_record

PROFILE = 'ct-signed-shock-conditional-qlike-score-t1-v1'
HELPERS = ('research/ct_signed_risk_score.py', 'research/verify_ct_signed_risk_score.py',
           'research/tests/test_ct_signed_risk_score.py', 'research/verify_risk_weak_teacher.py')
VERSIONS = ('numpy', 'pandas', 'scipy', 'pyarrow')


def basis(train_x, test_x, train_shock, test_shock):
    """Remove affine HAR and unsigned-shock size using training features only."""
    x, xt, s, st = [np.asarray(v, float) for v in (train_x, test_x, train_shock, test_shock)]
    if (x.ndim != 2 or xt.ndim != 2 or x.shape[1] != 3 or xt.shape[1] != 3
            or s.shape != (len(x),) or st.shape != (len(xt),)
            or not all(np.isfinite(v).all() for v in (x, xt, s, st))):
        raise ValueError('Aligned finite HAR and signed shocks required')
    raw = np.column_stack([x, np.abs(s)])
    raw_test = np.column_stack([xt, np.abs(st)])
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.column_stack([np.ones(len(x)), (raw-mean)/scale])
    zt = np.column_stack([np.ones(len(xt)), (raw_test-mean)/scale])
    coefficients, _, rank, _ = np.linalg.lstsq(z, s, rcond=None)
    residual = s-z@coefficients
    rms = float(np.sqrt(np.mean(residual**2)))
    if rank != 5 or rms <= 1e-12:
        raise ValueError('Signed information not identifiable beyond magnitude/HAR')
    orthogonality = float(np.max(np.abs(z.T@residual/len(x))))
    if orthogonality > 1e-9:
        raise ValueError('Past projection failed orthogonality')
    return (st-zt@coefficients)/rms, {
        'nuisance_mean': mean.tolist(), 'nuisance_scale': scale.tolist(),
        'projection_coef': coefficients.tolist(), 'residual_rms': rms,
        'orthogonality_inf': orthogonality, 'feature_only': True}


def score(actual, predicted, phi):
    y, p, z = [np.asarray(v, float) for v in (actual, predicted, phi)]
    if (y.shape != p.shape or z.shape != y.shape or not np.isfinite([y, p, z]).all()
            or (y < 0).any() or (p <= 0).any()):
        raise ValueError('Aligned nonnegative targets and positive forecasts required')
    return z*(y/p-1)


def gate(intervals, yearly, trimmed):
    direction = (1 if all(v[0] > 0 for v in intervals.values()) else
                 -1 if all(v[1] < 0 for v in intervals.values()) else 0)
    wins = sum(direction*v > 0 for v in yearly.values()) if direction else 0
    return bool(direction and wins >= 6 and direction*trimmed > 0), direction, wins


def source(repo):
    files = {n: digest(repo/n) for n in HELPERS}
    body = {'ml': research_source_identity(repo), 'helpers': files}
    return {'source_id': content_id(body), **body}


def prepare(repo, root, reference):
    if root.exists():
        raise ValueError('Preserve old or incomplete namespace')
    validate(repo/'research')
    proof_path = repo/'research/evidence/ct-ewma-har-result-20261014.json'
    proof = json.loads(proof_path.read_bytes())
    ready, _ = parent.verify(reference, reference/'source-snapshot')
    if (digest(reference/'reports/result.json') != proof['result_sha256']
            or digest(reference/'reports/paired-losses.csv') != proof['paired_losses_sha256']
            or ready['identity']['source_id'] != proof['source_id']
            or digest(Path(parent.__file__)) != digest(reference/'source-snapshot/ml/src/cottonlens_ml/research/ct_ewma_har.py')):
        raise ValueError('Registered native CT reference/module differs')
    related = check(*load_registry(repo/'research'), horizon=1, query='asymmetric')
    design = {'profile': PROFILE, 'new_forecast_fits': 0, 'new_data': False,
              'reference_profile': parent.PROFILE, 'origins': 2006,
              'feature': 'sign(current_return)*current_return^2/origin_EWMA',
              'projection': '97 outer-job past-only feature projections on intercept,3relativeHAR,unsigned current_return^2/origin_EWMA; residual training RMS',
              'primary': 'phi*(next_return_squared/native_raw_CT_HAR_variance-1)',
              'local_direction': 'native_raw_CT_HAR_variance*exp(gamma*phi), derivative at gamma=0 only',
              'signs': 'both signs fixed before run; no gamma learned, sign-flip forecast or grid',
              'criterion': 'both20/60 CIs wholly on same side;>=6/8same-direction years;top1pctEWMA-loss trim same direction',
              'bootstrap': {'replicates': 10000, 'seed': 42, 'blocks': [20, 60], 'preserve_year_boundaries': True},
              'prior': '20261013 positive inverse-leverage test had negative CIs but only5/8negativeyears; not QLIKE gain and not no-sign-information evidence',
              'scope': 'seen2016-2023 CT continuous proxy; not clean named futures, independent holdout or economic success',
              'clock_vintage_verified': False, 'independent_holdout': False,
              'multiplicity_adjusted': False, 'market_skill': False, 'training_allowed': False,
              'if_supported': 'only motivates separate frozen past-learned forecast contrast; no automatic fit',
              'if_unsupported': 'do not turn older ex-post covariance into a forecast-skill claim or open a sign grid'}
    root.mkdir(parents=True)
    shutil.copytree(reference, root/'reference')
    shutil.copyfile(proof_path, root/'parent-proof.json')
    files = {str(p.relative_to(reference)).replace('\\', '/'): digest(p)
             for p in sorted(reference.rglob('*')) if p.is_file()}
    freeze_record(root/'parent-files.json', {'files': files})
    freeze_record(root/'decision-contract.json', design)
    freeze_record(root/'history-check.json', {
        'status': related['status'], 'related': [r['id'] for r in related['experiments']],
        'prior_record': 'asymmetric-risk-preflight-20261013', 'no_match_is_not_novelty': True})
    manifest = source(repo)
    freeze_record(root/'source-manifest.json', manifest)
    identity = {'source_id': manifest['source_id'], 'design_id': content_id(design),
                'python': sys.version.split()[0],
                'versions': {p: importlib.metadata.version(p) for p in VERSIONS},
                'inputs_sha256': {n: digest(root/n) for n in ('parent-proof.json', 'parent-files.json', 'decision-contract.json')}}
    freeze_record(root/'ready.json', identity)
    return {'status': 'prepared', **identity, 'new_forecast_fits': 0}


def verify_inputs(root, repo):
    ready = read_record(root/'ready.json')
    design = read_record(root/'decision-contract.json')
    if (ready['source_id'] != source(repo)['source_id'] or ready['design_id'] != content_id(design)
            or ready['python'] != sys.version.split()[0]
            or ready['versions'] != {p: importlib.metadata.version(p) for p in VERSIONS}
            or design['training_allowed'] or design['new_forecast_fits']):
        raise ValueError('Frozen zero-fit source/environment/contract changed')
    for name, checksum in ready['inputs_sha256'].items():
        if digest(root/name) != checksum:
            raise ValueError('Frozen input changed')
    reference = root/'reference'
    files = read_record(root/'parent-files.json')['files']
    if {str(p.relative_to(reference)).replace('\\', '/') for p in reference.rglob('*') if p.is_file()} != set(files):
        raise ValueError('Parent file scope changed')
    for name, checksum in files.items():
        if digest(reference/name) != checksum:
            raise ValueError('Immutable parent payload changed')
    old, history = parent.verify(reference, reference/'source-snapshot')
    return ready, design, old, history


def run(root, repo):
    ready, _design, old, h = verify_inputs(root, repo)
    if any((root/n).exists() for n in ('result.json', 'scores.csv', 'transformations.json')):
        raise ValueError('Preserve completed/partial outputs')
    ref = root/'reference'
    ledger = Ledger(ref/'ledger', parent.ledger_identity(old['identity']))
    returns = h.cotton_ret_1.to_numpy(float)
    close = h.cotton_close.to_numpy(float)
    independent_returns = np.log(close[1:]/close[:-1])
    np.testing.assert_allclose(returns[1:], independent_returns, rtol=1e-10, atol=1e-14)
    h = h.copy()
    h['signed_shock'] = np.sign(returns)*returns**2/h.ct_ewma.to_numpy(float)
    indexed = h.set_index(h.date.dt.strftime('%Y-%m-%d'))
    transforms, outputs = {}, {}
    for job in design_jobs(old):
        train = parent.projection(h, job['train_dates'])
        test = parent.projection(h, job['origins'])
        if not train.date.lt(pd.Timestamp(job['cutoff'])).all() or not indexed.loc[job['train_dates']].target_date_5.lt(pd.Timestamp(job['cutoff'])).all():
            raise ValueError('Unavailable training row')
        predictions, fit_id, _gradient = parent.replay_native(ledger, job, train, test)
        x = parent.transform(train, True)[parent.FEATURES].to_numpy(float)
        xt = parent.transform(test, False)[parent.FEATURES].to_numpy(float)
        phi, metadata = basis(x, xt, indexed.loc[job['train_dates']].signed_shock,
                              indexed.loc[job['origins']].signed_shock)
        metadata.update(cutoff=job['cutoff'], train_count=len(train), train_dates_sha256=content_id(job['train_dates']))
        transforms[fit_id] = metadata
        for i, day in enumerate(job['origins']):
            if day in outputs:
                raise ValueError('Overlapping outer predictions')
            outputs[day] = (float(phi[i]), float(predictions[i]), fit_id)
    frame = pd.read_csv(ref/'reports/paired-losses.csv', float_precision='round_trip')
    origins = [d for fold in old['identity']['design']['split']['folds'] for d in fold['origins']]
    if frame.date.tolist() != origins or len(frame) != 2006 or len(transforms) != 97 or set(outputs) != set(origins):
        raise ValueError('Exact original cohort/97 outer fits required')
    rows = indexed.loc[origins]
    if frame.target_date.tolist() != rows.target_date_1.dt.strftime('%Y-%m-%d').tolist():
        raise ValueError('Target dates differ')
    np.testing.assert_allclose(frame.actual_variance, rows.variance_target_1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(frame.ewma_variance, rows.ct_ewma, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(frame.raw_variance, [outputs[d][1] for d in origins], rtol=1e-12, atol=1e-14)
    frame['phi'] = [outputs[d][0] for d in origins]
    frame['native_fit_id'] = [outputs[d][2] for d in origins]
    frame['signed_shock'] = rows.signed_shock.to_numpy()
    frame['score'] = score(frame.actual_variance, frame.raw_variance, frame.phi)
    module = importlib.util.spec_from_file_location('frozen_ct_score_bootstrap', ref/'inputs/bootstrap.py')
    bootstrap = importlib.util.module_from_spec(module); module.loader.exec_module(bootstrap)
    offset = max(1., 1-float(frame.score.min()))
    parts = [np.column_stack([g.score+offset, np.full(len(g), offset), np.ones(len(g))])
             for _, g in frame.groupby('year')]
    intervals = {str(b): bootstrap.weighted_bootstrap(parts, block=b)['difference_ci_95'] for b in (20, 60)}
    yearly = {str(y): float(g.score.mean()) for y, g in frame.groupby('year')}
    trimmed = frame.drop(frame.nlargest(math.ceil(len(frame)*.01), 'baseline_loss').index)
    trim_mean = float(trimmed.score.mean())
    passed, direction, wins = gate(intervals, yearly, trim_mean)
    result = {'profile': PROFILE, 'classification': 'DECISIVE_POSITIVE' if passed else 'INCONCLUSIVE',
              'classification_scope': 'one local conditional QLIKE direction; no forecast improvement established',
              'new_forecast_fits': 0, 'retained_origins': len(frame), 'native_outer_fits': len(transforms),
              'mean_score': float(frame.score.mean()), 'score_ci': intervals, 'yearly_score': yearly,
              'supported_direction': direction, 'same_direction_years': wins, 'trim_top1pct_score': trim_mean,
              'direction_gate_passed': passed, 'market_skill': False, 'training_allowed': False,
              'independent_holdout': False, 'source_id': ready['source_id'], 'design_id': ready['design_id']}
    frame.to_csv(root/'scores.csv', index=False, lineterminator='\n')
    freeze_record(root/'transformations.json', transforms)
    result.update(scores_sha256=digest(root/'scores.csv'), transformations_sha256=digest(root/'transformations.json'))
    freeze_record(root/'result.json', result)
    return result


def design_jobs(ready):
    return [j for j in ready['identity']['design']['jobs'] if j['role'] == 'outer']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        output = prepare(args.repo, args.root, args.reference)
    elif args.stage == 'run':
        output = run(args.root, args.repo)
    else:
        verify_inputs(args.root, args.repo)
        output = {'status': 'verified', 'new_forecast_fits': 0}
    print(json.dumps(output))
