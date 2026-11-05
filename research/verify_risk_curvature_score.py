"""Independent feature-only curvature projection and QLIKE-score audit."""

import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from research.verify_risk_weak_teacher import (
    difference_interval,
    fingerprint,
    read,
    sha,
)


def verify(root):
    result = read(root/'result.json'); transformations = read(root/'transformations.json')
    ready = read(root/'ready.json'); contract = read(root/'decision-contract.json')
    if (result['market_skill'] or result['new_forecast_model_fits'] != 0
            or result['feature_only_projections'] != 97 or contract['signs'] !=
            'both signs allowed before analysis; no ex-post direction/term/horizon search'):
        raise ValueError('Wrong zero-forecast-fit scope')
    if sha(root/'scores.csv') != result['scores_sha256'] or sha(root/'transformations.json') != result['transformations_sha256']:
        raise ValueError('Score/transformation hash mismatch')
    for name, checksum in ready['inputs_sha256'].items():
        if sha(root/name) != checksum:
            raise ValueError('Frozen reference changed')
    reference = root/'reference'; parent = read(reference/'ready.json')
    proof = read(root/'parent-proof.json')
    if (sha(reference/'reports/result.json') != proof['result_sha256']
            or sha(reference/'reports/paired-losses.csv') != proof['paired_losses_sha256']):
        raise ValueError('Registered parent differs')
    for name, checksum in parent['identity']['inputs_sha256'].items():
        if sha(reference/'inputs'/name) != checksum:
            raise ValueError('Parent source input differs')
    h = pd.read_parquet(reference/'inputs/history.parquet')
    h = h.set_index(h.date.dt.strftime('%Y-%m-%d'))
    options = {'float_precision': 'round_trip', 'keep_default_na': False, 'na_values': ['']}
    frame = pd.read_csv(root/'scores.csv', **options)
    source = pd.read_csv(reference/'reports/paired-losses.csv', **options)
    pd.testing.assert_frame_equal(frame[list(source)], source, check_exact=True)
    if len(frame) != 2006 or int(frame.weight.sum()) != 1989 or frame.date.duplicated().any():
        raise ValueError('Ordinal cohort mismatch')
    np.testing.assert_array_equal(frame.weight, frame.eligible.astype(int))
    np.testing.assert_array_equal(frame.target_date, h.loc[frame.date].target_date_1.dt.strftime('%Y-%m-%d'))
    expected_jobs = {(tuple(j['train_dates']), tuple(j['origins'])): j
                     for f in parent['identity']['design']['folds'] for j in f['jobs'] if j['role'] == 'outer'}
    if len(expected_jobs) != 97 or len(transformations) != 97:
        raise ValueError('Expected outer job count differs')
    seen = set(); predicted = {}
    max_basis_error, max_score_error = 0., 0.
    for fit_id, metadata in transformations.items():
        record = read(reference/'ledger/completed'/f'{fit_id}.json')
        spec = record['specification']
        if fingerprint({'identity': record['identity'], 'specification': spec}) != fit_id:
            raise ValueError('Native fit identity differs')
        key = tuple(spec['train_dates']), tuple(spec['test_dates'])
        if key not in expected_jobs or key in seen:
            raise ValueError('Native temporal split differs')
        job = expected_jobs[key]; seen.add(key)
        train, test = h.loc[list(key[0])], h.loc[list(key[1])]
        if (not (train.target_date_5 < pd.Timestamp(job['cutoff'])).all()
                or metadata['cutoff'] != job['cutoff'] or metadata['train_count'] != len(train)):
            raise ValueError('Basis uses unavailable training rows')
        for name, checksum in record['files'].items():
            if sha(reference/'ledger'/name) != checksum:
                raise ValueError('Native model hash differs')
        model = json.loads((reference/'ledger'/record['payload_root']/'variance-model.json').read_text())
        names = ['own_har_1', 'own_har_5', 'own_har_22']
        x = train[names].to_numpy()-np.log(train.named_ewma.to_numpy())[:, None]
        xt = test[names].to_numpy()-np.log(test.named_ewma.to_numpy())[:, None]
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.)
        z = np.column_stack([np.ones(len(train)), (x-mean)/scale])
        zt = np.column_stack([np.ones(len(test)), (xt-mean)/scale])
        # Moore-Penrose inverse independently checks the analysis least-squares projection.
        target = z[:, 2]**2-1
        coefficients = np.linalg.pinv(z)@target
        residual = target-z@coefficients
        rms = float(np.linalg.norm(residual)/math.sqrt(len(train)))
        if rms <= 1e-12 or np.max(np.abs(z.T@residual/len(train))) > 1e-9:
            raise ValueError('Curvature basis is not identifiable/orthogonal')
        phi = (zt[:, 2]**2-1-zt@coefficients)/rms
        np.testing.assert_allclose(coefficients, metadata['projection_coefficients'], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose([rms], [metadata['residual_rms']], rtol=1e-12)
        p = test.named_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), xt])@model['coef'])
        np.testing.assert_allclose(p, record['result']['variance_predictions'], rtol=1e-12)
        predicted.update({day: (float(v), float(b), fit_id) for day, v, b in zip(key[1], p, phi, strict=True)})
    if set(predicted) != set(frame.loc[frame.forecast_available, 'date']):
        raise ValueError('Native origin coverage differs')
    actual_scores = []
    with localcontext() as context:
        context.prec = 34
        for row in frame.itertuples():
            if row.forecast_available:
                p, phi, fit_id = predicted[row.date]
                if fit_id != row.native_fit_id:
                    raise ValueError('Origin assigned to wrong native model')
                np.testing.assert_allclose([p, phi], [row.raw, row.phi], rtol=1e-10, atol=1e-10)
                max_basis_error = max(max_basis_error, abs(phi-row.phi))
            if not row.eligible:
                if row.score != 0:
                    raise ValueError('Missing ordinal slot contributes a score')
                actual_scores.append(0.); continue
            source_row = h.loc[row.date]
            np.testing.assert_allclose([row.named_actual, row.baseline],
                                       [source_row.named_variance, source_row.named_ewma], rtol=1e-13)
            score = float(Decimal(str(row.phi))*(Decimal(str(row.named_actual))/Decimal(str(row.raw))-1))
            if not math.isclose(score, row.score, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Independent QLIKE derivative differs')
            max_score_error = max(max_score_error, abs(score-row.score)); actual_scores.append(score)
    audit = pd.DataFrame({'year': frame.year, 'score': actual_scores, 'weight': frame.weight})
    point = float(audit.score.sum()/audit.weight.sum())
    parts = [g[['score', 'weight']].to_numpy() for _, g in audit.groupby('year')]
    ci = {str(b): difference_interval(parts, b) for b in (20, 60)}
    np.testing.assert_allclose(point, result['mean_improvement_score'], atol=1e-11, rtol=0)
    for block, value in ci.items():
        np.testing.assert_allclose(value, result['score_ci_95'][block], atol=1e-10, rtol=0)
    yearly = {str(y): float(g.score.sum()/g.weight.sum()) for y, g in audit.groupby('year')}
    for year, value in yearly.items():
        np.testing.assert_allclose(value, result['yearly_score'][year], atol=1e-11, rtol=0)
    chosen = frame.loc[frame.eligible].nlargest(math.ceil(frame.weight.sum()*.01), 'baseline_loss').index
    trimmed = audit.drop(chosen)
    trim = float(trimmed.score.sum()/trimmed.weight.sum())
    direction = 1 if all(c[0] > 0 for c in ci.values()) else (-1 if all(c[1] < 0 for c in ci.values()) else 0)
    wins = sum(direction*v > 0 for v in yearly.values()) if direction else 0
    passed = bool(direction and wins >= 6 and direction*trim > 0)
    if (not math.isclose(trim, result['trim_top1pct_mean_score'], abs_tol=1e-11)
            or passed != result['score_gate_passed'] or direction != result['supported_direction']
            or wins != result['same_direction_years']):
        raise ValueError('Sensitivity/decision differs')
    return {'status': 'verified', 'new_forecast_model_fits': 0, 'native_outer_models': len(seen),
            'independent_feature_only_projections': len(seen), 'independent_scores': int(frame.weight.sum()),
            'independent_intervals': 2, 'retained_origins': len(frame),
            'max_basis_difference': max_basis_error, 'max_score_difference': max_score_error,
            'score_gate_passed': passed, 'market_skill': False,
            'result_sha256': sha(root/'result.json'), 'scores_sha256': sha(root/'scores.csv'),
            'verifier_sha256': sha(Path(__file__))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); receipt = verify(args.root)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
