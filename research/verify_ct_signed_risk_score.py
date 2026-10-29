"""Independent Decimal endpoints, pseudoinverse projection and signed QLIKE audit."""

import argparse
import json
import math
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from research.verify_risk_weak_teacher import (
    difference_interval,
    fingerprint,
    read,
    sha,
)


def independent_history(history):
    """Rebuild return, risk state and HAR from close; never use future labels."""
    rows = history.copy()
    prices = [Decimal(str(v)) for v in rows.cotton_close]
    with localcontext() as context:
        context.prec = 40
        returns = [None]+[(b/a).ln() for a, b in pairwise(prices)]
        squared = [None if v is None else v*v for v in returns]
        variance = None; ewma = []; har = []
        for i, value in enumerate(squared):
            if value is not None:
                variance = value if variance is None else Decimal('.94')*variance+Decimal('.06')*value
            ewma.append(np.nan if variance is None else float(max(variance, Decimal('1e-12'))))
            features = []
            for window in (1, 5, 22):
                part = squared[max(0, i-window+1):i+1]
                features.append(np.nan if len(part) != window or None in part else
                                float(max(sum(part)/window, Decimal('1e-12')).ln()))
            har.append(features)
        signed = [np.nan if r is None else float((1 if r >= 0 else -1)*v/Decimal(str(b)))
                  for r, v, b in zip(returns, squared, ewma, strict=True)]
    rows['checked_ewma'] = ewma
    rows['checked_shock'] = signed
    for k, name in enumerate(('har_1', 'har_5', 'har_22')):
        rows['checked_'+name] = np.array(har)[:, k]
        np.testing.assert_allclose(rows[name], rows['checked_'+name], rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(rows.ct_ewma.iloc[1:], rows.checked_ewma.iloc[1:], rtol=1e-11, atol=1e-14)
    rows['checked_actual'] = [float(v) if v is not None else np.nan for v in squared[1:]]+[np.nan]
    return rows.set_index(rows.date.dt.strftime('%Y-%m-%d'))


def verify(root):
    result = read(root/'result.json'); transforms = read(root/'transformations.json')
    ready = read(root/'ready.json'); contract = read(root/'decision-contract.json')
    if (result['market_skill'] or result['new_forecast_fits'] or contract['training_allowed']
            or result['retained_origins'] != 2006 or len(transforms) != 97):
        raise ValueError('Wrong zero-fit scope')
    for name, checksum in ready['inputs_sha256'].items():
        if sha(root/name) != checksum:
            raise ValueError('Frozen input mismatch')
    for name, checksum in read(root/'parent-files.json')['files'].items():
        if sha(root/'reference'/name) != checksum:
            raise ValueError('Immutable native parent mismatch')
    if sha(root/'scores.csv') != result['scores_sha256'] or sha(root/'transformations.json') != result['transformations_sha256']:
        raise ValueError('Score or projection hash mismatch')
    reference = root/'reference'; parent = read(reference/'ready.json')
    proof = json.loads((root/'parent-proof.json').read_bytes())
    if (sha(reference/'reports/result.json') != proof['result_sha256']
            or sha(reference/'reports/paired-losses.csv') != proof['paired_losses_sha256']
            or parent['identity']['source_id'] != proof['source_id']):
        raise ValueError('Registered reference differs')
    h = independent_history(pd.read_parquet(reference/'history.parquet'))
    frame = pd.read_csv(root/'scores.csv', float_precision='round_trip')
    old_frame = pd.read_csv(reference/'reports/paired-losses.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(frame[list(old_frame)], old_frame, check_exact=True)
    dates = [d for f in parent['identity']['design']['split']['folds'] for d in f['origins']]
    if frame.date.tolist() != dates or frame.target_date.tolist() != h.loc[dates].target_date_1.dt.strftime('%Y-%m-%d').tolist():
        raise ValueError('Origin or target endpoints differ')
    np.testing.assert_allclose(frame.actual_variance, h.loc[dates].checked_actual, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(frame.ewma_variance, h.loc[dates].checked_ewma, rtol=1e-10, atol=1e-14)
    jobs = {(tuple(j['train_dates']), tuple(j['origins'])): j
            for j in parent['identity']['design']['jobs'] if j['role'] == 'outer'}
    checked = {}; seen = set(); maximum_basis_error = 0.
    for fit_id, metadata in transforms.items():
        record = read(reference/'ledger/completed'/f'{fit_id}.json')
        spec = record['specification']; key = tuple(spec['train_dates']), tuple(spec['test_dates'])
        if (fingerprint({'identity': record['identity'], 'specification': spec}) != fit_id
                or key not in jobs or key in seen):
            raise ValueError('Native identity or split mismatch')
        job = jobs[key]; seen.add(key)
        for name, checksum in record['files'].items():
            if sha(reference/'ledger'/name) != checksum:
                raise ValueError('Native model payload mismatch')
        train, test = h.loc[list(key[0])], h.loc[list(key[1])]
        if (not train.target_date_5.lt(pd.Timestamp(job['cutoff'])).all()
                or metadata['cutoff'] != job['cutoff'] or metadata['train_count'] != len(train)
                or metadata['train_dates_sha256'] != fingerprint(list(key[0]))):
            raise ValueError('Projection uses unavailable history')
        names = ['checked_har_1', 'checked_har_5', 'checked_har_22']
        x = train[names].to_numpy()-np.log(train.checked_ewma.to_numpy())[:, None]
        xt = test[names].to_numpy()-np.log(test.checked_ewma.to_numpy())[:, None]
        raw = np.column_stack([x, np.abs(train.checked_shock)])
        raw_test = np.column_stack([xt, np.abs(test.checked_shock)])
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.)
        z = np.column_stack([np.ones(len(train)), (raw-mean)/scale])
        zt = np.column_stack([np.ones(len(test)), (raw_test-mean)/scale])
        # Separate Moore-Penrose algorithm; no production basis/score import.
        coefficients = np.linalg.pinv(z)@train.checked_shock.to_numpy()
        residual = train.checked_shock.to_numpy()-z@coefficients
        rms = float(np.linalg.norm(residual)/math.sqrt(len(train)))
        phi = (test.checked_shock.to_numpy()-zt@coefficients)/rms
        np.testing.assert_allclose(coefficients, metadata['projection_coef'], atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(rms, metadata['residual_rms'], atol=1e-10, rtol=1e-10)
        model = json.loads((reference/'ledger'/record['payload_root']/'variance-model.json').read_bytes())
        prediction = test.checked_ewma.to_numpy()*np.exp(np.column_stack([np.ones(len(test)), xt])@model['coef'])
        np.testing.assert_allclose(prediction, record['result']['variance_predictions'], rtol=1e-10, atol=1e-14)
        for i, day in enumerate(job['origins']):
            checked[day] = (phi[i], prediction[i], fit_id)
    if len(seen) != 97 or set(checked) != set(dates):
        raise ValueError('Native outer coverage differs')
    values = []
    for row in frame.itertuples():
        phi, prediction, fit_id = checked[row.date]
        maximum_basis_error = max(maximum_basis_error, abs(phi-row.phi))
        np.testing.assert_allclose(phi, row.phi, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(prediction, row.raw_variance, rtol=1e-10, atol=1e-14)
        if fit_id != row.native_fit_id:
            raise ValueError('Origin-native link mismatch')
        with localcontext() as context:
            context.prec = 40
            value = float(Decimal(str(row.phi))*(Decimal(str(row.actual_variance))/Decimal(str(row.raw_variance))-1))
        np.testing.assert_allclose(value, row.score, rtol=1e-11, atol=1e-11)
        values.append(value)
    audit = pd.DataFrame({'year': frame.year, 'value': values, 'weight': 1})
    intervals = {str(b): difference_interval([g[['value', 'weight']].to_numpy() for _, g in audit.groupby('year')], b)
                 for b in (20, 60)}
    for block, interval in intervals.items():
        np.testing.assert_allclose(interval, result['score_ci'][block], rtol=0, atol=1e-10)
    yearly = {str(y): float(g.value.mean()) for y, g in audit.groupby('year')}
    np.testing.assert_allclose(list(yearly.values()), list(result['yearly_score'].values()), rtol=1e-12, atol=1e-11)
    trimmed = audit.drop(frame.nlargest(math.ceil(len(frame)*.01), 'baseline_loss').index)
    trim = float(trimmed.value.mean())
    np.testing.assert_allclose(trim, result['trim_top1pct_score'], atol=1e-11, rtol=1e-12)
    direction = 1 if all(v[0] > 0 for v in intervals.values()) else -1 if all(v[1] < 0 for v in intervals.values()) else 0
    passed = bool(direction and sum(v*direction > 0 for v in yearly.values()) >= 6 and trim*direction > 0)
    if passed != result['direction_gate_passed'] or direction != result['supported_direction']:
        raise ValueError('Score conclusion differs')
    return {'status': 'verified', 'new_forecast_fits': 0, 'market_skill': False,
            'ordinal_origins': len(frame), 'native_outer_fits': len(seen),
            'past_only_feature_projections': len(seen), 'decimal_close_rows': len(h),
            'decimal_qlike_derivatives': len(values), 'independent_intervals': len(intervals),
            'maximum_basis_error': maximum_basis_error, 'direction_gate_passed': passed}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve independent receipt')
    receipt = verify(args.root)
    args.output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))
