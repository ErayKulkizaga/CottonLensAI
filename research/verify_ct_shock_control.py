"""Independent endpoints, native coefficients, past selections and paired losses."""

import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd

from research.verify_ct_signed_risk_score import independent_history
from research.verify_risk_weak_teacher import (
    difference_interval,
    fingerprint,
    read,
    sha,
)


def preflight(root):
    ready = read(root/'ready.json'); design = read(root/'decision-contract.json')
    identity = ready['identity']; manifest = read(root/'source-manifest.json')
    source_body = {k: v for k, v in manifest.items() if k != 'source_id'}
    if (fingerprint(source_body) != identity['source_id'] or fingerprint(design) != identity['design_id']
            or design['fit_budget'] != {'market': 338, 'synthetic': 1, 'processes': 1, 'threads': 2, 'session_minutes': 30}
            or design['primary_candidate'] != 'signed selected variance versus unchanged EWMA0.94'):
        raise ValueError('Frozen source/design/budget mismatch')
    for name, checksum in {**manifest['ml']['files'], **manifest['helpers']}.items():
        if sha(root/'source-snapshot'/name) != checksum:
            raise ValueError('Frozen source snapshot changed')
    for name, checksum in ready['frozen_hashes'].items():
        if sha(root/name) != checksum:
            raise ValueError('Frozen data/native input changed')
    parent = read(root/'reference/ready.json'); parent_identity = parent['identity']
    if parent_identity['design_id'] != design['reference_design_id']:
        raise ValueError('Native parent design differs')
    history = independent_history(pd.read_parquet(root/'reference/history.parquet'))
    augmented = pd.read_parquet(root/'history.parquet')
    np.testing.assert_allclose(augmented.signed_shock, history.checked_shock, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(augmented.unsigned_shock, np.abs(history.checked_shock), rtol=1e-10, atol=1e-12, equal_nan=True)
    jobs = parent_identity['design']['jobs']; dates = [d for f in parent_identity['design']['split']['folds'] for d in f['origins']]
    if len(jobs) != 169 or len(dates) != 2006 or len(set(dates)) != 2006:
        raise ValueError('Original job/origin population changed')
    counts = []
    for job in jobs:
        train = history.loc[job['train_dates']]
        if (len(train) < 500 or not train.date.lt(pd.Timestamp(job['cutoff'])).all()
                or not train.target_date_5.lt(pd.Timestamp(job['cutoff'])).all()
                or pd.Timestamp(job['origins'][0]) < pd.Timestamp(job['cutoff'])):
            raise ValueError('Unavailable or compressed training history')
        counts.append(len(train))
    return ready, design, parent, history, {'status': 'independently_verified_readiness', 'new_fits': 0,
        'ordinal_origins': len(dates), 'past_only_jobs': len(jobs), 'minimum_mature_train': min(counts),
        'decimal_close_rows': len(history), 'declared_market_fit_ceiling': 338,
        'market_skill': False, 'source_id': identity['source_id'], 'design_id': identity['design_id']}


def verify(root):
    ready, design, parent, h, receipt = preflight(root)
    report = read(root/'reports/result.json')
    frame = pd.read_csv(root/'reports/paired-losses.csv', float_precision='round_trip')
    original = pd.read_csv(root/'reference/reports/paired-losses.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(frame[list(original)], original, check_exact=True)
    if sha(root/'reports/paired-losses.csv') != report['paired_losses_sha256'] or len(frame) != 2006:
        raise ValueError('Forecast payload or population differs')
    learning_paths = list((root/'learning/ledger/completed').glob('*.json'))
    if len(learning_paths) != 1 or list((root/'learning/ledger/attempts').glob('*.json')):
        raise ValueError('Missing/failed native learning control')
    learning = read(learning_paths[0])
    if (learning['identity'] != {**ready['identity'], 'role': 'synthetic_not_market'}
            or fingerprint({'identity': learning['identity'], 'specification': learning['specification']}) != learning['experiment_id']
            or learning['specification']['recipe'] != design['recipes']['signed']):
        raise ValueError('Native learning control identity differs')
    for name, checksum in learning['files'].items():
        if sha(root/'learning/ledger'/name) != checksum:
            raise ValueError('Native learning payload changed')
    m = json.loads((root/'learning/ledger'/learning['payload_root']/'variance-model.json').read_bytes())
    rng = np.random.default_rng(42); x = rng.uniform(-2, 2, (1200, 4))
    baseline = np.exp(rng.uniform(-12, -4, 1200))
    expected = baseline[1000:]*np.exp(.1+x[1000:]@np.array([.2, .1, -.1, -.15]))
    replay = baseline[1000:]*np.exp(m['coef'][0]+x[1000:]@np.array(m['coef'][1:]))
    np.testing.assert_allclose(replay, expected, rtol=1e-5)
    np.testing.assert_allclose(replay, learning['result']['variance_predictions'], rtol=1e-12, atol=1e-14)
    jobs = {(tuple(j['train_dates']), tuple(j['origins'])): j for j in parent['identity']['design']['jobs']}
    forecasts = {'unsigned': {}, 'signed': {}}; seen = set(); prediction_count = 0; gradient_max = 0.
    paths = sorted((root/'ledger/completed').glob('*.json'))
    if len(paths) != 338 or list((root/'ledger/attempts').glob('*.json')):
        raise ValueError('Missing/failed native market fit')
    for path in paths:
        record = read(path); spec = record['specification']; arm = spec['recipe']['arm']
        key = tuple(spec['train_dates']), tuple(spec['test_dates'])
        if (arm not in forecasts or spec['recipe'] != design['recipes'][arm]
                or record['identity'] != ready['identity'] or key not in jobs or (arm, key) in seen
                or fingerprint({'identity': record['identity'], 'specification': spec}) != record['experiment_id']):
            raise ValueError('Native identity/recipe/split mismatch')
        seen.add((arm, key)); job = jobs[key]
        for name, checksum in record['files'].items():
            if sha(root/'ledger'/name) != checksum:
                raise ValueError('Native payload changed')
        model = json.loads((root/'ledger'/record['payload_root']/'variance-model.json').read_bytes())
        names = ['har_1', 'har_5', 'har_22', 'origin_shock']
        if model['features'] != names or len(model['coef']) != 5 or model['effective_parameters'] != 5:
            raise ValueError('Declared feature was not fitted')
        train, test = h.loc[list(key[0])], h.loc[list(key[1])]

        def features(rows, representation=arm):
            common = rows[['checked_har_1', 'checked_har_5', 'checked_har_22']].to_numpy()-np.log(rows.checked_ewma.to_numpy())[:, None]
            shock = rows.checked_shock.to_numpy()
            return np.column_stack([common, shock if representation == 'signed' else np.abs(shock)])

        x, xt = features(train), features(test); o = model['optimization']
        np.testing.assert_allclose(x.mean(axis=0), o['training_feature_mean'], rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(x.std(axis=0), o['training_feature_scale'], rtol=1e-10, atol=1e-11)
        eta = model['coef'][0]+x@np.array(model['coef'][1:])
        ratio = train.checked_actual.to_numpy()/train.checked_ewma.to_numpy()*np.exp(-eta)
        z = np.column_stack([np.ones(len(train)), (x-o['training_feature_mean'])/o['training_feature_scale']])
        gradient = z.T@(1-ratio)/len(train); norm = float(np.max(np.abs(gradient)))
        if not o['success'] or norm > 1.002e-6 or not math.isclose(float(np.mean(eta+ratio)), o['final_training_qlike'], abs_tol=1e-10):
            raise ValueError('Independent optimizer certificate differs')
        gradient_max = max(gradient_max, norm)
        predicted = test.checked_ewma.to_numpy()*np.exp(model['coef'][0]+xt@np.array(model['coef'][1:]))
        np.testing.assert_allclose(predicted, record['result']['variance_predictions'], rtol=1e-10, atol=1e-14)
        if record['result']['origins'] != job['origins']:
            raise ValueError('Native prediction date mismatch')
        role = 'inner' if job['role'].startswith('inner_') else 'outer'
        for day, value in zip(job['origins'], predicted, strict=True):
            k = job['year'], role, day
            if k in forecasts[arm]:
                raise ValueError('Overlapping native forecast')
            forecasts[arm][k] = float(value)
        prediction_count += len(predicted)
    if len(seen) != 338 or prediction_count != 7036:
        raise ValueError('Exact native prediction coverage required')
    selection_count = 0
    for arm, arm_forecasts in forecasts.items():
        for fold in parent['identity']['design']['split']['folds']:
            year = fold['year']; inner = {day: p for (y, role, day), p in arm_forecasts.items() if y == year and role == 'inner'}
            if len(inner) != 189:
                raise ValueError('Validation coverage differs')
            source = h.loc[list(inner)]; actual = source.checked_actual.to_numpy(); baseline = source.checked_ewma.to_numpy()
            candidates = []
            for w in (0., .25, .5, .75, 1.):
                p = w*np.array(list(inner.values()))+(1-w)*baseline
                candidates.append((float(np.mean(np.log(p)+actual/p)), w))
            chosen = min(candidates)[1]
            if chosen != report['past_selected_weights'][arm][str(year)]:
                raise ValueError('Past-only selection differs')
            sub = frame.loc[frame.year.eq(year)]
            for row in sub.itertuples():
                raw = arm_forecasts[year, 'outer', row.date]
                selected = chosen*raw+(1-chosen)*float(h.loc[row.date, 'checked_ewma'])
                np.testing.assert_allclose([raw, selected], [getattr(row, arm+'_raw'), getattr(row, arm+'_selected')], rtol=1e-10, atol=1e-14)
                if getattr(row, arm+'_weight') != chosen:
                    raise ValueError('Per-origin selection weight differs')
            selection_count += 1
    np.testing.assert_allclose(frame.actual_variance, h.loc[frame.date].checked_actual, rtol=1e-10, atol=1e-14)
    if frame.target_date.tolist() != h.loc[frame.date].target_date_1.dt.strftime('%Y-%m-%d').tolist():
        raise ValueError('Target endpoints differ')
    loss = {}; count = 0
    columns = {'baseline_loss': 'ewma_variance', **{f'{a}_{k}_loss': f'{a}_{k}' for a in forecasts for k in ('raw', 'selected')}}
    with localcontext() as context:
        context.prec = 40
        for name, column in columns.items():
            loss[name] = np.array([float(Decimal(str(p)).ln()+Decimal(str(y))/Decimal(str(p)))
                                   for y, p in zip(frame.actual_variance, frame[column], strict=True)])
            np.testing.assert_allclose(loss[name], frame[name], rtol=1e-10, atol=1e-12)
            count += len(frame)
    contrasts = {f'{a}_{k}_vs_EWMA': ('baseline_loss', f'{a}_{k}_loss') for a in forecasts for k in ('raw', 'selected')}
    contrasts.update({f'signed_{k}_vs_unsigned': (f'unsigned_{k}_loss', f'signed_{k}_loss') for k in ('raw', 'selected')})
    interval_count = 0
    for name, (a, b) in contrasts.items():
        delta = loss[a]-loss[b]
        audit = pd.DataFrame({'year': frame.year, 'delta': delta, 'weight': 1})
        ci = {str(k): difference_interval([g[['delta', 'weight']].to_numpy() for _, g in audit.groupby('year')], k) for k in (20, 60)}
        claimed = report['comparisons'][name]
        yearly = {str(y): float(g.delta.mean()) for y, g in audit.groupby('year')}
        trimmed = audit.drop(frame.nlargest(math.ceil(len(frame)*.01), 'baseline_loss').index).delta.mean()
        np.testing.assert_allclose(delta.mean(), claimed['gain'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(list(yearly.values()), list(claimed['yearly_gain'].values()), atol=1e-12, rtol=0)
        np.testing.assert_allclose(trimmed, claimed['trim_top1pct_gain'], atol=1e-12, rtol=0)
        for block, interval in ci.items():
            np.testing.assert_allclose(interval, claimed['gain_ci'][block], atol=1e-11, rtol=0)
            interval_count += 1
        passed = bool(delta.mean() > 0 and all(v[0] > 0 for v in ci.values()) and sum(v > 0 for v in yearly.values()) >= 6 and trimmed > 0)
        if passed != claimed['fixed_risk_gate_passed']:
            raise ValueError('Fixed risk gate differs')
    receipt.update(status='independently_verified_completed_forecasts', native_market_fits=338,
                   native_learning_fits=1, native_learning_predictions_reverified=200,
                   native_forecast_scalars=prediction_count, decimal_loss_scalars=count,
                   independent_loss_intervals=interval_count, past_only_year_selections=selection_count,
                   optimizer_gradient_inf_max=gradient_max, primary_gate_passed=report['primary_gate_passed'])
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve independent receipt')
    result = preflight(args.root)[-1] if args.preflight else verify(args.root)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
