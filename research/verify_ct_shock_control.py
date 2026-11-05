"""Independent endpoints, native coefficients, past selections and paired losses."""

import argparse
import json
import math
from bisect import bisect_right
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


def profile_definition(profile):
    # Deliberately independent of the engine's mapping and feature assembly.
    if profile == 'ct-signed-unsigned-shock-forecast-t1-v1':
        return ('unsigned', 'signed'), 'signed', 'unsigned', 'origin_shock', True
    if profile == 'ct-risk-level-dxy-control-t1-v1':
        return ('own_level', 'dxy_state'), 'dxy_state', 'own_level', 'origin_state', False
    raise ValueError('Unknown frozen profile')


def independent_dxy(market, dates):
    """Raw log-difference rolling window and merge_asof, not engine searchsorted."""
    dates = pd.DatetimeIndex(dates)
    quotes = market.loc[market.series.eq('dxy'), ['date', 'close']].copy().sort_values('date')
    quotes['date'] = pd.to_datetime(quotes.date)
    if (not len(quotes) or quotes.date.isna().any() or quotes.date.duplicated().any()
            or not quotes.date.eq(quotes.date.dt.normalize()).all()
            or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing):
        raise ValueError('Unique chronological source/origin dates required')
    price = quotes.close.where(np.isfinite(quotes.close) & quotes.close.gt(0))
    quotes['dxy_rv5'] = np.log(price).diff().pow(2).rolling(5, min_periods=5).mean()
    quotes['dxy_state_window_start'] = quotes.date.shift(5)
    quotes['dxy_state_available_at'] = pd.to_datetime(quotes.date, utc=True)+pd.Timedelta(days=1)
    quotes = quotes.rename(columns={'date': 'dxy_state_source_date'})
    origins = pd.DataFrame({'date': dates, 'decision_at': pd.to_datetime(dates, utc=True)+pd.Timedelta(days=1, minutes=15)})
    joined = pd.merge_asof(origins, quotes.drop(columns='close'), left_on='decision_at',
                           right_on='dxy_state_available_at', direction='backward', allow_exact_matches=True)
    joined['dxy_state_age'] = [np.nan if pd.isna(day) else i-(bisect_right(dates, day)-1)
                               for i, day in enumerate(joined.dxy_state_source_date)]
    joined['dxy_rv5'] = joined.dxy_rv5.where(joined.dxy_state_age.between(0, 3))
    return joined


def preflight(root):
    ready = read(root/'ready.json'); design = read(root/'decision-contract.json')
    identity = ready['identity']; manifest = read(root/'source-manifest.json')
    arms, primary, _control, extra_name, require_trim = profile_definition(identity['profile'])
    source_body = {k: v for k, v in manifest.items() if k != 'source_id'}
    if (fingerprint(source_body) != identity['source_id'] or fingerprint(design) != identity['design_id']
            or design['fit_budget'] != {'market': 338, 'synthetic': 1, 'processes': 1, 'threads': 2, 'session_minutes': 30}
            or design['profile'] != identity['profile']
            or set(design['recipes']) != set(arms)
            or design['primary_candidate'] != primary+' selected variance versus unchanged EWMA0.94'
            or design['trim_is_sensitivity'] != (not require_trim)):
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
    np.testing.assert_array_equal(augmented.date, history.date)
    if extra_name == 'origin_shock':
        np.testing.assert_allclose(augmented.signed_shock, history.checked_shock, rtol=1e-10, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(augmented.unsigned_shock, np.abs(history.checked_shock), rtol=1e-10, atol=1e-12, equal_nan=True)
    else:
        if sha(root/'inputs/market.parquet') != '23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c':
            raise ValueError('Declared held raw source changed')
        states = independent_dxy(pd.read_parquet(root/'inputs/market.parquet'), history.date)
        np.testing.assert_allclose(states.dxy_rv5, augmented.dxy_rv5, rtol=1e-10, atol=1e-14, equal_nan=True)
        np.testing.assert_allclose(states.dxy_state_age, augmented.dxy_state_age, rtol=0, atol=0, equal_nan=True)
        for name in ('dxy_state_source_date', 'dxy_state_available_at', 'dxy_state_window_start', 'decision_at'):
            pd.testing.assert_series_equal(states[name].reset_index(drop=True), augmented[name].reset_index(drop=True), check_names=False)
        for name in states.columns.drop('date'):
            history[name] = states[name].to_numpy()
        history['checked_state_own_level'] = np.log(history.checked_ewma)
        history['checked_state_dxy_state'] = np.log(np.maximum(history.dxy_rv5, 1e-12))
        for arm in arms:
            np.testing.assert_allclose(history['checked_state_'+arm], augmented['state_'+arm], rtol=1e-10, atol=1e-12, equal_nan=True)
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
        if (extra_name == 'origin_state'
                and not np.isfinite(history.loc[job['train_dates']+job['origins'], ['checked_state_'+a for a in arms]]).all().all()):
            raise ValueError('No source-unready train or forecast origin may be dropped')
    return ready, design, parent, history, {'status': 'independently_verified_readiness', 'new_fits': 0,
        'ordinal_origins': len(dates), 'past_only_jobs': len(jobs), 'minimum_mature_train': min(counts),
        'decimal_close_rows': len(history), 'declared_market_fit_ceiling': 338,
        'market_skill': False, 'source_id': identity['source_id'], 'design_id': identity['design_id']}


def verify(root):
    ready, design, parent, h, receipt = preflight(root)
    arms, primary, control, extra_name, require_trim = profile_definition(ready['identity']['profile'])
    report = read(root/'reports/result.json')
    frame = pd.read_csv(root/'reports/paired-losses.csv', float_precision='round_trip')
    original = pd.read_csv(root/'reference/reports/paired-losses.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(frame[list(original)], original, check_exact=True)
    if sha(root/'reports/paired-losses.csv') != report['paired_losses_sha256'] or len(frame) != 2006:
        raise ValueError('Forecast payload or population differs')
    if extra_name == 'origin_state':
        rows = h.loc[frame.date]
        for name in ('dxy_rv5', 'dxy_state_age'):
            np.testing.assert_allclose(frame[name], rows[name], rtol=1e-10, atol=1e-14)
        for name in ('dxy_state_source_date', 'dxy_state_available_at', 'dxy_state_window_start', 'decision_at'):
            np.testing.assert_array_equal(pd.to_datetime(frame[name], utc=True), pd.to_datetime(rows[name], utc=True))
    learning_paths = list((root/'learning/ledger/completed').glob('*.json'))
    if len(learning_paths) != 1 or list((root/'learning/ledger/attempts').glob('*.json')):
        raise ValueError('Missing/failed native learning control')
    learning = read(learning_paths[0])
    if (learning['identity'] != {**ready['identity'], 'role': 'synthetic_not_market'}
            or fingerprint({'identity': learning['identity'], 'specification': learning['specification']}) != learning['experiment_id']
            or learning['specification']['recipe'] != design['recipes'][primary]):
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
    forecasts = {a: {} for a in arms}; seen = set(); prediction_count = 0; gradient_max = 0.
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
        names = ['har_1', 'har_5', 'har_22', extra_name]
        if model['features'] != names or len(model['coef']) != 5 or model['effective_parameters'] != 5:
            raise ValueError('Declared feature was not fitted')
        train, test = h.loc[list(key[0])], h.loc[list(key[1])]

        def features(rows, representation=arm):
            common = rows[['checked_har_1', 'checked_har_5', 'checked_har_22']].to_numpy()-np.log(rows.checked_ewma.to_numpy())[:, None]
            if extra_name == 'origin_state':
                extra = rows['checked_state_'+representation].to_numpy()
            else:
                shock = rows.checked_shock.to_numpy()
                extra = shock if representation == 'signed' else np.abs(shock)
            return np.column_stack([common, extra])

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
    contrasts.update({f'{primary}_{k}_vs_{control}': (f'{control}_{k}_loss', f'{primary}_{k}_loss') for k in ('raw', 'selected')})
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
        passed = bool(delta.mean() > 0 and all(v[0] > 0 for v in ci.values()) and sum(v > 0 for v in yearly.values()) >= 6
                      and (not require_trim or trimmed > 0))
        if passed != claimed['fixed_risk_gate_passed']:
            raise ValueError('Fixed risk gate differs')
    if (report['primary_gate_passed'] != report['comparisons'][primary+'_selected_vs_EWMA']['fixed_risk_gate_passed']
            or report['trim_is_sensitivity'] != (not require_trim)):
        raise ValueError('Primary gate or sensitivity scope differs')
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
