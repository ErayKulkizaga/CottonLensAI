"""Bounded full-year stages on the existing Experiment/Ledger fit path."""
import importlib.metadata
import json
import platform
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from cottonlens_ml.code_identity import digest, research_source_identity
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.research.ledger import (
    FitBudgetReached,
    Ledger,
    freeze_record,
    read_record,
    writer,
)
from cottonlens_ml.research.protocol import full_year_manifest, rows_at
from cottonlens_ml.research.review import audit_market, verify_history
from cottonlens_ml.research.statistics import (
    bh_adjust,
    correlation,
    oos_r2,
    paired_bootstrap,
)
from cottonlens_ml.research.volatility import InvalidVarianceFit, ewma, features, qlike

SHRINKAGE = (0., .25, .5, .75, 1.)
HYPOTHESES = {
    'H1': 'T+1 selected shrunk price vs Naive price-MAE',
    'H2': 'T+5 selected shrunk price vs Naive price-MAE',
    'H3': 'T+1 inner-selected HAR/GARCH vs EWMA QLIKE',
    'H4': 'T+5 inner-selected HAR/GARCH vs EWMA QLIKE',
    'H5': 'T+1 normalized 90% interval vs Normal-EWMA interval score',
    'H6': 'T+5 normalized 90% interval vs Normal-EWMA interval score',
}


def price_recipes(h):
    base = {'horizon': h, 'task': 'price', 'device': 'cpu', 'seed': 42,
            'target': 'scaled_log', 'loss': 'reg:squarederror', 'window': 1,
            'features': list(FEATURE_NAMES), 'years': None, 'cadence': 21}
    return [*[{**base, 'family': 'ridge', 'params': {'alpha': a}} for a in (.1, 1., 10.)],
            *[{**base, 'family': 'elasticnet', 'params': {'alpha': a, 'l1_ratio': .5}} for a in (.001, .01)],
            {**base, 'family': 'xgboost', 'params': {'max_depth': 2, 'eta': .03, 'min_child_weight': 20,
                'alpha': 0, 'lambda': 1}, 'max_iterations': 600, 'patience': 50}]


def volatility_recipe(family, h):
    return {'family': family, 'horizon': h, 'device': 'cpu', 'seed': 42, 'cadence': 21,
            'task': 'variance', 'features': ['har_1', 'har_5', 'har_22'], 'years': None}


def prepare(repo, folder, reference):
    reference_ready, history = verify_history(reference)
    history, raw_audit = audit_market(reference, reference_ready, history)
    history = features(history)
    split = full_year_manifest(history)
    identity = {'version': 'full-year-pilot-v2-causal-intervals', 'profile': 'full-year-v1',
                'source_id': research_source_identity(repo)['source_id'],
                'reference_ready_sha256': digest(Path(reference) / 'ready.json'),
                'raw_audit': raw_audit,
                'research_data_id': frame_identity(history, list(history.columns)),
                'split': split, 'groups': {'base': list(FEATURE_NAMES)}, 'publication_sources': [],
                'python': sys.version.split()[0],
                'versions': {p: importlib.metadata.version(p) for p in
                             ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'arch', 'statsmodels')},
                'lock_sha256': digest(Path(repo) / 'ml/uv.lock'),
                'budgets': {'price_recipes_per_horizon': 6, 'volatility_families': 2, 'seeds': [42]},
                'execution': {'device': 'cpu', 'threads': 2},
                'interval_policy': INTERVAL_POLICY}
    with writer(folder):
        if (folder / 'ready.json').exists():
            ready = read_record(folder / 'ready.json')
            if ready['identity'] != identity or digest(folder / 'history.parquet') != ready['history_sha256']:
                raise ValueError('Frozen CPU source/environment changed; use a new namespace')
            return ready
        if (folder / 'history.parquet').exists():
            raise ValueError('Incomplete prepare; preserve and inspect it')
        history.to_parquet(folder / 'history.parquet', index=False)
        started = datetime.now(UTC)
        freeze_record(folder / 'preregistered.json', {'hypotheses': HYPOTHESES, 'created_at': started.isoformat(),
            'checkpoint_at': (started + timedelta(days=21)).isoformat(),
            'role': 'seen_historical_research_not_confirmatory_holdout',
            'price_gate': {'mae_gain_pct': 5, 'direction_pct': {'1': 53, '5': 55}, 'fold_wins': 6},
            'decision_table': {'P+U+': 'price_and_risk', 'P+U-': 'price', 'P-U+': 'naive_plus_risk', 'P-U-': 'bounded_new_information_diagnosis'},
            'cpu': {'threads': 2, 'parallel_jobs': 1}, 'platform': platform.platform()})
        ready = {'identity': identity, 'history_sha256': digest(folder / 'history.parquet'), 'created_at': started.isoformat()}
        freeze_record(folder / 'ready.json', ready)
        return ready


def chunks(history, origins, cadence=21):
    rows = rows_at(history, origins)
    start = rows.cotton_session_index.iloc[0]
    buckets = (rows.cotton_session_index - start) // cadence
    return [g for _, g in rows.groupby(buckets, sort=True)]


def predict_chunks(experiment, spec, origins, role, iterations=1):
    outputs = []
    field = 'variance_predictions' if spec['task'] == 'variance' else 'predictions'
    for i, chunk in enumerate(chunks(experiment.history, origins)):
        train = experiment.train_rows(chunk.date.min(), spec)
        record = experiment.fit(spec, train, None, chunk, f'{role}-{i}', iterations=iterations,
                                repeat=getattr(experiment, 'repeat_reason', None))
        outputs.extend(record['result'][field])
    return np.asarray(outputs)


def inner_price(experiment, spec, fold, *, weights=SHRINKAGE):
    blocks, counts = [], []
    for i, block in enumerate(fold['inner']):
        test = rows_at(experiment.history, block['origins'])
        count = 1
        if spec['family'] == 'xgboost':
            past = experiment.train_rows(test.date.min(), spec)
            if 'asset' in past:
                past = past.loc[past.asset.eq('cotton')]
            validation = past.tail(63)
            train = experiment.train_rows(validation.date.min(), spec)
            record = experiment.fit(spec, train, validation, validation, f'early-{fold["year"]}-{i}')
            count = record['result']['iterations']
        prediction = predict_chunks(experiment, spec, block['origins'], f'inner-{fold["year"]}-{i}', count)
        actual = test[f'target_return_{spec["horizon"]}'].to_numpy()
        prices = test.cotton_close.to_numpy()
        baseline = np.mean(prices * np.abs(np.expm1(actual)))
        if baseline <= 0:
            raise ValueError('Nonzero Naive error required')
        blocks.append([float(np.mean(prices * np.abs(np.exp(actual) - np.exp(w * prediction))) / baseline) for w in weights])
        counts.append(count)
    scores = np.mean(blocks, axis=0)
    index = int(np.argmin(scores))
    return {'recipe': spec, 'weight': weights[index], 'inner_score': float(scores[index]),
            'iterations': int(np.median(counts))}


def interval_rows(history, origin_frame, variance, variance_history, h):
    """252 strictly mature prequential standardized returns; no future outcomes."""
    rows = []
    available = variance_history.sort_values('date')
    for row, v in zip(origin_frame.itertuples(), variance, strict=True):
        past = available.loc[(available.date < row.date) & (available[f'target_date_{h}'] < row.date)].tail(252)
        if len(past) < 252:
            rows.append({'date': row.date, 'status': 'insufficient_mature_calibration'})
            continue
        residuals = past[f'target_return_{h}'].to_numpy() / np.sqrt(past.variance.to_numpy())
        raw = past[f'target_return_{h}'].to_numpy()
        item = {'date': row.date, 'status': 'available'}
        for coverage in (.8, .9):
            alpha = 1 - coverage
            levels = [alpha / 2, 1 - alpha / 2]
            for name, bounds in [('normalized', np.quantile(residuals, levels) * np.sqrt(v)),
                                 ('empirical', np.quantile(raw, levels)),
                                 ('normal_ewma', norm.ppf(levels) * np.sqrt(row.ewma_variance))]:
                item[f'{name}_{int(100*coverage)}_lo'], item[f'{name}_{int(100*coverage)}_hi'] = map(float, bounds)
        rows.append(item)
    return pd.DataFrame(rows)


def interval_metrics(actual, lo, hi, coverage):
    actual, lo, hi = (np.asarray(v, float) for v in (actual, lo, hi))
    alpha = 1 - coverage
    score = hi - lo + 2 / alpha * (np.maximum(lo - actual, 0) + np.maximum(actual - hi, 0))
    hit = (actual >= lo) & (actual <= hi)
    pin_lo = np.maximum((alpha / 2) * (actual - lo), (alpha / 2 - 1) * (actual - lo))
    pin_hi = np.maximum((1 - alpha / 2) * (actual - hi), (-alpha / 2) * (actual - hi))
    return {'coverage': float(hit.mean()), 'mean_width': float((hi-lo).mean()), 'interval_score': float(score.mean()),
            'pinball_lower': float(pin_lo.mean()), 'pinball_upper': float(pin_hi.mean())}, score, hit.astype(float)


INTERVAL_POLICY = {
    'version': 'causal-ewma-calibration-v1',
    'normalizer': 'fixed EWMA lambda=0.94 at every calibration and forecast origin',
    'mode': 'deterministic causal historical replay; not archived live predictions',
    'selection': 'independent of annual HAR/GARCH family and future outcomes',
}


def causal_intervals(history, origins, h):
    past = history.copy()
    past['variance'] = ewma(history, history.date, h)
    current = origins.copy()
    current['ewma_variance'] = ewma(history, current.date, h)
    return interval_rows(history, current, current.ewma_variance.to_numpy(), past, h)


def correct_intervals(folder):
    """Separate analysis identity: preserve all fitted prices/variances and old records."""
    originals = sorted((folder/'pilot-full-year').glob('*.json'))
    if len(originals) != 16:
        return folder
    expected_keys = {(year, h) for year in range(2016, 2024) for h in (1, 5)}
    if {(read_record(p)['year'], read_record(p)['horizon']) for p in originals} != expected_keys:
        raise ValueError('Sixteen unique frozen year/horizon outputs required')
    ready = read_record(folder/'ready.json')
    history = pd.read_parquet(folder/'history.parquet')
    if digest(folder/'history.parquet') != ready['history_sha256']:
        raise ValueError('Correction requires verified frozen history')
    source = {p.name: digest(p) for p in originals}
    policy = {**INTERVAL_POLICY, 'parent_ready_sha256': digest(folder/'ready.json'),
              'parent_outputs_sha256': source, 'correction_code_sha256': digest(Path(__file__)),
              'reason': 'annual family selection cannot determine earlier calibration scores',
              'model_training': False, 'price_and_variance_predictions_unchanged': True}
    target = folder/('interval-correction-'+content_id(policy)[:16])
    freeze_record(target/'ready.json', {**ready, 'derived_analysis': policy})
    for path in originals:
        original = read_record(path)
        h = original['horizon']
        frame = pd.DataFrame(original['records'])
        frame['date'] = pd.to_datetime(frame.date)
        expected = next(f['origins'] for f in ready['identity']['split']['folds'] if f['year'] == original['year'])
        if frame.date.dt.strftime('%Y-%m-%d').tolist() != expected:
            raise ValueError('Correction cannot change frozen origins')
        intervals = causal_intervals(history, frame, h)
        if not intervals.status.eq('available').all():
            raise ValueError('Incomplete causal interval replay; decision pending')
        for c in intervals:
            if c != 'date':
                frame[c] = intervals[c].to_numpy()
        frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
        freeze_record(target/'pilot-full-year'/path.name,
            {**original, 'records': json_predictions(frame), 'interval_policy': INTERVAL_POLICY,
             'parent_output_sha256': source[path.name]})
    return target


def json_predictions(data):
    """Optional GK gaps are null; primary predictions/targets must stay finite."""
    optional = {c for c in data if c.startswith('gk_target_')}
    required = data.select_dtypes(include='number').drop(columns=list(optional), errors='ignore')
    if not np.isfinite(required.to_numpy()).all():
        raise ValueError('Nonfinite primary prediction/label; output remains pending')
    return data.astype(object).where(data.notna(), None).to_dict('records')


def run(experiment, *, max_minutes=60):
    if not 0 < max_minutes <= 240:
        raise ValueError('CPU pilot session must be bounded by four hours')
    deadline = time.monotonic() + max_minutes * 60
    def before():
        if time.monotonic() >= deadline:
            raise FitBudgetReached('Planned CPU pause; rerun unchanged identity to resume')
    experiment.before_compute = before
    try:
        for fold in experiment.identity['split']['folds']:
            for h in (1, 5):
                namespace = 'reproduction/pilot-full-year' if getattr(experiment, 'repeat_reason', None) else 'pilot-full-year'
                marker = experiment.root / namespace / f't{h}-year{fold["year"]}.json'
                if marker.exists():
                    read_record(marker)
                    print(f'CACHE yearly output T+{h} {fold["year"]}', flush=True)
                    continue
                decision_path = experiment.root / 'decisions/full-year' / f't{h}-year{fold["year"]}.json'
                if decision_path.exists():
                    chosen = read_record(decision_path)
                else:
                    if getattr(experiment, 'repeat_reason', None):
                        raise ValueError('Reproduction requires frozen completed decisions')
                    prices = [inner_price(experiment, spec, fold) for spec in price_recipes(h)]
                    # Recipe order is complexity order; stable tie break.
                    price = min(enumerate(prices), key=lambda p: (p[1]['inner_score'], p[0]))[1]
                    vol, failures = [], []
                    for family in ('har', 'garch'):
                        spec = volatility_recipe(family, h)
                        scores = []
                        try:
                            for i, block in enumerate(fold['inner']):
                                test = rows_at(experiment.history, block['origins'])
                                pred = predict_chunks(experiment, spec, block['origins'], f'vol-inner-{fold["year"]}-{i}')
                                scores.append(float(qlike(test[f'variance_target_{h}'], pred).mean()))
                        except InvalidVarianceFit as exc:
                            failures.append({'family': family, 'status': 'unselectable', 'reason': str(exc)})
                            continue
                        vol.append({'recipe': spec, 'inner_score': float(np.mean(scores))})
                    if not vol:
                        raise ValueError('No complete variance candidate; pilot decision pending')
                    chosen = {'price': price, 'variance': min(vol, key=lambda v: v['inner_score']), 'year': fold['year'],
                              'horizon': h, 'all_price_candidates': prices, 'all_variance_candidates': vol, 'unselectable': failures}
                    freeze_record(decision_path, chosen)
                price, variance = chosen['price'], chosen['variance']
                test = rows_at(experiment.history, fold['origins']).copy()
                test['predicted_return'] = price['weight'] * predict_chunks(experiment, price['recipe'], fold['origins'],
                    f'price-outer-{fold["year"]}', price['iterations'])
                test['variance'] = predict_chunks(experiment, variance['recipe'], fold['origins'], f'vol-outer-{fold["year"]}')
                test['ewma_variance'] = ewma(experiment.history, test.date, h)
                intervals = causal_intervals(experiment.history, test, h)
                for c in intervals.columns:
                    if c != 'date':
                        test[c] = intervals[c].to_numpy()
                # Refit baselines on the exact same past cutoffs.
                test['median_return'], test['past_majority_sign'] = np.nan, np.nan
                for chunk in chunks(experiment.history, fold['origins']):
                    train = experiment.train_rows(chunk.date.min(), price['recipe'])
                    signs, counts = np.unique(np.sign(train[f'target_return_{h}']), return_counts=True)
                    use = test.date.isin(chunk.date)
                    test.loc[use, 'median_return'] = float(train[f'target_return_{h}'].median())
                    test.loc[use, 'past_majority_sign'] = float(signs[np.argmax(counts)])
                if not test.status.eq('available').all():
                    raise ValueError('Incomplete interval calibration: preserve fits; yearly output remains pending')
                columns = ['date', 'cotton_close', f'target_return_{h}', f'variance_target_{h}', 'predicted_return',
                           'variance', 'ewma_variance', 'median_return', 'past_majority_sign', 'status',
                           *[c for c in test if c.endswith(('_lo', '_hi'))]]
                if f'gk_target_{h}' in test:
                    columns.append(f'gk_target_{h}')
                data = test[columns].copy()
                data['date'] = data.date.dt.strftime('%Y-%m-%d')
                freeze_record(marker, {'year': fold['year'], 'horizon': h, 'decision_id': content_id(chosen),
                    'records': json_predictions(data), 'inner_score': price['inner_score'],
                    'price_family': price['recipe']['family'], 'volatility_family': variance['recipe']['family'],
                    'interval_policy': INTERVAL_POLICY})
                if getattr(experiment, 'mirror', None):
                    experiment.mirror.enqueue([marker.relative_to(experiment.root).as_posix(),
                                               decision_path.relative_to(experiment.root).as_posix()])
                    experiment.mirror.flush()
                print(f'SAVED {fold["year"]} T+{h}: {len(test)} origins', flush=True)
    except FitBudgetReached:
        return {'status': 'planned_pause', 'saved_year_horizon_outputs': len(list((experiment.root / 'pilot-full-year').glob('*.json')))}
    return {'status': 'complete', 'saved_year_horizon_outputs': 16}


def verify_reproduction(folder):
    comparisons = []
    for original in sorted((folder/'pilot-full-year').glob('*.json')):
        repeated = folder/'reproduction/pilot-full-year'/original.name
        if not repeated.exists():
            return {'status': 'pending', 'reason': 'fresh_selected_fits_not_all_complete'}
        a, b = (pd.DataFrame(read_record(p)['records']) for p in (original, repeated))
        if a.date.tolist() != b.date.tolist():
            raise ValueError('Reproduction changed frozen origins')
        price = float(np.max(np.abs(a.predicted_return-b.predicted_return)))
        variance = bool(np.allclose(a.variance, b.variance, rtol=1e-6, atol=1e-10))
        interval = max(float(np.max(np.abs(a[c]-b[c]))) for c in a if c.endswith(('_lo','_hi')))
        comparisons.append({'output': original.name, 'max_log_return_difference': price,
            'variance_parity': variance, 'max_interval_return_difference': interval,
            'passed': bool(price<=1e-6 and variance and interval<=1e-6)})
    if len(comparisons)!=16:
        return {'status': 'pending', 'reason': 'original_pilot_incomplete'}
    body = {'status': 'passed' if all(c['passed'] for c in comparisons) else 'failed',
            'fresh_fits': True, 'price_atol': 1e-6, 'variance_atol': 1e-10, 'variance_rtol': 1e-6,
            'interval_return_atol': 1e-6, 'comparisons': comparisons}
    freeze_record(folder/'reproduction/verification.json', body)
    return body


def report(folder, *, repetitions=10000):
    records = [read_record(p) for p in sorted((folder / 'pilot-full-year').glob('*.json'))]
    if len(records) != 16:
        return {'status': 'pending', 'complete_outputs': len(records), 'required_outputs': 16, 'decision': 'pending_not_negative'}
    hypotheses, horizons = {}, {}
    for h in (1, 5):
        parts, interval_parts, vol_parts, hit_parts = [], [], [], []
        price_wins, interval_wins, variance_wins, scores = 0, 0, 0, []
        all_frames = []
        for record in [r for r in records if r['horizon'] == h]:
            frame = pd.DataFrame(record['records'])
            a, pred, price = (frame[c].to_numpy() for c in (f'target_return_{h}', 'predicted_return', 'cotton_close'))
            naive, model = price * np.abs(np.expm1(a)), price * np.abs(np.exp(a) - np.exp(pred))
            parts.append(np.column_stack([naive, model]))
            price_wins += int(model.mean() < naive.mean())
            scores.append(record['inner_score'])
            vol_parts.append(np.column_stack([qlike(frame[f'variance_target_{h}'], frame.ewma_variance),
                                               qlike(frame[f'variance_target_{h}'], frame.variance)]))
            variance_wins += int(vol_parts[-1][:, 1].mean() < vol_parts[-1][:, 0].mean())
            if not frame.status.eq('available').all():
                raise ValueError('Insufficient interval coverage; incomplete output cannot decide')
            _, baseline, _ = interval_metrics(a, frame.normal_ewma_90_lo, frame.normal_ewma_90_hi, .9)
            _, candidate, hit = interval_metrics(a, frame.normalized_90_lo, frame.normalized_90_hi, .9)
            interval_parts.append(np.column_stack([baseline, candidate]))
            hit_parts.append(np.column_stack([hit, np.zeros(len(hit))]))
            interval_wins += int(candidate.mean() < baseline.mean())
            all_frames.append(frame)
        frame = pd.concat(all_frames, ignore_index=True)
        a = frame[f'target_return_{h}'].to_numpy()
        gk = None
        if f'gk_target_{h}' in frame:
            common = frame.loc[frame[f'gk_target_{h}'].notna() & (frame[f'gk_target_{h}']>=0)]
            if len(common):
                gk = {'count': len(common), 'model_qlike': float(qlike(common[f'gk_target_{h}'],common.variance).mean()),
                      'ewma_qlike': float(qlike(common[f'gk_target_{h}'],common.ewma_variance).mean()),
                      'role':'OHLC proxy common-origin sensitivity, not intraday realized variance'}
        metrics = {str(c): {method: interval_metrics(a, frame[f'{method}_{c}_lo'], frame[f'{method}_{c}_hi'], c/100)[0]
                   for method in ('normal_ewma', 'empirical', 'normalized')} for c in (80, 90)}
        direction_parts = [np.column_stack([np.sign(f[f'target_return_{h}']) == np.sign(f.predicted_return),
                         np.sign(f[f'target_return_{h}']) == f.past_majority_sign]).astype(float) for f in all_frames]
        flat = frame.predicted_return.to_numpy() == 0
        prices = frame.cotton_close.to_numpy()
        price_intervals = {str(c): {method: interval_metrics(prices * np.exp(a),
            prices * np.exp(frame[f'{method}_{c}_lo']), prices * np.exp(frame[f'{method}_{c}_hi']), c/100)[0]
            for method in ('normal_ewma', 'empirical', 'normalized')} for c in (80, 90)}
        bootstrap = {str(b): paired_bootstrap(parts, block=b, repetitions=repetitions,
                     normalization=float(np.concatenate(parts)[:, 0].mean())) for b in (10, 20, 40)}
        vol_bootstrap = paired_bootstrap(vol_parts, repetitions=repetitions)
        interval_bootstrap = paired_bootstrap(interval_parts, repetitions=repetitions)
        price_errors = np.concatenate(parts)
        gain = float(100 * (1 - price_errors[:, 1].mean()/price_errors[:, 0].mean()))
        direction = float(np.mean(np.sign(a) == np.sign(frame.predicted_return)) * 100)
        interval_gain = 100 * (1 - metrics['90']['normalized']['interval_score']/metrics['90']['normal_ewma']['interval_score'])
        p = np.mean(scores) <= .995 and gain > 0 and price_wins >= 5
        u = (.87 <= metrics['90']['normalized']['coverage'] <= .93
             and .77 <= metrics['80']['normalized']['coverage'] <= .83 and interval_gain >= 2 and interval_wins >= 5
             and metrics['90']['normalized']['interval_score'] <= metrics['90']['empirical']['interval_score'])
        horizons[str(h)] = {'mae_gain_pct': gain, 'direction_pct': direction, 'fold_wins': price_wins,
            'past_majority_direction_pct': float(np.mean(np.sign(a) == frame.past_majority_sign) * 100),
            'predicted_flat_count': int(flat.sum()),
            'nonflat_direction_pct_diagnostic_only': float(100*np.mean(np.sign(a[~flat]) == np.sign(frame.predicted_return.to_numpy()[~flat]))) if (~flat).any() else None,
            'spearman_ic': correlation(rankdata(a), rankdata(frame.predicted_return)),
            'paired_direction_vs_majority': paired_bootstrap(direction_parts, repetitions=repetitions),
            'price_mae': float(price_errors[:, 1].mean()), 'naive_price_mae': float(price_errors[:, 0].mean()),
            'inner_mae_gain_pct': float(100 * (1 - np.mean(scores))),
            'price_release_gate': bool(gain >= 5 and direction >= (53 if h == 1 else 55) and price_wins >= 6),
            'research_signals': {'P': bool(p), 'U': bool(u)},
            'decision': 'P' + ('+' if p else '-') + 'U' + ('+' if u else '-'),
            'price_bootstrap': bootstrap, 'variance_bootstrap': vol_bootstrap,
            'oos_r2_log_vs_naive': oos_r2(a, frame.predicted_return, np.zeros(len(a))),
            'oos_r2_price_vs_naive': oos_r2(prices*np.exp(a), prices*np.exp(frame.predicted_return), prices),
            'oos_r2_log_vs_median': oos_r2(a, frame.predicted_return, frame.median_return),
            'median_price_mae': float(np.mean(prices * np.abs(np.exp(a) - np.exp(frame.median_return)))),
            'price_intervals': price_intervals,
            'variance_mean_loss_difference': vol_bootstrap['paired_difference'],
            'variance_fold_wins': variance_wins, 'interval_fold_wins': interval_wins,
            'variance_research_signal': bool(vol_bootstrap['paired_difference'] > 0),
            'variance_confidence': 'positive_mean_but_uncertain' if vol_bootstrap['difference_ci_95'][0] <= 0 <= vol_bootstrap['difference_ci_95'][1] else 'historical_ci_excludes_zero',
            'gk_sensitivity': gk,
            'intervals': metrics, 'interval_gain_pct': interval_gain,
            'coverage_bootstrap': paired_bootstrap(hit_parts, repetitions=repetitions),
            'interval_bootstrap': interval_bootstrap}
        for index, body in ((h == 5, bootstrap['20']), (2 + (h == 5), vol_bootstrap), (4 + (h == 5), interval_bootstrap)):
            hypotheses[f'H{int(index)+1}'] = {'p_two_sided': body['centered_bootstrap_p_two_sided']}
    adjusted = bh_adjust([hypotheses[f'H{i}']['p_two_sided'] for i in range(1, 7)])
    for i, value in enumerate(adjusted, 1):
        hypotheses[f'H{i}']['bh_adjusted_p'] = value
    body = {'status': 'complete', 'evidence': 'reused_historical_research_not_independent_holdout',
            'horizons': horizons, 'hypotheses': hypotheses, 'wide_search_started': False,
            'interval_policy': records[0].get('interval_policy', {'status': 'invalid_annual_selection_calibration'}),
            'analysis_identity': read_record(folder/'ready.json').get('derived_analysis'),
            'reports_only_no_runtime_import': True,
            'ready_sha256': digest(folder / 'ready.json'),
            'input_outputs_sha256': {p.name: digest(p) for p in sorted((folder/'pilot-full-year').glob('*.json'))},
            'repetitions': repetitions, 'report_code_sha256': digest(Path(__file__))}
    from cottonlens_ml.research import statistics
    body['statistics_code_sha256'] = digest(Path(statistics.__file__))
    freeze_record(folder / 'reports' / ('pilot-decision-' + content_id(body)[:16] + '.json'), body)
    return body


def dispatch(args):
    from cottonlens_ml.research.engine import Experiment, root_path
    folder = root_path(args.drive_root, args.experiment)
    if args.mirror_root:
        from cottonlens_ml.research.mirror import Mirror
        metadata_only = args.stage in ('status', 'compare', 'report', 'pilot-plan')
        if not (metadata_only and (folder/'ready.json').exists()
                and len(list((folder/'pilot-full-year').glob('*.json'))) == 16):
            print('Restoring immutable metadata' if metadata_only else 'Restoring resumable experiment payloads', flush=True)
            Mirror(folder, root_path(args.mirror_root, args.experiment)).hydrate(metadata_only=metadata_only)
    if args.stage == 'prepare':
        if not args.reference_root:
            raise ValueError('Explicit immutable reference-root required')
        ready = prepare(args.repo, folder, args.reference_root)
        if args.mirror_root:
            from cottonlens_ml.research.mirror import Mirror
            mirror = Mirror(folder, root_path(args.mirror_root, args.experiment))
            mirror.enqueue(['ready.json','history.parquet','preregistered.json'])
            mirror.flush()
        print(json.dumps({'status': 'frozen', 'split_id': ready['identity']['split']['split_id'],
                          'origin_count': sum(len(f['origins']) for f in ready['identity']['split']['folds'])}))
    elif args.stage in ('compare', 'report'):
        corrected = correct_intervals(folder)
        print(json.dumps(report(corrected), indent=2))
        if args.mirror_root and corrected != folder:
            from cottonlens_ml.research.mirror import Mirror
            mirror = Mirror(folder, root_path(args.mirror_root, args.experiment))
            mirror.enqueue([p.relative_to(folder).as_posix() for p in corrected.rglob('*.json')])
            mirror.flush()
    elif args.stage == 'status':
        print(json.dumps({'prepared': (folder/'ready.json').exists(), 'writer_lock_present': (folder/'.writer-lock').exists(),
            'saved_outputs': len(list((folder/'pilot-full-year').glob('*.json'))), 'required_outputs': 16,
            'ledger': Ledger(folder/'ledger', {}).summary()}))
    elif args.stage == 'pilot-plan':
        ready = read_record(folder/'ready.json')
        history = pd.read_parquet(folder/'history.parquet')
        inner_chunks = sum(sum(len(chunks(history, b['origins'])) for b in f['inner']) for f in ready['identity']['split']['folds'])
        outer_chunks = sum(len(chunks(history, f['origins'])) for f in ready['identity']['split']['folds'])
        fit_count = 2 * (8 * inner_chunks + 3*8 + 2*outer_chunks)
        print(json.dumps({'hypotheses': HYPOTHESES, 'price_recipes': 12, 'seeds': [42],
            'maximum_fit_jobs_before_numerical_exclusions': fit_count,
            'origin_count': sum(len(f['origins']) for f in ready['identity']['split']['folds']),
            'refit_cadence': 21, 'max_session_minutes': 240, 'gate_changed': False}))
    elif args.stage == 'track':
        from cottonlens_ml.research.tracking import project
        with writer(folder):
            project(Experiment(folder, repo=args.repo))
    elif args.stage in ('pilot', 'reproduce'):
        with writer(folder):
            experiment = Experiment(folder, repo=args.repo)
            if args.stage == 'reproduce':
                if report(folder)['status'] != 'complete':
                    raise ValueError('Complete pilot report required before deliberate reproduction')
                experiment.repeat_reason = 'reproduction'
            if args.mirror_root:
                from cottonlens_ml.research.mirror import Mirror
                experiment.mirror = Mirror(folder, root_path(args.mirror_root, args.experiment))
                experiment.mirror.enqueue(['ready.json', 'history.parquet', 'preregistered.json'])
                experiment.mirror.flush()
                experiment.after_fit = experiment.mirror.fit
            print(json.dumps(run(experiment, max_minutes=args.max_minutes)))
            if args.stage == 'reproduce':
                print(json.dumps(verify_reproduction(folder)))
    else:
        raise ValueError('Full-year profile supports status/prepare/pilot-plan/pilot/compare/report; conditional release is not automatic')
