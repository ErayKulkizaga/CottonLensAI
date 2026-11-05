"""Fixed, past-only price-MAE calibration of already fitted Ridge models."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cottonlens_ml.code_identity import digest, research_source_identity, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.ledger import freeze_record, read_record
from cottonlens_ml.research.protocol import Preprocessor
from cottonlens_ml.research.review import training_rows
from cottonlens_ml.research.statistics import paired_bootstrap

PROFILE = 'price-mae-calibration-v1'
WEIGHTS = np.array([0., .25, .5, .75, 1.])


def factor(actual_price, predicted_price):
    """Exact minimizer of sum(abs(A-f*P)); deterministic lower weighted median."""
    a, p = np.asarray(actual_price, float), np.asarray(predicted_price, float)
    if (a.ndim != 1 or a.shape != p.shape or not len(a) or
            not np.isfinite([a, p]).all() or (a <= 0).any() or (p <= 0).any()):
        raise ValueError('Aligned finite positive prices required')
    ratio = a / p
    if not np.isfinite(ratio).all():
        raise ValueError('Nonfinite price ratio')
    order = np.argsort(ratio, kind='stable')
    cumulative = np.cumsum(p[order])
    if not np.isfinite(cumulative).all():
        raise ValueError('Nonfinite cumulative price weight')
    return float(ratio[order[np.searchsorted(cumulative, cumulative[-1] / 2, side='left')]])


def past_calibration(train, prediction, cutoff):
    cutoff = pd.Timestamp(cutoff)
    if (train.empty or train.date.max() >= cutoff or train.target_date_5.isna().any() or
            train.target_date_5.max() >= cutoff):
        raise ValueError('Only fully matured training labels before cutoff allowed')
    prediction = np.asarray(prediction, float)
    if prediction.shape != (len(train),) or not np.isfinite(prediction).all():
        raise ValueError('Aligned finite training inference required')
    with np.errstate(over='ignore', invalid='ignore'):
        actual = train.cotton_close.to_numpy() * np.exp(train.target_return_1.to_numpy())
        predicted = train.cotton_close.to_numpy() * np.exp(prediction)
    value = factor(actual, predicted)
    before, after = float(np.mean(np.abs(actual - predicted))), float(np.mean(np.abs(actual - value * predicted)))
    if after > before + 1e-12 * max(1., before):
        raise ValueError('Analytic calibration increased training MAE')
    return {'factor': value, 'log_shift': float(np.log(value)), 'training_rows': len(train),
            'training_mae_before': before, 'training_mae_after': after,
            'cutoff': cutoff.isoformat(), 'last_mature_label': train.target_date_5.max().isoformat()}


def inference(rows, adapter, linear):
    processor = Preprocessor(**adapter['processor'])
    if adapter['target']['kind'] != 'scaled_log':
        raise ValueError('Frozen standardized log-return adapter required')
    x = processor.transform(rows)
    z = x @ np.asarray(linear['coef'], np.float32) + np.float32(linear['intercept'])
    return z * adapter['target']['scale'] + adapter['target']['mean']


def errors(frame, field):
    actual = frame.cotton_close.to_numpy() * np.exp(frame.actual_return.to_numpy())
    predicted = frame.cotton_close.to_numpy() * np.exp(frame[field].to_numpy())
    if not np.isfinite([actual, predicted]).all():
        raise ValueError('Finite price reconstructions required')
    return np.abs(actual - frame.cotton_close.to_numpy()), np.abs(actual - predicted)


def metrics(frame, field):
    naive, loss = errors(frame, field)
    a, p = frame.actual_return.to_numpy(), frame[field].to_numpy()
    return {'count': len(frame), 'price_mae': float(loss.mean()), 'naive_mae': float(naive.mean()),
            'naive_gain_pct': float(100 * (1 - loss.mean() / naive.mean())),
            'direction_pct': float(100 * np.mean(np.sign(a) == np.sign(p))),
            'active_rate_pct': float(100 * np.mean(p != 0)),
            'active_direction_pct': float(100 * np.mean(np.sign(a[p != 0]) == np.sign(p[p != 0]))) if np.any(p != 0) else None}


def run(experiment, reference_manifest, output):
    if os.environ.get('COTTONLENS_ALLOW_LOCAL_CPU_TABULAR') != '1':
        raise RuntimeError('Explicit local CPU tabular allowance required')
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    if output == experiment or output.is_relative_to(experiment):
        raise ValueError('Do not write into old evidence')
    manifest = json.loads(Path(reference_manifest).read_bytes())
    inputs = {}

    def checked(name):
        key = 'experiment/' + name
        if not safe_member(name) or key not in manifest['members']:
            raise ValueError('Unregistered original artifact: ' + name)
        path = experiment / name
        expected = manifest['members'][key]['sha256']
        if digest(path) != expected:
            raise ValueError('Original artifact changed: ' + name)
        inputs[name] = expected
        return path

    ready = read_record(checked('ready.json'))
    history = pd.read_parquet(checked('history.parquet'))
    if digest(experiment / 'history.parquet') != ready['history_sha256']:
        raise ValueError('Ready/history mismatch')
    identity, design = ready['identity'], ready['identity']['design']
    if ([f['year'] for f in design['split']['folds']] != list(range(2016, 2024)) or
            design['shrinkage_weights'] != WEIGHTS.tolist()):
        raise ValueError('Fixed historical design required')
    recipe = {**design['recipe'], 'features': design['groups']['numeric_D0']}
    if recipe['family'] != 'ridge' or recipe['params'] != {'alpha': 1.} or recipe['horizon'] != 1:
        raise ValueError('Only the fixed reference Ridge T+1 recipe admitted')
    source = research_source_identity(Path(__file__).resolve().parents[4])
    settings = {'profile': PROFILE, 'source_id': source['source_id'], 'reference_manifest_sha256': digest(Path(reference_manifest)),
                'reference_ready_sha256': inputs['ready.json'], 'reference_source_id': identity['source_id'],
                'formula': 'lower weighted median of A/P with weights P, solely on matured model-training rows',
                'new_model_fits': 0, 'scalar_calibration_budget': 169, 'threads': 2,
                'shrinkage': WEIGHTS.tolist(), 'bootstrap_blocks': [20, 60], 'bootstrap_repetitions': 10000,
                'seed': 42, 'independent_holdout': False, 'release_allowed': False}
    output.mkdir(parents=True, exist_ok=True)
    freeze_record(output / 'source-manifest.json', source)
    freeze_record(output / 'settings.json', settings)
    indexed, updates, rows, weights = history.set_index('date', drop=False), {}, [], []

    def predictions(origins, role):
        test = indexed.loc[pd.to_datetime(origins)]
        buckets = (test.cotton_session_index - test.cotton_session_index.iloc[0]) // 21
        raw, calibrated = [], []
        for j, (_, chunk) in enumerate(test.groupby(buckets, sort=True)):
            train = training_rows(history, chunk.date.min(), recipe, identity['split'].get('coverage_start'))
            spec = {'recipe': recipe, 'role': role + '-' + str(j), 'iterations': 1, 'repeat_reason': None,
                    'train_dates': train.date.dt.strftime('%Y-%m-%d').tolist(), 'validation_dates': [],
                    'test_dates': chunk.date.dt.strftime('%Y-%m-%d').tolist(), 'train_identity': frame_identity(train, list(train))}
            key = content_id({'identity': identity, 'specification': spec})
            record = read_record(checked(f'ledger/completed/{key}.json'))
            if record['identity'] != identity or record['specification'] != spec or record['result']['origins'] != spec['test_dates']:
                raise ValueError('Reference receipt does not match exact cohort')
            payloads = {}
            for name, expected in record['files'].items():
                path = checked('ledger/' + name)
                if digest(path) != expected:
                    raise ValueError('Model payload changed')
                if name.endswith(('adapter.json', 'model.json')):
                    payloads[Path(name).name] = json.loads(path.read_bytes())
            adapter, linear = payloads['adapter.json'], payloads['model.json']
            if adapter['processor']['names'] != recipe['features']:
                raise ValueError('Feature representation changed')
            saved = np.asarray(record['result']['predictions'], float)
            np.testing.assert_allclose(inference(chunk, adapter, linear), saved, rtol=0, atol=1e-7)
            if key not in updates and len(updates) >= settings['scalar_calibration_budget']:
                raise ValueError('Scalar calibration budget exceeded')
            calibration = past_calibration(train, inference(train, adapter, linear), chunk.date.min())
            body = {'reference_fit_id': key, 'settings_id': content_id(settings), **calibration}
            freeze_record(output / f'calibration-updates/{key}.json', body)
            updates[key] = body
            raw.extend(saved)
            calibrated.extend(saved + calibration['log_shift'])
        return test, np.asarray(raw), np.asarray(calibrated)

    with threadpool_limits(limits=2):
        for fold in design['split']['folds']:
            original_scores, calibrated_scores = [], []
            for i, block in enumerate(fold['inner']):
                test, raw, calibrated = predictions(block['origins'], f'inner-{fold["year"]}-{i}')
                if test.target_date_5.max() >= pd.Timestamp(fold['origins'][0]):
                    raise ValueError('Inner labels not mature before outer decision')
                price, actual = test.cotton_close.to_numpy(), test.target_return_1.to_numpy()
                baseline = np.mean(price * np.abs(np.exp(actual) - 1))
                for values, scores in [(raw, original_scores), (calibrated, calibrated_scores)]:
                    scores.append([float(np.mean(price * np.abs(np.exp(actual) - np.exp(w * values))) / baseline) for w in WEIGHTS])
            original_index = int(np.argmin(np.mean(original_scores, axis=0)))
            calibrated_index = int(np.argmin(np.mean(calibrated_scores, axis=0)))
            decision = read_record(checked(f'nass-regional-decisions/numeric_D0-t1-year{fold["year"]}.json'))
            if decision['selection_used_outer'] or decision['selected']['weight'] != WEIGHTS[original_index]:
                raise ValueError('Original past-only selection mismatch')
            np.testing.assert_allclose(decision['selected']['inner_score'], np.mean(original_scores, axis=0)[original_index], rtol=1e-12, atol=1e-12)
            test, raw, calibrated = predictions(fold['origins'], f'path-outer-numeric_D0-{fold["year"]}')
            original = pd.DataFrame(read_record(checked(f'nass-regional-outputs/numeric_D0-t1-year{fold["year"]}.json'))['records'])
            np.testing.assert_array_equal(original.date, test.date.dt.strftime('%Y-%m-%d'))
            np.testing.assert_array_equal(pd.to_datetime(original.target_date), test.target_date_1)
            np.testing.assert_array_equal(original.actual_return, test.target_return_1)
            np.testing.assert_array_equal(original.raw_predicted_return, raw)
            np.testing.assert_array_equal(original.predicted_return, raw * WEIGHTS[original_index])
            frame = original.copy()
            if not frame.horizon.eq(1).all():
                raise ValueError('T+1 origins required')
            frame['year'] = fold['year']
            frame['control_raw'], frame['control_selected'] = raw, raw * WEIGHTS[original_index]
            frame['candidate_raw'], frame['candidate_selected'] = calibrated, calibrated * WEIGHTS[calibrated_index]
            frame['calibration_log_shift'] = calibrated - raw
            frame['candidate_weight'] = WEIGHTS[calibrated_index]
            rows.append(frame)
            weights.append({'year': fold['year'], 'control': float(WEIGHTS[original_index]), 'candidate': float(WEIGHTS[calibrated_index]),
                            'candidate_inner_scores': np.mean(calibrated_scores, axis=0).tolist()})
            print(f'VERIFIED T+1 year{fold["year"]}; scalar updates {len(updates)}', flush=True)
    combined = pd.concat(rows, ignore_index=True)
    if len(combined) != 2006 or combined.date.duplicated().any() or len(updates) != 169:
        raise ValueError('Exact 2006 origins/169 calibration updates required')
    report_metrics, comparisons = {}, {}
    keep = np.ones(len(combined), bool)
    naive = errors(combined, 'control_selected')[0]
    keep[np.argsort(-naive, kind='stable')[:21]] = False
    for field in ('control_raw', 'control_selected', 'candidate_raw', 'candidate_selected'):
        score = metrics(combined, field)
        score['years'] = {str(f.year.iloc[0]): metrics(f, field) for f in rows}
        score['year_wins'] = sum(v['naive_gain_pct'] > 1e-10 for v in score['years'].values())
        score['without_top_1pct_naive_errors'] = metrics(combined.loc[keep], field)
        parts = [np.column_stack(errors(f, field)) for f in rows]
        score['versus_naive'] = {str(b): paired_bootstrap(parts, block=b, repetitions=10000, seed=42, normalization=float(naive.mean())) for b in (20, 60)}
        report_metrics[field] = score
    for mode in ('raw', 'selected'):
        parts = [np.column_stack([errors(f, 'control_' + mode)[1], errors(f, 'candidate_' + mode)[1]]) for f in rows]
        comparisons[mode] = {str(b): paired_bootstrap(parts, block=b, repetitions=10000, seed=42, normalization=report_metrics['control_' + mode]['price_mae']) for b in (20, 60)}
    selected = report_metrics['candidate_selected']
    positive = all(comparisons['selected'][str(b)]['difference_ci_95'][0] > 0 for b in (20, 60))
    practical = selected['naive_gain_pct'] >= 5 and selected['direction_pct'] >= 53 and selected['year_wins'] >= 6
    upper_below_five = all(100 * selected['versus_naive'][str(b)]['difference_ci_95'][1] / selected['naive_mae'] < 5 for b in (20, 60))
    decision = ('STRONG_FIXED_RECIPE_CANDIDATE' if practical and positive else
                'SMALL_CALIBRATION_CONTRIBUTION' if positive else
                'CALIBRATION_DOES_NOT_RESCUE_FIXED_RECIPE' if upper_below_five else 'INCONCLUSIVE')
    export = output / 'outer-predictions.csv'
    encoded = combined.to_csv(index=False, lineterminator='\n').encode()
    if export.exists() and export.read_bytes() != encoded:
        raise ValueError('Frozen prediction export changed')
    if not export.exists():
        export.write_bytes(encoded)
    report = {'profile': PROFILE, 'settings': settings, 'input_hashes': inputs, 'new_model_fits': 0,
              'scalar_calibration_updates': len(updates), 'origin_count': 2006, 'metrics': report_metrics,
              'comparisons': comparisons, 'weights': weights, 'decision': decision,
              'predictions_sha256': digest(export), 'independent_holdout': False, 'release_allowed': False,
              'historical_source_admitted': False,
              'scope': 'Only a global price-MAE multiplicative bias correction is tested; full conditional quantile modeling and universal signal absence are not tested.'}
    freeze_record(output / 'calibration-report.json', report)
    return {'decision': decision, 'metrics': {k: v['naive_gain_pct'] for k, v in report_metrics.items()}, 'new_model_fits': 0, 'scalar_calibration_updates': len(updates)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.reference_manifest, args.output), indent=2))
