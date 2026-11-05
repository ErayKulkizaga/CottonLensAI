"""No-fit independent replay of the bounded synthetic Ridge receipts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cottonlens_ml.code_identity import digest, safe_member
from cottonlens_ml.cohort import content_id, frame_identity
from cottonlens_ml.research.ledger import read_record
from cottonlens_ml.research.review import training_rows


def replay(folder, namespace='weak-signal'):
    """Rebuild preprocessing, inference and selection without calling fit code."""
    folder = Path(folder)
    ready = read_record(folder / 'ready.json')
    identity = ready['identity']
    if digest(folder / 'history.parquet') != ready['history_sha256']:
        raise ValueError('Replay history changed')
    history = pd.read_parquet(folder / 'history.parquet')
    indexed = history.set_index('date', drop=False)
    design = identity['design']
    features = design['groups']['numeric_D0']
    expected_recipe = {**design['recipe'], 'features': features}
    paths = sorted((folder / 'ledger/completed').glob('*.json'))
    if len(paths) != 169:
        raise ValueError('All 169 synthetic fit receipts required')
    lookup, maximum = {}, 0.
    with threadpool_limits(limits=2):
        for path in paths:
            record = read_record(path)
            spec = record['specification']
            if (record['identity'] != identity or record['experiment_id'] != path.stem
                    or path.stem != content_id({'identity': identity, 'specification': spec})
                    or spec['recipe'] != expected_recipe or spec['validation_dates']
                    or not record['files'] or spec['role'] in lookup):
                raise ValueError('Replay identity/recipe/role changed')
            test = indexed.loc[pd.to_datetime(spec['test_dates'])]
            train = training_rows(history, test.date.min(), expected_recipe, identity['split'].get('coverage_start'))
            if (spec['train_dates'] != train.date.dt.strftime('%Y-%m-%d').tolist()
                    or spec['train_identity'] != frame_identity(train, list(train))
                    or train.target_date_5.max() >= test.date.min()):
                raise ValueError('Replay training cohort or label maturity changed')
            for name, expected in record['files'].items():
                if not safe_member(name) or digest(folder / 'ledger' / name) != expected:
                    raise ValueError('Replay model payload changed')
            adapter = json.loads((folder / 'ledger' / next(n for n in record['files'] if n.endswith('adapter.json'))).read_bytes())
            linear = json.loads((folder / 'ledger' / next(n for n in record['files'] if n.endswith('model.json'))).read_bytes())
            if adapter['processor']['names'] != features or adapter['target']['kind'] != 'scaled_log':
                raise ValueError('Replay adapter changed')
            matrix = train[features].to_numpy(dtype=float)
            finite = np.isfinite(matrix)
            median = np.array([np.median(matrix[finite[:, i], i]) if finite[:, i].any() else 0.
                               for i in range(len(features))])
            filled = np.where(finite, matrix, median)
            scale = np.where(filled.std(axis=0) > 1e-12, filled.std(axis=0), 1.)
            for name, value in [('median', median), ('mean', filled.mean(axis=0)), ('scale', scale)]:
                np.testing.assert_allclose(adapter['processor'][name], value, rtol=0, atol=1e-12)
            y = train.target_return_1.to_numpy(dtype=float)
            if adapter['target']['mean'] != float(y.mean()) or adapter['target']['scale'] != max(float(y.std()), 1e-8):
                raise ValueError('Replay target transformation used different labels')
            matrix = test[features].to_numpy(dtype=float)
            finite = np.isfinite(matrix)
            x = (np.where(finite, matrix, median) - filled.mean(axis=0)) / scale
            x = np.concatenate([x, (~finite).astype(float)], axis=1).astype(np.float32)
            standardized = x @ np.asarray(linear['coef'], dtype=np.float32) + np.float32(linear['intercept'])
            predicted = standardized * adapter['target']['scale'] + adapter['target']['mean']
            saved = np.asarray(record['result']['predictions'])
            np.testing.assert_allclose(predicted, saved, rtol=0, atol=1e-7)
            maximum = max(maximum, float(np.max(np.abs(predicted - saved))))
            lookup[spec['role']] = record
    used, selections = set(), []

    def reconstruct(dates, role):
        rows = indexed.loc[pd.to_datetime(dates)]
        buckets = (rows.cotton_session_index - rows.cotton_session_index.iloc[0]) // 21
        predictions = []
        for j, (_, chunk) in enumerate(rows.groupby(buckets, sort=True)):
            key = role + '-' + str(j)
            record = lookup[key]
            if record['specification']['test_dates'] != chunk.date.dt.strftime('%Y-%m-%d').tolist():
                raise ValueError('Replay forecast origins changed')
            predictions.extend(record['result']['predictions'])
            used.add(key)
        return rows, np.asarray(predictions)

    for fold in design['split']['folds']:
        scores = []
        for i, block in enumerate(fold['inner']):
            rows, raw = reconstruct(block['origins'], f'inner-{fold["year"]}-{i}')
            c, a = rows.cotton_close.to_numpy(), rows.target_return_1.to_numpy()
            baseline = np.mean(np.abs(c * np.exp(a) - c))
            scores.append([np.mean(np.abs(c * np.exp(a) - c * np.exp(w * raw))) / baseline
                           for w in design['shrinkage_weights']])
        scores = np.mean(scores, axis=0)
        name = f'numeric_D0-t1-year{fold["year"]}.json'
        selected = read_record(folder / f'{namespace}-decisions' / name)['selected']
        expected = design['shrinkage_weights'][int(np.argmin(scores))]
        if selected['weight'] != expected:
            raise ValueError('Replay shrinkage selection changed')
        np.testing.assert_allclose(selected['inner_score'], scores.min(), rtol=1e-12, atol=1e-12)
        rows, raw = reconstruct(fold['origins'], f'path-outer-numeric_D0-{fold["year"]}')
        frame = pd.DataFrame(read_record(folder / f'{namespace}-outputs' / name)['records'])
        np.testing.assert_array_equal(raw, frame.raw_predicted_return)
        np.testing.assert_array_equal(raw * expected, frame.predicted_return)
        selections.append({'year': fold['year'], 'weight': expected})
    if used != set(lookup):
        raise ValueError('Unregistered or unused fit receipts')
    return {'new_fits': 0, 'fit_payloads_replayed': len(lookup), 'past_only_selections_replayed': len(selections),
            'maximum_saved_inference_log_difference': maximum, 'selected_weights': selections}


def independent_interval(parts, block, repetitions):
    """Rebuild paired within-year draws without the research statistics helper."""
    rng = np.random.default_rng(42)
    count = sum(len(a) for a in parts)
    draws = []
    for start in range(0, repetitions, 128):
        batch = min(128, repetitions - start)
        totals = np.zeros((batch, 2))
        for array in parts:
            width = min(block, len(array))
            starts = rng.integers(0, len(array) - width + 1, (batch, int(np.ceil(len(array) / width))))
            indices = (starts[..., None] + np.arange(width)).reshape(batch, -1)[:, :len(array)]
            totals += array[indices].sum(axis=1)
        draws.append(totals / count)
    draws = np.concatenate(draws)
    combined = np.concatenate(parts)
    delta = combined[:, 0].mean() - combined[:, 1].mean()
    paired = combined[:, 0] - combined[:, 1]
    if np.ptp(paired) <= 32 * np.finfo(float).eps * max(1., np.max(np.abs(paired))):
        differences = np.full(repetitions, delta)
    else:
        differences = draws[:, 0] - draws[:, 1]
    noise = differences - differences.mean()
    return delta - np.quantile(noise, [.975, .025])


def verify_report(root):
    """Independent price arithmetic and bootstrap replay from the exported CSV."""
    root = Path(root)
    paths = list((root / 'reports').glob('weak-signal-*.json'))
    if len(paths) != 1:
        raise ValueError('Exactly one frozen synthetic report required')
    report = read_record(paths[0])
    export = root / 'reports/outer-predictions.csv'
    if digest(export) != report['predictions_sha256']:
        raise ValueError('Report CSV changed')
    # Pandas otherwise treats the literal null condition as a missing value and
    # silently drops all ten negative-control seeds during groupby.
    rows = pd.read_csv(export, float_precision='round_trip', keep_default_na=False)
    if len(rows) != 40120 or report['market_fits'] != 0 or report['synthetic_fits'] != 3380:
        raise ValueError('Report scope changed')
    names = {f'{condition}-seed{seed}' for condition, seed in rows.groupby(['condition', 'seed']).groups}
    if len(names) != 20 or names != set(report['scenarios']):
        raise ValueError('All twenty conditions/seeds required in independent report replay')
    metrics, intervals = {}, 0
    for (condition, seed), frame in rows.groupby(['condition', 'seed'], sort=True):
        if len(frame) != 2006 or not frame.horizon.eq(1).all() or frame.date.duplicated().any():
            raise ValueError('Report forecast cohort changed')
        name = f'{condition}-seed{seed}'
        scores = report['scenarios'][name]['modes']
        actual_price = frame.cotton_close.to_numpy() * np.exp(frame.actual_return.to_numpy())
        baseline = np.abs(actual_price - frame.cotton_close.to_numpy())
        for mode, field in [('raw', 'raw_predicted_return'), ('selected', 'predicted_return'), ('oracle', 'oracle_return')]:
            forecast = frame.cotton_close.to_numpy() * np.exp(frame[field].to_numpy())
            error = np.abs(actual_price - forecast)
            saved = scores[mode]
            np.testing.assert_allclose([error.mean(), baseline.mean()], [saved['price_mae'], saved['naive_mae']], rtol=0, atol=1e-12)
            gain = float(100 * (1 - error.mean() / baseline.mean()))
            np.testing.assert_allclose(gain, saved['naive_gain_pct'], rtol=0, atol=1e-10)
            direction = float(100 * np.mean(np.sign(frame.actual_return) == np.sign(frame[field])))
            np.testing.assert_allclose(direction, saved['direction_pct'], rtol=0, atol=1e-12)
            parts = []
            for year in sorted(frame.year.unique()):
                mask = frame.year.eq(year).to_numpy()
                parts.append(np.column_stack([baseline[mask], error[mask]]))
                np.testing.assert_allclose(error[mask].mean(), saved['years'][str(year)]['price_mae'], rtol=0, atol=1e-12)
            for block in (20, 60):
                rebuilt = independent_interval(parts, block, report['bootstrap_repetitions'])
                ci = saved['versus_naive'][str(block)]
                np.testing.assert_allclose(rebuilt, ci['difference_ci_95'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(100 * rebuilt / baseline.mean(), ci['gain_ci_pct'], rtol=0, atol=1e-10)
                intervals += 1
            metrics[name + '/' + mode] = {'price_mae': float(error.mean()), 'naive_mae': float(baseline.mean()),
                                         'gain_pct': gain}
    return {'verified': True, 'new_fits': 0, 'market_fits': 0, 'prediction_rows': len(rows),
            'scenario_modes_verified': len(metrics), 'independent_bootstrap_intervals': intervals,
            'report_sha256': digest(paths[0]), 'predictions_sha256': digest(export),
            'metrics': metrics, 'decision': report['decision'], 'synthetic_only': True}
