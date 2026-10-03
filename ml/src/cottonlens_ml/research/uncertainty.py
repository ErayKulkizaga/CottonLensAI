"""Delayed-label prequential residual intervals; empirical, not IID coverage claims."""
import math

import numpy as np
import pandas as pd

from cottonlens_ml.research.protocol import rows_at


def inner_predictions(chosen):
    if chosen['recipe']['family'] == 'ensemble':
        members = [inner_predictions(m['decision']) for m in chosen['members']]
        origins = members[0][0]
        if any(dates != origins for dates, _ in members):
            raise ValueError('Interval calibration requires matched inner origins')
        return origins, np.asarray([pred for _, pred in members]).T @ chosen['weights']
    seeds = chosen['seeds']
    origins = seeds[0]['origins']
    if any(s['origins'] != origins for s in seeds):
        raise ValueError('Interval calibration requires matched seed origins')
    return origins, np.mean([s['predictions'] for s in seeds], axis=0)


def residual_intervals(history, calibration_origins, calibration_predictions, origins, predictions, horizon,
                       *, window=126):
    if horizon not in (1, 5) or window < 126:
        raise ValueError('Use a supported horizon and at least 126 mature errors')
    calibration = rows_at(history, calibration_origins).copy()
    test = rows_at(history, origins).copy()
    predicted = np.asarray(predictions, dtype=float)
    previous = np.asarray(calibration_predictions, dtype=float)
    if len(predicted) != len(test) or len(previous) != len(calibration) or not np.isfinite(predicted).all() or not np.isfinite(previous).all():
        raise ValueError('Finite aligned predictions required')
    if set(calibration_origins) & set(origins) or not test.date.is_monotonic_increasing or test.date.duplicated().any():
        raise ValueError('Chronological disjoint calibration and test origins required')
    if calibration.date.max() >= test.date.min():
        raise ValueError('Initial calibration must precede the test block')
    calibration['predicted'] = previous
    test['predicted'] = predicted
    available = pd.concat([calibration, test], ignore_index=True).sort_values('date')
    output = []
    for row in test.itertuples():
        # Future labels exist in a backtest DataFrame but are never consulted until
        # their maturity date is strictly earlier than this prediction origin.
        mature = available.loc[(available.date < row.date) &
            (available[f'target_date_{horizon}'] < row.date)].tail(window)
        scores = np.abs(mature[f'target_return_{horizon}'].to_numpy() - mature.predicted.to_numpy())
        scores = scores[np.isfinite(scores)]
        item = {'origin': row.date.strftime('%Y-%m-%d'), 'mature_calibration_count': len(scores)}
        if len(scores) < window:
            item['status'] = 'insufficient_mature_calibration'
        else:
            item['status'] = 'available'
            for coverage in (.8, .9):
                rank = math.ceil((len(scores) + 1) * coverage)
                radius = float(np.sort(scores)[min(rank, len(scores)) - 1])
                bounds = row.cotton_close * np.exp([row.predicted - radius, row.predicted + radius])
                if not np.isfinite(bounds).all():
                    raise ValueError('Nonfinite interval bounds')
                item[str(int(coverage * 100))] = {'lower': float(bounds[0]), 'upper': float(bounds[1]),
                                                  'log_residual_radius': radius}
        output.append(item)
    return {'method': 'rolling_absolute_log_residual', 'window': window,
            'coverage_claim': 'empirical_only_dependent_data_and_selected_model',
            'label_availability': 'target_date_h strictly before origin', 'rows': output}


def evaluate_intervals(history, intervals, horizon):
    """Evaluation only: outcomes cannot modify the previously generated intervals."""
    result = {}
    valid = [r for r in intervals['rows'] if r['status'] == 'available']
    if not valid:
        return {'count': 0}
    rows = rows_at(history, [r['origin'] for r in valid])
    actual = rows.cotton_close.to_numpy() * np.exp(rows[f'target_return_{horizon}'].to_numpy())
    for coverage in (80, 90):
        lower = np.array([r[str(coverage)]['lower'] for r in valid])
        upper = np.array([r[str(coverage)]['upper'] for r in valid])
        alpha = 1 - coverage / 100
        score = upper - lower + 2 / alpha * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
        result[str(coverage)] = {'coverage': float(np.mean((actual >= lower) & (actual <= upper))),
                                'mean_width': float(np.mean(upper - lower)), 'interval_score': float(score.mean())}
    return {'count': len(valid), 'levels': result}


def for_candidate(history, chosen, origins, predictions):
    dates, past = inner_predictions(chosen)
    horizon = chosen['recipe']['horizon']
    intervals = residual_intervals(history, dates, past, origins, predictions, horizon)
    return {**intervals, 'evaluation': evaluate_intervals(history, intervals, horizon)}
