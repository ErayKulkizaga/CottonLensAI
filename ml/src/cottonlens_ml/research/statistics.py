"""Paired, fold-preserving historical diagnostics; no post-hoc gate changes."""
import numpy as np
from scipy.stats import rankdata


def oos_r2(actual, prediction, benchmark):
    a, p, b = (np.asarray(x, dtype=float) for x in (actual, prediction, benchmark))
    if a.shape != p.shape or a.shape != b.shape or not np.isfinite([a, p, b]).all():
        raise ValueError('Finite aligned vectors required')
    denominator = float(np.sum((a - b) ** 2))
    return 1 - float(np.sum((a - p) ** 2)) / denominator if denominator > 0 else None


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def paired_bootstrap(folds, *, block=20, repetitions=10000, seed=42, normalization=1.):
    """Each fold is an n x k aligned matrix. No block crosses a fold boundary.

    Bounded chunks avoid a 10000 x full-cohort allocation on the local laptop.
    Columns are resampled together; pairing is never broken.
    """
    arrays = [np.asarray(f, dtype=float) for f in folds]
    if (not arrays or block < 1 or repetitions < 20 or normalization <= 0
            or any(f.ndim != 2 or not len(f) or not np.isfinite(f).all() for f in arrays)
            or len({f.shape[1] for f in arrays}) != 1):
        raise ValueError('Finite aligned nonempty folds and valid bootstrap settings required')
    rng = np.random.default_rng(seed)
    n = sum(len(f) for f in arrays)
    samples = []
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        sums = np.zeros((count, arrays[0].shape[1]))
        for f in arrays:
            size = min(block, len(f))
            starts = rng.integers(0, len(f) - size + 1, (count, int(np.ceil(len(f) / size))))
            indexes = (starts[..., None] + np.arange(size)).reshape(count, -1)[:, :len(f)]
            sums += f[indexes].sum(axis=1)
        samples.append(sums / n)
    draws = np.concatenate(samples)
    observed = np.concatenate(arrays).mean(axis=0)
    difference = observed[0] - observed[1]
    paired_losses = np.concatenate(arrays)[:, 0] - np.concatenate(arrays)[:, 1]
    degenerate = np.ptp(paired_losses) <= 32 * np.finfo(float).eps * max(1., np.max(np.abs(paired_losses)))
    differences = np.full(repetitions, difference) if degenerate else draws[:, 0] - draws[:, 1]
    # Moving blocks underweight fold edges. Recenter their bootstrap distribution
    # before using it as null noise; basic CI and two-sided test use the same noise.
    noise = differences - differences.mean()
    se = float(np.std(noise, ddof=1))
    critical = float(np.quantile(np.abs(noise), .95))
    effects = [.005, .01, .02, .05]
    return {'count': n, 'block_length': block, 'replicates': repetitions,
            'paired_difference': float(difference),
            'difference_ci_95': (difference - np.quantile(noise, [.975, .025])).tolist(),
            'difference_ci_method': 'fold-preserving moving-block, recentered basic interval',
            'column_mean_ci_95': np.quantile(draws, [.025, .975], axis=0).tolist(),
            'standard_error': se,
            'centered_bootstrap_p_two_sided': float((1 + np.sum(np.abs(noise) >= abs(difference))) / (repetitions + 1)),
            'normal_approx_mde_80pct_relative': None if degenerate else 2.8 * se / normalization,
            'power_at_relative_effect': None if degenerate else {str(e): float(np.mean(np.abs(noise + e * normalization) > critical)) for e in effects},
            'power_identifiable': not degenerate,
            'power_limitation': 'Constant paired losses provide no empirical noise for power estimation' if degenerate else None,
            'power_assumptions': 'post-hoc paired noise held fixed under additive alternatives; selection not adjusted',
            'gate_changed': False}


def paired_prediction_diagnostics(rows, *, repetitions=10000):
    actual = rows.actual_return.to_numpy()
    predicted = rows.predicted_return.to_numpy()
    prices = rows.cotton_close.to_numpy()
    result = {'oos_r2_log_vs_naive': oos_r2(actual, predicted, np.zeros(len(rows))),
              'oos_r2_price_vs_naive': oos_r2(prices * np.exp(actual), prices * np.exp(predicted), prices),
              'oos_r2_log_vs_median': oos_r2(actual, predicted, rows.median_return),
              'oos_r2_price_vs_median': oos_r2(prices * np.exp(actual), prices * np.exp(predicted), prices * np.exp(rows.median_return)),
              'spearman_ic': correlation(rankdata(actual), rankdata(predicted)),
              'mae_difference_bootstrap': {}, 'direction_difference_bootstrap': {}}
    for block in (10, 20, 40):
        parts = [g[['naive_error', 'model_error']].to_numpy() for _, g in rows.groupby('fold')]
        result['mae_difference_bootstrap'][str(block)] = paired_bootstrap(
            parts, block=block, repetitions=repetitions, normalization=float(rows.naive_error.mean()) or 1.)
        parts = [g[['model_direction_hit', 'majority_direction_hit']].to_numpy(dtype=float)
                 for _, g in rows.groupby('fold')]
        result['direction_difference_bootstrap'][str(block)] = paired_bootstrap(parts, block=block, repetitions=repetitions)
    return result


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Valid p-values required')
    order = np.argsort(values)
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result.tolist()
