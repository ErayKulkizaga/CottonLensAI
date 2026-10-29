"""Daily variance proxies and small past-only CPU volatility estimators."""
import json

import numpy as np

from cottonlens_ml.runtime_guard import require_training

FLOOR = 1e-12


class InvalidVarianceFit(ValueError):
    """Numerical failure excludes this candidate, never a negative score."""


def features(history):
    rows = history.copy()
    returns = np.log(rows.cotton_close / rows.cotton_close.shift())
    squared = returns ** 2
    rows['daily_variance_proxy'] = squared
    for window in (1, 5, 22):
        rows[f'har_{window}'] = np.log(squared.rolling(window).mean().clip(lower=FLOOR))
    for h in (1, 5):
        rows[f'variance_target_{h}'] = sum(squared.shift(-i) for i in range(1, h + 1))
    names = ['cotton_open', 'cotton_high', 'cotton_low', 'cotton_close']
    if set(names).issubset(rows.columns):
        o, hi, lo, c = (rows[n] for n in names)
        valid = (lo > 0) & (hi >= np.maximum(o, c)) & (lo <= np.minimum(o, c))
        gk = (.5 * np.log(hi / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2).where(valid)
        for h in (1, 5):
            rows[f'gk_target_{h}'] = sum(gk.shift(-i) for i in range(1, h + 1))
    return rows


def ewma(history, origins, horizon):
    variance = history.daily_variance_proxy.ewm(alpha=.06, adjust=False).mean()
    return np.maximum(variance.loc[history.date.isin(origins)].to_numpy() * horizon, FLOOR)


def qlike(actual, predicted):
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    if a.shape != p.shape or not np.isfinite([a, p]).all() or (a < 0).any() or (p <= 0).any():
        raise ValueError('Nonnegative actual proxy and positive predicted variance required')
    return np.log(p) + a / p


def qlike_value_gradient(theta, design, actual):
    """Convex log-link criterion; actual zero is valid, not log-clipped."""
    eta = design @ theta
    with np.errstate(over='raise', invalid='raise'):
        try:
            ratio = actual * np.exp(-eta)
        except FloatingPointError as exc:
            raise InvalidVarianceFit('QLIKE optimizer left finite numeric domain') from exc
    return float(np.mean(eta + ratio)), design.T @ (1-ratio) / len(actual)


def qlike_features(names=None):
    """Explicit opt-in for a declared design; legacy callers retain three HARs."""
    if names is None:
        return ['har_1', 'har_5', 'har_22']
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(n, str) or not n or n.startswith(('target_', 'variance_target_')) for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('Unique declared QLIKE feature names required; targets cannot be features')
    return list(names)


def qlike_har(train, horizon, *, feature_names=None):
    """One declared log-link design, unchanged optimizer and past-only conditioning."""
    from scipy.optimize import minimize
    names = qlike_features(feature_names)
    raw = train[names].to_numpy(dtype=float)
    actual = train[f'variance_target_{horizon}'].to_numpy(dtype=float)
    if (not np.isfinite(raw).all() or not np.isfinite(actual).all()
            or (actual < 0).any() or not (actual > 0).any()):
        raise InvalidVarianceFit('Finite nonnegative variance labels with positive mass required')
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.column_stack([np.ones(len(train)), (raw-mean)/scale])
    # OLS+smearing is initialization only. It adds no parameters or trials.
    y = np.log(np.maximum(actual, FLOOR))
    initial = np.linalg.lstsq(design, y, rcond=None)[0]
    initial[0] += np.log(np.exp(y-design @ initial).mean())
    initial_loss, _ = qlike_value_gradient(initial, design, actual)
    fitted = minimize(qlike_value_gradient, initial, args=(design, actual), jac=True,
                      method='BFGS', options={'maxiter': 500, 'gtol': 1e-6})
    loss, gradient = qlike_value_gradient(fitted.x, design, actual)
    if (not fitted.success or not np.isfinite(fitted.x).all()
            or np.max(np.abs(gradient)) > 1e-6 or loss > initial_loss + 1e-10
            or (design @ fitted.x < np.log(FLOOR)).any()):
        raise InvalidVarianceFit('QLIKE optimization not certified; no OLS fallback')
    coefficients = fitted.x.copy()
    coefficients[1:] /= scale
    coefficients[0] -= mean @ coefficients[1:]
    return coefficients, {'method': 'BFGS', 'success': bool(fitted.success), 'iterations': int(fitted.nit),
                          'gradient_inf_norm': float(np.max(np.abs(gradient))),
                          'initial_training_qlike': initial_loss, 'final_training_qlike': loss,
                          'training_feature_mean': mean.tolist(), 'training_feature_scale': scale.tolist()}


def fit_predict(history, train, test, spec, workspace):
    require_training(spec['family'], spec.get('device', 'cpu'))
    h = spec['horizon']
    if spec['family'] == 'har' and spec.get('training_loss', 'log_mse') not in ('log_mse', 'qlike'):
        raise ValueError('Explicit supported HAR training loss required')
    if spec.get('variance_features') is not None and (spec['family'] != 'har' or spec.get('training_loss') != 'qlike'):
        raise ValueError('Explicit variance features require the QLIKE HAR path')
    if spec['family'] == 'har' and spec.get('training_loss', 'log_mse') == 'qlike':
        names = qlike_features(spec.get('variance_features'))
        coefficients, optimization = qlike_har(train, h, feature_names=names)
        predicted = np.maximum(np.exp(np.column_stack([np.ones(len(test)), test[names]]) @ coefficients), FLOOR)
        model = {'kind': 'log_har', 'features': names, 'coef': coefficients.tolist(), 'smearing': 1.,
                 'training_loss': 'qlike', 'optimization': optimization}
    elif spec['family'] == 'har':
        names = ['har_1', 'har_5', 'har_22']
        x = np.column_stack([np.ones(len(train)), train[names]])
        y = np.log(np.maximum(train[f'variance_target_{h}'].to_numpy(), FLOOR))
        if not np.isfinite([x]).all() or not np.isfinite(y).all():
            raise ValueError('Finite mature HAR inputs/labels required')
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        # Duan smearing: entirely training residuals, no lognormal assumption.
        smear = float(np.mean(np.exp(y - x @ coefficients)))
        predicted = np.maximum(np.exp(np.column_stack([np.ones(len(test)), test[names]]) @ coefficients) * smear, FLOOR)
        model = {'kind': 'log_har', 'features': names, 'coef': coefficients.tolist(), 'smearing': smear}
    elif spec['family'] == 'garch':
        from arch import arch_model
        values = train.cotton_ret_1.dropna().to_numpy() * 100
        fitted = arch_model(values, mean='Zero', vol='GARCH', p=1, q=1, dist='StudentsT', rescale=False).fit(disp='off')
        if fitted.convergence_flag != 0:
            raise InvalidVarianceFit('GARCH did not converge; incomplete result is not selectable')
        omega = float(fitted.params['omega']) / 10000
        alpha, beta = float(fitted.params['alpha[1]']), float(fitted.params['beta[1]'])
        if alpha + beta >= 1 or omega <= 0:
            raise InvalidVarianceFit('Stationary positive GARCH required')
        fitted_variance = float(np.asarray(fitted.conditional_volatility)[-1] ** 2 / 10000)
        variance = omega + alpha * float(train.cotton_ret_1.iloc[-1]) ** 2 + beta * fitted_variance
        last = train.date.max()
        desired = set(test.date)
        predicted = []
        for row in history.loc[(history.date > last) & (history.date <= test.date.max())].itertuples():
            next_variance = omega + alpha * float(row.cotton_ret_1) ** 2 + beta * variance
            if row.date in desired:
                total, step = 0., next_variance
                for _ in range(h):
                    total += step
                    step = omega + (alpha + beta) * step
                predicted.append(max(total, FLOOR))
            variance = next_variance
        predicted = np.asarray(predicted)
        model = {'kind': 'garch', 'omega': omega, 'alpha': alpha, 'beta': beta,
                 'nu': float(fitted.params['nu']), 'last_training_origin': last.isoformat(),
                 'conditional_variance_at_last_origin': fitted_variance,
                 'last_observed_return': float(train.cotton_ret_1.iloc[-1])}
    else:
        raise ValueError('Unsupported volatility family')
    if len(predicted) != len(test) or not np.isfinite(predicted).all():
        raise ValueError('Invalid volatility prediction')
    (workspace / 'variance-model.json').write_text(json.dumps(model), encoding='utf-8')
    return {'variance_predictions': predicted.tolist(), 'iterations': 1,
            'details': {'effective_device': 'cpu', 'thread_limit': 2},
            'origins': test.date.dt.strftime('%Y-%m-%d').tolist()}
