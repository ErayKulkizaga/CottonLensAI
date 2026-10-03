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


def fit_predict(history, train, test, spec, workspace):
    require_training(spec['family'], spec.get('device', 'cpu'))
    h = spec['horizon']
    if spec['family'] == 'har':
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
