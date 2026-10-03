"""Deterministic, bounded candidate generation and research decisions."""
import numpy as np

from cottonlens_ml.cohort import content_id

SEEDS = (17, 42, 101)
BUDGETS = {'A': {'xgboost': 128, 'catboost': 128, 'ridge': 24, 'elasticnet': 24},
           'B': {'xgboost': 64, 'catboost': 64, 'logistic': 64},
           'C': {'mlp': 48, 'lstm': 48, 'tcn': 48}}


def recipes(family, horizon, features, count, *, task='price', offset=0):
    rng = np.random.default_rng(int(content_id({'family': family, 'horizon': horizon, 'task': task})[:8], 16))
    seen, result = set(), []
    def log(low, high):
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    for index in range(count + offset):
        if family == 'xgboost':
            parameters = {'max_depth': int(rng.integers(1, 9)), 'eta': log(.005, .15),
                          'min_child_weight': log(1, 100), 'alpha': 0. if index % 3 == 0 else log(1e-6, 10),
                          'lambda': log(1e-3, 100), 'subsample': float(rng.uniform(.6, 1)),
                          'colsample_bytree': float(rng.uniform(.6, 1))}
        elif family == 'catboost':
            parameters = {'depth': int(rng.integers(4, 9)), 'learning_rate': log(.01, .15), 'l2_leaf_reg': log(1, 100)}
        elif family in ('mlp', 'lstm', 'tcn'):
            parameters = {'width': int(rng.choice([32, 64, 128])), 'dropout': float(rng.uniform(0, .4)),
                          'batch_size': int(rng.choice([32, 64, 128])), 'learning_rate': log(1e-4, 3e-3)}
        elif family in ('ridge', 'elasticnet'):
            parameters = {'alpha': log(1e-5, 100)}
            if family == 'elasticnet':
                parameters['l1_ratio'] = float(rng.uniform(.05, .95))
        elif family == 'logistic':
            parameters = {'C': log(.001, 100)}
        else:
            raise ValueError('Unknown model family')
        spec = {'family': family, 'horizon': horizon, 'task': task, 'params': parameters, 'seed': 42,
                'target': ('scaled_log', 'price_delta')[index % 2],
                'loss': ('reg:squarederror', 'reg:absoluteerror')[(index // 2) % 2],
                'window': int(rng.choice([20, 60, 120])) if family in ('mlp', 'lstm', 'tcn') else 1,
                'features': list(features), 'years': None, 'cadence': 126}
        spec['hypothesis'] = f'{family} {task}: candidate {index}; target={spec["target"]}; loss={spec["loss"]}'
        if index >= offset:
            identity = content_id(spec)
            if identity in seen:
                raise ValueError('Duplicate search recipe')
            seen.add(identity)
            result.append(spec)
    return result


def diagnostic_recipes(horizon, features):
    result = []
    for scaled in (False, True):
        for relaxed in (False, True):
            result.append({'family': 'xgboost', 'horizon': horizon, 'task': 'price', 'features': list(features),
                'seed': 42, 'target': 'scaled_log' if scaled else 'raw_log', 'loss': 'reg:squarederror',
                'window': 1, 'years': None, 'cadence': 126,
                'params': {'max_depth': 3, 'eta': .03, 'subsample': 1., 'colsample_bytree': 1.,
                           'min_child_weight': 1 if relaxed else 20, 'alpha': 0 if relaxed else 1,
                           'lambda': 1 if relaxed else 10},
                'hypothesis': f'target_scaled={scaled};regularization_relaxed={relaxed}'})
    return result


def historical_gate(metrics, fold_wins, horizon):
    return {'passes': bool(metrics['relative_mae_improvement_pct'] >= 5 and fold_wins >= 6
            and metrics['directional_accuracy'] >= (53 if horizon == 1 else 55)),
            'required_fold_wins': 6, 'total_folds': 8, 'evidence': 'seen_historical_research',
            'fold_wins': int(fold_wins),
            'prospective_verified': False}


def next_round(gains):
    """Relative inner-validation improvements, expressed as fractions."""
    if len(gains) >= 2 and all(g < .005 for g in gains[-2:]):
        return 'new_information_or_diagnosis'
    return 'extend_top_two_families_64_each' if gains and (len(gains) > 1 or gains[0] > 0) else 'diagnose_or_new_information'


def simplex_weights(predictions, prices, actual):
    """At most three components; weights fitted on historical OOF data only."""
    from scipy.optimize import minimize
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or not 1 <= predictions.shape[1] <= 3:
        raise ValueError('Ensemble supports one to three components')
    def loss(weights):
        return float(np.mean(prices * np.abs(np.exp(actual) - np.exp(predictions @ weights))))
    n = predictions.shape[1]
    fitted = minimize(loss, np.full(n, 1 / n), method='SLSQP', bounds=[(0., 1.)] * n,
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}, options={'maxiter': 1000, 'ftol': 1e-10})
    if not fitted.success or (fitted.x < -1e-8).any():
        raise ValueError('Constrained ensemble optimizer failed')
    values = np.maximum(fitted.x, 0)
    return (values / values.sum()).tolist()
