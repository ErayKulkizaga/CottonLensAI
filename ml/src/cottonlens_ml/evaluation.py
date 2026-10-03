"""Training-free metric helpers shared by fold, audit and diagnostic reports."""

import numpy as np


def price_mae_metric(prices: np.ndarray):
    """XGBoost sklearn callback; its only eval_set must match these prices."""
    prices = np.asarray(prices, dtype=float).copy()

    def price_mae(actual, predicted) -> float:
        if np.asarray(actual).shape != prices.shape or np.asarray(predicted).shape != prices.shape:
            raise ValueError("Price MAE callback requires the matching validation rows")
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("Price MAE callback received non-finite predictions")
        return float(np.mean(prices * np.abs(np.exp(actual) - np.exp(predicted))))

    return price_mae


def evaluate(current_price, actual_return, predicted_return) -> dict[str, float]:
    prices, actual, predicted = (
        np.asarray(values, dtype=float) for values in (current_price, actual_return, predicted_return)
    )
    if prices.ndim != 1 or not (prices.shape == actual.shape == predicted.shape) or not len(prices):
        raise ValueError("Metrics require nonempty aligned one-dimensional arrays")
    if not all(np.isfinite(values).all() for values in (prices, actual, predicted)) or np.any(prices <= 0):
        raise ValueError("Metrics require finite returns and positive current prices")
    actual_price, predicted_price = prices * np.exp(actual), prices * np.exp(predicted)
    if not np.isfinite(actual_price).all() or not np.isfinite(predicted_price).all():
        raise ValueError("Price conversion overflowed")
    error = np.abs(actual_price - predicted_price)
    signs, predicted_signs = np.sign(actual), np.sign(predicted)
    # A zero forecast abstains from up/down; do not silently count it as 'down'.
    recalls = [float(np.mean(predicted_signs[signs == sign] == sign)) for sign in (-1, 1) if np.any(signs == sign)]
    return {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(error / actual_price) * 100),
        "directional_accuracy": float(np.mean(signs == predicted_signs) * 100),
        "balanced_accuracy": float(np.mean(recalls) * 100) if recalls else 0.0,
        "majority_direction_accuracy": float(max(np.mean(signs == sign) for sign in (-1, 0, 1)) * 100),
        "sample_count": len(prices),
        "predicted_up_pct": float(np.mean(predicted_signs > 0) * 100),
        "predicted_flat_pct": float(np.mean(predicted_signs == 0) * 100),
        "actual_up_pct": float(np.mean(signs > 0) * 100),
        "predicted_return_mean_pct": float(predicted.mean() * 100),
        "actual_return_mean_pct": float(actual.mean() * 100),
        "predicted_return_std_pct": float(predicted.std() * 100),
        "actual_return_std_pct": float(actual.std() * 100),
    }
