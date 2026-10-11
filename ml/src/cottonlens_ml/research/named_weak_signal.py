"""Fixed named-cohort weak teacher readiness; no model fitting or market claim."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from cottonlens_ml.research.named_history import FEATURES
from cottonlens_ml.research.protocol import rows_at
from cottonlens_ml.research.weak_signal import expected_absolute_log_price

PROFILE = "named-weak-signal-control-t5-v1"
SOURCE = FEATURES[0]
SEEDS = (703, 1703, 2703)
CONDITIONS = ("null", "injected")
ORACLE_GAIN = 10.0


def teacher(frame, threshold):
    """One shallow split of an origin-known field; missing values inject no signal."""
    values = frame[SOURCE].to_numpy(dtype=float)
    return np.where(np.isfinite(values), np.where(values >= threshold, 1.0, -1.0), 0.0)


def calibrate(history, dates):
    """Only the earliest frozen mature training block sets threshold/noise/beta."""
    train = rows_at(history, dates)
    if not train.named_eligible.all():
        raise ValueError("Calibration requires mature admitted training labels")
    values = train[SOURCE].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 20 or np.ptp(finite) <= 1e-8:
        raise ValueError("Insufficient early source variation")
    threshold = float(np.median(finite))
    sigma = float(train.named_return.std(ddof=0))
    prices = train.named_price.to_numpy(dtype=float)
    if (
        not sigma > 1e-8
        or not np.isfinite(sigma)
        or not np.isfinite(prices).all()
        or (prices <= 0).any()
    ):
        raise ValueError("Finite positive early prices and target noise scale required")
    direction = teacher(train, threshold)

    def gain(beta):
        mean = beta * direction
        baseline = expected_absolute_log_price(prices, mean, sigma, 0.0).mean()
        oracle = expected_absolute_log_price(prices, mean, sigma, mean).mean()
        return float(100 * (1 - oracle / baseline))

    if gain(0.1) <= ORACLE_GAIN:
        raise ValueError(
            "Frozen oracle-strength bracket insufficient; no adaptive widening"
        )
    beta = float(brentq(lambda b: gain(b) - ORACLE_GAIN, 0.0, 0.1, xtol=1e-15))
    return {
        "source": SOURCE,
        "threshold": threshold,
        "sigma": sigma,
        "beta": beta,
        "oracle_calibration_gain_pct": gain(beta),
        "requested_oracle_gain_pct": ORACLE_GAIN,
        "origins": list(dates),
        "finite_source_rows": len(finite),
        "reference_period_end": train.date.max().strftime("%Y-%m-%d"),
        "calibration_uses_evaluation": False,
    }


def build(history, calibration, condition, seed):
    """Semi-synthetic overlapping T5 labels; not a coherent traded price path."""
    if (
        condition not in CONDITIONS
        or seed not in SEEDS
        or calibration["source"] != SOURCE
    ):
        raise ValueError("Only frozen condition/seed/source allowed")
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError("Full unique Cotton ordinal calendar required")
    result = history.copy()
    rng = np.random.default_rng(seed)
    daily = rng.standard_normal(len(history) + 5)
    # Each label uses its five FUTURE innovations. Model inputs stay original;
    # the frozen target-date maturity rule determines when these labels enter fit.
    noise = np.array(
        [daily[i + 1 : i + 6].sum() / np.sqrt(5.0) for i in range(len(history))]
    )
    mean = (
        calibration["beta"] * teacher(history, calibration["threshold"])
        if condition == "injected"
        else np.zeros(len(history))
    )
    returns = mean + calibration["sigma"] * noise
    result["named_return"] = returns
    result["target_return_5"] = returns
    result["named_target_price"] = result.named_price * np.exp(returns)
    unchanged = [
        n
        for n in history
        if n not in ("named_return", "target_return_5", "named_target_price")
    ]
    pd.testing.assert_frame_equal(
        result[unchanged], history[unchanged], check_exact=True
    )
    return result, mean, noise


def expected(frame, prediction, mean, sigma):
    valid = frame.named_eligible.to_numpy(dtype=bool)
    price = frame.named_price.to_numpy(dtype=float)[valid]
    mu = np.asarray(mean, dtype=float)[valid]
    p = np.asarray(prediction, dtype=float)[valid]
    baseline = expected_absolute_log_price(price, mu, sigma, 0.0)
    loss = expected_absolute_log_price(price, mu, sigma, p)
    return {
        "eligible": int(valid.sum()),
        "naive": float(baseline.mean()),
        "model": float(loss.mean()),
        "gain_pct": float(100 * (1 - loss.mean() / baseline.mean())),
    }
