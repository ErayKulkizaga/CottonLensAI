"""Pre-audit rolling-origin evaluation; this module never runs on the local backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cottonlens_ml.config import FEATURE_NAMES

if TYPE_CHECKING:
    from cottonlens_ml.training import Candidate

AUDIT_START = pd.Timestamp("2024-06-18")
FOLD_SIZE = 126
FOLD_COUNT = 4
CORE = [name for name in FEATURE_NAMES if not name.startswith(("dxy_", "wti_", "cotton_dxy_", "cotton_wti_"))]
CORE = [name for name in CORE if name not in ("cotton_volume_z20", "cotton_volatility_regime_20_60")]
MACRO = [name for name in FEATURE_NAMES if name not in (
    "cotton_volume_z20", "cotton_volatility_regime_20_60",
    "cotton_dxy_corr_60", "cotton_wti_corr_60",
)]


@dataclass(frozen=True)
class Fold:
    number: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def purged_validation_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundary = max(60, int(len(frame) * 0.85))
    validation = frame.iloc[boundary:].copy()
    if validation.empty:
        raise ValueError("not enough pre-audit history for validation")
    train = frame.iloc[:boundary].loc[
        frame.iloc[:boundary].target_date_5 < validation.date.min()
    ].copy()
    if len(train) < 120:
        raise ValueError("not enough purged training history")
    return train, validation


def build_folds(modeling: pd.DataFrame) -> list[Fold]:
    previous = modeling.loc[
        (modeling.date < AUDIT_START) & (modeling.target_date_5 < AUDIT_START)
    ].reset_index(drop=True)
    required = FOLD_SIZE * FOLD_COUNT + 180
    if len(previous) < required:
        raise ValueError(f"walk-forward requires at least {required} pre-audit rows")
    first = len(previous) - FOLD_SIZE * FOLD_COUNT
    folds: list[Fold] = []
    for index in range(FOLD_COUNT):
        start = first + index * FOLD_SIZE
        end = start + FOLD_SIZE
        test = previous.iloc[start:end].copy()
        eligible = previous.iloc[:start].loc[
            previous.iloc[:start].target_date_5 < test.date.min()
        ].copy()
        train, validation = purged_validation_split(eligible)
        if train.target_date_5.max() >= validation.date.min():
            raise AssertionError("training targets cross the inner validation boundary")
        if validation.target_date_5.max() >= test.date.min():
            raise AssertionError("inner validation targets cross the fold boundary")
        folds.append(Fold(index + 1, train, validation, test))
    return folds


def audit_split(modeling: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pre_audit = modeling.loc[modeling.date < AUDIT_START].copy()
    audit = modeling.loc[modeling.date >= AUDIT_START].copy()
    if len(audit) < 60:
        raise ValueError("historical audit period needs at least 60 completed targets")
    pre_audit = pre_audit.loc[pre_audit.target_date_5 < audit.date.min()]
    train, validation = purged_validation_split(pre_audit)
    return {"train": train, "validation": validation, "test": audit}


def _direction_stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_up = actual > 0
    predicted_up = predicted > 0
    true_positive = np.mean(predicted_up[actual_up]) if actual_up.any() else 0.0
    true_negative = np.mean(~predicted_up[~actual_up]) if (~actual_up).any() else 0.0
    return {
        "balanced_accuracy": float((true_positive + true_negative) * 50),
        "majority_direction_accuracy": float(max(actual_up.mean(), (~actual_up).mean()) * 100),
    }


def _bootstrap_mae_interval(errors: np.ndarray, *, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(errors)
    starts = np.arange(max(1, n - 19))
    draws = np.empty(1000)
    for index in range(len(draws)):
        selected: list[int] = []
        while len(selected) < n:
            start = int(rng.choice(starts))
            selected.extend(range(start, min(start + 20, n)))
        draws[index] = np.mean(errors[np.asarray(selected[:n])])
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def summarize(predictions: list[tuple[pd.DataFrame, Candidate]]) -> dict:
    from cottonlens_ml.training import evaluate
    prices = np.concatenate([frame.cotton_close.to_numpy() for frame, _ in predictions])
    actual = np.concatenate([
        frame[f"target_return_{candidate.horizon}"].to_numpy()
        for frame, candidate in predictions
    ])
    predicted = np.concatenate([candidate.predictions for _, candidate in predictions])
    result = evaluate(prices, actual, predicted)
    result.update(_direction_stats(actual, predicted))
    result["sample_count"] = len(actual)
    result["mae_ci_95"] = _bootstrap_mae_interval(
        np.abs(prices * np.exp(actual) - prices * np.exp(predicted))
    )
    return result


def _ablation_mae(fold: Fold, names: list[str], horizon: int) -> float:
    import xgboost as xgb

    from cottonlens_ml.training import evaluate
    model = xgb.XGBRegressor(
        n_estimators=1200, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=42,
        early_stopping_rounds=50, n_jobs=2,
    )
    target = f"target_return_{horizon}"
    model.fit(
        fold.train[names], fold.train[target],
        eval_set=[(fold.validation[names], fold.validation[target])], verbose=False,
    )
    return evaluate(
        fold.test.cotton_close.to_numpy(), fold.test[target].to_numpy(),
        model.predict(fold.test[names]),
    )["mae"]


def run_walkforward(modeling: pd.DataFrame, features: pd.DataFrame, checkpoint_root) -> dict:
    from cottonlens_ml.training import (
        naive_candidates,
        ridge_candidates,
        train_lstm,
        train_xgboost,
    )
    collected: dict[tuple[str, int], list[tuple[pd.DataFrame, Candidate]]] = {}
    folds_report: list[dict] = []
    ablation: list[dict] = []
    for fold in build_folds(modeling):
        print(f"Walk-forward fold {fold.number}/{FOLD_COUNT}: {fold.test.date.min().date()} to {fold.test.date.max().date()}", flush=True)
        folder = checkpoint_root / f"walkforward-fold-{fold.number}"
        folder.mkdir(parents=True, exist_ok=True)
        baseline = naive_candidates(fold.test, fold.validation)
        ridge = ridge_candidates(fold.train, fold.validation, fold.test)
        trees = train_xgboost(fold.train, fold.validation, fold.test, folder)
        _, _, sequences = train_lstm(
            fold.train, fold.validation, fold.test, folder, feature_history=features
        )
        fold_metrics: dict[str, dict] = {}
        for candidate in [*baseline.values(), *ridge.values(), *trees.values(), *sequences.values()]:
            key = (candidate.name, candidate.horizon)
            collected.setdefault(key, []).append((fold.test, candidate))
            fold_metrics[f"{candidate.name}-T+{candidate.horizon}"] = candidate.metrics
        folds_report.append({
            "fold": fold.number,
            "train_end": str(fold.train.date.max().date()),
            "validation_end": str(fold.validation.date.max().date()),
            "test_start": str(fold.test.date.min().date()),
            "test_end": str(fold.test.date.max().date()),
            "sample_count": len(fold.test),
            "metrics": fold_metrics,
            "experiments": {
                f"{candidate.name}-T+{candidate.horizon}": candidate.parameters
                for candidate in [*ridge.values(), *trees.values(), *sequences.values()]
            },
        })
        for horizon in (1, 5):
            ablation.append({
                "fold": fold.number, "horizon": horizon,
                "cotton_mae": _ablation_mae(fold, CORE, horizon),
                "cotton_macro_mae": _ablation_mae(fold, MACRO, horizon),
                "full_mae": _ablation_mae(fold, FEATURE_NAMES, horizon),
            })
    aggregate = {
        f"{name}-T+{horizon}": summarize(rows)
        for (name, horizon), rows in collected.items()
    }
    return {
        "folds": folds_report, "aggregate": aggregate, "feature_ablation": ablation,
        "feature_groups": {"cotton": CORE, "cotton_macro": MACRO, "full": FEATURE_NAMES},
        "cftc_candidate": "excluded: source archive lacks verified actual publication timestamp",
        "period": "pre-2024-06-18",
    }
