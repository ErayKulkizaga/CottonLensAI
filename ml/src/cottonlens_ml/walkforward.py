"""Pre-audit rolling-origin evaluation; this module never runs on the local backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.evaluation import evaluate, price_mae_metric
from cottonlens_ml.features import coverage_report, select_feature_rows
from cottonlens_ml.protocol import (
    PROTOCOL_VERSION,
    VALIDATION_ROWS,
    split_diagnostics,
)
from cottonlens_ml.runtime_guard import require_colab_training

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
    boundary = len(frame) - VALIDATION_ROWS
    if boundary < 125:
        raise ValueError("not enough history for 126-row validation and purged training")
    validation = frame.iloc[boundary:].copy()
    if validation.empty:
        raise ValueError("not enough pre-audit history for validation")
    train = frame.iloc[:boundary].loc[
        frame.iloc[:boundary].target_date_5 < validation.date.min()
    ].copy()
    if len(train) < 120:
        raise ValueError("not enough purged training history")
    return train, validation


def build_folds(modeling: pd.DataFrame, cohort: dict | None = None) -> list[Fold]:
    if modeling.date.isna().any() or modeling.date.duplicated().any() or not modeling.date.is_monotonic_increasing:
        raise ValueError("Fold source dates must be unique, present and ascending")
    if cohort is not None:
        from cottonlens_ml.cohort import content_id, frame_identity

        payload = {key: value for key, value in cohort.items() if key != "cohort_id"}
        if cohort.get("cohort_id") != content_id(payload) or cohort.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("Frozen cohort identity/protocol mismatch")
        if len(cohort.get("folds", [])) != FOLD_COUNT:
            raise ValueError("Frozen cohort requires four folds")
        folds = []
        previous_end = pd.Timestamp.min
        indexed = modeling.set_index("date", drop=False)
        label_columns = ["date", "cotton_close", "target_return_1", "target_return_5", "target_date_1", "target_date_5"]
        for number, entry in enumerate(cohort["folds"], start=1):
            origins = pd.DatetimeIndex(entry["origins"])
            if (entry["fold"] != number or len(origins) != FOLD_SIZE or origins.has_duplicates
                    or not origins.is_monotonic_increasing or origins.min() <= previous_end):
                raise ValueError("Frozen fold origins must be 126 ordered, disjoint observations")
            missing = origins.difference(indexed.index)
            if len(missing):
                raise ValueError(f"Frozen fold {number} origins missing from model inputs: {missing.strftime('%Y-%m-%d').tolist()}")
            test = indexed.loc[origins].reset_index(drop=True)
            if (test.date >= AUDIT_START).any() or (test.target_date_5 >= AUDIT_START).any():
                raise ValueError("Frozen fold overlaps the seen historical audit")
            if frame_identity(test, label_columns) != entry["label_identity"]:
                raise ValueError("Frozen fold prices/labels changed")
            for horizon in (1, 5):
                if [str(value.date()) for value in test[f"target_date_{horizon}"]] != entry["target_dates"][str(horizon)]:
                    raise ValueError("Frozen fold target dates changed")
            eligible = modeling.loc[(modeling.date < test.date.min()) & (modeling.target_date_5 < test.date.min())].copy()
            train, validation = purged_validation_split(eligible)
            folds.append(Fold(number, train, validation, test))
            previous_end = origins.max()
        return folds
    # Compatibility for standalone split tests. Production execution requires a
    # frozen cohort and never uses this dynamic selection path.
    previous = modeling.loc[
        (modeling.date < AUDIT_START) & (modeling.target_date_5 < AUDIT_START)
    ].reset_index(drop=True)
    required = FOLD_SIZE * FOLD_COUNT + VALIDATION_ROWS + 130
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


def paired_block_bootstrap(
    folds: list[tuple[np.ndarray, np.ndarray]], *, block_size: int = 20,
    replicates: int = 1000, seed: int = 42,
) -> dict:
    """Resample paired model/Naive errors within each fold, never across boundaries."""
    if not folds or block_size < 1 or replicates < 1:
        raise ValueError("Bootstrap needs folds, positive block size and replicate count")
    values = []
    for model, baseline in folds:
        model, baseline = np.asarray(model, dtype=float), np.asarray(baseline, dtype=float)
        if (model.ndim != 1 or model.shape != baseline.shape or not len(model)
                or not np.isfinite(model).all() or not np.isfinite(baseline).all()
                or (model < 0).any() or (baseline < 0).any()):
            raise ValueError("Bootstrap requires finite paired nonnegative fold errors")
        values.append((model, baseline))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 3))
    for index in range(replicates):
        model_draw, baseline_draw = [], []
        for model, baseline in values:
            n, width = len(model), min(block_size, len(model))
            starts = rng.integers(0, n - width + 1, size=(n + width - 1) // width)
            selected = (starts[:, None] + np.arange(width)).ravel()[:n]
            model_draw.append(model[selected])
            baseline_draw.append(baseline[selected])
        model_mae = float(np.concatenate(model_draw).mean())
        baseline_mae = float(np.concatenate(baseline_draw).mean())
        draws[index] = [model_mae, baseline_mae - model_mae,
                        (1 - model_mae / baseline_mae) * 100 if baseline_mae > 0 else np.nan]

    def interval(column):
        return [float(value) for value in np.quantile(draws[:, column], [.025, .975])] if np.isfinite(draws[:, column]).all() else None

    return {"block_size": block_size, "replicates": replicates, "seed": seed,
            "fold_boundaries_preserved": True, "model_mae_ci_95": interval(0),
            "paired_gain_ci_95_cents_per_lb": interval(1), "relative_gain_ci_95_pct": interval(2)}


def summarize(predictions: list[tuple[pd.DataFrame, Candidate]], *, bootstrap_replicates: int = 1000) -> dict:
    prices = np.concatenate([frame.cotton_close.to_numpy() for frame, _ in predictions])
    actual = np.concatenate([
        frame[f"target_return_{candidate.horizon}"].to_numpy()
        for frame, candidate in predictions
    ])
    predicted = np.concatenate([candidate.predictions for _, candidate in predictions])
    result = evaluate(prices, actual, predicted)
    folds, records = [], []
    for number, (frame, candidate) in enumerate(predictions, start=1):
        current = frame.cotton_close.to_numpy()
        actual_prices = current * np.exp(frame[f"target_return_{candidate.horizon}"].to_numpy())
        predicted_prices = current * np.exp(candidate.predictions)
        model_error, naive_error = np.abs(actual_prices - predicted_prices), np.abs(actual_prices - current)
        folds.append((model_error, naive_error))
        for index, row in enumerate(frame.itertuples()):
            records.append({
                "fold": number, "origin": row.date.isoformat(),
                "target_date": getattr(row, f"target_date_{candidate.horizon}").isoformat(),
                "horizon": candidate.horizon, "model": candidate.name,
                "fit_cutoff": (candidate.parameters or {}).get("fit_cutoff"),
                "current_price": float(current[index]), "actual_price": float(actual_prices[index]),
                "predicted_price": float(predicted_prices[index]), "naive_price": float(current[index]),
                "absolute_error": float(model_error[index]), "naive_absolute_error": float(naive_error[index]),
                "paired_gain": float(naive_error[index] - model_error[index]),
            })
    result["naive_mae"] = float(np.concatenate([values[1] for values in folds]).mean())
    result["relative_mae_improvement_pct"] = (1 - result["mae"] / result["naive_mae"]) * 100 if result["naive_mae"] > 0 else None
    sensitivity = {str(size): paired_block_bootstrap(folds, block_size=size, replicates=bootstrap_replicates)
                   for size in (10, 20, 40)}
    result["paired_bootstrap"] = sensitivity["20"]
    result["paired_bootstrap_sensitivity"] = sensitivity
    result["mae_ci_95"] = sensitivity["20"]["model_mae_ci_95"]
    result["paired_mae_gain_ci_95_cents_per_lb"] = sensitivity["20"]["paired_gain_ci_95_cents_per_lb"]
    result["origin_records"] = records
    return result


def ablation_partitions(fold: Fold, names: list[str], history: pd.DataFrame) -> tuple[Fold, pd.DataFrame, dict]:
    """Group-specific natural fit coverage with the original common test origins."""
    natural = select_feature_rows(history, names)
    eligible = natural.loc[(natural.date < fold.test.date.min()) & (natural.target_date_5 < fold.test.date.min())]
    train, validation = purged_validation_split(eligible)
    indexed = natural.set_index("date", drop=False)
    if not set(fold.test.date).issubset(set(indexed.index)):
        raise ValueError("Ablation cannot drop frozen common evaluation origins")
    test = indexed.loc[fold.test.date].reset_index(drop=True)
    refit = natural.loc[(natural.date >= train.date.min()) & (natural.date <= validation.date.max())].copy()
    if train.target_date_5.max() >= validation.date.min() or refit.target_date_5.max() >= test.date.min():
        raise ValueError("Ablation refit crosses validation/evaluation boundary")
    natural_period = natural.loc[natural.date.between(test.date.min(), test.date.max())]
    coverage = {"natural_fit_rows": len(refit), "natural_period_origins": len(natural_period),
                "common_evaluation_origins": len(test), "scored_population": "frozen_common_origins_only"}
    return Fold(fold.number, train, validation, test), refit, coverage


def _ablation_mae(fold: Fold, names: list[str], horizon: int, history: pd.DataFrame) -> float:
    require_colab_training()
    import xgboost as xgb

    fold, refit, _ = ablation_partitions(fold, names, history)

    model = xgb.XGBRegressor(
        n_estimators=1200, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=42,
        early_stopping_rounds=50, n_jobs=2,
        eval_metric=price_mae_metric(fold.validation.cotton_close.to_numpy()),
    )
    target = f"target_return_{horizon}"
    model.fit(
        fold.train[names], fold.train[target],
        eval_set=[(fold.validation[names], fold.validation[target])], verbose=False,
    )
    locked_trees = int(model.best_iteration) + 1
    model = xgb.XGBRegressor(
        n_estimators=locked_trees, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.9,
        objective="reg:squarederror", random_state=42, n_jobs=2,
    )
    model.fit(refit[names], refit[target], verbose=False)
    return evaluate(
        fold.test.cotton_close.to_numpy(), fold.test[target].to_numpy(),
        model.predict(fold.test[names]),
    )["mae"]


def run_walkforward(
    modeling: pd.DataFrame, features: pd.DataFrame, checkpoint_root, *, cohort: dict,
    include_ablation: bool = False,
) -> tuple[dict, dict[str, dict[int, Candidate]]]:
    require_colab_training()
    from cottonlens_ml.cohort import verify_cohort
    from cottonlens_ml.training import (
        naive_candidates,
        ridge_candidates,
        train_lstm,
        train_xgboost,
    )
    data_id = features.attrs.get("data_identity", {}).get("data_id")
    verify_cohort(cohort, features, data_id)
    collected: dict[tuple[str, int], list[tuple[pd.DataFrame, Candidate]]] = {}
    folds_report: list[dict] = []
    ablation: list[dict] = []
    last_candidates = {}
    for fold in build_folds(modeling, cohort):
        print(f"Walk-forward fold {fold.number}/{FOLD_COUNT}: {fold.test.date.min().date()} to {fold.test.date.max().date()}", flush=True)
        folder = checkpoint_root / f"walkforward-fold-{fold.number}"
        folder.mkdir(parents=True, exist_ok=True)
        baseline = naive_candidates(fold.test, fold.validation)
        ridge = ridge_candidates(fold.train, fold.validation, fold.test, feature_history=features)
        trees = train_xgboost(fold.train, fold.validation, fold.test, folder, feature_history=features)
        _, _, sequences = train_lstm(
            fold.train, fold.validation, fold.test, folder, feature_history=features
        )
        last_candidates = {"Naive": baseline, "Ridge": ridge, "XGBoost": trees, "LSTM": sequences}
        fold_metrics: dict[str, dict] = {}
        for candidate in [*baseline.values(), *ridge.values(), *trees.values(), *sequences.values()]:
            key = (candidate.name, candidate.horizon)
            collected.setdefault(key, []).append((fold.test, candidate))
            fold_metrics[f"{candidate.name}-T+{candidate.horizon}"] = candidate.metrics
        folds_report.append({
            "fold": fold.number,
            "train_end": str(fold.train.date.max().date()),
            "validation_end": str(fold.validation.date.max().date()),
            "refit_end": str(fold.validation.date.max().date()),
            "validation_rows": len(fold.validation),
            "inner_validation_diagnostics": split_diagnostics(fold.train, fold.validation),
            "test_start": str(fold.test.date.min().date()),
            "test_end": str(fold.test.date.max().date()),
            "sample_count": len(fold.test),
            "lstm_validation_training_history": sequences[1].training_history,
            "metrics": fold_metrics,
            "experiments": {
                f"{candidate.name}-T+{candidate.horizon}": candidate.parameters
                for candidate in [*ridge.values(), *trees.values(), *sequences.values()]
            },
        })
        for horizon in (1, 5) if include_ablation else ():
            ablation.append({
                "fold": fold.number, "horizon": horizon,
                "cotton_mae": _ablation_mae(fold, CORE, horizon, features),
                "cotton_macro_mae": _ablation_mae(fold, MACRO, horizon, features),
                "full_mae": _ablation_mae(fold, FEATURE_NAMES, horizon, features),
                "coverage": {
                    name: ablation_partitions(fold, names, features)[2]
                    for name, names in {"cotton": CORE, "cotton_macro": MACRO, "full": FEATURE_NAMES}.items()
                },
            })
    aggregate = {
        f"{name}-T+{horizon}": summarize(rows)
        for (name, horizon), rows in collected.items()
    }
    report = {
        "protocol": PROTOCOL_VERSION,
        "evaluation_status": "development_evidence: these folds have already informed revisions",
        "validation_rows": VALIDATION_ROWS,
        "cohort_id": cohort["cohort_id"],
        "folds": folds_report, "aggregate": aggregate, "feature_ablation": ablation,
        "feature_ablation_status": "enabled" if include_ablation else "disabled_day1_no_feature_search",
        "feature_coverage": coverage_report(features, {"cotton": CORE, "cotton_macro": MACRO, "full": FEATURE_NAMES}),
        "feature_groups": {"cotton": CORE, "cotton_macro": MACRO, "full": FEATURE_NAMES},
        "cftc_candidate": "excluded: source archive lacks verified actual publication timestamp",
        "period": "pre-2024-06-18",
    }
    return report, last_candidates
