from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.selection import select_walkforward_name
from cottonlens_ml.tracking import tracked_run

SEED = 42


def _data_fingerprint(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    columns = ["date", *FEATURE_NAMES, "target_return_1", "target_return_5"]
    for frame in frames:
        digest.update(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes())
    return digest.hexdigest()[:12]


@dataclass
class Candidate:
    name: str
    horizon: int
    model: object
    predictions: np.ndarray
    metrics: dict[str, float]
    validation_metrics: dict[str, float] | None = None
    scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None
    training_history: list[dict[str, float]] | None = None
    parameters: dict | None = None


def evaluate(
    current_price: np.ndarray, actual_return: np.ndarray, predicted_return: np.ndarray
) -> dict[str, float]:
    actual_price = current_price * np.exp(actual_return)
    predicted_price = current_price * np.exp(predicted_return)
    return {
        "mae": float(mean_absolute_error(actual_price, predicted_price)),
        "rmse": float(math.sqrt(mean_squared_error(actual_price, predicted_price))),
        "mape": float(np.mean(np.abs((actual_price - predicted_price) / actual_price)) * 100),
        "directional_accuracy": float(np.mean(np.sign(actual_return) == np.sign(predicted_return)) * 100),
    }


def naive_candidates(test: pd.DataFrame, validation: pd.DataFrame) -> dict[int, Candidate]:
    candidates: dict[int, Candidate] = {}
    for horizon in (1, 5):
        predictions = np.zeros(len(test), dtype=np.float64)
        metrics = evaluate(
            test["cotton_close"].to_numpy(),
            test[f"target_return_{horizon}"].to_numpy(),
            predictions,
        )
        validation_metrics = evaluate(
            validation["cotton_close"].to_numpy(),
            validation[f"target_return_{horizon}"].to_numpy(),
            np.zeros(len(validation), dtype=np.float64),
        )
        candidates[horizon] = Candidate(
            "Naive", horizon, None, predictions, metrics, validation_metrics=validation_metrics
        )
    return candidates


def ridge_candidates(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[int, Candidate]:
    """Fixed-parameter linear reference, never eligible as a production model."""
    scaler = StandardScaler().fit(train[FEATURE_NAMES])
    train_x = scaler.transform(train[FEATURE_NAMES])
    validation_x = scaler.transform(validation[FEATURE_NAMES])
    test_x = scaler.transform(test[FEATURE_NAMES])
    result = {}
    for horizon in (1, 5):
        model = Ridge(alpha=1.0).fit(train_x, train[f"target_return_{horizon}"])
        predicted = model.predict(test_x)
        result[horizon] = Candidate(
            "Ridge", horizon, model, predicted,
            evaluate(test.cotton_close.to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predicted),
            validation_metrics=evaluate(
                validation.cotton_close.to_numpy(),
                validation[f"target_return_{horizon}"].to_numpy(),
                model.predict(validation_x),
            ),
            parameters={"alpha": 1.0, "feature_scaler_fit": "train_only"},
        )
    return result


def train_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    checkpoint_root: Path,
) -> dict[int, Candidate]:
    grid = list(
        itertools.product(
            (3, 5),
            (0.03, 0.07),
            (0.8, 1.0),
        )
    )[:10]
    fingerprint = _data_fingerprint(train, validation)
    result: dict[int, Candidate] = {}
    for horizon in (1, 5):
        best: tuple[float, xgb.XGBRegressor, dict] | None = None
        for depth, learning_rate, subsample in grid:
            print(f"XGBoost T+{horizon}: depth={depth}, lr={learning_rate}, subsample={subsample}", flush=True)
            with tracked_run(run_name=f"xgboost-t{horizon}", nested=True):
                checkpoint = checkpoint_root / (
                    f"xgb-{fingerprint}-t{horizon}-d{depth}-lr{learning_rate}-s{subsample}.json"
                )
                model = xgb.XGBRegressor()
                resumed = checkpoint.exists()
                if resumed:
                    model.load_model(checkpoint)
                else:
                    model = xgb.XGBRegressor(
                        n_estimators=1200,
                        max_depth=depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=SEED,
                        early_stopping_rounds=50,
                        n_jobs=2,
                    )
                    model.fit(
                        train[FEATURE_NAMES],
                        train[f"target_return_{horizon}"],
                        eval_set=[
                            (
                                validation[FEATURE_NAMES],
                                validation[f"target_return_{horizon}"],
                            )
                        ],
                        verbose=False,
                    )
                    model.save_model(checkpoint)
                validation_predictions = model.predict(validation[FEATURE_NAMES])
                validation_mae = evaluate(
                    validation["cotton_close"].to_numpy(),
                    validation[f"target_return_{horizon}"].to_numpy(),
                    validation_predictions,
                )["mae"]
                mlflow.log_params(
                    {
                        "horizon": horizon,
                        "max_depth": depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "resumed_from_drive": resumed,
                    }
                )
                mlflow.log_metric("validation_mae", validation_mae)
                if best is None or validation_mae < best[0]:
                    best = validation_mae, model, {
                        "max_depth": depth, "learning_rate": learning_rate,
                        "subsample": subsample, "best_iteration": model.best_iteration,
                    }
        assert best is not None
        selected = best[1]
        predictions = selected.predict(test[FEATURE_NAMES])
        metrics = evaluate(
            test["cotton_close"].to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predictions
        )
        validation_metrics = evaluate(
            validation["cotton_close"].to_numpy(),
            validation[f"target_return_{horizon}"].to_numpy(),
            selected.predict(validation[FEATURE_NAMES]),
        )
        result[horizon] = Candidate(
            "XGBoost",
            horizon,
            selected,
            predictions,
            metrics,
            validation_metrics=validation_metrics,
            parameters=best[2],
        )
    return result


def _sequences(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    window: int = 60,
    history: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    preceding = frame.iloc[:0] if history is None else history.loc[history.date < frame.date.min()].tail(window - 1)
    if history is not None and len(preceding) != window - 1:
        raise ValueError("60-step evaluation needs 59 earlier feature snapshots")
    combined = pd.concat([preceding, frame], ignore_index=True)
    values = scaler.transform(combined[FEATURE_NAMES]).astype(np.float32)
    targets = frame[["target_return_1", "target_return_5"]].to_numpy(dtype=np.float32)
    start = window - 1 if history is None else len(preceding)
    return (
        np.asarray(
            [values[index - window + 1 : index + 1] for index in range(start, len(combined))],
            dtype=np.float32,
        ),
        targets[window - 1 :] if history is None else targets,
    )


def train_lstm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    checkpoint_root: Path,
    feature_history: pd.DataFrame | None = None,
) -> tuple[tf.keras.Model, StandardScaler, dict[int, Candidate]]:
    tf.keras.utils.set_random_seed(SEED)
    scaler = StandardScaler().fit(train[FEATURE_NAMES])
    train_x, train_y_raw = _sequences(train, scaler)
    context = feature_history if feature_history is not None else pd.concat([train, validation, test])
    validation_x, validation_y_raw = _sequences(validation, scaler, history=context)
    test_x, _ = _sequences(test, scaler, history=context)
    target_scaler = StandardScaler().fit(train_y_raw)
    train_y = target_scaler.transform(train_y_raw).astype(np.float32)
    validation_y = target_scaler.transform(validation_y_raw).astype(np.float32)
    fingerprint = _data_fingerprint(train, validation)
    configs = [(32, 0.10), (32, 0.20), (64, 0.10), (64, 0.20)]
    best_loss = float("inf")
    best_history: list[dict[str, float]] | None = None
    best_parameters: dict | None = None
    best_path = checkpoint_root / f"lstm-{fingerprint}-best.keras"
    for units, dropout in configs:
        print(f"LSTM: units={units}, dropout={dropout}", flush=True)
        with tracked_run(run_name="lstm-multi-horizon", nested=True):
            checkpoint = checkpoint_root / (
                f"lstm-{fingerprint}-{units}-{int(dropout * 100)}.keras"
            )
            resumed = checkpoint.exists()
            if resumed:
                model = tf.keras.models.load_model(checkpoint)
                loss = float(model.evaluate(validation_x, validation_y, verbose=0))
                history_file = checkpoint.with_suffix(".history.json")
                history_rows = json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else []
            else:
                inputs = tf.keras.Input(shape=(60, len(FEATURE_NAMES)), name="features")
                hidden = tf.keras.layers.LSTM(units)(inputs)
                hidden = tf.keras.layers.Dropout(dropout)(hidden)
                hidden = tf.keras.layers.Dense(32, activation="relu")(hidden)
                outputs = tf.keras.layers.Dense(2, name="returns")(hidden)
                model = tf.keras.Model(inputs, outputs)
                model.compile(optimizer="adam", loss="mae")
                history = model.fit(
                    train_x,
                    train_y,
                    validation_data=(validation_x, validation_y),
                    epochs=100,
                    batch_size=64,
                    verbose=2,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        tf.keras.callbacks.ModelCheckpoint(checkpoint, save_best_only=True),
                    ],
                )
                loss = float(min(history.history["val_loss"]))
                history_rows = [
                    {"epoch": epoch + 1, "loss": float(train_loss), "val_loss": float(val_loss)}
                    for epoch, (train_loss, val_loss) in enumerate(
                        zip(history.history["loss"], history.history["val_loss"], strict=True)
                    )
                ]
                (checkpoint.with_suffix(".history.json")).write_text(
                    json.dumps(history_rows), encoding="utf-8"
                )
                model = tf.keras.models.load_model(checkpoint)
            mlflow.log_params(
                {"units": units, "dropout": dropout, "window": 60, "resumed_from_drive": resumed}
            )
            mlflow.log_metric("validation_loss", loss)
            if loss < best_loss:
                best_loss = loss
                model.save(best_path)
                best_history = history_rows
                best_parameters = {"units": units, "dropout": dropout, "batch_size": 64,
                                   "epochs_run": len(history_rows), "optimizer": "adam", "loss": "mae"}
    selected = tf.keras.models.load_model(best_path)
    predictions = target_scaler.inverse_transform(selected.predict(test_x, verbose=0))
    validation_predictions = target_scaler.inverse_transform(selected.predict(validation_x, verbose=0))
    aligned = test
    aligned_validation = validation
    candidates: dict[int, Candidate] = {}
    for column, horizon in enumerate((1, 5)):
        metrics = evaluate(
            aligned["cotton_close"].to_numpy(),
            aligned[f"target_return_{horizon}"].to_numpy(),
            predictions[:, column],
        )
        validation_metrics = evaluate(
            aligned_validation["cotton_close"].to_numpy(),
            aligned_validation[f"target_return_{horizon}"].to_numpy(),
            validation_predictions[:, column],
        )
        candidates[horizon] = Candidate(
            "LSTM",
            horizon,
            selected,
            predictions[:, column],
            metrics,
            validation_metrics=validation_metrics,
            scaler=scaler,
            target_scaler=target_scaler,
            training_history=best_history,
            parameters=best_parameters,
        )
    joblib.dump(scaler, checkpoint_root / "lstm-scaler.joblib")
    return selected, scaler, candidates


def select_production(
    naive: dict[int, Candidate],
    xgboost_candidates: dict[int, Candidate],
    lstm_candidates: dict[int, Candidate],
    walkforward_report: dict,
) -> tuple[dict[int, Candidate], dict[int, dict]]:
    selected: dict[int, Candidate] = {}
    audit: dict[int, dict] = {}
    for horizon in (1, 5):
        baseline = naive[horizon]
        tree = xgboost_candidates[horizon]
        sequence = lstm_candidates[horizon]
        model_name, horizon_audit = select_walkforward_name(
            walkforward_report, horizon,
            {"Naive": baseline.metrics, "XGBoost": tree.metrics, "LSTM": sequence.metrics},
        )
        selected[horizon] = {"Naive": baseline, "XGBoost": tree, "LSTM": sequence}[model_name]
        audit[horizon] = horizon_audit
    return selected, audit
