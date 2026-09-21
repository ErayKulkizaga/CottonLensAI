from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from cottonlens_ml.config import FEATURE_NAMES

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
    scaler: StandardScaler | None = None


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


def naive_candidates(test: pd.DataFrame) -> dict[int, Candidate]:
    candidates: dict[int, Candidate] = {}
    for horizon in (1, 5):
        predictions = np.zeros(len(test), dtype=np.float64)
        metrics = evaluate(
            test["cotton_close"].to_numpy(),
            test[f"target_return_{horizon}"].to_numpy(),
            predictions,
        )
        candidates[horizon] = Candidate("Naive", horizon, None, predictions, metrics)
    return candidates


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
        best: tuple[float, xgb.XGBRegressor] | None = None
        for depth, learning_rate, subsample in grid:
            with mlflow.start_run(run_name=f"xgboost-t{horizon}", nested=True):
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
                    best = validation_mae, model
        assert best is not None
        selected = best[1]
        predictions = selected.predict(test[FEATURE_NAMES])
        metrics = evaluate(
            test["cotton_close"].to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predictions
        )
        result[horizon] = Candidate("XGBoost", horizon, selected, predictions, metrics)
    return result


def _sequences(frame: pd.DataFrame, scaler: StandardScaler, window: int = 60) -> tuple[np.ndarray, np.ndarray]:
    values = scaler.transform(frame[FEATURE_NAMES]).astype(np.float32)
    targets = frame[["target_return_1", "target_return_5"]].to_numpy(dtype=np.float32)
    return (
        np.asarray(
            [values[index - window + 1 : index + 1] for index in range(window - 1, len(frame))]
        ),
        targets[window - 1 :],
    )


def train_lstm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    checkpoint_root: Path,
) -> tuple[tf.keras.Model, StandardScaler, dict[int, Candidate]]:
    tf.keras.utils.set_random_seed(SEED)
    scaler = StandardScaler().fit(train[FEATURE_NAMES])
    train_x, train_y = _sequences(train, scaler)
    validation_x, validation_y = _sequences(validation, scaler)
    test_x, _ = _sequences(test, scaler)
    fingerprint = _data_fingerprint(train, validation)
    configs = [(32, 0.10), (48, 0.15), (64, 0.20), (64, 0.30), (96, 0.20), (96, 0.30)]
    best_loss = float("inf")
    best_path = checkpoint_root / f"lstm-{fingerprint}-best.keras"
    for units, dropout in configs:
        with mlflow.start_run(run_name="lstm-multi-horizon", nested=True):
            checkpoint = checkpoint_root / (
                f"lstm-{fingerprint}-{units}-{int(dropout * 100)}.keras"
            )
            resumed = checkpoint.exists()
            if resumed:
                model = tf.keras.models.load_model(checkpoint)
                loss = float(model.evaluate(validation_x, validation_y, verbose=0))
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
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                        tf.keras.callbacks.ModelCheckpoint(checkpoint, save_best_only=True),
                    ],
                )
                loss = float(min(history.history["val_loss"]))
                model = tf.keras.models.load_model(checkpoint)
            mlflow.log_params(
                {"units": units, "dropout": dropout, "window": 60, "resumed_from_drive": resumed}
            )
            mlflow.log_metric("validation_loss", loss)
            if loss < best_loss:
                best_loss = loss
                model.save(best_path)
    selected = tf.keras.models.load_model(best_path)
    predictions = selected.predict(test_x, verbose=0)
    aligned = test.iloc[59:].copy()
    candidates: dict[int, Candidate] = {}
    for column, horizon in enumerate((1, 5)):
        metrics = evaluate(
            aligned["cotton_close"].to_numpy(),
            aligned[f"target_return_{horizon}"].to_numpy(),
            predictions[:, column],
        )
        candidates[horizon] = Candidate(
            "LSTM", horizon, selected, predictions[:, column], metrics, scaler=scaler
        )
    joblib.dump(scaler, checkpoint_root / "lstm-scaler.joblib")
    return selected, scaler, candidates


def select_production(
    naive: dict[int, Candidate],
    xgboost_candidates: dict[int, Candidate],
    lstm_candidates: dict[int, Candidate],
) -> dict[int, Candidate]:
    selected: dict[int, Candidate] = {}
    for horizon in (1, 5):
        baseline = naive[horizon]
        tree = xgboost_candidates[horizon]
        sequence = lstm_candidates[horizon]
        if tree.metrics["mae"] >= baseline.metrics["mae"] and sequence.metrics["mae"] >= baseline.metrics["mae"]:
            selected[horizon] = baseline
            continue
        materially_better = sequence.metrics["mae"] <= tree.metrics["mae"] * 0.95
        direction_not_worse = (
            sequence.metrics["directional_accuracy"] >= tree.metrics["directional_accuracy"]
        )
        selected[horizon] = sequence if materially_better and direction_not_worse else tree
    return selected
