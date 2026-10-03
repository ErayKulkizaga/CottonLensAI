from __future__ import annotations

import importlib
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.checkpoints import (
    completed_checkpoint,
    history_path,
    identity_hash,
    save_completed_checkpoint,
)
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.evaluation import evaluate, price_mae_metric
from cottonlens_ml.protocol import (
    PROTOCOL_VERSION,
    complete_feature_rows,
    experiment_identity,
    refit_rows,
    validate_partition,
)
from cottonlens_ml.runtime_guard import require_colab_training
from cottonlens_ml.selection import select_walkforward_name
from cottonlens_ml.sequences import sequence_diagnostics
from cottonlens_ml.sequences import sequences as _sequences
from cottonlens_ml.tracking import tracked_run

SEED = 42


def _runtime(name):
    """Keep no-training tests and protocol imports free of training runtimes."""
    return importlib.import_module(name)


def _scaler():
    return _runtime("sklearn.preprocessing").StandardScaler()


@dataclass
class Candidate:
    name: str
    horizon: int
    model: object
    predictions: np.ndarray
    metrics: dict[str, float]
    validation_metrics: dict[str, float] | None = None
    scaler: object | None = None
    target_scaler: object | None = None
    training_history: list[dict[str, float]] | None = None
    parameters: dict | None = None


def _identity(*frames, family, horizon, role, config, cutoff):
    return experiment_identity(*frames, settings={
        "model_family": family, "horizon": horizon, "role": role,
        "seed": SEED, "config": config, "prediction_cutoff": pd.Timestamp(cutoff).isoformat(),
    })


def _fit_metadata(refit, cutoff, identity, config, role="evaluation_backtest"):
    return {
        "protocol": PROTOCOL_VERSION, "model_role": role,
        "fit_cutoff": pd.Timestamp(cutoff).isoformat(),
        "fit_origin_cutoff": refit.date.max().isoformat(),
        "fit_label_cutoff": refit.target_date_5.max().isoformat(),
        "refit_end": str(refit.date.max().date()), "refit_rows": len(refit),
        "recipe_identity": identity_hash(config), "locked_config": config,
        "experiment_identity": identity, "model_identity": identity_hash(identity),
    }


def _sequence_identity_context(context, cutoff):
    # Labels on history-only observations are never consumed by the sequence
    # builder. In particular, pending/audit labels cannot alter a locked model.
    columns = ["date", *FEATURE_NAMES]
    if "cotton_session_index" in context:
        columns.append("cotton_session_index")
    return context.loc[context.date <= cutoff, columns].copy()


def _restored(path, identity, load, *, history=False):
    if completed_checkpoint(path, identity, require_history=history):
        try:
            return load(path)
        except Exception:  # noqa: BLE001 - a checksummed payload can still fail model-format validation
            return None
    return None


def naive_candidates(test: pd.DataFrame, validation: pd.DataFrame) -> dict[int, Candidate]:
    candidates = {}
    for horizon in (1, 5):
        predictions = np.zeros(len(test), dtype=np.float64)
        candidates[horizon] = Candidate(
            "Naive", horizon, None, predictions,
            evaluate(test.cotton_close.to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predictions),
            validation_metrics=evaluate(
                validation.cotton_close.to_numpy(), validation[f"target_return_{horizon}"].to_numpy(),
                np.zeros(len(validation), dtype=np.float64),
            ), parameters={"protocol": PROTOCOL_VERSION, "model_role": "evaluation_backtest",
                           "prediction_semantics": "zero log return; last observed Cotton price"},
        )
    return candidates


def ridge_candidates(train, validation, test, feature_history=None) -> dict[int, Candidate]:
    """Fixed-parameter linear reference with the same safe refit origins as XGB/LSTM."""
    require_colab_training()
    refit = refit_rows(train, validation, test, history=feature_history)
    ridge = _runtime("sklearn.linear_model").Ridge
    scaler = _scaler().fit(train[FEATURE_NAMES])
    refit_scaler = _scaler().fit(refit[FEATURE_NAMES])
    result = {}
    for horizon in (1, 5):
        model = ridge(alpha=1.0).fit(scaler.transform(train[FEATURE_NAMES]), train[f"target_return_{horizon}"])
        validation_metrics = evaluate(
            validation.cotton_close.to_numpy(), validation[f"target_return_{horizon}"].to_numpy(),
            model.predict(scaler.transform(validation[FEATURE_NAMES])),
        )
        model = ridge(alpha=1.0).fit(refit_scaler.transform(refit[FEATURE_NAMES]), refit[f"target_return_{horizon}"])
        predictions = model.predict(refit_scaler.transform(test[FEATURE_NAMES]))
        config = {"alpha": 1.0}
        identity = _identity(refit, family="Ridge", horizon=horizon, role="evaluation_refit", config=config, cutoff=test.date.min())
        result[horizon] = Candidate(
            "Ridge", horizon, model, predictions,
            evaluate(test.cotton_close.to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predictions),
            validation_metrics=validation_metrics, scaler=refit_scaler,
            parameters={**config, **_fit_metadata(refit, test.date.min(), identity, config),
                        "feature_scaler_fit": "tuning_train_then_refit_train_plus_validation"},
        )
    return result


def _load_xgb(path):
    model = _runtime("xgboost").XGBRegressor()
    model.load_model(path)
    return model


def train_xgboost(train, validation, test, checkpoint_root: Path, feature_history=None) -> dict[int, Candidate]:
    require_colab_training()
    refit = refit_rows(train, validation, test, history=feature_history)
    xgb = _runtime("xgboost")
    mlflow = _runtime("mlflow")
    grid = list(itertools.product((3, 5), (0.03, 0.07), (0.8, 1.0)))
    result = {}
    for horizon in (1, 5):
        best = None
        trials = []
        for depth, learning_rate, subsample in grid:
            config = {
                "n_estimators": 1200, "max_depth": depth, "learning_rate": learning_rate,
                "subsample": subsample, "colsample_bytree": 0.9,
                "objective": "reg:squarederror", "random_state": SEED,
                "early_stopping_rounds": 50, "n_jobs": 2,
                "early_stopping_metric": "price_mae",
            }
            identity = _identity(train, validation, family="XGBoost", horizon=horizon,
                                 role="tuning", config=config, cutoff=validation.date.min())
            checkpoint = checkpoint_root / f"xgb-{identity_hash(identity)}.json"
            print(f"XGBoost T+{horizon}: depth={depth}, lr={learning_rate}, subsample={subsample}", flush=True)
            with tracked_run(run_name=f"xgboost-t{horizon}", nested=True):
                model = _restored(checkpoint, identity, _load_xgb)
                resumed = model is not None
                if model is None:
                    kwargs = {key: value for key, value in config.items() if key != "early_stopping_metric"}
                    model = xgb.XGBRegressor(**kwargs, eval_metric=price_mae_metric(validation.cotton_close.to_numpy()))
                    model.fit(train[FEATURE_NAMES], train[f"target_return_{horizon}"],
                              eval_set=[(validation[FEATURE_NAMES], validation[f"target_return_{horizon}"])], verbose=False)
                metrics = evaluate(validation.cotton_close.to_numpy(), validation[f"target_return_{horizon}"].to_numpy(),
                                   model.predict(validation[FEATURE_NAMES]))
                if not resumed:
                    save_completed_checkpoint(checkpoint, identity, model.save_model)
                trial = {"max_depth": depth, "learning_rate": learning_rate, "subsample": subsample,
                         "best_iteration": int(model.best_iteration), "validation_mae": metrics["mae"],
                         "resumed_from_drive": resumed, "experiment_identity": identity_hash(identity)}
                mlflow.log_params({**config, "horizon": horizon, "resumed_from_drive": resumed,
                                   "experiment_identity": identity_hash(identity), "protocol": PROTOCOL_VERSION})
                mlflow.log_metric("validation_mae", metrics["mae"])
                trials.append(trial)
                if best is None or metrics["mae"] < best[0]:
                    best = metrics["mae"], model, trial, metrics
        assert best is not None
        locked = {
            "n_estimators": best[2]["best_iteration"] + 1, "max_depth": best[2]["max_depth"],
            "learning_rate": best[2]["learning_rate"], "subsample": best[2]["subsample"],
            "colsample_bytree": 0.9, "objective": "reg:squarederror", "random_state": SEED, "n_jobs": 2,
        }
        identity = _identity(refit, family="XGBoost", horizon=horizon, role="evaluation_refit", config=locked, cutoff=test.date.min())
        path = checkpoint_root / f"xgb-{identity_hash(identity)}-refit.json"
        selected = _restored(path, identity, _load_xgb)
        if selected is None:
            selected = xgb.XGBRegressor(**locked)
            selected.fit(refit[FEATURE_NAMES], refit[f"target_return_{horizon}"], verbose=False)
            save_completed_checkpoint(path, identity, selected.save_model)
        predictions = selected.predict(test[FEATURE_NAMES])
        result[horizon] = Candidate(
            "XGBoost", horizon, selected, predictions,
            evaluate(test.cotton_close.to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predictions),
            validation_metrics=best[3], parameters={
                **best[2], "seed": SEED, "trials": trials,
                "training_objective": "log_return_squared_error_surrogate",
                "early_stopping_metric": "price_mae", "selection_metric": "validation_price_mae",
                **_fit_metadata(refit, test.date.min(), identity, locked),
            },
        )
    return result


def _new_lstm(units: int, dropout: float):
    tf = _runtime("tensorflow")
    tf.keras.utils.set_random_seed(SEED)
    inputs = tf.keras.Input(shape=(60, len(FEATURE_NAMES)), name="features")
    hidden = tf.keras.layers.LSTM(units)(inputs)
    hidden = tf.keras.layers.Dropout(dropout)(hidden)
    hidden = tf.keras.layers.Dense(32, activation="relu")(hidden)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2, name="returns")(hidden))
    model.compile(optimizer="adam", loss="mae")
    return model


def _validated_history(path):
    rows = json.loads(history_path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("LSTM completion requires nonempty history")
    for index, row in enumerate(rows):
        if row["epoch"] != index + 1 or not np.isfinite([row["loss"], row["val_loss"]]).all():
            raise ValueError("LSTM completion history is invalid")
    return rows


def _finite_sequences(x, y):
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("LSTM training requires nonempty finite sequences and labels")


def train_lstm(train, validation, test, checkpoint_root: Path, feature_history=None):
    require_colab_training()
    refit = refit_rows(train, validation, test, history=feature_history)
    tf = _runtime("tensorflow")
    mlflow = _runtime("mlflow")
    context = feature_history if feature_history is not None else pd.concat([train, validation, test]).sort_values("date")
    scaler = _scaler().fit(train[FEATURE_NAMES])
    train_x, train_y_raw = _sequences(train, scaler, history=context, require_all=False)
    validation_x, validation_y_raw = _sequences(validation, scaler, history=context)
    _finite_sequences(train_x, train_y_raw)
    _finite_sequences(validation_x, validation_y_raw)
    target_scaler = _scaler().fit(train_y_raw)
    train_y = target_scaler.transform(train_y_raw).astype(np.float32)
    validation_y = target_scaler.transform(validation_y_raw).astype(np.float32)
    best = None
    trials = []
    for units, dropout in [(32, 0.10), (32, 0.20), (64, 0.10), (64, 0.20)]:
        config = {"units": units, "dropout": dropout, "batch_size": 64, "epochs": 100,
                  "optimizer": "adam", "loss": "standardized_joint_log_return_mae", "patience": 10, "window": 60}
        identity = _identity(train, validation, _sequence_identity_context(context, validation.date.max()),
                             family="LSTM", horizon=[1, 5], role="tuning", config=config, cutoff=validation.date.min())
        path = checkpoint_root / f"lstm-{identity_hash(identity)}.keras"
        with tracked_run(run_name="lstm-multi-horizon", nested=True):
            model = _restored(path, identity, tf.keras.models.load_model, history=True)
            history_rows = None
            if model is not None:
                try:
                    history_rows = _validated_history(path)
                except (ValueError, TypeError, KeyError, OSError):
                    model = None
            resumed = model is not None
            if model is None:
                model = _new_lstm(units, dropout)
                history = model.fit(train_x, train_y, validation_data=(validation_x, validation_y),
                                    epochs=100, batch_size=64, verbose=2,
                                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
                history_rows = [
                    {"epoch": epoch + 1, "loss": float(loss), "val_loss": float(val_loss)}
                    for epoch, (loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"], strict=True))
                ]
                if not history_rows or not all(np.isfinite([row["loss"], row["val_loss"]]).all() for row in history_rows):
                    raise ValueError("LSTM trial did not produce a finite completed history")
                save_completed_checkpoint(path, identity, model.save, history=history_rows)
            loss = min(row["val_loss"] for row in history_rows)
            trial = {"units": units, "dropout": dropout, "validation_loss": loss,
                     "epochs_run": len(history_rows), "best_epoch": min(history_rows, key=lambda row: row["val_loss"])["epoch"],
                     "resumed_from_drive": resumed, "experiment_identity": identity_hash(identity)}
            trials.append(trial)
            mlflow.log_params({**config, "seed": SEED, "resumed_from_drive": resumed,
                               "experiment_identity": identity_hash(identity), "protocol": PROTOCOL_VERSION})
            mlflow.log_metric("validation_loss", loss)
            if best is None or loss < best[0]:
                best = loss, model, history_rows, trial
    assert best is not None
    validation_predictions = target_scaler.inverse_transform(best[1].predict(validation_x, verbose=0))
    locked = {"units": best[3]["units"], "dropout": best[3]["dropout"], "epochs": best[3]["best_epoch"],
              "batch_size": 64, "optimizer": "adam", "loss": "standardized_joint_log_return_mae", "window": 60}
    scaler = _scaler().fit(refit[FEATURE_NAMES])
    refit_x, refit_y_raw = _sequences(refit, scaler, history=context, require_all=False)
    _finite_sequences(refit_x, refit_y_raw)
    target_scaler = _scaler().fit(refit_y_raw)
    identity = _identity(refit, _sequence_identity_context(context, refit.date.max()), family="LSTM", horizon=[1, 5],
                         role="evaluation_refit", config=locked, cutoff=test.date.min())
    path = checkpoint_root / f"lstm-{identity_hash(identity)}-refit.keras"
    selected = _restored(path, identity, tf.keras.models.load_model)
    if selected is None:
        selected = _new_lstm(locked["units"], locked["dropout"])
        selected.fit(refit_x, target_scaler.transform(refit_y_raw).astype(np.float32),
                     epochs=locked["epochs"], batch_size=64, verbose=2)
        save_completed_checkpoint(path, identity, selected.save)
    test_x, _ = _sequences(test, scaler, history=context)
    predictions = target_scaler.inverse_transform(selected.predict(test_x, verbose=0))
    parameters = {**locked, "seed": SEED, "trials": trials, "refit_epochs": locked["epochs"],
                  "training_objective": "standardized_joint_log_return_mae",
                  "selection_metric": "validation_standardized_joint_log_return_mae",
                  "scaler_fit": "tuning_train_then_refit_train_plus_validation",
                  "refit_sequence_rows": len(refit_x), "sequence_diagnostics": sequence_diagnostics(test, context),
                  **_fit_metadata(refit, test.date.min(), identity, locked)}
    candidates = {}
    for column, horizon in enumerate((1, 5)):
        candidates[horizon] = Candidate(
            "LSTM", horizon, selected, predictions[:, column],
            evaluate(test.cotton_close.to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predictions[:, column]),
            validation_metrics=evaluate(validation.cotton_close.to_numpy(), validation[f"target_return_{horizon}"].to_numpy(), validation_predictions[:, column]),
            scaler=scaler, target_scaler=target_scaler, training_history=best[2], parameters=parameters.copy(),
        )
    return selected, scaler, candidates


def _refit_locked(candidates, features, cutoff, checkpoint_root, *, role):
    cutoff = pd.Timestamp(cutoff)
    context = features.loc[features.date <= cutoff].copy()
    refit = complete_feature_rows(context)
    refit = refit.loc[(refit.target_date_5 < cutoff)].dropna(subset=["target_return_1", "target_return_5"]).copy()
    validate_partition(refit, "Deployment refit")
    result = {}
    lstm_cache = {}
    for horizon, candidate in candidates.items():
        params = candidate.parameters or {}
        config = params.get("locked_config")
        if candidate.name == "Naive":
            result[horizon] = replace(candidate, predictions=np.zeros(0), metrics={}, validation_metrics=None,
                                      parameters={**params, "model_role": role, "fit_cutoff": cutoff.isoformat(),
                                                  "fit_origin_cutoff": None, "fit_label_cutoff": None})
            continue
        if not config:
            raise ValueError("Deployment refit requires an evaluation-locked configuration")
        identity = _identity(refit, _sequence_identity_context(context, cutoff) if candidate.name == "LSTM" else refit,
                             family=candidate.name, horizon=[1, 5] if candidate.name == "LSTM" else horizon,
                             role=role, config=config, cutoff=cutoff)
        metadata = _fit_metadata(refit, cutoff, identity, config, role=role)
        scaler = target_scaler = None
        if candidate.name == "XGBoost":
            path = checkpoint_root / f"xgb-{identity_hash(identity)}-{role}.json"
            model = _restored(path, identity, _load_xgb)
            if model is None:
                model = _runtime("xgboost").XGBRegressor(**config)
                model.fit(refit[FEATURE_NAMES], refit[f"target_return_{horizon}"], verbose=False)
                save_completed_checkpoint(path, identity, model.save_model)
        elif candidate.name == "LSTM":
            key = identity_hash(identity)
            if key not in lstm_cache:
                scaler = _scaler().fit(refit[FEATURE_NAMES])
                x, y = _sequences(refit, scaler, history=context, require_all=False)
                _finite_sequences(x, y)
                target_scaler = _scaler().fit(y)
                path = checkpoint_root / f"lstm-{key}-{role}.keras"
                model = _restored(path, identity, _runtime("tensorflow").keras.models.load_model)
                if model is None:
                    model = _new_lstm(config["units"], config["dropout"])
                    model.fit(x, target_scaler.transform(y).astype(np.float32), epochs=config["epochs"], batch_size=config["batch_size"], verbose=2)
                    save_completed_checkpoint(path, identity, model.save)
                lstm_cache[key] = model, scaler, target_scaler, len(x)
            model, scaler, target_scaler, count = lstm_cache[key]
            metadata["refit_sequence_rows"] = count
        elif candidate.name == "Ridge":
            scaler = _scaler().fit(refit[FEATURE_NAMES])
            model = _runtime("sklearn.linear_model").Ridge(**config).fit(scaler.transform(refit[FEATURE_NAMES]), refit[f"target_return_{horizon}"])
        else:
            raise ValueError(f"Unsupported deployment family: {candidate.name}")
        result[horizon] = replace(candidate, model=model, predictions=np.zeros(0), metrics={}, validation_metrics=None,
                                  scaler=scaler, target_scaler=target_scaler, training_history=None,
                                  parameters={**params, **metadata})
    return result


def refit_deployment(candidates: dict[int, Candidate], features: pd.DataFrame, cutoff, checkpoint_root: Path) -> dict[int, Candidate]:
    """Fresh current weights with locked settings; no selection, tuning or audit decisions."""
    require_colab_training()
    return _refit_locked(candidates, features, cutoff, checkpoint_root, role="deployment_live")


def refit_locked_evaluation(candidates: dict[int, Candidate], features: pd.DataFrame,
                            test: pd.DataFrame, checkpoint_root: Path) -> dict[int, Candidate]:
    """Score already-locked recipes on seen audit without any new inner tuning."""
    require_colab_training()
    validate_partition(test, "Locked evaluation")
    result = _refit_locked(candidates, features, test.date.min(), checkpoint_root, role="evaluation_backtest")
    for horizon, candidate in result.items():
        if candidate.name == "Naive":
            predictions = np.zeros(len(test))
        elif candidate.name == "LSTM":
            x, _ = _sequences(test, candidate.scaler, history=features)
            predictions = candidate.target_scaler.inverse_transform(candidate.model.predict(x, verbose=0))[:, (0 if horizon == 1 else 1)]
            candidate.parameters["sequence_diagnostics"] = sequence_diagnostics(test, features)
        else:
            x = test[FEATURE_NAMES] if candidate.scaler is None else candidate.scaler.transform(test[FEATURE_NAMES])
            predictions = candidate.model.predict(x)
        candidate.predictions = predictions
        candidate.metrics = evaluate(test.cotton_close.to_numpy(), test[f"target_return_{horizon}"].to_numpy(), predictions)
        candidate.parameters["evidence_label"] = "seen_historical_audit"
        candidate.parameters["audit_policy"] = "descriptive_only_no_selection_veto_or_tuning"
    return result


def select_production(naive, xgboost_candidates, lstm_candidates, walkforward_report):
    selected = {}
    audit = {}
    for horizon in (1, 5):
        baseline, tree, sequence = naive[horizon], xgboost_candidates[horizon], lstm_candidates[horizon]
        model_name, horizon_audit = select_walkforward_name(
            walkforward_report, horizon,
            {"Naive": baseline.metrics, "XGBoost": tree.metrics, "LSTM": sequence.metrics},
        )
        selected[horizon] = {"Naive": baseline, "XGBoost": tree, "LSTM": sequence}[model_name]
        audit[horizon] = horizon_audit
    return selected, audit
