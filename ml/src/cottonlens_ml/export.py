from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.features import schema_hash
from cottonlens_ml.onnx_export import export_lstm
from cottonlens_ml.training import Candidate
from cottonlens_ml.xgb_export import inference_booster


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_matrix(frame: pd.DataFrame, scaler: object) -> np.ndarray:
    values = scaler.transform(frame[FEATURE_NAMES]).astype(np.float32)
    return np.asarray(
        [values[index - 59 : index + 1] for index in range(59, len(values))],
        dtype=np.float32,
    )


def _sequence_matrix_with_history(
    history: pd.DataFrame, frame: pd.DataFrame, scaler: object
) -> np.ndarray:
    prior = history.loc[history.date < frame.date.min()].tail(59)
    if len(prior) != 59:
        raise ValueError("LSTM explanation needs 59 preceding feature snapshots")
    values = scaler.transform(pd.concat([prior, frame])[FEATURE_NAMES]).astype(np.float32)
    return np.asarray([values[index:index + 60] for index in range(len(frame))], dtype=np.float32)


def _display_contributions(values: np.ndarray, features: pd.Series) -> list[dict]:
    top = np.argsort(np.abs(values))[-8:]
    rows = [
        {
            "feature": FEATURE_NAMES[index],
            "display_name": FEATURE_NAMES[index].replace("_", " ").title(),
            "feature_value": float(features[FEATURE_NAMES[index]]),
            "contribution_pct": float(values[index] * 100),
        }
        for index in top
    ]
    rows.append({
        "feature": "other", "display_name": "Other features", "feature_value": 0.0,
        "contribution_pct": float((values.sum() - values[top].sum()) * 100),
    })
    return rows


def _lstm_shap(
    candidate: Candidate,
    full_features: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, float]:
    if candidate.scaler is None:
        raise ValueError("LSTM explanations require the train-fitted scaler")
    modeling_rows = full_features.dropna(subset=["target_return_1", "target_return_5"])
    train_rows = modeling_rows.iloc[: int(len(modeling_rows) * 0.65)]
    background = _sequence_matrix(train_rows.tail(160), candidate.scaler)
    if len(background) > 64:
        background = background[np.linspace(0, len(background) - 1, 64, dtype=int)]
    test_inputs = _sequence_matrix_with_history(full_features, test, candidate.scaler)
    live_values = candidate.scaler.transform(full_features[FEATURE_NAMES].tail(60)).astype(
        np.float32
    )
    explained_inputs = np.concatenate([test_inputs, live_values[None, ...]], axis=0)
    explainer = shap.GradientExplainer(candidate.model, background)
    batches: list[np.ndarray] = []
    output_index = 0 if horizon == 1 else 1
    for start in range(0, len(explained_inputs), 64):
        raw = explainer.shap_values(explained_inputs[start : start + 64], nsamples=64)
        if isinstance(raw, list):
            selected_output = np.asarray(raw[output_index])
        else:
            selected_output = np.asarray(raw)
            if selected_output.ndim == 4 and selected_output.shape[-1] == 2:
                selected_output = selected_output[..., output_index]
            elif selected_output.ndim == 4 and selected_output.shape[0] == 2:
                selected_output = selected_output[output_index]
        if selected_output.ndim != 3:
            raise ValueError(f"unexpected LSTM SHAP shape: {selected_output.shape}")
        batches.append(selected_output.sum(axis=1))
    contributions = np.concatenate(batches, axis=0)
    base_value = float(
        np.mean(candidate.model.predict(background, verbose=0)[:, output_index])
    )
    if candidate.target_scaler is not None:
        scale = float(candidate.target_scaler.scale_[output_index])
        offset = float(candidate.target_scaler.mean_[output_index])
        contributions *= scale
        base_value = base_value * scale + offset
    return contributions, base_value


def export_release(
    release_root: Path,
    full_features: pd.DataFrame,
    test: pd.DataFrame,
    market: pd.DataFrame,
    all_candidates: list[Candidate],
    selected: dict[int, Candidate],
    selection_audit: dict[int, dict],
    walkforward_report: dict,
) -> Path:
    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M")
    with tempfile.TemporaryDirectory(prefix="cottonlens-release-") as temp:
        root = Path(temp) / f"cottonlens-model-{version}"
        model_root = root / "model"
        preprocessing_root = root / "preprocessing"
        model_root.mkdir(parents=True)
        preprocessing_root.mkdir()
        model_entries: list[dict] = []
        for horizon, candidate in selected.items():
            if candidate.name == "XGBoost":
                path = model_root / f"xgboost-t{horizon}.json"
                inference_booster(candidate.model).save_model(path)
                model_entries.append(
                    {"horizon": horizon, "name": candidate.name, "format": "xgboost_json", "path": f"model/{path.name}"}
                )
            elif candidate.name == "LSTM":
                path = model_root / f"lstm-t{horizon}.onnx"
                history = full_features.loc[full_features.date <= test.date.max()]
                sampled = np.linspace(0, len(test) - 1, min(16, len(test)), dtype=int)
                raw_sequences = np.stack([
                    pd.concat([history.loc[history.date < test.date.min()].tail(59), test])[FEATURE_NAMES]
                    .iloc[index:index + 60].to_numpy(dtype=np.float32)
                    for index in sampled
                ])
                parity = export_lstm(candidate.model, candidate.scaler, horizon, path, raw_sequences, candidate.target_scaler)
                model_entries.append(
                    {
                        "horizon": horizon,
                        "name": candidate.name,
                        "format": "onnx",
                        "path": f"model/{path.name}",
                        **parity,
                    }
                )
            else:
                tree_fallback = next(
                    item for item in all_candidates if item.horizon == horizon and item.name == "XGBoost"
                )
                path = model_root / f"xgboost-t{horizon}-experimental.json"
                inference_booster(tree_fallback.model).save_model(path)
                model_entries.append(
                    {
                        "horizon": horizon,
                        "name": "XGBoost experimental sensitivity model",
                        "primary_forecast": "Naive",
                        "format": "xgboost_json",
                        "path": f"model/{path.name}",
                    }
                )

        schema = {"schema_hash": schema_hash(), "features": FEATURE_NAMES}
        (root / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        metrics = [
            {
                "model": item.name,
                "horizon": item.horizon,
                **item.metrics,
                "validation_metrics": item.validation_metrics,
                "training_history": item.training_history,
                "parameters": item.parameters,
                "walkforward": walkforward_report["aggregate"][f"{item.name}-T+{item.horizon}"],
                "selected": selected[item.horizon].name == item.name,
            }
            for item in all_candidates
        ]
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        modeling_rows = full_features.dropna(subset=["target_return_1", "target_return_5"])
        from cottonlens_ml.walkforward import audit_split
        split_frames = audit_split(modeling_rows)

        def date_range(frame: pd.DataFrame) -> dict[str, str]:
            return {"start": str(frame.date.min().date()), "end": str(frame.date.max().date())}

        manifest = {
            "artifact_version": version,
            "generated_at": datetime.now(UTC).isoformat(),
            "git_sha": _git_sha(),
            "dataset_range": {"start": str(full_features.date.min().date()), "end": str(full_features.date.max().date())},
            "split_ranges": {
                "train": date_range(split_frames["train"]),
                "validation": date_range(split_frames["validation"]),
                "historical_audit": date_range(split_frames["test"]),
            },
            "feature_schema_hash": schema_hash(),
            "production_models": model_entries,
            "selection_policy": {
                "minimum_mae_improvement_pct_vs_naive": 5.0,
                "minimum_directional_accuracy": {"T+1": 53.0, "T+5": 55.0},
                "lstm_minimum_mae_improvement_pct_vs_xgboost": 5.0,
                "selection_basis": "four pre-2024-06-18 rolling-origin folds",
                "historical_audit_role": "rejection only; previously observed, not independent",
            },
            "selection_audit": selection_audit,
            "walkforward_report": walkforward_report,
            "required_runtimes": {"python": ">=3.12,<3.14", "xgboost": ">=3,<4", "onnxruntime": ">=1.20,<2"},
            "source_freshness": {
                "market_as_of": str(pd.to_datetime(market.date).max().date()),
                "features_as_of": str(full_features.date.max().date()),
            },
            "data_quality": "historical_audit",
            "source_note": "Yahoo Finance continuous futures proxy; not official ICE settlement data.",
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        metric_rows = "\n".join(
            "| {model} | T+{horizon} | {mae:.4f} | {rmse:.4f} | {directional_accuracy:.2f}% | {selected} |".format(
                **{**metric, "selected": "yes" if metric["selected"] else "no"}
            )
            for metric in metrics
        )
        selection_rows = "\n".join(
            f"- T+{horizon}: **{candidate.name}**" for horizon, candidate in selected.items()
        )
        (root / "model_card.md").write_text(
            "\n".join(
                [
                    f"# CottonLens model card — {version}",
                    "",
                    "## Intended use",
                    "Research and demonstration forecasts for Cotton No. 2 continuous-futures proxy data. Not trading advice or an official ICE settlement forecast.",
                    "",
                    "## Selected production models",
                    selection_rows,
                    "",
                    "## Historical audit metrics (2024 onward; previously observed)",
                    "| Model | Horizon | MAE (¢/lb) | RMSE (¢/lb) | Directional accuracy | Selected |",
                    "|---|---:|---:|---:|---:|---:|",
                    metric_rows,
                    "",
                    "## Evaluation boundary",
                    "Four pre-2024-06-18 rolling-origin folds select the model. Each fold purges T+5 boundary targets and fits transforms on its train portion. The 2024 onward period is a previously seen historical rejection audit, not a new independent test. See metrics.json and manifest.json for fold evidence.",
                    "",
                    "## Limitations",
                    "Yahoo Finance continuous futures are a research proxy. Results can be affected by roll construction, revised upstream data, regime shifts, and missing exogenous drivers. Sensitivity output is not causal inference.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        finite_market = market.loc[
            np.isfinite(pd.to_numeric(market.close, errors="coerce"))
            & ((market.series == "wti") | (market.close > 0))
        ]
        finite_market.rename(columns={"date": "observed_on"}).to_parquet(
            root / "market_history.parquet", index=False
        )
        full_features[["date", *FEATURE_NAMES]].tail(500).to_parquet(
            root / "feature_snapshots.parquet", index=False
        )
        forecast_rows: list[dict] = []
        explanation_rows: list[dict] = []
        for horizon, candidate in selected.items():
            aligned_test = test.copy()
            predictions = candidate.predictions
            lstm_contributions: np.ndarray | None = None
            lstm_base_value = 0.0
            if candidate.name == "LSTM":
                lstm_contributions, lstm_base_value = _lstm_shap(
                    candidate, full_features, test, horizon
                )
            for row_index, (_, row) in enumerate(aligned_test.iterrows()):
                predicted_return = float(predictions[row_index])
                forecast_id = f"{version}-{horizon}-{row_index}"
                forecast_rows.append(
                    {
                        "id": forecast_id,
                        "as_of_date": row.date,
                        "target_date": row[f"target_date_{horizon}"],
                        "horizon": horizon,
                        "current_price": row.cotton_close,
                        "predicted_price": row.cotton_close * np.exp(predicted_return),
                        "predicted_return_pct": predicted_return * 100,
                        "actual_price": row.cotton_close * np.exp(row[f"target_return_{horizon}"]),
                        "origin_type": "backtest",
                    }
                )
            if candidate.name == "XGBoost":
                dmatrix = xgb.DMatrix(aligned_test[FEATURE_NAMES], feature_names=FEATURE_NAMES)
                contributions = inference_booster(candidate.model).predict(dmatrix, pred_contribs=True)
                for row_index, values in enumerate(contributions):
                    explanation_rows.append(
                        {
                            "forecast_id": f"{version}-{horizon}-{row_index}",
                            "base_value": float(values[-1] * 100),
                            "explainer": "XGBoost pred_contribs (TreeSHAP)",
                            "contributions": json.dumps(_display_contributions(values[:-1], aligned_test.iloc[row_index])),
                        }
                    )
            elif candidate.name == "LSTM" and lstm_contributions is not None:
                for row_index, values in enumerate(lstm_contributions[:-1]):
                    error = float(candidate.predictions[row_index] - lstm_base_value - values.sum())
                    explanation_rows.append(
                        {
                            "forecast_id": f"{version}-{horizon}-{row_index}",
                            "base_value": float(lstm_base_value * 100),
                            "explainer": "SHAP GradientExplainer (precomputed in Colab)",
                            "contributions": json.dumps(_display_contributions(values, aligned_test.iloc[row_index])),
                            "approximation_error_pct": error * 100,
                        }
                    )
            else:
                for row_index in range(len(aligned_test)):
                    explanation_rows.append(
                        {
                            "forecast_id": f"{version}-{horizon}-{row_index}",
                            "base_value": 0.0,
                            "explainer": "Naive persistence baseline",
                            "contributions": "[]",
                        }
                    )
            latest = full_features.iloc[-1]
            if candidate.name == "XGBoost":
                latest_prediction = float(candidate.model.predict(latest[FEATURE_NAMES].to_frame().T)[0])
            elif candidate.name == "LSTM":
                sequence = candidate.scaler.transform(full_features[FEATURE_NAMES].tail(60)).astype(np.float32)
                outputs = candidate.model.predict(sequence.reshape(1, 60, len(FEATURE_NAMES)), verbose=0)[0]
                output_column = 0 if horizon == 1 else 1
                latest_prediction = float(outputs[output_column])
                if candidate.target_scaler is not None:
                    latest_prediction = (
                        latest_prediction * candidate.target_scaler.scale_[output_column]
                        + candidate.target_scaler.mean_[output_column]
                    )
            else:
                latest_prediction = 0.0
            live_id = f"{version}-{horizon}-live"
            forecast_rows.append(
                {
                    "id": live_id,
                    "as_of_date": latest.date,
                    "target_date": latest.date + pd.offsets.BDay(horizon),
                    "horizon": horizon,
                    "current_price": latest.cotton_close,
                    "predicted_price": latest.cotton_close * np.exp(latest_prediction),
                    "predicted_return_pct": latest_prediction * 100,
                    "actual_price": None,
                    "origin_type": "live",
                }
            )
            if candidate.name == "XGBoost":
                latest_contributions = inference_booster(candidate.model).predict(
                    xgb.DMatrix(latest[FEATURE_NAMES].to_frame().T, feature_names=FEATURE_NAMES),
                    pred_contribs=True,
                )[0]
                explanation_rows.append(
                    {
                        "forecast_id": live_id,
                        "base_value": float(latest_contributions[-1] * 100),
                        "explainer": "XGBoost pred_contribs (TreeSHAP)",
                        "contributions": json.dumps(_display_contributions(latest_contributions[:-1], latest)),
                    }
                )
            elif candidate.name == "LSTM" and lstm_contributions is not None:
                values = lstm_contributions[-1]
                error = float(latest_prediction - lstm_base_value - values.sum())
                explanation_rows.append(
                    {
                        "forecast_id": live_id,
                        "base_value": float(lstm_base_value * 100),
                        "explainer": "SHAP GradientExplainer (precomputed in Colab)",
                        "contributions": json.dumps(_display_contributions(values, latest)),
                        "approximation_error_pct": error * 100,
                    }
                )
            else:
                explanation_rows.append(
                    {
                        "forecast_id": live_id,
                        "base_value": 0.0,
                        "explainer": "Naive persistence baseline",
                        "contributions": "[]",
                    }
                )
        pd.DataFrame(forecast_rows).to_parquet(root / "forecasts.parquet", index=False)
        pd.DataFrame(explanation_rows).to_parquet(root / "explanations.parquet", index=False)
        checksum_lines = []
        for path in sorted(file for file in root.rglob("*") if file.is_file()):
            if path.name == "checksums.sha256":
                continue
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
        (root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        output = release_root / f"{root.name}.zip"
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, arcname=f"{root.name}/{path.relative_to(root)}")
        output.with_suffix(f"{output.suffix}.sha256").write_text(
            f"{_sha256(output)}  {output.name}\n", encoding="utf-8"
        )
        shutil.copy2(root / "manifest.json", release_root / f"{version}-manifest.json")
        return output
