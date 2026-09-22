"""Colab-only synthetic compatibility tests. Never calls the full training pipeline."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import site
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import tomllib


def environment(repo: Path, require_gpu: bool) -> dict:
    if sys.version_info[:2] != (3, 12) or sys.prefix == sys.base_prefix or site.ENABLE_USER_SITE:
        raise RuntimeError("Smoke requires isolated Python 3.12 without user/system packages")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Synthetic training smoke runs only in Colab after Drive mount")
    project = tomllib.loads((repo / "ml/pyproject.toml").read_text())
    versions = {}
    for requirement in project["project"]["dependencies"]:
        name, expected = requirement.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Version mismatch: {name} expected {expected}, got {actual}")
        versions[name] = actual
    for name in ("numpy", "pandas", "pyarrow", "tensorflow", "keras", "tf2onnx", "onnx", "onnxruntime", "mlflow", "xgboost", "shap", "matplotlib", "yfinance"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f"Global package leaked into venv: {name}: {module.__file__}")
    import matplotlib
    import tensorflow as tf

    if os.environ.get("MPLBACKEND") != "Agg" or matplotlib.get_backend().lower() != "agg":
        raise RuntimeError("Headless Agg backend is required before TensorFlow imports")
    gpus = tf.config.list_physical_devices("GPU")
    if require_gpu and not gpus:
        raise RuntimeError("TensorFlow sees no GPU. Select T4 GPU and reconnect; inspect CUDA diagnostics above")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return {"python": sys.version, "executable": sys.executable, "versions": versions,
            "gpus": [gpu.name for gpu in gpus], "mpl_backend": matplotlib.get_backend(),
            "lock_sha256": hashlib.sha256((repo / "ml/uv.lock").read_bytes()).hexdigest()}


def inference_probe(repo: Path, python: Path, root: Path, models: list[dict], cases: list[dict], names: list[str]):
    (root / "manifest.json").write_text(json.dumps({"production_models": models}), encoding="utf-8")
    (root / "feature_schema.json").write_text(json.dumps({"features": names}), encoding="utf-8")
    (root / "probe_cases.json").write_text(json.dumps(cases), encoding="utf-8")
    command = [str(python), "-I", str(repo / "ml/inference_probe.py"), str(root),
               "--backend-runtime", str(repo / "backend/app/runtime.py")]
    subprocess.run(command, check=True)  # Inherits stdout/stderr, retaining the actual traceback.


def synthetic_checks(repo: Path, inference_python: Path, work: Path, tracking_root: Path) -> dict:
    import mlflow
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    from cottonlens_ml.config import FEATURE_NAMES
    from cottonlens_ml.onnx_export import export_lstm
    from cottonlens_ml.tracking import tracked_run, tracking_session
    from cottonlens_ml.xgb_export import inference_booster

    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    names = FEATURE_NAMES
    raw = rng.normal(10, 3, (12, 60, len(names))).astype(np.float32)
    scaler = StandardScaler().fit(raw[:8].reshape(-1, len(names)))
    scaled = scaler.transform(raw.reshape(-1, len(names))).reshape(raw.shape).astype(np.float32)
    targets = rng.normal(0, 0.01, (12, 2)).astype(np.float32)
    target_scaler = StandardScaler().fit(targets[:8])
    scaled_targets = target_scaler.transform(targets).astype(np.float32)
    inputs = tf.keras.Input(shape=(60, len(names)))
    hidden = tf.keras.layers.LSTM(4)(inputs)
    hidden = tf.keras.layers.Dropout(0.1)(hidden)
    hidden = tf.keras.layers.Dense(32, activation="relu")(hidden)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2)(hidden))
    model.compile(optimizer="adam", loss="mae")
    model.fit(scaled[:8], scaled_targets[:8], validation_data=(scaled[8:], scaled_targets[8:]), epochs=1, batch_size=4, verbose=2)
    expected = target_scaler.inverse_transform(np.asarray(model(scaled[8:], training=False)))
    path = work / "tiny.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(target_scaler.inverse_transform(np.asarray(restored(scaled[8:], training=False))), expected, atol=1e-6)
    entries, parity = [], {}
    for horizon in (1, 5):
        path = work / f"lstm-t{horizon}.onnx"
        parity[str(horizon)] = export_lstm(restored, scaler, horizon, path, raw[8:], target_scaler)
        entries.append({"horizon": horizon, "format": "onnx", "path": path.name})
    cases = [{"features": [dict(zip(names, row.tolist(), strict=True)) for row in sequence],
              "expected": {str(h): float(expected[i, col]) for col, h in enumerate((1, 5))}}
             for i, sequence in enumerate(raw[8:])]
    inference_probe(repo, inference_python, work, entries, cases, names)

    frame = pd.DataFrame(raw[:, -1, :], columns=names)
    tree = xgb.XGBRegressor(n_estimators=8, max_depth=2, random_state=42, n_jobs=1, early_stopping_rounds=2)
    tree.fit(frame.iloc[:8], targets[:8, 0], eval_set=[(frame.iloc[8:], targets[8:, 0])], verbose=False)
    tree_path = work / "tiny-xgboost.json"
    tree.save_model(tree_path)
    loaded = xgb.XGBRegressor()
    loaded.load_model(tree_path)
    tree_expected = tree.predict(frame.iloc[8:])
    np.testing.assert_allclose(loaded.predict(frame.iloc[8:]), tree_expected, atol=1e-7)
    inference_booster(tree).save_model(tree_path)
    tree_entries = [{"horizon": h, "format": "xgboost_json", "path": tree_path.name} for h in (1, 5)]
    tree_cases = [{"features": frame.iloc[i].to_dict(), "expected": {"1": float(tree_expected[i-8]), "5": float(tree_expected[i-8])}} for i in range(8, 12)]
    inference_probe(repo, inference_python, work, tree_entries, tree_cases, names)
    parquet = work / "roundtrip.parquet"
    frame.to_parquet(parquet, index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(parquet))

    with (
        tracking_session(tracking_root, "smoke"),
        tracked_run(run_name="parent") as parent,
        tracked_run(run_name="child", nested=True) as child,
    ):
        mlflow.log_metric("smoke", 1.0)
        mlflow.log_artifact(str(parquet))
        child_id = child.info.run_id
        parent_id = parent.info.run_id
    with tracking_session(tracking_root, "smoke"):
        saved = mlflow.get_run(child_id)
        if saved.data.tags["mlflow.parentRunId"] != parent_id or saved.info.status != "FINISHED":
            raise ValueError("MLflow nested run did not survive SQLite restore")
    return {"lstm_save_load": "passed", "onnx": parity, "backend_inference_without_tensorflow": "passed",
            "xgboost_save_load": "passed", "parquet": "passed", "mlflow_nested_run_restore": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    args = parser.parse_args()
    report = {"status": "failed", "evidence": "synthetic compatibility only; no market performance claim"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["environment"] = environment(args.repo, require_gpu=True)
        # Non-finite feature behavior is a separate lightweight regression suite.
        subprocess.run([sys.executable, "-m", "pytest", str(args.repo / "ml/tests"), "-q"], check=True)
        with tempfile.TemporaryDirectory(prefix="cottonlens-smoke-") as temporary:
            report["checks"] = synthetic_checks(args.repo, args.inference_python, Path(temporary), args.report.parent / "smoke-tracking")
        report["status"] = "passed"
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Smoke report: {args.report} — {report['status']}", flush=True)


if __name__ == "__main__":
    main()
