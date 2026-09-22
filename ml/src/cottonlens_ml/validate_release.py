"""Validate the exact just-exported ZIP with the existing backend contract."""

import argparse
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    args = parser.parse_args()
    releases = args.drive_root / "artifacts/releases"
    name = (releases / "latest.txt").read_text().strip()
    if Path(name).name != name or not name.endswith(".zip"):
        raise ValueError("Invalid release pointer")
    spec = importlib.util.spec_from_file_location("backend_artifacts", args.repo / "backend/app/artifacts.py")
    artifacts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(artifacts)
    with tempfile.TemporaryDirectory(prefix="cottonlens-release-check-") as temporary:
        root = Path(temporary) / "current"
        manifest = artifacts.install_bundle(releases / name, root)
        for filename in ("market_history", "feature_snapshots", "forecasts", "explanations"):
            pd.read_parquet(root / f"{filename}.parquet")
        schema = json.loads((root / "feature_schema.json").read_text())
        snapshots = pd.read_parquet(root / "feature_snapshots.parquet").sort_values("date")
        forecasts = pd.read_parquet(root / "forecasts.parquet")
        cases = {}
        for entry in manifest["production_models"]:
            horizon = int(entry["horizon"])
            forecast = forecasts[(forecasts.horizon == horizon) & (forecasts.origin_type == "live")].iloc[-1]
            rows = snapshots[snapshots.date <= forecast.as_of_date][schema["features"]]
            raw = rows.tail(60).to_numpy(dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError("Non-finite release inference input")
            cases[str(horizon)] = {
                "features": rows.tail(60).to_dict("records") if entry["format"] == "onnx" else rows.iloc[-1].to_dict(),
                "expected": float(forecast.predicted_return_pct / 100),
                "experimental": entry.get("primary_forecast") == "Naive",
            }
        (root / "release_cases.json").write_text(json.dumps(cases), encoding="utf-8")
        subprocess.run([str(args.inference_python), "-I", str(args.repo / "ml/inference_probe.py"), str(root),
                        "--backend-runtime", str(args.repo / "backend/app/runtime.py"), "--release"], check=True)
    print(f"Artifact validated: {name}", flush=True)


if __name__ == "__main__":
    main()
