"""Validate the exact just-exported ZIP with the existing backend contract."""

import argparse
import importlib.util
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.code_identity import source_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--inference-python", type=Path, required=True)
    args = parser.parse_args()
    releases = args.drive_root / "artifacts/releases"
    release = args.release.resolve()
    if release.parent != releases.resolve() or release.suffix != ".zip":
        raise ValueError("Release must be an explicit ZIP in this Drive root's release directory")
    name = release.name
    spec = importlib.util.spec_from_file_location("backend_artifacts", args.repo / "backend/app/artifacts.py")
    artifacts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(artifacts)
    with tempfile.TemporaryDirectory(prefix="cottonlens-release-check-") as temporary:
        root = Path(temporary) / "current"
        manifest = artifacts.install_bundle(release, root)
        if manifest.get("artifact_schema_version") == 2 and manifest["code_identity"] != source_identity(args.repo):
            raise ValueError("Release source identity differs from the exact verified Colab workspace")
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
                "expected": (entry["runtime_reference_return"] if entry.get("primary_forecast") == "Naive"
                             else float(forecast.predicted_return_pct / 100)),
                "experimental": entry.get("primary_forecast") == "Naive",
            }
        case_path = Path(temporary) / "release_cases.json"
        case_path.write_text(json.dumps(cases, allow_nan=False), encoding="utf-8")
        subprocess.run([str(args.inference_python), "-I", str(args.repo / "ml/inference_probe.py"), str(root),
                        "--backend-runtime", str(args.repo / "backend/app/runtime.py"), "--release",
                        "--cases", str(case_path)], check=True)
    receipt = {
        "status": "passed", "artifact_version": manifest["artifact_version"],
        "zip_sha256": artifacts.sha256_file(release), "validated_at": datetime.now(UTC).isoformat(),
        "checks": {"complete_checksum_inventory": "passed", "backend_runtime_parity": "passed",
                   "tensorflow_free_inference": "passed"},
        "tolerance_log_return": 1e-4,
    }
    receipt_path = release.with_suffix(".validation.json")
    staged = receipt_path.with_suffix(".pending")
    staged.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
    staged.replace(receipt_path)
    pointer = releases / "latest.pending"
    pointer.write_text(name, encoding="utf-8")
    pointer.replace(releases / "latest.txt")
    print(f"Artifact validated: {name}", flush=True)


if __name__ == "__main__":
    main()
