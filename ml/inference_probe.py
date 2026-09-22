"""Run with the tiny inference venv, without installing the backend application."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--backend-runtime", type=Path, required=True)
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    for name in ("tensorflow", "keras", "tf2onnx", "mlflow"):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"Inference environment must not contain {name}")
    spec = importlib.util.spec_from_file_location("cottonlens_backend_runtime", args.backend_runtime)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.ArtifactRuntime(str(args.bundle))
    runtime.load()
    if args.release:
        cases = json.loads((args.bundle / "release_cases.json").read_text())
        for horizon, case in cases.items():
            prediction = runtime.predict_return(int(horizon), case["features"])
            if not np.isfinite(prediction):
                raise ValueError("Non-finite runtime prediction")
            if not case["experimental"] and abs(prediction - case["expected"]) >= 1e-4:
                raise ValueError(f"Release forecast/runtime parity failed: T+{horizon}")
        print("Release loads with backend runtime; TensorFlow absent; forecast parity passed", flush=True)
        return
    cases = json.loads((args.bundle / "probe_cases.json").read_text())
    errors = {}
    for entry in runtime.manifest["production_models"]:
        horizon = int(entry["horizon"])
        predictions = [runtime.predict_return(horizon, case["features"]) for case in cases]
        expected = [case["expected"][str(horizon)] for case in cases]
        error = float(np.max(np.abs(np.asarray(predictions) - expected)))
        if not np.isfinite(predictions).all() or error >= 1e-4:
            raise ValueError(f"Backend runtime parity failed for T+{horizon}: {error}")
        errors[str(horizon)] = error
    print(json.dumps({"tensorflow_installed": False, "backend_runtime_max_abs_errors": errors}), flush=True)


if __name__ == "__main__":
    main()
