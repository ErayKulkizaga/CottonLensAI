import importlib.util
import json
import math
from datetime import date
from pathlib import Path

import numpy as np


class ArtifactRuntimeError(RuntimeError):
    pass


class ArtifactRuntime:
    """Loads only lightweight inference artifacts; no training stack is imported."""

    def __init__(self, artifact_dir: str) -> None:
        self.root = Path(artifact_dir)
        self.manifest: dict = {}
        self.payload_checksum: str | None = None
        self.feature_names: list[str] = []
        self.models: dict[int, tuple[str, object]] = {}
        self._feature_snapshots: dict[date, dict[str, float]] = {}

    @property
    def ready(self) -> bool:
        return bool(self.manifest and self.models)

    def load(self, *, with_snapshots: bool = False) -> None:
        self.manifest = {}
        self.payload_checksum = None
        self.models = {}
        self._feature_snapshots = {}
        manifest_path = self.root / "manifest.json"
        schema_path = self.root / "feature_schema.json"
        if not manifest_path.exists() or not schema_path.exists():
            raise ArtifactRuntimeError("manifest.json or feature_schema.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload_checksum = None
        if with_snapshots or manifest.get("artifact_schema_version") in (2, 3) or (self.root / "checksums.sha256").exists():
            spec = importlib.util.spec_from_file_location("cottonlens_artifact_verifier", Path(__file__).with_name("artifacts.py"))
            verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier)
            manifest, payload_checksum = verifier.verify_directory_with_identity(self.root)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.feature_names = schema["features"]
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ArtifactRuntimeError("feature names must be nonempty and unique")
        expected_hash = manifest.get("feature_schema_hash")
        if expected_hash and schema.get("schema_hash") != expected_hash:
            raise ArtifactRuntimeError("feature schema hash does not match the manifest")
        models: dict[int, tuple[str, object]] = {}
        for horizon_config in manifest.get("production_models", []):
            horizon = int(horizon_config["horizon"])
            model_format = horizon_config["format"]
            model_path = self.root / horizon_config["path"]
            if model_format == "research_adapter_v1":
                from app.research_runtime import ResearchModel
                model = ResearchModel(self.root, horizon_config)
            elif model_format == "xgboost_json":
                import xgboost as xgb

                model = xgb.Booster()
                model.load_model(model_path)
            elif model_format == "onnx":
                import onnxruntime as ort

                model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            else:
                raise ArtifactRuntimeError(f"unsupported model format: {model_format}")
            if horizon in models:
                raise ArtifactRuntimeError("duplicate runtime horizon")
            models[horizon] = (model_format, model)
        if not models:
            raise ArtifactRuntimeError("manifest does not define a production model")
        if with_snapshots:
            self._feature_snapshots = self._read_feature_snapshots()
        if payload_checksum is not None and verifier.payload_identity(self.root) != payload_checksum:
            raise ArtifactRuntimeError("artifact payload changed while loading the runtime")
        self.models = models
        self.payload_checksum = payload_checksum
        self.manifest = manifest

    def _read_feature_snapshots(self) -> dict[date, dict[str, float]]:
        import pyarrow.parquet as pq

        snapshots = {}
        for row in pq.read_table(self.root / "feature_snapshots.parquet").to_pylist():
            observed = date.fromisoformat(str(row.pop("date"))[:10])
            if observed in snapshots:
                raise ArtifactRuntimeError("duplicate release feature snapshot date")
            snapshots[observed] = {name: float(value) for name, value in row.items() if value is not None}
        return dict(sorted(snapshots.items()))

    def feature_window(self, origin: date, window: int) -> list[dict[str, float]]:
        if origin not in self._feature_snapshots:
            raise ArtifactRuntimeError("baseline feature snapshot is unavailable in the loaded release")
        dates = [observed for observed in self._feature_snapshots if observed <= origin]
        if len(dates) < window:
            raise ArtifactRuntimeError(f"{window} feature snapshots are required in the loaded release")
        return [dict(self._feature_snapshots[observed]) for observed in dates[-window:]]

    def model_format(self, horizon: int) -> str:
        if horizon not in self.models:
            raise ArtifactRuntimeError(f"no runtime model for horizon {horizon}")
        return self.models[horizon][0]

    def model_name(self, horizon: int) -> str:
        for entry in self.manifest.get("production_models", []):
            if int(entry["horizon"]) == horizon:
                return str(entry["name"])
        raise ArtifactRuntimeError(f"no runtime model metadata for horizon {horizon}")

    def required_window(self, horizon: int) -> int:
        kind, model = self.models[horizon]
        return model.window if kind == 'research_adapter_v1' else (60 if kind == 'onnx' else 1)

    def predict_return(
        self, horizon: int, features: dict[str, float] | list[dict[str, float]]
    ) -> float:
        if horizon not in self.models:
            raise ArtifactRuntimeError(f"no runtime model for horizon {horizon}")
        model_format, model = self.models[horizon]
        if model_format == 'research_adapter_v1':
            return float(model.predict(features))
        if model_format == "xgboost_json":
            import xgboost as xgb

            current = features[-1] if isinstance(features, list) else features
            vector = np.asarray([[current[name] for name in self.feature_names]], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ArtifactRuntimeError("non-finite inference features")
            return float(model.predict(xgb.DMatrix(vector, feature_names=self.feature_names))[0])
        if not isinstance(features, list) or len(features) != 60:
            raise ArtifactRuntimeError("ONNX LSTM inference requires exactly 60 ordered feature snapshots")
        vector = np.asarray(
            [[[row[name] for name in self.feature_names] for row in features]], dtype=np.float32
        ).reshape(1, 60, len(self.feature_names))
        if not np.isfinite(vector).all():
            raise ArtifactRuntimeError("non-finite inference features")
        input_name = model.get_inputs()[0].name
        return float(np.asarray(model.run(None, {input_name: vector})[0]).reshape(-1)[0])

    @staticmethod
    def price_from_log_return(current_price: float, predicted_log_return: float) -> float:
        return current_price * math.exp(predicted_log_return)
