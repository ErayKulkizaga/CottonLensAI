import json
import math
from pathlib import Path

import numpy as np


class ArtifactRuntimeError(RuntimeError):
    pass


class ArtifactRuntime:
    """Loads only lightweight inference artifacts; no training stack is imported."""

    def __init__(self, artifact_dir: str) -> None:
        self.root = Path(artifact_dir)
        self.manifest: dict = {}
        self.feature_names: list[str] = []
        self.models: dict[int, tuple[str, object]] = {}

    @property
    def ready(self) -> bool:
        return bool(self.manifest and self.models)

    def load(self) -> None:
        manifest_path = self.root / "manifest.json"
        schema_path = self.root / "feature_schema.json"
        if not manifest_path.exists() or not schema_path.exists():
            raise ArtifactRuntimeError("manifest.json or feature_schema.json is missing")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.feature_names = schema["features"]
        expected_hash = self.manifest.get("feature_schema_hash")
        if expected_hash and schema.get("schema_hash") != expected_hash:
            raise ArtifactRuntimeError("feature schema hash does not match the manifest")
        for horizon_config in self.manifest.get("production_models", []):
            horizon = int(horizon_config["horizon"])
            model_format = horizon_config["format"]
            model_path = self.root / horizon_config["path"]
            if model_format == "xgboost_json":
                import xgboost as xgb

                model = xgb.Booster()
                model.load_model(model_path)
            elif model_format == "onnx":
                import onnxruntime as ort

                model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            else:
                raise ArtifactRuntimeError(f"unsupported model format: {model_format}")
            self.models[horizon] = (model_format, model)
        if not self.models:
            raise ArtifactRuntimeError("manifest does not define a production model")

    def model_format(self, horizon: int) -> str:
        if horizon not in self.models:
            raise ArtifactRuntimeError(f"no runtime model for horizon {horizon}")
        return self.models[horizon][0]

    def model_name(self, horizon: int) -> str:
        for entry in self.manifest.get("production_models", []):
            if int(entry["horizon"]) == horizon:
                return str(entry["name"])
        raise ArtifactRuntimeError(f"no runtime model metadata for horizon {horizon}")

    def predict_return(
        self, horizon: int, features: dict[str, float] | list[dict[str, float]]
    ) -> float:
        if horizon not in self.models:
            raise ArtifactRuntimeError(f"no runtime model for horizon {horizon}")
        model_format, model = self.models[horizon]
        if model_format == "xgboost_json":
            import xgboost as xgb

            current = features[-1] if isinstance(features, list) else features
            vector = np.asarray([[current[name] for name in self.feature_names]], dtype=np.float32)
            return float(model.predict(xgb.DMatrix(vector, feature_names=self.feature_names))[0])
        if not isinstance(features, list) or len(features) != 60:
            raise ArtifactRuntimeError("ONNX LSTM inference requires exactly 60 ordered feature snapshots")
        vector = np.asarray(
            [[[row[name] for name in self.feature_names] for row in features]], dtype=np.float32
        ).reshape(1, 60, len(self.feature_names))
        input_name = model.get_inputs()[0].name
        return float(np.asarray(model.run(None, {input_name: vector})[0]).reshape(-1)[0])

    @staticmethod
    def price_from_log_return(current_price: float, predicted_log_return: float) -> float:
        return current_price * math.exp(predicted_log_return)
