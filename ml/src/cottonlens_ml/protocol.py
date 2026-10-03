"""Time boundaries and experiment identity; never imports a training runtime."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.checkpoints import identity_hash
from cottonlens_ml.config import FEATURE_NAMES

PROTOCOL_VERSION = "v2-day1-frozen-cohort-refit-identity"
VALIDATION_ROWS = 126


def experiment_identity(*frames: pd.DataFrame, settings: dict | None = None) -> dict:
    """Bind reuse to data, all protocol code, dependency lock and actual runtime."""
    package = Path(__file__).parent
    source = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    dependencies = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "tensorflow", "keras", "mlflow"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not-installed"
    locks = {}
    project = package.parent.parent
    for path in [project / "uv.lock", project / "pyproject.toml", *sorted((project / "constraints").glob("*"))]:
        if path.is_file():
            locks[str(path.relative_to(project))] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = []
    for frame in frames:
        digest = hashlib.sha256(json.dumps({
            "columns": list(frame.columns), "dtypes": [str(value) for value in frame.dtypes],
        }, sort_keys=True).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        data.append({"sha256": digest.hexdigest(), "rows": len(frame),
                     "snapshot_identity": frame.attrs.get("data_identity")})
    return {
        "protocol": PROTOCOL_VERSION, "feature_schema": list(FEATURE_NAMES),
        "source_sha256": source.hexdigest(), "data": data,
        "settings": settings or {}, "dependency_locks": locks,
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "dependencies": dependencies},
    }


def fingerprint(*frames: pd.DataFrame, settings: dict | None = None) -> str:
    return identity_hash(experiment_identity(*frames, settings=settings))


def complete_feature_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Retained feature observations, including rows whose labels are not available."""
    return frame.loc[np.isfinite(frame[FEATURE_NAMES].to_numpy(dtype=float)).all(axis=1)].copy()


def validate_partition(frame: pd.DataFrame, name: str, *, labels: bool = True) -> None:
    if frame.empty or frame.date.isna().any() or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError(f"{name} requires nonempty unique ascending dates")
    numeric = FEATURE_NAMES + (["cotton_close", "target_return_1", "target_return_5"] if labels else [])
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains non-finite features or labels")
    if labels:
        if (frame.cotton_close <= 0).any() or frame.target_date_5.isna().any() or (frame.target_date_5 <= frame.date).any():
            raise ValueError(f"{name} contains invalid label dates or prices")
        if "target_date_1" in frame and (
            frame.target_date_1.isna().any() or (frame.target_date_1 <= frame.date).any()
            or (frame.target_date_1 > frame.target_date_5).any()
        ):
            raise ValueError(f"{name} contains invalid T+1 label dates")


def refit_rows(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """After settings are locked, validation labels become fit data, never test labels."""
    if train.empty or validation.empty or test.empty:
        raise ValueError("Refit requires nonempty train, validation and evaluation partitions")
    validate_partition(train, "Training")
    validate_partition(validation, "Validation")
    validate_partition(test, "Evaluation", labels=False)
    if train.target_date_5.max() >= validation.date.min():
        raise ValueError("Training labels overlap the tuning validation period")
    if validation.target_date_5.max() >= test.date.min():
        raise ValueError("Refit labels cross the evaluation boundary")
    frame = pd.concat([train, validation]).sort_values("date").reset_index(drop=True)
    if history is not None:
        frame = complete_feature_rows(history).loc[
            (history.date >= train.date.min()) & (history.date <= validation.date.max())
        ].dropna(subset=["target_return_1", "target_return_5"]).copy().reset_index(drop=True)
    if frame.date.duplicated().any() or frame.target_date_5.isna().any():
        raise ValueError("Invalid refit dates")
    if frame.target_date_5.max() >= test.date.min():
        raise ValueError("Refit labels cross the evaluation boundary")
    validate_partition(frame, "Refit")
    return frame


def split_diagnostics(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Describe training/inner-validation drift without inspecting audit outcomes."""
    deviations = train[FEATURE_NAMES].std().replace(0, float("nan"))
    shifts = ((validation[FEATURE_NAMES].mean() - train[FEATURE_NAMES].mean()) / deviations).abs().dropna()
    return {
        "train_rows": len(train), "validation_rows": len(validation),
        "largest_feature_mean_shifts_train_std": {
            key: float(value) for key, value in shifts.nlargest(6).items()
        },
        "targets": {
            f"T+{horizon}": {
                "train_mean_pct": float(train[f"target_return_{horizon}"].mean() * 100),
                "validation_mean_pct": float(validation[f"target_return_{horizon}"].mean() * 100),
                "train_std_pct": float(train[f"target_return_{horizon}"].std() * 100),
                "validation_std_pct": float(validation[f"target_return_{horizon}"].std() * 100),
            }
            for horizon in (1, 5)
        },
    }
