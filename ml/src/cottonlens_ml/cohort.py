"""Freeze evaluation origins before any model fit or metric inspection."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.features import schema_hash, select_feature_rows
from cottonlens_ml.protocol import PROTOCOL_VERSION

AUDIT_START = pd.Timestamp("2024-06-18")
FOLD_ENDS = ("2022-10-21", "2023-05-11", "2023-11-23", "2024-06-10")
FOLD_SIZE = 126


def content_id(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frame_identity(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash ordered rows and dtypes; never round floating-point observations."""
    digest = hashlib.sha256(json.dumps([(name, str(frame[name].dtype)) for name in columns]).encode())
    digest.update(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes())
    return digest.hexdigest()


def create_cohort(history: pd.DataFrame, data_id: str) -> dict:
    if not data_id:
        raise ValueError("A frozen data identity is required")
    if history.date.duplicated().any() or not history.date.is_monotonic_increasing:
        raise ValueError("Cohort source dates must be unique and sorted")
    # A scored origin must have the complete locked feature schema available at
    # its decision time. We retain the fixed anchors and record the resulting
    # 504 dates before any fit; keeping label-valid but unscorable dates would
    # make a supposedly frozen cohort impossible to evaluate.
    eligible = select_feature_rows(history, FEATURE_NAMES)
    eligible = eligible.loc[(eligible.cotton_close > 0) & (eligible.target_date_5 < AUDIT_START)]
    folds = []
    previous_end = pd.Timestamp.min
    columns = ["date", "cotton_close", "target_return_1", "target_return_5", "target_date_1", "target_date_5"]
    for number, end in enumerate(FOLD_ENDS, start=1):
        rows = eligible.loc[(eligible.date > previous_end) & (eligible.date <= pd.Timestamp(end))].tail(FOLD_SIZE)
        if len(rows) != FOLD_SIZE:
            raise ValueError(f"Frozen fold {number} needs 126 eligible Cotton origins before {end}; found {len(rows)}")
        folds.append({
            "fold": number, "end_anchor": end,
            "origins": [str(value.date()) for value in rows.date],
            "target_dates": {
                str(h): [str(value.date()) for value in rows[f"target_date_{h}"]] for h in (1, 5)
            },
            "sample_count": len(rows), "label_identity": frame_identity(rows, columns),
        })
        previous_end = pd.Timestamp(end)
    payload = {
        "protocol_version": PROTOCOL_VERSION, "data_id": data_id,
        "feature_schema_hash": schema_hash(), "audit_start": str(AUDIT_START.date()),
        "evidence_status": "corrected_v2_historical_development_not_exact_v1_reproduction",
        "origin_policy": (
            "last_126_complete_feature_and_price_label_valid_origins_per_fixed_end_anchor; "
            "feature availability is frozen before fitting and cannot change after cohort creation"
        ),
        "v1_evidence": "Original raw/processed snapshot, 504 origins, tracking/checkpoints and full release must be verified separately; aggregate report alone is insufficient.",
        "source_feature_identity": frame_identity(history, ["date", *FEATURE_NAMES]),
        "folds": folds,
    }
    return {**payload, "cohort_id": content_id(payload)}


def verify_cohort(manifest: dict, history: pd.DataFrame, data_id: str) -> None:
    payload = {key: value for key, value in manifest.items() if key != "cohort_id"}
    if manifest.get("cohort_id") != content_id(payload):
        raise ValueError("Cohort manifest checksum mismatch")
    if manifest != create_cohort(history, data_id):
        raise ValueError("Frozen cohort/data/feature/label identity changed; create a new experiment")


def freeze_cohort(path: Path, history: pd.DataFrame, data_id: str) -> dict:
    manifest = create_cohort(history, data_id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        verify_cohort(existing, history, data_id)
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an existing experiment being replaced.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return manifest
