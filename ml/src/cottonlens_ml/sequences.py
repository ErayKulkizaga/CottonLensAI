"""Construct past-only windows without dropping context when a target is missing."""

import numpy as np
import pandas as pd

from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.protocol import complete_feature_rows

SEQUENCE_SEMANTICS = "60 retained complete-feature observations, not 60 calendar or exchange sessions"


def _context(frame: pd.DataFrame, history: pd.DataFrame | None, window: int, require_all: bool):
    if window < 1 or frame.empty:
        raise ValueError("Sequence window and origins must be nonempty")
    if frame.date.isna().any() or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError("Sequence origins must have unique ascending dates")
    source = (frame if history is None else history).copy()
    if source.date.isna().any() or source.date.duplicated().any() or not source.date.is_monotonic_increasing:
        raise ValueError("Sequence history must have unique ascending dates")
    if "cotton_session_index" not in source:
        source["cotton_session_index"] = np.arange(len(source))
    source = complete_feature_rows(source.loc[source.date <= frame.date.max()])
    ordinals = source.cotton_session_index.to_numpy(dtype=float)
    if not np.isfinite(ordinals).all() or np.any(np.diff(ordinals) <= 0):
        raise ValueError("Sequence source session ordinals must be strictly increasing")
    positions = pd.Index(source.date).get_indexer(frame.date)
    if np.any(positions < 0):
        raise ValueError("Sequence history is missing a complete-feature evaluation origin")
    eligible = positions >= window - 1
    if history is not None and require_all and not eligible.all():
        raise ValueError(f"{window}-step evaluation needs {window - 1} earlier feature snapshots")
    return source, positions, eligible


def sequences(
    frame: pd.DataFrame, scaler, window: int = 60, history: pd.DataFrame | None = None,
    require_all: bool = True,
):
    source, positions, eligible = _context(frame, history, window, require_all)
    values = source[FEATURE_NAMES].to_numpy() if scaler is None else scaler.transform(source[FEATURE_NAMES])
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Sequence scaler produced non-finite features")
    targets = frame[["target_return_1", "target_return_5"]].to_numpy(dtype=np.float32)
    return (
        np.asarray([values[position - window + 1:position + 1] for position in positions[eligible]], dtype=np.float32).reshape(-1, window, len(FEATURE_NAMES)),
        targets[eligible],
    )


def sequence_diagnostics(frame: pd.DataFrame, history: pd.DataFrame, window: int = 60) -> dict:
    source, positions, eligible = _context(frame, history, window, False)
    ordinals = source.cotton_session_index.to_numpy(dtype=int)
    skipped = [int(ordinals[position] - ordinals[position - window + 1] + 1 - window)
               for position in positions[eligible]]
    return {"semantics": SEQUENCE_SEMANTICS, "window": window,
            "requested_origins": len(frame), "sample_count": int(eligible.sum()),
            "origins_without_context": int((~eligible).sum()),
            "windows_with_missing_feature_observations": sum(value > 0 for value in skipped),
            "max_missing_feature_observations_in_window": max(skipped, default=0),
            "ordinal_basis": "source Cotton observations; not independently verified exchange sessions"}
