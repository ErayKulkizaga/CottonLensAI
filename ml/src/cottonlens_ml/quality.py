"""Explicit input validation before log-return feature generation."""

import numpy as np
import pandas as pd


def clean_market(market: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = market.copy(deep=True)
    required = {"date", "series", "close", "high", "low", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing market columns: {sorted(required - set(frame.columns))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.date.isna().any() or frame.duplicated(["date", "series"]).any():
        raise ValueError("Market dates must be present and (date, series) keys unique")
    if "cotton" not in set(frame.series):
        raise ValueError("Cotton source observations are required")
    issues = []
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        # Negative WTI is real market history. Preserve it here and mask it only
        # when a transformation explicitly requires positive prices.
        invalid = ~np.isfinite(numeric) | (
            numeric < 0 if column == "volume" else (numeric <= 0) & (frame.series != "wti")
        )
        for index in frame.index[invalid]:
            issues.append({
                "series": str(frame.at[index, "series"]),
                "date": frame.at[index, "date"].isoformat(),
                "field": column,
                "value": str(frame.at[index, column]),
                "reason": "non_finite_or_negative_volume" if column == "volume" else "non_positive_or_non_finite_price",
            })
        frame[column] = numeric.mask(invalid)
    return frame, {
        "missing_external_series": sorted({"dxy", "wti"} - set(frame.series)),
        "invalid_market_values": issues,
        "invalid_market_value_count": len(issues),
        "policy": "Mask non-finite inputs and non-positive Cotton/DXY prices. "
        "Never fill Cotton prices/targets. Preserve finite negative WTI in raw data; "
        "positive-domain features use bounded past-only fill after a one-observation lag.",
    }
