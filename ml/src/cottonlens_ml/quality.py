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
    if not {"cotton", "dxy", "wti"}.issubset(set(frame.series)):
        raise ValueError("Cotton, DXY and WTI sources are required")
    issues = []
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = ~np.isfinite(numeric) | (numeric < 0 if column == "volume" else numeric <= 0)
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
        "invalid_market_values": issues,
        "invalid_market_value_count": len(issues),
        "policy": "Mask invalid prices as missing. Never fill Cotton prices/targets. "
        "DXY/WTI use past-only forward fill. Drop incomplete feature rows. "
        "Negative WTI can be a real observation but is outside the log-return domain.",
    }
