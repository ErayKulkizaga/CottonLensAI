from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from cottonlens_ml.config import (
    DATA_POLICY_VERSION,
    EXTERNAL_MAX_AGE_SESSIONS,
    FEATURE_NAMES,
    HORIZON_SEMANTICS,
)
from cottonlens_ml.quality import clean_market

LEGACY_ALIGNMENT = "preceding-cotton-observation"
AVAILABILITY_ALIGNMENT = "assumed-next-day-0015-utc"


def _external_alignment(
    market: pd.DataFrame, cotton_dates: pd.DatetimeIndex, series: str,
    *, policy: str = LEGACY_ALIGNMENT, decision_times=None,
) -> pd.DataFrame:
    """Positive log-domain prices known no later than the preceding Cotton row.

    Source daily bars are considered available at 00:00 UTC the following day.
    No same-origin external close enters a feature; age counts Cotton observations
    since the last usable source close, including that mandatory lag.
    """
    if policy not in (LEGACY_ALIGNMENT, AVAILABILITY_ALIGNMENT):
        raise ValueError("Unknown external alignment policy")
    rows = market.loc[market.series == series].set_index("date").sort_index()
    valid = rows.close.where(rows.close > 0).dropna()
    result = pd.DataFrame(index=cotton_dates)
    if valid.empty:
        result["close"] = np.nan
        result["source_date"] = pd.NaT
        result["age_sessions"] = np.nan
    else:
        if policy == LEGACY_ALIGNMENT:
            prior = pd.Series(cotton_dates, index=cotton_dates).shift(1)
            positions = valid.index.searchsorted(pd.DatetimeIndex(prior), side="right") - 1
            usable = prior.notna().to_numpy() & (positions >= 0)
        else:
            decisions = pd.DatetimeIndex(decision_times) if decision_times is not None else (
                pd.to_datetime(cotton_dates, utc=True) + pd.Timedelta(days=1, minutes=15))
            if len(decisions) != len(cotton_dates) or decisions.hasnans or decisions.tz is None:
                raise ValueError("One timezone-aware decision timestamp per Cotton observation required")
            availability = pd.to_datetime(valid.index, utc=True) + pd.Timedelta(days=1)
            positions = availability.searchsorted(decisions, side="right") - 1
            usable = positions >= 0
        chosen = np.maximum(positions, 0)
        result["close"] = np.where(usable, valid.to_numpy()[chosen], np.nan)
        source_dates = pd.Series(valid.index.take(chosen), index=cotton_dates).where(usable)
        result["source_date"] = source_dates
        source_positions = cotton_dates.searchsorted(pd.DatetimeIndex(source_dates), side="right") - 1
        result["age_sessions"] = pd.Series(
            np.arange(len(cotton_dates)) - source_positions, index=cotton_dates
        ).where(usable)
    result["available_at"] = pd.to_datetime(result.source_date, utc=True) + pd.Timedelta(days=1)
    result["stale"] = result.age_sessions.gt(EXTERNAL_MAX_AGE_SESSIONS)
    result.loc[result.stale, "close"] = np.nan
    return result


def build_feature_history(market: pd.DataFrame, cftc: pd.DataFrame, *,
                          alignment_policy: str = LEGACY_ALIGNMENT) -> pd.DataFrame:
    """Keep every recorded Cotton observation, including incomplete feature rows."""
    market, quality = clean_market(market)
    cotton_rows = market[market["series"] == "cotton"].set_index("date").sort_index()
    features = pd.DataFrame(index=cotton_rows.index)
    cotton = cotton_rows["close"]
    for window in (1, 5, 10, 20):
        features[f"cotton_ret_{window}"] = np.log(cotton / cotton.shift(window))
    features["cotton_momentum_5"] = cotton / cotton.shift(5) - 1
    features["cotton_momentum_20"] = cotton / cotton.shift(20) - 1
    features["cotton_sma_ratio_5_20"] = cotton.rolling(5).mean() / cotton.rolling(20).mean() - 1
    features["cotton_sma_ratio_20_50"] = cotton.rolling(20).mean() / cotton.rolling(50).mean() - 1
    daily_return = np.log(cotton / cotton.shift(1))
    features["cotton_volatility_5"] = daily_return.rolling(5).std()
    features["cotton_volatility_20"] = daily_return.rolling(20).std()
    volatility_60 = daily_return.rolling(60).std()
    features["cotton_volatility_regime_20_60"] = features["cotton_volatility_20"] / volatility_60.where(volatility_60 > 0)
    features["cotton_range"] = (cotton_rows["high"] - cotton_rows["low"]) / cotton_rows["close"]
    volume = cotton_rows["volume"]
    features["cotton_volume_change"] = volume / volume.shift(1).where(volume.shift(1) > 0) - 1
    volume_std = volume.rolling(20).std()
    features["cotton_volume_z20"] = (volume - volume.rolling(20).mean()) / volume_std.where(volume_std > 0)
    alignments = {}
    for series in ("dxy", "wti"):
        alignment = _external_alignment(market, cotton_rows.index, series, policy=alignment_policy)
        alignments[series] = alignment
        aligned = alignment.close
        for window in (1, 5, 20):
            features[f"{series}_ret_{window}"] = np.log(aligned / aligned.shift(window))
        features[f"cotton_{series}_corr_60"] = daily_return.rolling(60).corr(
            features[f"{series}_ret_1"]
        )

    cftc_values = cftc.set_index("available_date")["cftc_managed_money_net"].sort_index()
    if not cftc_values.empty:
        cftc_values.index = pd.to_datetime(cftc_values.index)
    cftc_values = pd.to_numeric(cftc_values, errors="coerce")
    available = cftc_values.reindex(features.index.union(cftc_values.index)).sort_index().ffill()
    features["cftc_managed_money_net"] = available.reindex(features.index)
    weekly = cftc_values.diff()
    four_week = cftc_values.diff(4)
    deviation = cftc_values.rolling(52).std()
    z52 = (cftc_values - cftc_values.rolling(52).mean()) / deviation.where(deviation > 0)
    for name, values in (
        ("cftc_net_change_1w", weekly),
        ("cftc_net_change_4w", four_week),
        ("cftc_net_z52", z52),
    ):
        aligned = values.reindex(features.index.union(values.index)).sort_index().ffill()
        features[name] = aligned.reindex(features.index)
    features["month_sin"] = np.sin(2 * np.pi * features.index.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * features.index.month / 12)
    features["target_return_1"] = np.log(cotton.shift(-1) / cotton)
    features["target_return_5"] = np.log(cotton.shift(-5) / cotton)
    features["cotton_close"] = cotton
    # Keep the newest rows even though their future targets are not known yet; they are
    # required for live inference. Modeling code drops missing targets before splitting.
    infinite = np.isinf(features.to_numpy(dtype=float))
    quality["infinite_feature_values"] = [
        {"date": features.index[row].isoformat(), "feature": features.columns[column]}
        for row, column in zip(*np.where(infinite), strict=True)
    ]
    cotton_dates = pd.Series(features.index, index=features.index)
    features["target_date_1"] = cotton_dates.shift(-1)
    features["target_date_5"] = cotton_dates.shift(-5)
    features = features.replace([np.inf, -np.inf], np.nan)
    features["cotton_session_index"] = np.arange(len(features), dtype=np.int64)
    features["decision_time"] = pd.to_datetime(features.index, utc=True) + pd.Timedelta(days=1)
    if alignment_policy == AVAILABILITY_ALIGNMENT:
        features["decision_time"] += pd.Timedelta(minutes=15)
    for series, alignment in alignments.items():
        for name in ("source_date", "available_at", "age_sessions", "stale"):
            features[f"{series}_{name}"] = alignment[name]
    result = features.reset_index().rename(columns={"index": "date"})
    complete = features[FEATURE_NAMES].notna().all(axis=1) & features.cotton_close.notna()
    quality["dropped_incomplete_feature_rows"] = int((~complete).sum())
    quality["missing_feature_counts_before_drop"] = {
        name: int(count) for name, count in
        features[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).isna().sum().items()
        if count
    }
    quality["missing_features_by_origin"] = [
        {"date": date.isoformat(), "features": row.index[row.isna()].tolist()}
        for date, row in features.loc[~complete, FEATURE_NAMES].iterrows()
    ]
    retained_positions = np.flatnonzero(complete.to_numpy())
    quality["nonconsecutive_retained_cotton_steps"] = int(np.sum(np.diff(retained_positions) > 1))
    quality["sequence_calendar_note"] = (
        "60 feature observations, not 60 calendar days; cotton_session_index exposes dropped rows. "
        "Absent exchange sessions cannot be distinguished from holidays without a trusted ICE calendar."
    )
    quality["date_contract"] = {
        "policy_version": DATA_POLICY_VERSION,
        "horizon_semantics": HORIZON_SEMANTICS,
        "decision_clock": ("00:15" if alignment_policy == AVAILABILITY_ALIGNMENT else "00:00")
            + " UTC on the day after the Cotton source observation date",
        "source_close_availability": "Daily source dates conservatively usable next UTC day; not vendor publication proof",
        "trusted_exchange_calendar": False,
        "missing_exchange_session_detection": "unknown; holidays and absent provider rows are not inferred",
        "live_target_date_policy": "unknown until the future Cotton observation is recorded",
    }
    quality["external_policy"] = {
        "alignment_policy": alignment_policy,
        "availability_evidence": "assumption, not verified historical publication or ingestion",
        "lag_cotton_observations": 1 if alignment_policy == LEGACY_ALIGNMENT else 0,
        "max_age_cotton_observations_including_lag": EXTERNAL_MAX_AGE_SESSIONS,
        "nonpositive_log_domain": "mask only transformed input, use bounded earlier positive close",
        "stale_counts": {name: int(value.stale.sum()) for name, value in alignments.items()},
        "unavailable_counts": {name: int(value.close.isna().sum()) for name, value in alignments.items()},
        "log_domain_masked_observations": [
            {"series": row.series, "date": row.date.isoformat(), "value": float(row.close)}
            for row in market.loc[market.series.isin(["dxy", "wti"]) & (market.close <= 0)].itertuples()
        ],
    }
    result.attrs["data_quality"] = quality
    return result


def select_feature_rows(
    history: pd.DataFrame, names: list[str], *, require_targets: bool = True
) -> pd.DataFrame:
    """Filter a group's own availability; callers separately lock common origins."""
    if not names or not set(names).issubset(FEATURE_NAMES):
        raise ValueError("Only verified model feature names are accepted (CFTC is audit-only)")
    required = [*names, "cotton_close"]
    if require_targets:
        required += ["target_return_1", "target_return_5", "target_date_1", "target_date_5"]
    result = history.dropna(subset=required).copy()
    if not np.isfinite(result[[*names, "cotton_close"]].to_numpy(dtype=float)).all():
        raise ValueError("Selected feature rows contain non-finite values")
    result.attrs = history.attrs.copy()
    return result


def coverage_report(history: pd.DataFrame, groups: dict[str, list[str]] | None = None) -> dict:
    groups = groups or {
        "cotton_only": [name for name in FEATURE_NAMES if not name.startswith(
            ("dxy_", "wti_", "cotton_dxy_", "cotton_wti_")
        )],
        "all_features": FEATURE_NAMES,
    }
    selected = {name: select_feature_rows(history, names) for name, names in groups.items()}
    common = set.intersection(*(set(frame.date) for frame in selected.values()))
    return {
        "comparison_policy": "Performance uses frozen common origins; natural availability is coverage only",
        "recorded_cotton_observations": len(history),
        "common_origin_count": len(common),
        "groups": {
            name: {
                "natural_origin_count": len(frame),
                "common_origin_count": len(common),
                "additional_natural_origin_count": len(frame) - len(common),
                "first_origin": frame.date.min().isoformat() if not frame.empty else None,
                "last_origin": frame.date.max().isoformat() if not frame.empty else None,
            }
            for name, frame in selected.items()
        },
    }


def build_features(market: pd.DataFrame, cftc: pd.DataFrame) -> pd.DataFrame:
    """Compatibility API: complete full-schema rows, including live-only origins."""
    history = build_feature_history(market, cftc)
    return select_feature_rows(history, FEATURE_NAMES, require_targets=False).reset_index(drop=True)


def require_latest_live_row(history: pd.DataFrame, names: list[str] | None = None) -> pd.Series:
    """Fail instead of silently forecasting from an older complete observation."""
    complete = select_feature_rows(history, names or FEATURE_NAMES, require_targets=False)
    if complete.empty or complete.date.max() != history.date.max():
        raise ValueError("Latest Cotton row has invalid/incomplete features; refusing a stale live forecast")
    return complete.iloc[-1]


def schema_hash() -> str:
    payload = json.dumps(FEATURE_NAMES, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_end = int(len(frame) * 0.65)
    validation_end = int(len(frame) * 0.80)
    validation_start = frame.iloc[train_end]["date"]
    test_start = frame.iloc[validation_end]["date"]
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    # An origin in the preceding split is not trainable when its T+5 outcome
    # lands in the following split, even if its feature timestamp is earlier.
    train = train.loc[train.target_date_5 < validation_start]
    validation = validation.loc[validation.target_date_5 < test_start]
    return {"train": train, "validation": validation, "test": frame.iloc[validation_end:].copy()}
