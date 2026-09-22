from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.quality import clean_market


def build_features(market: pd.DataFrame, cftc: pd.DataFrame) -> pd.DataFrame:
    market, quality = clean_market(market)
    indexed = market.pivot(index="date", columns="series", values="close").sort_index()
    cotton_rows = market[market["series"] == "cotton"].set_index("date").sort_index()
    # Fill external series on their full source timeline before taking Cotton sessions.
    indexed[["dxy", "wti"]] = indexed[["dxy", "wti"]].ffill()
    indexed = indexed.reindex(cotton_rows.index)
    features = pd.DataFrame(index=indexed.index)
    cotton = indexed["cotton"]
    for window in (1, 5, 10, 20):
        features[f"cotton_ret_{window}"] = np.log(cotton / cotton.shift(window))
    features["cotton_momentum_5"] = cotton / cotton.shift(5) - 1
    features["cotton_momentum_20"] = cotton / cotton.shift(20) - 1
    features["cotton_sma_ratio_5_20"] = cotton.rolling(5).mean() / cotton.rolling(20).mean() - 1
    features["cotton_sma_ratio_20_50"] = cotton.rolling(20).mean() / cotton.rolling(50).mean() - 1
    daily_return = np.log(cotton / cotton.shift(1))
    features["cotton_volatility_5"] = daily_return.rolling(5).std()
    features["cotton_volatility_20"] = daily_return.rolling(20).std()
    features["cotton_range"] = (cotton_rows["high"] - cotton_rows["low"]) / cotton_rows["close"]
    volume = cotton_rows["volume"]
    features["cotton_volume_change"] = volume / volume.shift(1).where(volume.shift(1) > 0) - 1
    for series in ("dxy", "wti"):
        aligned = indexed[series].ffill()
        for window in (1, 5, 20):
            features[f"{series}_ret_{window}"] = np.log(aligned / aligned.shift(window))

    cftc_values = cftc.set_index("available_date")["cftc_managed_money_net"].sort_index()
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
    result = (
        features.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURE_NAMES, "cotton_close"])
        .reset_index()
        .rename(columns={"index": "date"})
    )
    quality["dropped_incomplete_feature_rows"] = len(features) - len(result)
    result.attrs["data_quality"] = quality
    return result


def schema_hash() -> str:
    payload = json.dumps(FEATURE_NAMES, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    train_end = int(len(frame) * 0.65)
    validation_end = int(len(frame) * 0.80)
    return {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }
