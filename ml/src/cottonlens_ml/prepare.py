"""Validate cached/downloaded sources and produce features without training imports."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cottonlens_ml.config import PipelinePaths
from cottonlens_ml.data import cache_sources
from cottonlens_ml.features import build_features, chronological_split


def prepare(paths: PipelinePaths, refresh: bool = False):
    paths.create()
    market, cftc = cache_sources(paths.raw, refresh=refresh)
    # Yahoo can return an in-progress current-day candle. Delay at most one
    # session rather than treating an incomplete close as an observed target.
    utc_today = datetime.now(UTC).date()
    incomplete_rows = int((market.date.dt.date >= utc_today).sum())
    market = market.loc[market.date.dt.date < utc_today].copy()
    if not cftc.empty and not np.isfinite(cftc["cftc_managed_money_net"]).all():
        raise ValueError("CFTC source contains non-finite net positions")
    if not cftc.empty and (cftc.available_date.isna().any() or cftc.available_date.duplicated().any()):
        raise ValueError("CFTC availability dates must be present and unique")
    features = build_features(market, cftc)
    modeling = features.dropna(subset=["target_return_1", "target_return_5"]).copy()
    report = {
        **features.attrs["data_quality"],
        "market_rows": len(market), "cftc_rows": len(cftc), "modeling_rows": len(modeling),
        "latest_market_date": str(market.date.max()),
        "latest_feature_date": str(features.date.max()),
        "incomplete_current_day_rows_excluded": incomplete_rows,
        "source_first_dates": {
            series: str(group.date.min().date()) for series, group in market.groupby("series")
        },
        "modeling_first_date": str(modeling.date.min().date()),
        "cftc_release_timestamp_verified": False,
        "cftc_model_policy": "excluded until actual per-report publication times are verified",
        "cftc_source_available": not cftc.empty,
    }
    (paths.processed / "data_quality.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    splits = chronological_split(modeling)
    if any(len(frame) < 61 for frame in splits.values()):
        raise ValueError("Insufficient complete data for 60-step train/validation/test sequences; see data_quality.json")
    latest_cotton_date = market.loc[market.series == "cotton", "date"].max()
    if features.date.max() != latest_cotton_date:
        raise ValueError("Latest Cotton row has invalid/incomplete features; refusing a stale live forecast")
    features.to_parquet(paths.processed / "training_dataset.parquet", index=False)
    split_manifest = {
        name: {"start": str(frame.date.min()), "end": str(frame.date.max()), "rows": len(frame)}
        for name, frame in splits.items()
    }
    (paths.processed / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(f"Sources validated: {len(modeling)} modeling rows; {report['invalid_market_value_count']} invalid input values recorded", flush=True)
    return market, features, splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    prepare(PipelinePaths(args.drive_root), args.refresh)


if __name__ == "__main__":
    main()
