"""Validate cached/downloaded sources and produce features without training imports."""

import argparse
import json
from pathlib import Path

import numpy as np

from cottonlens_ml.config import PipelinePaths
from cottonlens_ml.data import cache_sources
from cottonlens_ml.features import build_features, chronological_split


def prepare(paths: PipelinePaths, refresh: bool = False):
    paths.create()
    market, cftc = cache_sources(paths.raw, refresh=refresh)
    if cftc.empty or not np.isfinite(cftc["cftc_managed_money_net"]).all():
        raise ValueError("CFTC source is empty or contains non-finite net positions")
    if cftc.available_date.isna().any() or cftc.available_date.duplicated().any():
        raise ValueError("CFTC availability dates must be present and unique")
    features = build_features(market, cftc)
    modeling = features.dropna(subset=["target_return_1", "target_return_5"]).copy()
    report = {
        **features.attrs["data_quality"],
        "market_rows": len(market), "cftc_rows": len(cftc), "modeling_rows": len(modeling),
        "latest_market_date": str(market.date.max()),
        "latest_feature_date": str(features.date.max()),
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
