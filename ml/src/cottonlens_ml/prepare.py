"""Validate cached/downloaded sources and produce features without training imports."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cottonlens_ml.config import DATA_POLICY_VERSION, FEATURE_NAMES, PipelinePaths
from cottonlens_ml.data import cache_sources
from cottonlens_ml.features import (
    build_feature_history,
    coverage_report,
    require_latest_live_row,
    select_feature_rows,
)
from cottonlens_ml.snapshots import write_snapshot
from cottonlens_ml.walkforward import audit_split


def prepare(paths: PipelinePaths, refresh: bool = False):
    paths.create()
    previous_dataset = paths.processed / "training_dataset.parquet"
    if previous_dataset.exists():
        evidence = {}
        for name in ("data_quality", "split_manifest"):
            path = paths.processed / f"{name}.json"
            if path.exists():
                evidence[name] = json.loads(path.read_text(encoding="utf-8"))
        write_snapshot(
            paths.processed / "snapshots", {"previous_training_dataset": pd.read_parquet(previous_dataset)},
            {"kind": "pre_prepare_processed_cache", **evidence},
        )
    market, cftc = cache_sources(paths.raw, refresh=refresh)
    raw_snapshot = write_snapshot(
        paths.raw / "snapshots", {"market": market, "cftc": cftc}, {"kind": "source_cache"}
    )
    # Yahoo can return an in-progress current-day candle. Delay at most one
    # session rather than treating an incomplete close as an observed target.
    utc_today = datetime.now(UTC).date()
    incomplete_rows = int((market.date.dt.date >= utc_today).sum())
    market = market.loc[market.date.dt.date < utc_today].copy()
    if not cftc.empty and not np.isfinite(cftc["cftc_managed_money_net"]).all():
        raise ValueError("CFTC source contains non-finite net positions")
    if not cftc.empty and (cftc.available_date.isna().any() or cftc.available_date.duplicated().any()):
        raise ValueError("CFTC availability dates must be present and unique")
    features = build_feature_history(market, cftc)
    modeling = select_feature_rows(features, FEATURE_NAMES)
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
        "feature_coverage": coverage_report(features),
    }
    features.attrs["data_quality"] = report
    snapshot = write_snapshot(
        paths.processed / "snapshots", {"feature_history": features},
        {"raw_id": raw_snapshot["snapshot_id"], "data_policy": DATA_POLICY_VERSION,
         "feature_schema": FEATURE_NAMES, "data_quality": report},
    )
    features.attrs["data_identity"] = {
        "data_id": snapshot["snapshot_id"], "raw_id": raw_snapshot["snapshot_id"],
        "manifest_path": snapshot["manifest_path"], "raw_manifest_path": raw_snapshot["manifest_path"],
        "checksums": snapshot["checksums"], "source_checksums": raw_snapshot["checksums"],
    }
    (paths.processed / "data_identity.json").write_text(
        json.dumps(features.attrs["data_identity"], indent=2, allow_nan=False), encoding="utf-8"
    )
    (paths.processed / "data_quality.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    splits = audit_split(modeling)
    if any(len(frame) < 61 for frame in splits.values()):
        raise ValueError("Insufficient data for 60-step train/validation/audit sequences; see data_quality.json")
    require_latest_live_row(features)
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
