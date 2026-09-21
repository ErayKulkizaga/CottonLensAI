from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from cottonlens_ml.config import PipelinePaths
from cottonlens_ml.data import cache_sources
from cottonlens_ml.export import export_release
from cottonlens_ml.features import build_features, chronological_split
from cottonlens_ml.training import (
    naive_candidates,
    select_production,
    train_lstm,
    train_xgboost,
)


def run(drive_root: Path, refresh: bool) -> Path:
    paths = PipelinePaths(drive_root)
    paths.create()
    market, cftc = cache_sources(paths.raw, refresh=refresh)
    features = build_features(market, cftc)
    features.to_parquet(paths.processed / "training_dataset.parquet", index=False)
    modeling_features = features.dropna(subset=["target_return_1", "target_return_5"]).copy()
    quality_report = {
        "market_rows": len(market),
        "cftc_rows": len(cftc),
        "modeling_rows": len(modeling_features),
        "duplicate_market_keys": int(market.duplicated(["date", "series"]).sum()),
        "feature_nulls": {
            column: int(features[column].isna().sum()) for column in features.columns
        },
        "latest_market_date": str(market.date.max().date()),
        "latest_feature_date": str(features.date.max().date()),
    }
    (paths.processed / "data_quality.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )
    splits = chronological_split(modeling_features)
    split_manifest = {
        name: {"start": str(frame.date.min().date()), "end": str(frame.date.max().date()), "rows": len(frame)}
        for name, frame in splits.items()
    }
    (paths.processed / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )
    mlflow.set_tracking_uri(paths.mlruns.as_uri())
    mlflow.set_experiment("cottonlens-forecasting")
    with mlflow.start_run(run_name="locked-model-comparison"):
        naive = naive_candidates(splits["test"], splits["validation"])
        tree = train_xgboost(
            splits["train"], splits["validation"], splits["test"], paths.checkpoints
        )
        _, _, sequence = train_lstm(
            splits["train"], splits["validation"], splits["test"], paths.checkpoints
        )
        selected, selection_audit = select_production(naive, tree, sequence)
        candidates = [*naive.values(), *tree.values(), *sequence.values()]
        for candidate in candidates:
            mlflow.log_metrics(
                {f"{candidate.name.lower()}_t{candidate.horizon}_{key}": value for key, value in candidate.metrics.items()}
            )
        return export_release(
            paths.releases,
            features,
            splits["test"],
            market,
            candidates,
            selected,
            selection_audit,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CottonLens training in Google Colab")
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/CottonLensAI"),
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    output = run(args.drive_root, args.refresh)
    print(f"Validated artifact bundle: {output}")


if __name__ == "__main__":
    main()
