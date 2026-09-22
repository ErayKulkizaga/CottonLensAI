from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from cottonlens_ml.config import PipelinePaths
from cottonlens_ml.export import export_release
from cottonlens_ml.prepare import prepare
from cottonlens_ml.tracking import tracked_run, tracking_session
from cottonlens_ml.training import (
    naive_candidates,
    select_production,
    train_lstm,
    train_xgboost,
)


def run(drive_root: Path, refresh: bool) -> Path:
    paths = PipelinePaths(drive_root)
    market, features, splits = prepare(paths, refresh)
    with tracking_session(paths.root), tracked_run(run_name="locked-model-comparison"):
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
        output = export_release(
            paths.releases,
            features,
            splits["test"],
            market,
            candidates,
            selected,
            selection_audit,
        )
        (paths.releases / "latest.txt").write_text(output.name, encoding="utf-8")
        return output


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
    print(f"Exported bundle (run validate_release before use): {output}")


if __name__ == "__main__":
    main()
