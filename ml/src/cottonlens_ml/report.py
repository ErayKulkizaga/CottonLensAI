"""Create a copyable, evidence-backed Colab study report without moving the model ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def _number(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _metric_line(name: str, metrics: dict, naive_mae: float) -> str:
    gain = (1 - metrics["mae"] / naive_mae) * 100
    interval = metrics.get("mae_ci_95")
    ci = f" [{_number(interval[0])}, {_number(interval[1])}]" if interval else ""
    balanced = (
        f"{_number(metrics['balanced_accuracy'], 2)}%"
        if "balanced_accuracy" in metrics else "not measured"
    )
    majority = (
        f"{_number(metrics['majority_direction_accuracy'], 2)}%"
        if "majority_direction_accuracy" in metrics else "not measured"
    )
    return (
        f"  {name}: n={metrics.get('sample_count', 'not recorded')}, "
        f"MAE={_number(metrics['mae'])} cents/lb{ci}, "
        f"vs Naive={gain:+.2f}%, RMSE={_number(metrics['rmse'])}, "
        f"MAPE={_number(metrics['mape'])}%, direction={_number(metrics['directional_accuracy'], 2)}%, "
        f"balanced={balanced}, majority={majority}"
    )


def _trial_lines(name: str, parameters: dict | None, prefix: str = "  ") -> list[str]:
    if not parameters:
        return []
    trials = parameters.get("trials", [])
    if name.startswith("XGBoost"):
        return [
            f"{prefix}depth={trial['max_depth']} lr={trial['learning_rate']} "
            f"subsample={trial['subsample']} trees={trial['best_iteration']} "
            f"validation_MAE={_number(trial['validation_mae'])} "
            f"resumed={trial['resumed_from_drive']}"
            for trial in trials
        ]
    if name.startswith("LSTM"):
        return [
            f"{prefix}units={trial['units']} dropout={trial['dropout']} "
            f"epochs={trial['epochs_run']} best_epoch={trial['best_epoch']} "
            f"validation_scaled_loss={_number(trial['validation_loss'], 5)} "
            f"resumed={trial['resumed_from_drive']}"
            for trial in trials
        ]
    return [f"{prefix}parameters={json.dumps(parameters, sort_keys=True)}"]


def build_report(evidence: dict) -> str:
    """Render all model-selection evidence; raise on absent core results."""
    manifest = evidence["manifest"]
    quality = evidence["data_quality"]
    smoke = evidence["environment_smoke"]
    metrics = evidence["metrics"]
    walk = manifest["walkforward_report"]
    audit = manifest["selection_audit"]
    if len(walk["folds"]) != 4 or not evidence["outer_checksum_verified"]:
        raise ValueError("Expected four folds and a verified release checksum")
    if smoke["status"] != "passed":
        raise ValueError("Environment smoke did not pass")

    lines = [
        "BEGIN COTTONLENS RESULTS — send this whole text in the chat",
        f"Artifact: {manifest['artifact_version']} | Git: {manifest['git_sha']}",
        f"Generated: {manifest['generated_at']} | ZIP SHA-256: {evidence['zip_sha256']}",
        "Selection outcome: " + ", ".join(
            f"T+{horizon}={audit[str(horizon)]['selected']} "
            f"({'learned-model gate passed' if audit[str(horizon)]['selected'] != 'Naive' else 'learned-model gate not met'})"
            for horizon in (1, 5)
        ),
        "Outer ZIP checksum: PASS. Backend runtime validation: see successful preceding Colab cell.",
        "Evidence type: historical research; 2024+ audit has been viewed before and is NOT an independent holdout.",
        "No financial-return or trading-performance claim is implied by price/direction metrics.",
        "",
        "ENVIRONMENT AND REPRODUCIBILITY",
        f"Python: {smoke['environment']['python'].split()[0]} | GPU: {smoke['environment']['gpus']}",
        f"Smoke checks: {json.dumps(smoke.get('checks', {}), sort_keys=True)}",
        f"Pinned packages: {json.dumps(smoke['environment']['versions'], sort_keys=True)}",
        f"Lock SHA-256: {smoke['environment']['lock_sha256']}",
        "",
        "DATA AND AVAILABILITY",
        f"Dataset: {manifest['dataset_range']} | modeling rows: {quality['modeling_rows']}",
        f"First source dates: {json.dumps(quality['source_first_dates'], sort_keys=True)}",
        f"Modeling first date: {quality['modeling_first_date']} | latest market: {quality['latest_market_date']}",
        f"Market rows: {quality['market_rows']} | CFTC rows: {quality['cftc_rows']}",
        (
            f"Invalid market values: {quality['invalid_market_value_count']} | "
            f"dropped feature rows: {quality['dropped_incomplete_feature_rows']} | "
            f"incomplete current-day rows: {quality['incomplete_current_day_rows_excluded']}"
        ),
        f"CFTC model policy: {quality['cftc_model_policy']}",
        f"Source note: {manifest['source_note']}",
    ]
    for issue in quality.get("invalid_market_values", [])[:20]:
        lines.append(f"  invalid input: {json.dumps(issue, sort_keys=True)}")
    if quality["invalid_market_value_count"] > 20:
        lines.append("  Additional invalid inputs are in data/processed/data_quality.json on Drive.")
    lines.extend([
        "",
        "PROTOCOL",
        f"Split ranges: {json.dumps(manifest['split_ranges'], sort_keys=True)}",
        f"Selection policy: {json.dumps(manifest['selection_policy'], sort_keys=True)}",
        f"Feature schema hash: {manifest['feature_schema_hash']}",
        f"Feature groups: {json.dumps(walk['feature_groups'], sort_keys=True)}",
        f"CFTC candidate: {walk['cftc_candidate']}",
        "All horizons use matching forecast dates within a fold; T+5 labels are purged at boundaries.",
        "",
        "FOUR PRE-AUDIT WALK-FORWARD FOLDS (selection evidence)",
    ])
    for fold in walk["folds"]:
        lines.append(
            f"Fold {fold['fold']}: train end={fold['train_end']}, validation end={fold['validation_end']}, "
            f"test={fold['test_start']}..{fold['test_end']}, n={fold['sample_count']}"
        )
        for horizon in (1, 5):
            naive = fold["metrics"][f"Naive-T+{horizon}"]["mae"]
            for name in ("Naive", "Ridge", "XGBoost", "LSTM"):
                key = f"{name}-T+{horizon}"
                lines.append(_metric_line(key, fold["metrics"][key], naive))
    lines.extend(["", "AGGREGATE WALK-FORWARD METRICS (20-session block-bootstrap MAE interval)"])
    for horizon in (1, 5):
        naive = walk["aggregate"][f"Naive-T+{horizon}"]["mae"]
        for name in ("Naive", "Ridge", "XGBoost", "LSTM"):
            key = f"{name}-T+{horizon}"
            lines.append(_metric_line(key, walk["aggregate"][key], naive))
    lines.extend(["", "FEATURE ABLATION — fold MAE in cents/lb"])
    for row in walk["feature_ablation"]:
        lines.append(
            f"Fold {row['fold']} T+{row['horizon']}: cotton={_number(row['cotton_mae'])}, "
            f"cotton+macro={_number(row['cotton_macro_mae'])}, full={_number(row['full_mae'])}"
        )
    lines.extend(["", "LOCKED CANDIDATE, HISTORICAL AUDIT, AND FINAL DECISION"])
    for horizon in (1, 5):
        decision = audit[str(horizon)]
        lines.append(
            f"T+{horizon}: locked={decision['locked_candidate']}, "
            f"production={decision['selected']}, fold wins={decision['fold_wins_vs_naive']}, "
            f"XGB gate={decision['xgboost_walkforward_gate']}, "
            f"LSTM gate={decision['lstm_walkforward_gate']}, "
            f"LSTM beats XGB={decision['lstm_beats_xgboost']}"
        )
        naive = decision["historical_audit"]["Naive"]["mae"]
        for name in ("Naive", "XGBoost", "LSTM"):
            lines.append(_metric_line(f"audit {name}-T+{horizon}", decision["historical_audit"][name], naive))
    lines.extend([
        "The historical audit can reject a locked learned candidate, never select a new one.",
        "If a threshold fails, do not tune against this already viewed audit period.",
        "",
        "FINAL-FIT MODEL SETTINGS AND ALL VALIDATION TRIALS",
    ])
    for row in metrics:
        name = f"{row['model']}-T+{row['horizon']}"
        lines.append(
            f"{name}: selected={row['selected']}, "
            f"settings={json.dumps({k: v for k, v in (row.get('parameters') or {}).items() if k != 'trials'}, sort_keys=True)}, "
            f"historical_audit_MAE={_number(row['mae'])}, "
            f"validation_MAE={_number(row['validation_metrics']['mae']) if row.get('validation_metrics') else 'n/a'}"
        )
        lines.extend(_trial_lines(name, row.get("parameters"), "    "))
    lines.extend(["", "PER-FOLD VALIDATION TRIALS"])
    for fold in walk["folds"]:
        for name, parameters in fold.get("experiments", {}).items():
            if name.startswith(("XGBoost", "LSTM")):
                lines.append(f"Fold {fold['fold']} {name}:")
                lines.extend(_trial_lines(name, parameters, "    "))
    lines.extend(["", "BEST LSTM CONFIGURATION TRAINING CURVE (scaled-target loss; not price MAE)"])
    lstm = next(row for row in metrics if row["model"] == "LSTM" and row["horizon"] == 1)
    curve = lstm.get("training_history") or []
    lines.append(f"Epochs recorded: {len(curve)}")
    lines.extend(
        f"  epoch={point['epoch']} train_loss={_number(point['loss'], 5)} "
        f"val_loss={_number(point['val_loss'], 5)}"
        for point in curve
    )
    lines.extend(["", "EXPORTED RUNTIME MODELS"])
    for model in manifest["production_models"]:
        lines.append(json.dumps(model, sort_keys=True))
    lines.extend([
        "",
        "INTERPRETATION",
        "Naive production means the learned models did not satisfy the predeclared evidence gates.",
        "A positive walk-forward result is historical evidence, not a guarantee of future accuracy.",
        "The ZIP remains in Drive; send only this report now. Transfer ZIP + .sha256 once integration is approved.",
        "END COTTONLENS RESULTS",
    ])
    return "\n".join(lines) + "\n"


def collect_evidence(drive_root: Path) -> dict:
    release_root = drive_root / "artifacts" / "releases"
    release_name = (release_root / "latest.txt").read_text(encoding="utf-8").strip()
    if Path(release_name).name != release_name or not release_name.endswith(".zip"):
        raise ValueError("Invalid release name in latest.txt")
    archive_path = release_root / release_name
    expected_checksum = (release_root / f"{release_name}.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    with archive_path.open("rb") as release_file:
        actual_checksum = hashlib.file_digest(release_file, "sha256").hexdigest()
    if actual_checksum != expected_checksum:
        raise ValueError("Release ZIP checksum mismatch")
    prefix = archive_path.stem
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read(f"{prefix}/manifest.json"))
        metrics = json.loads(archive.read(f"{prefix}/metrics.json"))
    if "walkforward_report" not in manifest:
        raise ValueError("Release predates the walk-forward report; run the updated Colab notebook")
    return {
        "manifest": manifest,
        "metrics": metrics,
        "data_quality": json.loads((drive_root / "data/processed/data_quality.json").read_text(encoding="utf-8")),
        "environment_smoke": json.loads((drive_root / "reports/environment-smoke.json").read_text(encoding="utf-8")),
        "zip_sha256": actual_checksum,
        "outer_checksum_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True, type=Path)
    args = parser.parse_args()
    evidence = collect_evidence(args.drive_root)
    content = build_report(evidence)
    output = args.drive_root / "reports" / f"cottonlens-results-{evidence['manifest']['artifact_version']}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    evidence_path = output.with_suffix(".json")
    evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    print(content, flush=True)
    print(f"Saved shareable report: {output}", flush=True)
    print(f"Saved complete machine-readable evidence: {evidence_path}", flush=True)


if __name__ == "__main__":
    main()
