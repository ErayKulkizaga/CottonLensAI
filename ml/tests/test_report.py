import hashlib
import json
import zipfile

import pytest
from cottonlens_ml.report import build_report, collect_evidence


def _scores(mae):
    return {
        "mae": mae, "rmse": mae + 0.3, "mape": 1.1,
        "directional_accuracy": 55, "balanced_accuracy": 54,
        "majority_direction_accuracy": 51, "sample_count": 126,
        "mae_ci_95": [mae - 0.1, mae + 0.1],
    }


def _evidence():
    scores = {
        f"{name}-T+{horizon}": _scores(mae)
        for horizon in (1, 5)
        for name, mae in (("Naive", 1.0), ("Ridge", 0.99), ("XGBoost", 0.9), ("LSTM", 0.88))
    }
    folds = [
        {
            "fold": index, "train_end": "2020-01-01", "validation_end": "2021-01-01",
            "test_start": "2022-01-01", "test_end": "2022-06-01", "sample_count": 126,
            "metrics": scores,
            "experiments": {
                "XGBoost-T+1": {"trials": [{
                    "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
                    "best_iteration": 51, "validation_mae": 0.9, "resumed_from_drive": False,
                }]},
            },
        }
        for index in range(1, 5)
    ]
    decision = {
        str(horizon): {
            "locked_candidate": "XGBoost", "selected": "XGBoost",
            "fold_wins_vs_naive": {"XGBoost": 4, "LSTM": 4},
            "xgboost_walkforward_gate": {"passes": True},
            "lstm_walkforward_gate": {"passes": True},
            "lstm_beats_xgboost": False,
            "historical_audit": {"Naive": _scores(1.0), "XGBoost": _scores(0.9), "LSTM": _scores(0.88)},
        }
        for horizon in (1, 5)
    }
    metrics = [
        {
            "model": name, "horizon": horizon, "selected": name == "XGBoost",
            "mae": mae, "validation_metrics": _scores(mae),
            "parameters": {"units": 32, "trials": []} if name == "LSTM" else None,
            "training_history": [{"epoch": 1, "loss": 0.9, "val_loss": 1.1}] if name == "LSTM" else None,
        }
        for horizon in (1, 5)
        for name, mae in (("Naive", 1.0), ("Ridge", 0.99), ("XGBoost", 0.9), ("LSTM", 0.88))
    ]
    return {
        "manifest": {
            "artifact_version": "v20260923-1200", "git_sha": "abc123",
            "generated_at": "2026-09-23T12:00:00+00:00",
            "dataset_range": {"start": "2016-01-01", "end": "2026-09-22"},
            "split_ranges": {"train": {}, "validation": {}, "historical_audit": {}},
            "selection_policy": {"minimum_mae_improvement_pct_vs_naive": 5.0},
            "feature_schema_hash": "schema123", "source_note": "research proxy",
            "walkforward_report": {
                "folds": folds, "aggregate": scores,
                "feature_ablation": [{
                    "fold": 1, "horizon": 1, "cotton_mae": 1.0,
                    "cotton_macro_mae": 0.9, "full_mae": 0.8,
                }],
                "feature_groups": {"cotton": ["cotton_ret_1"]},
                "cftc_candidate": "excluded",
            },
            "selection_audit": decision,
            "production_models": [{"name": "XGBoost", "format": "xgboost_json", "horizon": 1}],
        },
        "metrics": metrics,
        "data_quality": {
            "modeling_rows": 2000, "source_first_dates": {"cotton": "2016-01-01"},
            "modeling_first_date": "2016-04-01", "latest_market_date": "2026-09-22",
            "market_rows": 6000, "cftc_rows": 0, "invalid_market_value_count": 0,
            "dropped_incomplete_feature_rows": 60, "incomplete_current_day_rows_excluded": 0,
            "cftc_model_policy": "excluded", "invalid_market_values": [],
        },
        "environment_smoke": {
            "status": "passed", "checks": {"onnx": "passed"},
            "environment": {
                "python": "3.12.11 (main)", "gpus": ["GPU:0"],
                "versions": {"tensorflow": "2.20.0"}, "lock_sha256": "lock123",
            },
        },
        "zip_sha256": "digest123", "outer_checksum_verified": True,
    }


def test_shareable_report_contains_selection_data_trials_and_curve():
    report = build_report(_evidence())
    assert "BEGIN COTTONLENS RESULTS" in report
    assert "Fold 4" in report
    assert "vs Naive=+10.00%" in report
    assert "depth=3 lr=0.03" in report
    assert "epoch=1 train_loss=0.90000" in report
    assert "NOT an independent holdout" in report
    assert "Transfer ZIP + .sha256 once" in report


def test_report_refuses_incomplete_or_unverified_evidence():
    evidence = _evidence()
    evidence["manifest"]["walkforward_report"]["folds"].pop()
    with pytest.raises(ValueError, match="four folds"):
        build_report(evidence)
    evidence = _evidence()
    evidence["outer_checksum_verified"] = False
    with pytest.raises(ValueError, match="verified release checksum"):
        build_report(evidence)


def test_collect_evidence_rejects_tampered_zip(tmp_path):
    release_root = tmp_path / "artifacts" / "releases"
    release_root.mkdir(parents=True)
    name = "cottonlens-model-v20260923-1200.zip"
    (release_root / "latest.txt").write_text(name, encoding="utf-8")
    with zipfile.ZipFile(release_root / name, "w") as archive:
        archive.writestr("cottonlens-model-v20260923-1200/manifest.json", json.dumps({}))
    (release_root / f"{name}.sha256").write_text("0" * 64 + "  " + name, encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        collect_evidence(tmp_path)


def test_collect_evidence_reads_matching_release_and_reports(tmp_path):
    sample = _evidence()
    release_root = tmp_path / "artifacts" / "releases"
    release_root.mkdir(parents=True)
    name = "cottonlens-model-v20260923-1200.zip"
    prefix = name.removesuffix(".zip")
    (release_root / "latest.txt").write_text(name, encoding="utf-8")
    with zipfile.ZipFile(release_root / name, "w") as archive:
        archive.writestr(f"{prefix}/manifest.json", json.dumps(sample["manifest"]))
        archive.writestr(f"{prefix}/metrics.json", json.dumps(sample["metrics"]))
    digest = hashlib.sha256((release_root / name).read_bytes()).hexdigest()
    (release_root / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8")
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "data_quality.json").write_text(json.dumps(sample["data_quality"]), encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "environment-smoke.json").write_text(
        json.dumps(sample["environment_smoke"]), encoding="utf-8"
    )
    collected = collect_evidence(tmp_path)
    assert collected["zip_sha256"] == digest
    assert "Fold 4" in build_report(collected)
