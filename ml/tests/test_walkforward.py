from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.cohort import create_cohort
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.features import select_feature_rows
from cottonlens_ml.walkforward import (
    AUDIT_START,
    CORE,
    ablation_partitions,
    audit_split,
    build_folds,
    paired_block_bootstrap,
    summarize,
)


def _calendar() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2026-09-01")
    frame = pd.DataFrame({"date": dates})
    frame["target_date_5"] = frame.date.shift(-5)
    return frame.dropna().reset_index(drop=True)


def test_four_walkforward_folds_are_purged_and_pre_audit() -> None:
    folds = build_folds(_calendar())
    assert len(folds) == 4
    assert all(len(fold.test) == 126 for fold in folds)
    for fold in folds:
        assert fold.train.target_date_5.max() < fold.validation.date.min()
        assert fold.validation.target_date_5.max() < fold.test.date.min()
        assert fold.test.target_date_5.max() < AUDIT_START
    assert folds[0].test.date.max() < folds[1].test.date.min()


def test_historical_audit_cannot_supply_pre_audit_labels() -> None:
    split = audit_split(_calendar())
    assert split["train"].target_date_5.max() < split["validation"].date.min()
    assert split["validation"].target_date_5.max() < split["test"].date.min()
    assert split["test"].date.min() >= AUDIT_START


def _history():
    from test_cohort import history

    frame = history()
    frame.attrs["data_identity"] = {"data_id": "synthetic-data"}
    return frame


def test_frozen_folds_use_exact_origins_and_reject_missing_or_changed_labels():
    history = _history()
    cohort = create_cohort(history, "synthetic-data")
    modeling = select_feature_rows(history, FEATURE_NAMES)
    folds = build_folds(modeling, cohort)
    assert len(folds) == 4
    for fold, entry in zip(folds, cohort["folds"], strict=True):
        assert fold.test.date.dt.strftime("%Y-%m-%d").tolist() == entry["origins"]
        assert len(fold.validation) == 126
        assert fold.train.target_date_5.max() < fold.validation.date.min()
        assert fold.validation.target_date_5.max() < fold.test.date.min()
    missing = modeling.loc[modeling.date != folds[0].test.date.iloc[10]]
    with pytest.raises(ValueError, match="origins missing"):
        build_folds(missing, cohort)
    changed = modeling.copy()
    changed.loc[changed.date == folds[0].test.date.iloc[10], "cotton_close"] += 1
    with pytest.raises(ValueError, match="prices/labels changed"):
        build_folds(changed, cohort)


def test_ablation_recovers_natural_cotton_fit_rows_without_changing_test_origins():
    history = _history()
    history.loc[history.date.between("2020-01-01", "2020-02-01"), "dxy_ret_1"] = np.nan
    cohort = create_cohort(history, "synthetic-data")
    fold = build_folds(select_feature_rows(history, FEATURE_NAMES), cohort)[0]
    cotton, cotton_refit, coverage = ablation_partitions(fold, CORE, history)
    _, full_refit, _ = ablation_partitions(fold, FEATURE_NAMES, history)
    assert len(cotton_refit) > len(full_refit)
    assert cotton.test.date.tolist() == fold.test.date.tolist()
    assert cotton_refit.target_date_5.max() < cotton.test.date.min()
    assert cotton.train.target_date_5.max() < cotton.validation.date.min()
    assert coverage["common_evaluation_origins"] == 126
    assert coverage["scored_population"] == "frozen_common_origins_only"


def test_paired_bootstrap_preserves_folds_and_recomputes_ratio():
    # With constant errors per fold, preserving fold sample sizes gives an exact
    # estimate. Sampling arbitrary blocks across the concatenation would not.
    folds = [(np.ones(12), np.full(12, 2.0)), (np.full(8, 3.0), np.full(8, 6.0))]
    for size in (10, 20, 40):
        result = paired_block_bootstrap(folds, block_size=size, replicates=30)
        assert result["model_mae_ci_95"] == pytest.approx([1.8, 1.8])
        assert result["paired_gain_ci_95_cents_per_lb"] == pytest.approx([1.8, 1.8])
        assert result["relative_gain_ci_95_pct"] == pytest.approx([50, 50])
        assert result["fold_boundaries_preserved"]
        assert result == paired_block_bootstrap(folds, block_size=size, replicates=30)
    zero = paired_block_bootstrap([(np.zeros(3), np.zeros(3))], replicates=5)
    assert zero["relative_gain_ci_95_pct"] is None
    with pytest.raises(ValueError, match="paired"):
        paired_block_bootstrap([(np.ones(3), np.ones(2))])


def test_summary_keeps_aligned_origin_naive_error_and_fit_cutoff():
    dates = pd.bdate_range("2024-01-01", periods=4)
    frame = pd.DataFrame({"date": dates[:3], "target_date_1": dates[1:],
                          "cotton_close": [100.0, 100.0, 100.0],
                          "target_return_1": np.log([1.02, 1.04, .96])})
    candidate = SimpleNamespace(name="Synthetic", horizon=1, predictions=np.log([1.01, 1.02, .98]),
                                parameters={"fit_cutoff": "2023-12-29T00:00:00"})
    report = summarize([(frame, candidate)])
    assert report["sample_count"] == 3
    assert report["mae"] == pytest.approx(5 / 3)
    assert report["naive_mae"] == pytest.approx(10 / 3)
    assert report["relative_mae_improvement_pct"] == pytest.approx(50)
    assert len(report["origin_records"]) == 3
    for record in report["origin_records"]:
        assert record["fit_cutoff"] == "2023-12-29T00:00:00"
        assert record["naive_price"] == record["current_price"]
        assert record["paired_gain"] == pytest.approx(record["naive_absolute_error"] - record["absolute_error"])
    assert set(report["paired_bootstrap_sensitivity"]) == {"10", "20", "40"}


def test_walkforward_passes_same_history_to_all_model_families(tmp_path, monkeypatch):
    from cottonlens_ml import training, walkforward

    history = _history()
    cohort = create_cohort(history, "synthetic-data")
    received = []
    monkeypatch.setattr(walkforward, "require_colab_training", lambda: None)

    def candidates(name, test):
        return {h: SimpleNamespace(name=name, horizon=h, predictions=np.zeros(len(test)),
                                   metrics={"mae": 0.1}, parameters={},
                                   training_history=[{"epoch": 1, "loss": 0.8, "val_loss": 0.9}]) for h in (1, 5)}

    def fitted(name):
        def fake(train, validation, test, *args, feature_history):
            received.append((name, feature_history))
            result = candidates(name, test)
            return (None, None, result) if name == "LSTM" else result
        return fake

    monkeypatch.setattr(training, "naive_candidates", lambda test, validation: candidates("Naive", test))
    monkeypatch.setattr(training, "ridge_candidates", fitted("Ridge"))
    monkeypatch.setattr(training, "train_xgboost", fitted("XGBoost"))
    monkeypatch.setattr(training, "train_lstm", fitted("LSTM"))
    monkeypatch.setattr(walkforward, "summarize", lambda rows: {"sample_count": sum(len(frame) for frame, _ in rows)})
    report, locked = walkforward.run_walkforward(
        select_feature_rows(history, FEATURE_NAMES), history, tmp_path, cohort=cohort,
    )
    assert len(received) == 12
    assert all(value is history for _, value in received)
    assert set(locked) == {"Naive", "Ridge", "XGBoost", "LSTM"}
    assert report["feature_ablation"] == []
    assert all(fold["lstm_validation_training_history"][0]["val_loss"] == 0.9 for fold in report["folds"])
    assert all(value["sample_count"] == 504 for value in report["aggregate"].values())
