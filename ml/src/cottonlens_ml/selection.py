from __future__ import annotations

from collections.abc import Mapping

MIN_MAE_IMPROVEMENT_PCT = 5.0
MIN_DIRECTIONAL_ACCURACY = {1: 53.0, 5: 55.0}
LSTM_VS_XGBOOST_MAE_IMPROVEMENT_PCT = 5.0

Metrics = Mapping[str, float]


def mae_improvement_pct(baseline: Metrics, candidate: Metrics) -> float:
    baseline_mae = baseline["mae"]
    if baseline_mae <= 0:
        return 0.0
    return (baseline_mae - candidate["mae"]) / baseline_mae * 100


def quality_gate(baseline: Metrics, candidate: Metrics, horizon: int) -> dict[str, float | bool]:
    improvement = mae_improvement_pct(baseline, candidate)
    directional_accuracy = candidate["directional_accuracy"]
    return {
        "mae_improvement_pct": improvement,
        "directional_accuracy": directional_accuracy,
        "min_directional_accuracy": MIN_DIRECTIONAL_ACCURACY[horizon],
        "passes_mae": improvement >= MIN_MAE_IMPROVEMENT_PCT,
        "passes_direction": directional_accuracy >= MIN_DIRECTIONAL_ACCURACY[horizon],
        "passes": (
            improvement >= MIN_MAE_IMPROVEMENT_PCT
            and directional_accuracy >= MIN_DIRECTIONAL_ACCURACY[horizon]
        ),
    }


def select_model_name(
    naive_validation: Metrics,
    naive_test: Metrics,
    xgboost_validation: Metrics,
    xgboost_test: Metrics,
    lstm_validation: Metrics,
    lstm_test: Metrics,
    horizon: int,
) -> tuple[str, dict]:
    tree_validation_gate = quality_gate(naive_validation, xgboost_validation, horizon)
    tree_test_gate = quality_gate(naive_test, xgboost_test, horizon)
    lstm_validation_gate = quality_gate(naive_validation, lstm_validation, horizon)
    lstm_test_gate = quality_gate(naive_test, lstm_test, horizon)
    tree_qualified = bool(tree_validation_gate["passes"] and tree_test_gate["passes"])
    lstm_qualified = bool(lstm_validation_gate["passes"] and lstm_test_gate["passes"])
    lstm_beats_tree = (
        mae_improvement_pct(xgboost_test, lstm_test) >= LSTM_VS_XGBOOST_MAE_IMPROVEMENT_PCT
        and lstm_test["directional_accuracy"] >= xgboost_test["directional_accuracy"]
    )

    if lstm_qualified and (not tree_qualified or lstm_beats_tree):
        selected = "LSTM"
    elif tree_qualified:
        selected = "XGBoost"
    else:
        selected = "Naive"

    return selected, {
        "horizon": horizon,
        "thresholds": {
            "min_mae_improvement_pct": MIN_MAE_IMPROVEMENT_PCT,
            "min_directional_accuracy": MIN_DIRECTIONAL_ACCURACY[horizon],
            "lstm_vs_xgboost_min_mae_improvement_pct": LSTM_VS_XGBOOST_MAE_IMPROVEMENT_PCT,
        },
        "xgboost": {
            "validation": tree_validation_gate,
            "test": tree_test_gate,
            "qualified": tree_qualified,
        },
        "lstm": {
            "validation": lstm_validation_gate,
            "test": lstm_test_gate,
            "qualified": lstm_qualified,
            "beats_xgboost": lstm_beats_tree,
        },
        "selected": selected,
    }


def select_walkforward_name(
    report: dict,
    horizon: int,
    historical_audit: Mapping[str, Metrics],
) -> tuple[str, dict]:
    aggregate = report["aggregate"]
    naive = aggregate[f"Naive-T+{horizon}"]
    tree = aggregate[f"XGBoost-T+{horizon}"]
    sequence = aggregate[f"LSTM-T+{horizon}"]
    fold_wins = {
        name: sum(
            fold["metrics"][f"{name}-T+{horizon}"]["mae"]
            < fold["metrics"][f"Naive-T+{horizon}"]["mae"]
            for fold in report["folds"]
        )
        for name in ("XGBoost", "LSTM")
    }
    tree_gate = quality_gate(naive, tree, horizon)
    sequence_gate = quality_gate(naive, sequence, horizon)
    tree_ok = bool(tree_gate["passes"] and fold_wins["XGBoost"] >= 3)
    sequence_ok = bool(sequence_gate["passes"] and fold_wins["LSTM"] >= 3)
    lstm_beats_tree = (
        mae_improvement_pct(tree, sequence) >= LSTM_VS_XGBOOST_MAE_IMPROVEMENT_PCT
        and sequence["directional_accuracy"] >= tree["directional_accuracy"]
    )
    if sequence_ok and (not tree_ok or lstm_beats_tree):
        locked = "LSTM"
    elif tree_ok:
        locked = "XGBoost"
    else:
        locked = "Naive"
    selected = locked
    if locked != "Naive" and not quality_gate(
        historical_audit["Naive"], historical_audit[locked], horizon
    )["passes"]:
        selected = "Naive"
    return selected, {
        "locked_from": "four_pre_audit_walkforward_folds",
        "locked_candidate": locked,
        "selected": selected,
        "audit_role": "historical_rejection_only_not_independent_test",
        "fold_wins_vs_naive": fold_wins,
        "xgboost_walkforward_gate": tree_gate,
        "lstm_walkforward_gate": sequence_gate,
        "lstm_beats_xgboost": lstm_beats_tree,
        "historical_audit": dict(historical_audit),
    }
