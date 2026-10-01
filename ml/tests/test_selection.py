from cottonlens_ml.selection import select_model_name, select_walkforward_name


def metrics(mae: float, directional_accuracy: float) -> dict[str, float]:
    return {"mae": mae, "directional_accuracy": directional_accuracy}


def test_xgboost_requires_both_split_gates() -> None:
    selected, audit = select_model_name(
        metrics(10, 50),
        metrics(10, 50),
        metrics(9.4, 54),
        metrics(9.4, 54),
        metrics(9.3, 54),
        metrics(9.2, 54),
        horizon=1,
    )
    assert selected == "XGBoost"
    assert audit["xgboost"]["qualified"] is True
    assert audit["lstm"]["beats_xgboost"] is False


def test_lstm_must_clear_its_own_gate_and_beat_xgboost_by_five_percent() -> None:
    selected, audit = select_model_name(
        metrics(10, 50),
        metrics(10, 50),
        metrics(9.3, 56),
        metrics(9.3, 56),
        metrics(9.0, 56),
        metrics(8.8, 56),
        horizon=5,
    )
    assert selected == "LSTM"
    assert audit["lstm"]["qualified"] is True
    assert audit["lstm"]["beats_xgboost"] is True


def test_naive_remains_primary_when_directional_gate_fails() -> None:
    selected, audit = select_model_name(
        metrics(10, 50),
        metrics(10, 50),
        metrics(8.5, 52.9),
        metrics(8.5, 52.9),
        metrics(8.0, 52.0),
        metrics(8.0, 52.0),
        horizon=1,
    )
    assert selected == "Naive"
    assert audit["xgboost"]["qualified"] is False
    assert audit["lstm"]["qualified"] is False


def test_t5_uses_higher_directional_accuracy_threshold() -> None:
    selected, audit = select_model_name(
        metrics(10, 50),
        metrics(10, 50),
        metrics(9.0, 54.0),
        metrics(9.0, 54.0),
        metrics(8.5, 54.0),
        metrics(8.5, 54.0),
        horizon=5,
    )
    assert selected == "Naive"
    assert audit["xgboost"]["test"]["min_directional_accuracy"] == 55.0


def _walkforward_report(wins: int = 3) -> dict:
    folds = []
    for index in range(4):
        folds.append({"metrics": {
            "Naive-T+1": metrics(10, 0),
            "XGBoost-T+1": metrics(9 if index < wins else 11, 56),
            "LSTM-T+1": metrics(8 if index < wins else 12, 56),
        }})
    return {"aggregate": {
        "Naive-T+1": metrics(10, 0),
        "XGBoost-T+1": metrics(9.2, 56),
        "LSTM-T+1": metrics(8.6, 56),
    }, "folds": folds}


def test_walkforward_selects_lstm_before_historical_audit() -> None:
    selected, audit = select_walkforward_name(
        _walkforward_report(), 1,
        {"Naive": metrics(10, 0), "XGBoost": metrics(9.2, 56), "LSTM": metrics(8.8, 56)},
    )
    assert selected == "LSTM"
    assert audit["locked_candidate"] == "LSTM"
    assert audit["fold_wins_vs_naive"]["LSTM"] == 3


def test_seen_historical_audit_cannot_reject_or_replace_locked_model() -> None:
    selected, audit = select_walkforward_name(
        _walkforward_report(), 1,
        {"Naive": metrics(10, 0), "XGBoost": metrics(7, 70), "LSTM": metrics(11, 56)},
    )
    assert audit["locked_candidate"] == "LSTM"
    assert selected == "LSTM"  # Seen history has no selection or veto authority.


def test_period_consistency_rejects_two_of_four_wins() -> None:
    selected, audit = select_walkforward_name(
        _walkforward_report(wins=2), 1,
        {"Naive": metrics(10, 0), "XGBoost": metrics(9, 56), "LSTM": metrics(8, 56)},
    )
    assert selected == "Naive"
    assert audit["fold_wins_vs_naive"]["XGBoost"] == 2


def test_lstm_cannot_bypass_extra_gate_when_xgboost_direction_fails():
    report = _walkforward_report()
    report["aggregate"]["XGBoost-T+1"] = metrics(9.0, 52.0)
    report["aggregate"]["LSTM-T+1"] = metrics(9.3, 54.0)
    selected, audit = select_walkforward_name(report, 1, {})
    assert selected == "Naive"
    assert not audit["lstm_beats_xgboost"]


def test_legacy_selector_also_enforces_extra_lstm_gate():
    selected, _ = select_model_name(
        metrics(10, 0), metrics(10, 0), metrics(9, 52), metrics(9, 52),
        metrics(9.3, 54), metrics(9.3, 54), horizon=1,
    )
    assert selected == "Naive"


def test_seen_audit_values_never_change_selection():
    report = _walkforward_report()
    decisions = [select_walkforward_name(report, 1, audit)[0] for audit in (
        {}, {"Naive": metrics(1, 0), "LSTM": metrics(100, 0)},
        {"Naive": metrics(100, 0), "XGBoost": metrics(.01, 100)},
    )]
    assert decisions == ["LSTM"] * 3


def test_horizons_are_selected_independently():
    report = _walkforward_report()
    for row in [report["aggregate"], *(fold["metrics"] for fold in report["folds"])]:
        for model in ("Naive", "XGBoost", "LSTM"):
            row[f"{model}-T+5"] = metrics(10 if model == "Naive" else 9, 54)
    assert select_walkforward_name(report, 1, {})[0] == "LSTM"
    assert select_walkforward_name(report, 5, {})[0] == "Naive"
