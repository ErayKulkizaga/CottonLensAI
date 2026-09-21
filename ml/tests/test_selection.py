from cottonlens_ml.selection import select_model_name


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
