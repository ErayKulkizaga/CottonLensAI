import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.evaluation import evaluate, price_mae_metric
from cottonlens_ml.protocol import experiment_identity, fingerprint, refit_rows
from cottonlens_ml.sequences import sequence_diagnostics, sequences
from cottonlens_ml.walkforward import purged_validation_split


def _frame(n=400):
    frame = pd.DataFrame({name: np.arange(n, dtype=float) for name in FEATURE_NAMES})
    frame["date"] = pd.bdate_range("2020-01-01", periods=n)
    frame["cotton_close"] = 80.0
    frame["target_return_1"] = 0.01
    frame["target_return_5"] = 0.03
    frame["target_date_5"] = frame.date.shift(-5)
    return frame


def test_recent_validation_and_refit_never_use_evaluation_labels():
    frame = _frame()
    train, validation = purged_validation_split(frame.iloc[:300])
    test = frame.iloc[310:]
    assert len(validation) == 126
    refit = refit_rows(train, validation, test, history=frame)
    assert refit.date.max() == validation.date.max()
    assert refit.target_date_5.max() < test.date.min()
    assert len(refit) == 300  # restores the now-safe inner purge rows
    with pytest.raises(ValueError, match="evaluation boundary"):
        refit_rows(train, validation, frame.iloc[301:])


def test_refit_rejects_overlapping_tuning_partitions():
    frame = _frame()
    with pytest.raises(ValueError, match="tuning validation"):
        refit_rows(frame.iloc[:200], frame.iloc[200:300], frame.iloc[310:])


def test_checkpoint_identity_changes_with_price_and_recipe():
    frame = _frame()
    original = fingerprint(frame, settings={"seed": 42})
    assert original == fingerprint(frame.copy(), settings={"seed": 42})
    changed = frame.copy()
    changed.loc[0, "cotton_close"] = 90
    assert fingerprint(changed, settings={"seed": 42}) != original
    assert fingerprint(frame, settings={"seed": 17}) != original


def test_price_early_stopping_matches_reported_mae():
    prices = np.array([30.0, 200.0])
    actual = np.array([0.1, -0.02])
    predicted = np.array([0.02, 0.01])
    assert price_mae_metric(prices)(actual, predicted) == pytest.approx(evaluate(prices, actual, predicted)["mae"])
    with pytest.raises(ValueError, match="matching validation"):
        price_mae_metric(prices)(actual[:1], predicted[:1])


def test_naive_direction_abstains_and_metrics_count_observations():
    result = evaluate(np.full(4, 80.0), [-0.01, 0.02, -0.01, 0.02], np.zeros(4))
    assert result["sample_count"] == 4
    assert result["directional_accuracy"] == 0
    assert result["balanced_accuracy"] == 0
    assert result["predicted_flat_pct"] == 100
    assert result["majority_direction_accuracy"] == 50


def test_sequence_keeps_feature_context_for_dates_without_valid_targets():
    history = _frame(100)
    origins = history.iloc[[60, 63, 67]].copy()
    inputs, targets = sequences(origins, None, history=history)
    assert inputs.shape == (3, 60, len(FEATURE_NAMES))
    np.testing.assert_array_equal(inputs[1, :, 0], np.arange(4, 64))
    np.testing.assert_array_equal(inputs[:, -1, 0], [60, 63, 67])
    assert targets.shape == (3, 2)
    future_changed = history.copy()
    future_changed.loc[68:, FEATURE_NAMES] = 99999
    np.testing.assert_array_equal(sequences(origins, None, history=future_changed)[0], inputs)


def test_sequence_requires_all_evaluation_origins_to_have_past_context():
    frame = _frame(100)
    with pytest.raises(ValueError, match="59 earlier"):
        sequences(frame.iloc[55:65], None, history=frame)


def test_sequence_missing_features_are_retained_observations_with_explicit_gaps():
    history = _frame(100)
    history["cotton_session_index"] = np.arange(100)
    history.loc[20, FEATURE_NAMES[0]] = np.nan
    history.loc[21, "target_return_1"] = np.nan  # target missing does not remove context
    origins = history.iloc[[65, 67]]
    inputs, _ = sequences(origins, None, history=history)
    assert inputs.shape == (2, 60, len(FEATURE_NAMES))
    assert 20 not in inputs[0, :, 1] and 21 in inputs[0, :, 1]
    report = sequence_diagnostics(origins, history)
    assert report["sample_count"] == 2
    assert report["windows_with_missing_feature_observations"] == 2
    assert report["max_missing_feature_observations_in_window"] == 1
    assert "not 60 calendar" in report["semantics"]


@pytest.mark.parametrize("problem", ["duplicate", "unsorted", "missing_origin", "ordinal"])
def test_sequence_rejects_invalid_source_order_and_missing_origins(problem):
    history = _frame(100)
    origins = history.iloc[[65, 67]].copy()
    if problem == "duplicate":
        history.loc[20, "date"] = history.date.iloc[19]
    elif problem == "unsorted":
        history = history.iloc[::-1]
    elif problem == "missing_origin":
        history.loc[65, FEATURE_NAMES[0]] = np.inf
    else:
        history["cotton_session_index"] = np.arange(100)
        history.loc[20, "cotton_session_index"] = 19
    with pytest.raises(ValueError):
        sequences(origins, None, history=history)


def test_feature_missing_refit_rows_match_the_same_complete_schema():
    frame = _frame()
    train, validation = purged_validation_split(frame.iloc[:300])
    # A safe inner-purge origin is restored only when its model inputs are complete.
    missing = train.index.max() + 1
    history = frame.copy()
    history.loc[missing, FEATURE_NAMES[0]] = np.nan
    refit = refit_rows(train, validation, frame.iloc[310:], history=history)
    assert len(refit) == 299
    assert history.date.iloc[missing] not in set(refit.date)


def test_fingerprint_covers_source_environment_dependencies_schema_and_config():
    frame = _frame()
    identity = experiment_identity(frame, settings={"horizon": 5, "model_family": "XGBoost", "seed": 42})
    assert len(identity["source_sha256"]) == 64
    assert len(identity["data"][0]["sha256"]) == 64
    assert "uv.lock" in identity["dependency_locks"]
    assert identity["feature_schema"] == FEATURE_NAMES
    assert identity["environment"]["python"]
    assert "tensorflow" in identity["environment"]["dependencies"]
    changed = frame.copy()
    changed["target_date_1"] = frame.date.shift(-1)
    assert fingerprint(changed) != fingerprint(frame)
