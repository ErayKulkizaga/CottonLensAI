"""Only fake estimators/scalers. No XGBoost, sklearn estimator or TensorFlow fits."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml import training
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.protocol import refit_rows
from cottonlens_ml.walkforward import purged_validation_split


class FakeScaler:
    fits: ClassVar[list] = []

    def fit(self, data):
        self.values = np.asarray(data).copy()
        self.fits.append(self.values)
        return self

    def transform(self, data):
        return np.asarray(data, dtype=float)

    def inverse_transform(self, data):
        return np.asarray(data, dtype=float)


class FakeXGB:
    fits: ClassVar[list] = []

    def __init__(self, **config):
        self.config = config
        self.best_iteration = 1

    def fit(self, x, y, **kwargs):
        self.fits.append((x.copy(), np.asarray(y).copy(), kwargs, self.config))
        return self

    def predict(self, x):
        return np.zeros(len(x))

    def save_model(self, path):
        path.write_text(json.dumps({"best_iteration": self.best_iteration}))

    def load_model(self, path):
        self.best_iteration = json.loads(path.read_text())["best_iteration"]


class FakeRidge(FakeXGB):
    fits: ClassVar[list] = []


class FakeLSTM:
    fits: ClassVar[list] = []

    def fit(self, x, y, **kwargs):
        self.fits.append((x.copy(), y.copy(), kwargs))
        return SimpleNamespace(history={"loss": [2.0, 1.0], "val_loss": [1.0, 0.5]})

    def predict(self, x, **kwargs):
        return np.zeros((len(x), 2))

    def save(self, path):
        path.write_text("completed fake LSTM")


@pytest.fixture
def fakes(monkeypatch):
    FakeXGB.fits = []
    FakeRidge.fits = []
    FakeLSTM.fits = []
    FakeScaler.fits = []
    tf = SimpleNamespace(keras=SimpleNamespace(
        models=SimpleNamespace(load_model=lambda path: FakeLSTM()),
        callbacks=SimpleNamespace(EarlyStopping=lambda **kwargs: object()),
    ))
    runtime = {
        "xgboost": SimpleNamespace(XGBRegressor=FakeXGB),
        "sklearn.linear_model": SimpleNamespace(Ridge=FakeRidge),
        "sklearn.preprocessing": SimpleNamespace(StandardScaler=FakeScaler),
        "tensorflow": tf,
        "mlflow": SimpleNamespace(log_params=lambda *_: None, log_metric=lambda *_: None),
    }
    monkeypatch.setattr(training, "require_colab_training", lambda: None)
    monkeypatch.setattr(training, "_runtime", runtime.__getitem__)
    monkeypatch.setattr(training, "_new_lstm", lambda *_: FakeLSTM())
    monkeypatch.setattr(training, "tracked_run", lambda **_: nullcontext())


def frames():
    n = 420
    frame = pd.DataFrame({name: np.arange(n, dtype=float) for name in FEATURE_NAMES})
    frame["date"] = pd.bdate_range("2020-01-01", periods=n)
    frame["cotton_session_index"] = np.arange(n)
    frame["cotton_close"] = 80.0
    frame["target_return_1"] = 0.01
    frame["target_return_5"] = 0.03
    frame["target_date_1"] = frame.date.shift(-1)
    frame["target_date_5"] = frame.date.shift(-5)
    train, validation = purged_validation_split(frame.iloc[60:360])
    test = frame.iloc[370:390].copy()
    return frame, train, validation, test


def test_training_entrypoints_block_local_fits_before_loading_runtimes(monkeypatch, tmp_path):
    from cottonlens_ml import runtime_guard

    monkeypatch.setattr(runtime_guard.sys, "platform", "win32")
    frame, train, validation, test = frames()
    monkeypatch.setattr(training, "_runtime", lambda *_: pytest.fail("Training runtime was imported"))
    for action in (
        lambda: training.train_xgboost(train, validation, test, tmp_path),
        lambda: training.train_lstm(train, validation, test, tmp_path),
        lambda: training.ridge_candidates(train, validation, test),
        lambda: training.refit_deployment({}, frame, test.date.min(), tmp_path),
        lambda: training.refit_locked_evaluation({}, frame, test, tmp_path),
    ):
        with pytest.raises(RuntimeError, match="Google Colab"):
            action()


def test_isolated_colab_environment_guard_does_not_require_google_module(monkeypatch):
    from cottonlens_ml import runtime_guard

    monkeypatch.setattr(runtime_guard.sys, "platform", "linux")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-test")
    monkeypatch.setattr(runtime_guard.Path, "is_dir", lambda self: str(self).replace("\\", "/") in ("/content", "/content/drive/MyDrive"))
    # This check performs no imports of google.colab, tensorflow or xgboost.
    runtime_guard.require_colab_training()
    monkeypatch.delenv("COLAB_RELEASE_TAG")
    monkeypatch.delenv("COLAB_BACKEND_VERSION", raising=False)
    with pytest.raises(RuntimeError, match="Google Colab"):
        runtime_guard.require_colab_training()


def test_same_refit_origins_scaler_cutoffs_and_locked_counts(fakes, tmp_path):
    frame, train, validation, test = frames()
    expected = refit_rows(train, validation, test, history=frame)
    ridge = training.ridge_candidates(train, validation, test, feature_history=frame)
    trees = training.train_xgboost(train, validation, test, tmp_path / "xgb", feature_history=frame)
    _, _, lstms = training.train_lstm(train, validation, test, tmp_path / "lstm", feature_history=frame)
    for candidates in (ridge, trees, lstms):
        for candidate in candidates.values():
            assert candidate.parameters["refit_rows"] == len(expected) == 300
            assert pd.Timestamp(candidate.parameters["fit_label_cutoff"]) < test.date.min()
    for fit in FakeXGB.fits:
        x, _y, kwargs, config = fit
        if "eval_set" in kwargs:
            assert list(x.index) == list(train.index)
            assert list(kwargs["eval_set"][0][0].index) == list(validation.index)
            assert config["eval_metric"](np.zeros(len(validation)), np.zeros(len(validation))) == 0
        else:
            assert list(x.iloc[:, 0]) == list(expected.iloc[:, 0])
            assert config["n_estimators"] == 2
            assert "early_stopping_rounds" not in config
    for x, _, kwargs in FakeLSTM.fits:
        if "validation_data" not in kwargs:
            np.testing.assert_array_equal(x[:, -1, 0], expected[FEATURE_NAMES[0]])
            assert kwargs["epochs"] == 2
    # Feature scalers see only train or safe refit, never validation/test future values.
    for values in FakeScaler.fits:
        if values.shape[1] == len(FEATURE_NAMES):
            assert values[:, 0].max() in (train[FEATURE_NAMES[0]].max(), expected[FEATURE_NAMES[0]].max())
    assert trees[1].parameters["training_objective"] == "log_return_squared_error_surrogate"
    assert lstms[1].parameters["refit_sequence_rows"] == len(expected)


def test_completed_trials_resume_without_fitting_and_corruption_refits(fakes, tmp_path):
    frame, train, validation, test = frames()
    training.train_xgboost(train, validation, test, tmp_path, feature_history=frame)
    original_count = len(FakeXGB.fits)
    training.train_xgboost(train, validation, test, tmp_path, feature_history=frame)
    assert len(FakeXGB.fits) == original_count
    next(path for path in tmp_path.glob("xgb-*.json") if ".complete." not in path.name and "refit" not in path.name).write_text("corrupt")
    training.train_xgboost(train, validation, test, tmp_path, feature_history=frame)
    assert len(FakeXGB.fits) == original_count + 1


def test_lstm_requires_completed_finite_history_to_resume(fakes, tmp_path):
    frame, train, validation, test = frames()
    training.train_lstm(train, validation, test, tmp_path, feature_history=frame)
    original_count = len(FakeLSTM.fits)
    training.train_lstm(train, validation, test, tmp_path, feature_history=frame)
    assert len(FakeLSTM.fits) == original_count
    next(tmp_path.glob("*.history.json")).write_text("[]")
    training.train_lstm(train, validation, test, tmp_path, feature_history=frame)
    assert len(FakeLSTM.fits) == original_count + 1


def test_deployment_refits_locked_recipe_at_current_cutoff_without_tuning(fakes, tmp_path):
    frame, train, validation, test = frames()
    trees = training.train_xgboost(train, validation, test, tmp_path / "evaluation", feature_history=frame)
    original_models = {h: c.model for h, c in trees.items()}
    start = len(FakeXGB.fits)
    cutoff = frame.date.iloc[410]
    deployed = training.refit_deployment(trees, frame, cutoff, tmp_path / "deployment")
    assert len(FakeXGB.fits) == start + 2
    for horizon, candidate in deployed.items():
        assert candidate.model is not original_models[horizon]
        assert not len(candidate.predictions) and candidate.metrics == {}
        assert candidate.parameters["model_role"] == "deployment_live"
        assert pd.Timestamp(candidate.parameters["fit_label_cutoff"]) < cutoff
        assert candidate.parameters["locked_config"] == trees[horizon].parameters["locked_config"]
        assert candidate.parameters["model_identity"] != trees[horizon].parameters["model_identity"]
    for x, _, kwargs, _ in FakeXGB.fits[start:]:
        assert "eval_set" not in kwargs
        assert x.index.max() == 404
    assert all(c.parameters["model_role"] == "evaluation_backtest" for c in trees.values())


def test_deployment_lstm_shared_two_output_model_is_fitted_once(fakes, tmp_path):
    frame, train, validation, test = frames()
    _, _, candidates = training.train_lstm(train, validation, test, tmp_path / "evaluation", feature_history=frame)
    start = len(FakeLSTM.fits)
    deployed = training.refit_deployment(candidates, frame, frame.date.iloc[410], tmp_path / "deployment")
    assert len(FakeLSTM.fits) == start + 1
    assert deployed[1].model is deployed[5].model
    assert "validation_data" not in FakeLSTM.fits[-1][2]


def test_nonfinite_validation_fails_before_training_runtime(fakes, monkeypatch, tmp_path):
    frame, train, validation, test = frames()
    validation.loc[validation.index[0], FEATURE_NAMES[0]] = np.inf
    monkeypatch.setattr(training, "_runtime", lambda *_: pytest.fail("Runtime loaded before validation"))
    with pytest.raises(ValueError, match="non-finite"):
        training.train_xgboost(train, validation, test, tmp_path, feature_history=frame)


def test_locked_seen_audit_does_not_tune_or_use_audit_labels_for_fit(fakes, tmp_path):
    frame, train, validation, test = frames()
    locked = training.train_xgboost(train, validation, test, tmp_path / "evaluation", feature_history=frame)
    count = len(FakeXGB.fits)
    audited = training.refit_locked_evaluation(locked, frame, test, tmp_path / "audit")
    assert len(FakeXGB.fits) == count + 2
    for _, _, kwargs, config in FakeXGB.fits[count:]:
        assert "eval_set" not in kwargs and "early_stopping_rounds" not in config
    changed = test.copy()
    changed["target_return_1"] = 0.25
    changed["target_return_5"] = -0.25
    changed_history = frame.copy()
    changed_history.loc[test.index, "target_return_1"] = 0.25
    changed_history.loc[test.index, "target_return_5"] = -0.25
    repeated = training.refit_locked_evaluation(locked, changed_history, changed, tmp_path / "audit")
    assert len(FakeXGB.fits) == count + 2
    for horizon in (1, 5):
        assert audited[horizon].parameters["model_identity"] == repeated[horizon].parameters["model_identity"]
        assert audited[horizon].parameters["model_role"] == "evaluation_backtest"
        assert audited[horizon].parameters["evidence_label"] == "seen_historical_audit"
        assert audited[horizon].parameters["fit_label_cutoff"] < test.date.min().isoformat()
        assert audited[horizon].validation_metrics is None
        assert len(audited[horizon].predictions) == len(test)
        assert audited[horizon].metrics != repeated[horizon].metrics


def test_lstm_locked_audit_identity_excludes_unavailable_context_labels(fakes, tmp_path):
    frame, train, validation, test = frames()
    _, _, locked = training.train_lstm(train, validation, test, tmp_path / "evaluation", feature_history=frame)
    count = len(FakeLSTM.fits)
    audited = training.refit_locked_evaluation(locked, frame, test, tmp_path / "audit")
    assert len(FakeLSTM.fits) == count + 1
    changed = frame.copy()
    changed.loc[changed.target_date_5 >= test.date.min(), ["target_return_1", "target_return_5"]] = 0.5
    repeated = training.refit_locked_evaluation(locked, changed, changed.loc[test.index], tmp_path / "audit")
    assert len(FakeLSTM.fits) == count + 1
    assert audited[1].parameters["model_identity"] == repeated[1].parameters["model_identity"]
    assert "validation_data" not in FakeLSTM.fits[-1][2]
