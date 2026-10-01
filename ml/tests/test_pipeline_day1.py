"""Fake-only orchestration check: selection is frozen before audit scoring."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
from cottonlens_ml import pipeline


def test_pipeline_freezes_independent_winners_before_seen_audit_and_pending_release(tmp_path, monkeypatch):
    import cottonlens_ml.export as exporting
    import cottonlens_ml.walkforward as walking
    from cottonlens_ml import tracking, training

    events = []
    dates = pd.bdate_range("2026-09-01", periods=10)
    features = pd.DataFrame({"date": dates, "cotton_close": np.full(10, 80.0)})
    features.attrs["data_identity"] = {"data_id": "data"}
    ready = {"readiness_id": "ready", "code_identity": {"source_id": "code"},
             "data_identity": {"data_id": "data"}, "cohort_identity": {"cohort_id": "cohort"},
             "protocol_identity": {"version": "frozen"}, "data_quality": {"status": "synthetic"},
             "environment_smoke": {"status": "passed"}}
    drive = tmp_path / "drive"
    (drive / "experiments/corrected-v2").mkdir(parents=True)
    monkeypatch.setattr(pipeline, "require_colab_training", lambda: None)
    monkeypatch.setattr(pipeline, "verify_readiness", lambda *_: (ready, features, features))
    monkeypatch.setattr(pipeline, "select_feature_rows", lambda *_, **__: features)
    monkeypatch.setattr(walking, "audit_split", lambda _: {"test": features.iloc[-3:]})
    monkeypatch.setattr(tracking, "tracking_session", lambda *_: nullcontext())
    monkeypatch.setattr(tracking, "tracked_run", lambda **_: nullcontext())
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(
        log_params=lambda *_: None, log_artifact=lambda *_: None, log_metrics=lambda *_: None,
    ))

    def candidate(name, horizon, audit=False):
        return SimpleNamespace(name=name, horizon=horizon, metrics={"mae": 0.01 if audit else 1.0,
                                                                     "directional_accuracy": 99.0},
                               parameters={"model_role": "evaluation_backtest"})

    locked = {name: {h: candidate(name, h) for h in (1, 5)} for name in ("Naive", "Ridge", "XGBoost", "LSTM")}
    monkeypatch.setattr(walking, "run_walkforward", lambda *_, **__: ({"folds": [1, 2, 3, 4]}, locked))

    def choose(_, horizon, historical_audit):
        assert not historical_audit and not any(event.startswith("audit") for event in events)
        events.append(f"select-{horizon}")
        name = "Naive" if horizon == 1 else "XGBoost"
        return name, {"selected": name, "historical_audit": {}}

    monkeypatch.setattr(pipeline, "select_walkforward_name", choose)

    def audit(candidates, *_):
        assert (drive / "experiments/corrected-v2/selection-before-audit.json").is_file()
        events.append("audit-" + candidates[1].name)
        return {h: candidate(candidates[h].name, h, audit=True) for h in (1, 5)}

    monkeypatch.setattr(training, "refit_locked_evaluation", audit)

    def deployment(candidates, *_):
        events.append("deployment")
        return {h: candidate(candidates[h].name, h) for h in (1, 5)}

    monkeypatch.setattr(training, "refit_deployment", deployment)

    def export(_root, _history, _test, _market, _all, selected, decisions, _report, **kwargs):
        assert [selected[h].name for h in (1, 5)] == ["Naive", "XGBoost"]
        assert decisions[1]["historical_audit"]["LSTM"]["directional_accuracy"] == 99.0
        assert set(kwargs["deployment_selected"]) == {1, 5}
        assert kwargs["evidence_identity"]["cohort_identity"] == ready["cohort_identity"]
        events.append("export")
        return drive / "artifacts/releases/pending.zip"

    monkeypatch.setattr(exporting, "export_release", export)
    result = pipeline.run(drive, repo=tmp_path, experiment="corrected-v2")
    assert result.name == "pending.zip"
    assert events[:2] == ["select-1", "select-5"]
    assert events.index("export") > events.index("deployment") > events.index("audit-LSTM")
    frozen = json.loads((drive / "experiments/corrected-v2/selection-before-audit.json").read_text())
    assert [frozen[str(h)]["selected"] for h in (1, 5)] == ["Naive", "XGBoost"]
    assert not (drive / "artifacts/releases/latest.txt").exists()
