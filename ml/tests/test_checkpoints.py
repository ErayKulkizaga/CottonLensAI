import json

import pytest
from cottonlens_ml.checkpoints import (
    completed_checkpoint,
    completion_path,
    history_path,
    save_completed_checkpoint,
)


def _save(path):
    path.write_text("synthetic model bytes", encoding="utf-8")


def test_completion_requires_identity_and_payload_checksums(tmp_path):
    path = tmp_path / "model.json"
    identity = {"protocol": "test", "horizon": 5, "settings": {"seed": 42}}
    path.write_text("partial", encoding="utf-8")
    assert not completed_checkpoint(path, identity)
    save_completed_checkpoint(path, identity, _save)
    assert completed_checkpoint(path, identity)
    assert not completed_checkpoint(path, {**identity, "horizon": 1})
    assert not completed_checkpoint(path, {**identity, "settings": {"seed": 17}})
    path.write_text("corrupt", encoding="utf-8")
    assert not completed_checkpoint(path, identity)


def test_lstm_completion_checks_both_model_and_history(tmp_path):
    path = tmp_path / "model.keras"
    identity = {"model": "LSTM", "horizon": [1, 5]}
    history = [{"epoch": 1, "loss": 2.0, "val_loss": 3.0}]
    save_completed_checkpoint(path, identity, _save, history=history)
    assert completed_checkpoint(path, identity, require_history=True)
    history_path(path).write_text("[]", encoding="utf-8")
    assert not completed_checkpoint(path, identity, require_history=True)
    save_completed_checkpoint(path, identity, _save)
    assert not completed_checkpoint(path, identity, require_history=True)


def test_interrupted_rewrite_invalidates_previous_completion(tmp_path):
    path = tmp_path / "model.json"
    identity = {"model": "test"}
    save_completed_checkpoint(path, identity, _save)

    def interrupt(staged):
        staged.write_text("half written", encoding="utf-8")
        raise RuntimeError("interrupted fit/save")

    with pytest.raises(RuntimeError, match="interrupted"):
        save_completed_checkpoint(path, identity, interrupt)
    assert not completed_checkpoint(path, identity)
    assert not list(tmp_path.glob("*.pending*"))


def test_malformed_and_false_completion_markers_rejected(tmp_path):
    path = tmp_path / "model.json"
    identity = {"protocol": "test"}
    save_completed_checkpoint(path, identity, _save)
    marker = completion_path(path)
    record = json.loads(marker.read_text())
    record["state"] = "running"
    marker.write_text(json.dumps(record))
    assert not completed_checkpoint(path, identity)
    marker.write_text("not-json")
    assert not completed_checkpoint(path, identity)
    marker.write_text("[]")
    assert not completed_checkpoint(path, identity)


def test_completion_marker_is_committed_only_after_payloads(tmp_path, monkeypatch):
    from cottonlens_ml import checkpoints

    path = tmp_path / "model.keras"
    real_replace = checkpoints.os.replace

    def fail_history(source, destination):
        if destination == history_path(path):
            raise OSError("Drive disconnected")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoints.os, "replace", fail_history)
    with pytest.raises(OSError, match="Drive disconnected"):
        save_completed_checkpoint(path, {"run": 1}, _save, history=[{"epoch": 1}])
    assert not completion_path(path).exists()
    assert not completed_checkpoint(path, {"run": 1}, require_history=True)
