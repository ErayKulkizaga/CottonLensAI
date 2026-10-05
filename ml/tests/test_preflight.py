"""No-fit checks for snapshot, date, source and frozen evaluation readiness."""

import json

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.code_identity import digest, manifest_id
from cottonlens_ml.config import FEATURE_NAMES, PipelinePaths
from cottonlens_ml.preflight import (
    freeze_readiness,
    verify_readiness,
    verify_target_contract,
)
from cottonlens_ml.snapshots import write_snapshot


def _history():
    dates = pd.bdate_range("2016-01-01", "2026-09-22")
    frame = pd.DataFrame({"date": dates, "cotton_close": 70 + np.arange(len(dates)) / 1000})
    for name in FEATURE_NAMES:
        frame[name] = 0.01
    for horizon in (1, 5):
        frame[f"target_date_{horizon}"] = frame.date.shift(-horizon)
        frame[f"target_return_{horizon}"] = np.log(frame.cotton_close.shift(-horizon) / frame.cotton_close)
    frame["cotton_session_index"] = np.arange(len(frame))
    frame["decision_time"] = pd.to_datetime(frame.date, utc=True) + pd.Timedelta(days=1)
    for series in ("dxy", "wti"):
        frame[f"{series}_source_date"] = frame.date.shift(1)
        frame[f"{series}_available_at"] = pd.to_datetime(frame.date.shift(1), utc=True) + pd.Timedelta(days=1)
        frame[f"{series}_age_sessions"] = np.where(frame.index > 0, 1.0, np.nan)
        frame[f"{series}_stale"] = False
    return frame


def _source(repo):
    package = repo / "ml/src/cottonlens_ml"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# captured code\n", encoding="utf-8")
    (repo / "ml/uv.lock").write_text("lock\n", encoding="utf-8")
    files = {name: digest(repo / name) for name in ("ml/src/cottonlens_ml/__init__.py", "ml/uv.lock")}
    payload = {"format": 1, "git_head": "synthetic", "working_tree_dirty": True, "files": files}
    manifest = {**payload, "source_id": manifest_id(payload)}
    (repo / ".source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _inputs(tmp_path, history):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = _source(repo)
    paths = PipelinePaths(tmp_path / "drive")
    paths.create()
    raw = write_snapshot(paths.raw / "snapshots", {"market": pd.DataFrame({
        "date": history.date, "series": "cotton", "close": history.cotton_close,
    })}, {"kind": "source_cache"})
    processed = write_snapshot(paths.processed / "snapshots", {"feature_history": history}, {
        "raw_id": raw["snapshot_id"], "data_quality": {"source": "synthetic unit fixture"},
    })
    identity = {"data_id": processed["snapshot_id"], "raw_id": raw["snapshot_id"],
                "manifest_path": processed["manifest_path"], "raw_manifest_path": raw["manifest_path"],
                "checksums": processed["checksums"], "source_checksums": raw["checksums"]}
    (paths.processed / "data_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps({"status": "passed", "source_id": source["source_id"],
                                      "environment": {"lock_sha256": digest(repo / "ml/uv.lock")}}), encoding="utf-8")
    return repo, paths, smoke_path


def test_preflight_freezes_504_origins_and_rejects_modified_source_or_snapshot(tmp_path):
    repo, paths, smoke = _inputs(tmp_path, _history())
    ready = freeze_readiness(repo, paths, "corrected-v2", smoke)
    assert ready["checks"]["evaluation_count_per_horizon"] == 504
    assert verify_readiness(repo, paths, "corrected-v2")[0] == ready
    assert freeze_readiness(repo, paths, "corrected-v2", smoke) == ready
    source_file = repo / "ml/src/cottonlens_ml/__init__.py"
    source_file.write_text("# different code\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source file"):
        verify_readiness(repo, paths, "corrected-v2")
    source_file.write_text("# captured code\n", encoding="utf-8")
    parquet = next((paths.processed / "snapshots").glob("*/feature_history.parquet"))
    with parquet.open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        verify_readiness(repo, paths, "corrected-v2")


def test_preflight_freezes_replacement_origin_when_feature_is_unavailable_before_fitting(tmp_path):
    history = _history()
    history.loc[history.date == pd.Timestamp("2022-10-21"), FEATURE_NAMES[0]] = np.nan
    repo, paths, smoke = _inputs(tmp_path, history)
    ready = freeze_readiness(repo, paths, "missing-origin", smoke)
    origins = [date for fold in ready["cohort_identity"]["folds"] for date in fold["origins"]]
    assert "2022-10-21" not in origins
    assert ready["checks"]["evaluation_count_per_horizon"] == 504


def test_target_and_external_timestamps_block_future_information():
    history = _history()
    verify_target_contract(history)
    wrong_target = history.copy()
    wrong_target.loc[10, "target_date_5"] = wrong_target.loc[11, "target_date_5"]
    with pytest.raises(ValueError, match=r"T\+5 target dates"):
        verify_target_contract(wrong_target)
    future_source = history.copy()
    future_source.loc[10, "dxy_source_date"] = future_source.loc[10, "date"]
    with pytest.raises(ValueError, match="same-origin/future"):
        verify_target_contract(future_source)
    future_availability = history.copy()
    future_availability.loc[10, "wti_available_at"] += pd.Timedelta(days=2)
    with pytest.raises(ValueError, match="availability timestamp"):
        verify_target_contract(future_availability)
