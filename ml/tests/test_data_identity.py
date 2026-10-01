"""Small synthetic data evidence tests; no source download or model fitting."""

import json

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.config import PipelinePaths
from cottonlens_ml.snapshots import file_digest, frame_digest, write_snapshot


def test_data_digest_does_not_round_small_price_changes():
    original = pd.DataFrame({"price": [0.1234567890123456]})
    changed = pd.DataFrame({"price": [np.nextafter(original.price.iloc[0], np.inf)]})
    assert frame_digest(original) != frame_digest(changed)


def test_data_snapshot_reuse_is_content_addressed_and_does_not_overwrite(tmp_path):
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "close": [80.0, 81.0]})
    first = write_snapshot(tmp_path, {"market": frame}, {"policy": "test"})
    manifest_bytes = (tmp_path / first["snapshot_id"] / "manifest.json").read_bytes()
    repeated = write_snapshot(tmp_path, {"market": frame.copy()}, {"policy": "test"})
    assert first == repeated
    assert (tmp_path / first["snapshot_id"] / "manifest.json").read_bytes() == manifest_bytes
    changed = frame.copy()
    changed.loc[0, "close"] = 82.0
    second = write_snapshot(tmp_path, {"market": changed}, {"policy": "test"})
    assert first["snapshot_id"] != second["snapshot_id"]
    assert (tmp_path / first["snapshot_id"] / "manifest.json").read_bytes() == manifest_bytes


def test_data_snapshot_corruption_is_blocking_and_never_repaired_in_place(tmp_path):
    frame = pd.DataFrame({"close": [80.0, 81.0]})
    result = write_snapshot(tmp_path, {"market": frame}, {"policy": "test"})
    evidence = tmp_path / result["snapshot_id"] / "market.parquet"
    evidence.write_bytes(b"interrupted-or-corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_snapshot(tmp_path, {"market": frame}, {"policy": "test"})
    assert evidence.read_bytes() == b"interrupted-or-corrupt"


def test_data_snapshot_requires_completion_and_exact_manifest_identity(tmp_path):
    frame = pd.DataFrame({"close": [80.0, 81.0]})
    result = write_snapshot(tmp_path, {"market": frame}, {"policy": "test"})
    path = tmp_path / result["snapshot_id"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["complete"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity/checksum mismatch"):
        write_snapshot(tmp_path, {"market": frame}, {"policy": "test"})
    assert not any(tmp_path.glob(".pending-*"))


def test_refresh_preserves_old_raw_cache_before_replacing_aliases(tmp_path, monkeypatch):
    from cottonlens_ml import data

    previous = pd.DataFrame({"close": [80.0, -37.63]})
    previous.to_parquet(tmp_path / "market.parquet", index=False)
    replacement = pd.DataFrame({"close": [81.0, 82.0]})
    cftc = pd.DataFrame(columns=["available_date", "cftc_managed_money_net"])
    monkeypatch.setattr(data, "download_market_data", lambda: replacement)
    monkeypatch.setattr(data, "download_cftc", lambda: cftc)
    market, _ = data.cache_sources(tmp_path, refresh=True)
    pd.testing.assert_frame_equal(market, replacement)
    manifests = list((tmp_path / "snapshots").glob("*/manifest.json"))
    archived = next(path for path in manifests if json.loads(path.read_text())["identity"]["metadata"]["kind"] == "pre_refresh_source_cache")
    pd.testing.assert_frame_equal(pd.read_parquet(archived.parent / "market.parquet"), previous)


def test_prepare_retains_history_and_binds_immutable_raw_processed_evidence(tmp_path, monkeypatch):
    from cottonlens_ml import prepare
    from test_features import _cftc_fixture, _market_fixture

    market = _market_fixture()
    paths = PipelinePaths(tmp_path)
    monkeypatch.setattr(prepare, "cache_sources", lambda *args, **kwargs: (market, _cftc_fixture()))
    monkeypatch.setattr(prepare, "audit_split", lambda frame: {name: frame.copy() for name in ("train", "validation", "test")})
    _, history, _ = prepare.prepare(paths)
    assert len(history) == len(market.loc[market.series == "cotton"])
    assert history.target_return_5.isna().sum() == 5
    identity = history.attrs["data_identity"]
    manifest = json.loads((paths.processed / "snapshots" / identity["data_id"] / "manifest.json").read_text())
    assert manifest["identity"]["metadata"]["raw_id"] == identity["raw_id"]
    assert file_digest(paths.processed / "snapshots" / identity["data_id"] / "feature_history.parquet") == identity["checksums"]["feature_history.parquet"]
    assert history.attrs["data_quality"]["feature_coverage"]["groups"]["cotton_only"]["natural_origin_count"] > 0
