import copy

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.cohort import create_cohort, freeze_cohort, verify_cohort
from cottonlens_ml.config import FEATURE_NAMES


def history():
    dates = pd.bdate_range("2016-01-01", "2026-09-22")
    result = pd.DataFrame({"date": dates, "cotton_close": 70 + np.arange(len(dates)) / 1000})
    for name in FEATURE_NAMES:
        result[name] = 0.01
    for h in (1, 5):
        result[f"target_date_{h}"] = result.date.shift(-h)
        result[f"target_return_{h}"] = np.log(result.cotton_close.shift(-h) / result.cotton_close)
    return result


def test_cohort_freezes_504_complete_feature_origins_before_any_fit(tmp_path):
    frame = history()
    manifest = freeze_cohort(tmp_path / "cohort.json", frame, "snapshot-a")
    origins = [d for fold in manifest["folds"] for d in fold["origins"]]
    assert len(origins) == len(set(origins)) == 504
    assert freeze_cohort(tmp_path / "cohort.json", frame, "snapshot-a") == manifest
    changed = frame.copy()
    changed.loc[changed.date == origins[12], FEATURE_NAMES] = np.nan
    new = create_cohort(changed, "snapshot-b")
    new_origins = [date for fold in new["folds"] for date in fold["origins"]]
    assert len(new_origins) == len(set(new_origins)) == 504
    assert origins[12] not in new_origins
    assert new["folds"][0]["end_anchor"] == manifest["folds"][0]["end_anchor"]
    assert "complete_feature" in new["origin_policy"]


def test_frozen_cohort_rejects_changed_data_or_manifest(tmp_path):
    frame = history()
    path = tmp_path / "cohort.json"
    original = freeze_cohort(path, frame, "snapshot-a")
    with pytest.raises(ValueError, match="changed"):
        freeze_cohort(path, frame, "snapshot-b")
    altered = copy.deepcopy(original)
    altered["folds"][0]["origins"][0] = "2000-01-01"
    with pytest.raises(ValueError, match="checksum"):
        verify_cohort(altered, frame, "snapshot-a")
    assert freeze_cohort(path, frame, "snapshot-a") == original


def test_changed_label_cannot_reuse_cohort(tmp_path):
    frame = history()
    manifest = create_cohort(frame, "data")
    frame.loc[frame.date == manifest["folds"][0]["origins"][0], "target_return_5"] += 0.01
    with pytest.raises(ValueError, match="changed"):
        verify_cohort(manifest, frame, "data")


def test_insufficient_data_cannot_invent_replacement_origins():
    with pytest.raises(ValueError, match="126"):
        create_cohort(history().tail(200), "data")
