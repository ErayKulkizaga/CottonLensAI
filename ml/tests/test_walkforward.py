import pandas as pd
from cottonlens_ml.walkforward import AUDIT_START, audit_split, build_folds


def _calendar() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", "2026-09-01")
    frame = pd.DataFrame({"date": dates})
    frame["target_date_5"] = frame.date.shift(-5)
    return frame.dropna().reset_index(drop=True)


def test_four_walkforward_folds_are_purged_and_pre_audit() -> None:
    folds = build_folds(_calendar())
    assert len(folds) == 4
    assert all(len(fold.test) == 126 for fold in folds)
    for fold in folds:
        assert fold.train.target_date_5.max() < fold.validation.date.min()
        assert fold.validation.target_date_5.max() < fold.test.date.min()
        assert fold.test.target_date_5.max() < AUDIT_START
    assert folds[0].test.date.max() < folds[1].test.date.min()


def test_historical_audit_cannot_supply_pre_audit_labels() -> None:
    split = audit_split(_calendar())
    assert split["train"].target_date_5.max() < split["validation"].date.min()
    assert split["validation"].target_date_5.max() < split["test"].date.min()
    assert split["test"].date.min() >= AUDIT_START
