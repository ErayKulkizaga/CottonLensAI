import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.config import FEATURE_NAMES
from cottonlens_ml.features import build_features, chronological_split


def _market_fixture() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=620)
    rows: list[dict] = []
    for series, base, slope in (
        ("cotton", 72.0, 0.015),
        ("dxy", 101.0, 0.01),
        ("wti", 78.0, 0.02),
    ):
        for index, observed_on in enumerate(dates):
            close = base + index * slope + np.sin(index / 17) * 0.4
            rows.append(
                {
                    "date": observed_on,
                    "series": series,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 100_000 + index * 11,
                }
            )
    return pd.DataFrame(rows)


def _cftc_fixture() -> pd.DataFrame:
    available_dates = pd.date_range("2021-01-08", periods=150, freq="W-FRI")
    return pd.DataFrame(
        {
            "available_date": available_dates,
            "cftc_managed_money_net": np.arange(len(available_dates)) * 1_000,
        }
    )


def test_cftc_value_changes_on_friday_not_before() -> None:
    features = build_features(_market_fixture(), _cftc_fixture()).set_index("date")
    friday = next(
        value
        for value in _cftc_fixture().available_date
        if value in features.index and value - pd.offsets.BDay(1) in features.index
    )
    thursday = friday - pd.offsets.BDay(1)
    expected = _cftc_fixture().set_index("available_date")["cftc_managed_money_net"]
    assert features.loc[thursday, "cftc_managed_money_net"] == expected.loc[friday - pd.Timedelta(days=7)]
    assert features.loc[friday, "cftc_managed_money_net"] == expected.loc[friday]


def test_latest_rows_remain_available_for_live_inference() -> None:
    market = _market_fixture()
    features = build_features(market, _cftc_fixture())
    latest_cotton_date = market.loc[market.series == "cotton", "date"].max()
    assert features.date.max() == latest_cotton_date
    assert pd.isna(features.iloc[-1].target_return_1)
    assert pd.isna(features.iloc[-1].target_return_5)


def test_split_is_chronological_and_65_15_20() -> None:
    features = build_features(_market_fixture(), _cftc_fixture()).dropna(
        subset=["target_return_1", "target_return_5"]
    )
    splits = chronological_split(features)
    assert len(splits["train"]) == int(len(features) * 0.65) - 5
    assert len(splits["validation"]) == int(len(features) * 0.80) - int(len(features) * 0.65) - 5
    assert splits["train"].date.max() < splits["validation"].date.min()
    assert splits["validation"].date.max() < splits["test"].date.min()
    assert splits["train"].target_date_5.max() < splits["validation"].date.min()
    assert splits["validation"].target_date_5.max() < splits["test"].date.min()


@pytest.mark.parametrize("bad_value", [-37.63, 0.0, float("inf"), float("nan")])
def test_invalid_external_prices_are_reported_and_only_past_filled(bad_value) -> None:
    market = _market_fixture()
    wti = market.index[market.series == "wti"]
    index = wti[300]
    bad_date = market.at[index, "date"]
    market.at[index, "close"] = bad_value
    with np.errstate(invalid="raise", divide="raise"):
        result = build_features(market, _cftc_fixture())
    assert np.isfinite(result[FEATURE_NAMES].to_numpy()).all()
    issue = next(item for item in result.attrs["data_quality"]["invalid_market_values"] if item["field"] == "close")
    assert issue["series"] == "wti"
    assert issue["date"] == bad_date.isoformat()
    assert result.set_index("date").loc[bad_date + pd.offsets.BDay(1), "wti_ret_1"] == pytest.approx(0.0)
    # Altering future prices must not affect any feature already available.
    changed = market.copy()
    changed.loc[(changed.series == "wti") & (changed.date > bad_date), "close"] *= 2
    future = build_features(changed, _cftc_fixture())
    pd.testing.assert_frame_equal(
        result.loc[result.date <= bad_date, FEATURE_NAMES],
        future.loc[future.date <= bad_date, FEATURE_NAMES],
    )


def test_invalid_cotton_is_not_filled_or_removed_from_target_calendar() -> None:
    market = _market_fixture()
    rows = market.index[market.series == "cotton"]
    bad_date = market.at[rows[300], "date"]
    preceding_date = market.at[rows[295], "date"]
    market.at[rows[300], "close"] = -1
    result = build_features(market, _cftc_fixture()).set_index("date")
    assert bad_date not in result.index
    assert pd.isna(result.loc[preceding_date, "target_return_5"])


def test_zero_volume_and_constant_cftc_never_produce_infinite_features() -> None:
    market = _market_fixture()
    market.loc[market.index[market.series == "cotton"][300], "volume"] = 0
    result = build_features(market, _cftc_fixture())
    assert np.isfinite(result[FEATURE_NAMES].to_numpy()).all()
    cftc = _cftc_fixture()
    cftc["cftc_managed_money_net"] = 1.0
    result = build_features(market, cftc)
    assert not result.empty  # Unverified CFTC values are not model inputs.
    assert result.cftc_net_z52.isna().all()


def test_same_day_external_close_cannot_change_cotton_feature() -> None:
    market = _market_fixture()
    original = build_features(market, _cftc_fixture()).set_index("date")
    decision_date = pd.Timestamp("2023-04-03")
    market.loc[(market.series == "dxy") & (market.date == decision_date), "close"] *= 1.1
    changed = build_features(market, _cftc_fixture()).set_index("date")
    pd.testing.assert_series_equal(
        original.loc[decision_date, ["dxy_ret_1", "dxy_ret_5", "dxy_ret_20"]],
        changed.loc[decision_date, ["dxy_ret_1", "dxy_ret_5", "dxy_ret_20"]],
    )


def test_missing_audit_only_cftc_does_not_block_model_features() -> None:
    cftc = pd.DataFrame(columns=["available_date", "cftc_managed_money_net"])
    result = build_features(_market_fixture(), cftc)
    assert not result.empty
    assert result.cftc_managed_money_net.isna().all()
    assert np.isfinite(result[FEATURE_NAMES].to_numpy()).all()
