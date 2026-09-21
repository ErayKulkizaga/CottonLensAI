import numpy as np
import pandas as pd
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
    assert len(splits["train"]) == int(len(features) * 0.65)
    assert len(splits["validation"]) == int(len(features) * 0.80) - len(splits["train"])
    assert splits["train"].date.max() < splits["validation"].date.min()
    assert splits["validation"].date.max() < splits["test"].date.min()
