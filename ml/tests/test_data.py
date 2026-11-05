import pandas as pd
import pytest
from cottonlens_ml import data
from cottonlens_ml.data import MarketDownloadError, _download_ticker_with_retry


def _yahoo_frame(close, ticker=None):
    values = {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [close], "Volume": [10]}
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(["2020-04-20"], name="Date"))
    if ticker is not None:
        frame.columns = pd.MultiIndex.from_product([frame.columns, [ticker]], names=["Price", "Ticker"])
    return frame


@pytest.mark.parametrize("corruption", ["all_missing_close", "wrong_ticker", "missing_field", "duplicate_date"])
def test_market_refresh_preserves_existing_cache_when_provider_payload_is_invalid(tmp_path, monkeypatch, corruption):
    market_path = tmp_path / "market.parquet"
    prior = _yahoo_frame(80.0).reset_index().rename(columns=str.lower).assign(series="cotton")
    prior.to_parquet(market_path, index=False)
    before = market_path.read_bytes()
    calls = []

    def download(ticker, **_):
        calls.append(ticker)
        frame = _yahoo_frame(80.0, ticker)
        if ticker != "DX-Y.NYB":
            return frame
        if corruption == "all_missing_close":
            return _yahoo_frame(float("nan"), ticker)
        if corruption == "wrong_ticker":
            return _yahoo_frame(100.0, "CT=F")
        if corruption == "missing_field":
            return frame.drop(columns=("Close", ticker))
        return pd.concat([frame, frame])

    def fetch(ticker, start):
        return _download_ticker_with_retry(ticker, start, download=download, sleep=lambda _: None)

    monkeypatch.setattr(data, "_download_ticker_with_retry", fetch)
    monkeypatch.setattr(data, "download_cftc", pd.DataFrame)
    with pytest.raises(MarketDownloadError, match="DX-Y.NYB"):
        data.cache_sources(tmp_path, refresh=True)
    assert calls == ["CT=F"] + ["DX-Y.NYB"] * data.YAHOO_ATTEMPTS
    assert market_path.read_bytes() == before


def test_market_download_retries_empty_response_then_returns_data() -> None:
    calls = []
    sleeps = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame() if len(calls) < 3 else _yahoo_frame(1.0)

    result = _download_ticker_with_retry("CT=F", "2010-01-01", download=download, sleep=sleeps.append)
    assert not result.empty
    assert sleeps == [10, 30]


def test_market_download_exhaustion_explains_safe_retry() -> None:
    with pytest.raises(MarketDownloadError, match="wait 5–10 minutes and rerun only the data cell"):
        _download_ticker_with_retry(
            "CT=F", "2010-01-01", download=lambda *args, **kwargs: pd.DataFrame(), sleep=lambda _: None
        )


def test_market_download_preserves_daily_schema_and_negative_wti(monkeypatch):
    def download(ticker, **_):
        return _yahoo_frame(-37.63 if ticker == "CL=F" else 80.0, ticker)

    monkeypatch.setattr(data, "_download_ticker_with_retry", lambda ticker, start: _download_ticker_with_retry(
        ticker, start, download=download, sleep=lambda _: pytest.fail("valid bar should not retry"),
    ))
    market = data.download_market_data()
    assert list(market.columns) == ["date", "series", "open", "high", "low", "close", "volume"]
    assert market.set_index("series").close.to_dict() == {"cotton": 80.0, "dxy": 80.0, "wti": -37.63}
    assert market.date.eq(pd.Timestamp("2020-04-20")).all()
