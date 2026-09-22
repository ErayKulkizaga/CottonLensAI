import pandas as pd
import pytest
from cottonlens_ml.data import MarketDownloadError, _download_ticker_with_retry


def test_market_download_retries_empty_response_then_returns_data() -> None:
    calls = []
    sleeps = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return pd.DataFrame() if len(calls) < 3 else pd.DataFrame({"Close": [1.0]})

    result = _download_ticker_with_retry("CT=F", "2010-01-01", download=download, sleep=sleeps.append)
    assert not result.empty
    assert sleeps == [10, 30]


def test_market_download_exhaustion_explains_safe_retry() -> None:
    with pytest.raises(MarketDownloadError, match="wait 5–10 minutes and rerun only the data cell"):
        _download_ticker_with_retry(
            "CT=F", "2010-01-01", download=lambda *args, **kwargs: pd.DataFrame(), sleep=lambda _: None
        )
