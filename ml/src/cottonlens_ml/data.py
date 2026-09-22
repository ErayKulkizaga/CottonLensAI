from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

TICKERS = {"cotton": "CT=F", "dxy": "DX-Y.NYB", "wti": "CL=F"}
CFTC_MARKET_CODE = "033661"
YAHOO_ATTEMPTS = 4
YAHOO_RETRY_DELAYS_SECONDS = (10, 30, 60)


class MarketDownloadError(RuntimeError):
    """A source failed after visible, bounded retries; no partial dataset is returned."""


def _download_ticker_with_retry(
    ticker: str,
    start: str,
    *,
    download=None,
    sleep=time.sleep,
) -> pd.DataFrame:
    if download is None:
        import yfinance as yf

        download = yf.download
    failures: list[str] = []
    for attempt in range(1, YAHOO_ATTEMPTS + 1):
        try:
            raw = download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
            if not raw.empty:
                return raw
            failures.append("empty response (Yahoo may have rate-limited this request)")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        if attempt < YAHOO_ATTEMPTS:
            delay = YAHOO_RETRY_DELAYS_SECONDS[attempt - 1]
            print(
                f"Yahoo request for {ticker} failed (attempt {attempt}/{YAHOO_ATTEMPTS}); "
                f"retrying in {delay}s...",
                flush=True,
            )
            sleep(delay)
    raise MarketDownloadError(
        f"Yahoo returned no usable data for {ticker} after {YAHOO_ATTEMPTS} attempts. "
        "It is temporarily rate-limited; wait 5–10 minutes and rerun only the data cell. "
        f"Details: {' | '.join(failures)}"
    )


def download_market_data(start: str = "2010-01-01") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series, ticker in TICKERS.items():
        raw = _download_ticker_with_retry(ticker, start)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        normalized = raw.rename(columns=str.lower).reset_index()
        date_column = "Date" if "Date" in normalized.columns else "date"
        normalized = normalized.rename(columns={date_column: "date"})
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
        normalized["series"] = series
        frames.append(normalized[["date", "series", "open", "high", "low", "close", "volume"]])
    return pd.concat(frames, ignore_index=True).sort_values(["date", "series"])


def _normalized_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def download_cftc(start_year: int = 2010, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or date.today().year
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        url = f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_name = next(name for name in archive.namelist() if name.lower().endswith((".txt", ".csv")))
            with archive.open(csv_name) as handle:
                frame = pd.read_csv(handle, low_memory=False)
        frame.columns = [_normalized_column(column) for column in frame.columns]
        code_column = next(column for column in frame if "cftc_contract_market_code" in column)
        date_column = next(column for column in frame if column.startswith("report_date_as"))
        long_column = next(column for column in frame if column.startswith("m_money_positions_long_all"))
        short_column = next(column for column in frame if column.startswith("m_money_positions_short_all"))
        code = frame[code_column].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
        cotton = frame.loc[code == CFTC_MARKET_CODE, [date_column, long_column, short_column]].copy()
        cotton.columns = ["report_date", "managed_money_long", "managed_money_short"]
        frames.append(cotton)
    result = pd.concat(frames, ignore_index=True)
    result["report_date"] = pd.to_datetime(result["report_date"])
    result["available_date"] = result["report_date"] + pd.Timedelta(days=3)
    result["cftc_managed_money_net"] = result["managed_money_long"] - result["managed_money_short"]
    return result.sort_values("available_date").drop_duplicates("available_date", keep="last")


def cache_sources(root: Path, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_path = root / "market.parquet"
    cftc_path = root / "cftc.parquet"
    if refresh or not market_path.exists():
        market = download_market_data()
        temporary = market_path.with_suffix(".pending.parquet")
        market.to_parquet(temporary, index=False)
        temporary.replace(market_path)
    else:
        market = pd.read_parquet(market_path)
    if refresh or not cftc_path.exists():
        cftc = download_cftc()
        temporary = cftc_path.with_suffix(".pending.parquet")
        cftc.to_parquet(temporary, index=False)
        temporary.replace(cftc_path)
    else:
        cftc = pd.read_parquet(cftc_path)
    return market, cftc
