from __future__ import annotations

import io
import re
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

TICKERS = {"cotton": "CT=F", "dxy": "DX-Y.NYB", "wti": "CL=F"}
CFTC_MARKET_CODE = "033661"


def download_market_data(start: str = "2010-01-01") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series, ticker in TICKERS.items():
        raw = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
        if raw.empty:
            raise RuntimeError(f"no data returned for {ticker}")
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
        market.to_parquet(market_path, index=False)
    else:
        market = pd.read_parquet(market_path)
    if refresh or not cftc_path.exists():
        cftc = download_cftc()
        cftc.to_parquet(cftc_path, index=False)
    else:
        cftc = pd.read_parquet(cftc_path)
    return market, cftc

