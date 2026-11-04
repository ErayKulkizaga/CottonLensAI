from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TICKERS = {"cotton": "CT=F", "dxy": "DX-Y.NYB", "wti": "CL=F"}
MARKET_BAR_FIELDS = ["open", "high", "low", "close", "volume"]
CFTC_MARKET_CODE = "033661"
YAHOO_ATTEMPTS = 4
YAHOO_RETRY_DELAYS_SECONDS = (10, 30, 60)


class MarketDownloadError(RuntimeError):
    """A source failed after visible, bounded retries; no partial dataset is returned."""


def _normalize_market_bars(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if raw.columns.nlevels != 2 or set(raw.columns.get_level_values(1)) != {ticker}:
            raise ValueError("response ticker does not match the requested ticker")
        raw = raw.droplevel(1, axis="columns")
    bars = raw.rename(columns=str.lower)
    missing = set(MARKET_BAR_FIELDS) - set(bars.columns)
    if missing or not bars.columns.is_unique:
        raise ValueError(f"response has missing or duplicate bar fields (missing: {sorted(missing)})")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("daily bars require a date index")
    dates = bars.index.tz_localize(None)
    if dates.hasnans or dates.has_duplicates or not dates.equals(dates.normalize()):
        raise ValueError("daily bar dates must be valid, unique and normalized")

    close = pd.to_numeric(bars["close"], errors="coerce")
    usable_close = np.isfinite(close)
    if ticker != TICKERS["wti"]:
        usable_close &= close > 0
    if not usable_close.any():
        raise ValueError("response has no usable close prices")

    bars = bars[MARKET_BAR_FIELDS].copy()
    bars.insert(0, "date", dates)
    return bars.reset_index(drop=True)


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
                return _normalize_market_bars(raw, ticker)
            failures.append("empty response (Yahoo may have rate-limited this request)")
        except Exception as exc:  # noqa: BLE001 - preserve provider failure for bounded retry
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
        "If rate-limited, wait 5–10 minutes and rerun only the data cell. "
        f"Details: {' | '.join(failures)}"
    )


def download_market_data(start: str = "2010-01-01") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series, ticker in TICKERS.items():
        bars = _download_ticker_with_retry(ticker, start)
        bars.insert(1, "series", series)
        frames.append(bars)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "series"])


def _normalized_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def download_cftc(start_year: int = 2010, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or datetime.now(UTC).year
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
    from cottonlens_ml.snapshots import write_snapshot

    root.mkdir(parents=True, exist_ok=True)
    market_path = root / "market.parquet"
    cftc_path = root / "cftc.parquet"
    # Preserve the previous cache before a refresh can replace it. An incomplete
    # old source set is still evidence and is labelled accordingly.
    if refresh and (market_path.exists() or cftc_path.exists()):
        prior = {
            name: pd.read_parquet(path)
            for name, path in (("market", market_path), ("cftc", cftc_path)) if path.exists()
        }
        write_snapshot(root / "snapshots", prior, {"kind": "pre_refresh_source_cache"})
    if refresh or not market_path.exists():
        market = download_market_data()
        temporary = market_path.with_suffix(".pending.parquet")
        market.to_parquet(temporary, index=False)
        temporary.replace(market_path)
    else:
        market = pd.read_parquet(market_path)
    if refresh or not cftc_path.exists():
        try:
            cftc = download_cftc()
            temporary = cftc_path.with_suffix(".pending.parquet")
            cftc.to_parquet(temporary, index=False)
            temporary.replace(cftc_path)
        except (OSError, ValueError, requests.RequestException, zipfile.BadZipFile) as exc:
            print(f"CFTC audit-only source unavailable: {exc}; model inputs exclude CFTC", flush=True)
            cftc = pd.read_parquet(cftc_path) if cftc_path.exists() else pd.DataFrame(
                columns=["available_date", "cftc_managed_money_net"]
            )
    else:
        cftc = pd.read_parquet(cftc_path)
    write_snapshot(root / "snapshots", {"market": market, "cftc": cftc}, {"kind": "source_cache"})
    return market, cftc
