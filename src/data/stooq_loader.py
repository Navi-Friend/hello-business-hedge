from __future__ import annotations

from pathlib import Path
from typing import Iterable
import io
import logging
import os
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_prices_stooq(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    base_urls = _stooq_base_urls()
    apikey = os.getenv("STOOQ_API_KEY", "").strip()
    for ticker in tickers:
        symbol = ticker.lower()
        if "." not in symbol:
            symbol = f"{symbol}.us"
        text = ""
        for base_url in base_urls:
            url = f"{base_url}?s={symbol}&i=d&lang=en"
            if apikey:
                url = f"{url}&apikey={apikey}"
            try:
                req = Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
                        "Accept": "text/csv, text/plain;q=0.9,*/*;q=0.8",
                    },
                )
                with urlopen(req, timeout=20) as resp:
                    raw = resp.read()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                logger.warning("Stooq download failed for %s: %s", ticker, exc)
                continue
            text = raw.decode("utf-8", errors="replace")
            first_line = text.splitlines()[0] if text else ""
            # Accept both English (Date, Close) and Polish (Data, Zamkniecie) headers
            if "Date" in first_line or "Data" in first_line:
                break
            # Stooq may return a landing page asking for an apikey instead of CSV.
            lower = first_line.lower()
            if "apikey" in lower or "api key" in lower or "uzyskaj" in lower or "get your" in lower:
                if apikey:
                    logger.warning(
                        "Stooq still asks apikey for %s even though STOOQ_API_KEY is set (not logging key)",
                        ticker,
                    )
                else:
                    logger.warning(
                        "Stooq requires apikey for %s; set STOOQ_API_KEY (not stored in repo)",
                        ticker,
                    )
                text = ""
                break

            # Some networks/proxies can return an HTML landing page instead of CSV.
            # Treat any non-CSV response as a transient failure and try the next base URL.
            logger.warning(
                "Stooq response not CSV for %s; first line=%s",
                ticker,
                first_line[:120],
            )
            text = ""
        if not text:
            continue
        df = _parse_stooq_csv(text)
        if df.empty or "Date" not in df.columns:
            logger.warning("Stooq returned no data for %s", ticker)
            continue
        df = df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        df = df[(df["date"] >= start_ts) & (df["date"] < end_ts)]
        if df.empty:
            logger.warning("Stooq data out of range for %s", ticker)
            continue
        df["ticker"] = ticker
        df["volume"] = df.get("volume", 0)
        frames.append(df[["date", "ticker", "close", "volume"]])
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume"])
    return pd.concat(frames, ignore_index=True)


def _stooq_base_urls() -> list[str]:
    override = os.getenv("STOOQ_BASE_URLS", "").strip()
    if override:
        return [u.strip().rstrip("/") + "/q/d/l/" for u in override.split(",") if u.strip()]
    return [
        "https://stooq.pl/q/d/l/",
        "https://stooq.com/q/d/l/",
    ]


def _parse_stooq_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=",", engine="python", on_bad_lines="skip")
    if "Date" in df.columns:
        return df
    # Handle Polish headers
    if "Data" in df.columns:
        df = df.rename(columns={
            "Data": "Date",
            "Zamkniecie": "Close",
            "Wolumen": "Volume",
        })
        return df
    # Try semicolon separator
    if ";" in (text.splitlines()[0] if text else ""):
        df = pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
        if "Data" in df.columns:
            df = df.rename(columns={
                "Data": "Date",
                "Zamkniecie": "Close",
                "Wolumen": "Volume",
            })
    return df


def fetch_fundamentals(tickers: Iterable[str]) -> pd.DataFrame:
    # Stooq daily endpoint does not provide fundamentals.
    # We keep the schema so Spark feature pipeline can run unchanged.
    return pd.DataFrame.from_records(
        [
            {
                "ticker": t,
                "market_cap": None,
                "trailing_pe": None,
                "price_to_book": None,
                "beta": None,
            }
            for t in tickers
        ]
    )


def build_market_proxy(prices: pd.DataFrame, market_ticker: str) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume"])

    market = (
        prices.groupby("date", as_index=False)
        .agg(close=("close", "mean"), volume=("volume", "sum"))
        .assign(ticker=market_ticker)
    )
    return market[["date", "ticker", "close", "volume"]]


def ensure_parquet(path: str, df: pd.DataFrame) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        target,
        index=False,
        engine="pyarrow",
        coerce_timestamps="ms",
        allow_truncated_timestamps=True,
    )
    return target
