"""
data/fetcher.py
===============
Downloads and caches price data for the asset universe.

Design decisions:
  - Uses yfinance (free, covers all our ETF universe)
  - Caches to CSV so you don't hammer the API on every run
  - Cache invalidation: if today's date > last cached date, re-fetch
  - Returns Adjusted Close prices (accounts for dividends & splits)
  - Handles missing data with forward-fill then back-fill (standard practice)
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config import UNIVERSE, DATA_START_DATE, CACHE_DIR, PRICE_FIELD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    """Returns the local CSV path for a given ticker."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return Path(CACHE_DIR) / f"{ticker}.csv"


def _cache_is_fresh(ticker: str, max_age_days: int = 1) -> bool:
    """Returns True if cached file exists and was updated within max_age_days."""
    p = _cache_path(ticker)
    if not p.exists():
        return False
    mod_time = datetime.fromtimestamp(p.stat().st_mtime)
    return (datetime.now() - mod_time) < timedelta(days=max_age_days)


def _read_cache(ticker: str) -> pd.DataFrame:
    p = _cache_path(ticker)
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df


def _write_cache(ticker: str, df: pd.DataFrame) -> None:
    df.to_csv(_cache_path(ticker))


# ─────────────────────────────────────────────
# SINGLE TICKER FETCH
# ─────────────────────────────────────────────

def fetch_ticker(
    ticker: str,
    start: str = DATA_START_DATE,
    force_refresh: bool = False,
) -> pd.Series:
    """
    Fetches Adjusted Close prices for a single ticker.

    Returns a pd.Series indexed by date, named by ticker.
    Uses cache if available and fresh; downloads otherwise.
    """
    if not force_refresh and _cache_is_fresh(ticker):
        logger.debug(f"Cache hit: {ticker}")
        df = _read_cache(ticker)
        return df["price"].rename(ticker)

    logger.info(f"Fetching {ticker} from {start} ...")
    try:
        raw = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=True,   # Adjusted Close
        )
        if raw.empty:
            raise ValueError(f"No data returned for {ticker}")

        prices = raw["Close"].squeeze()
        prices.name = ticker
        prices.index = pd.to_datetime(prices.index)

        # Cache it
        _write_cache(ticker, prices.rename("price").to_frame())
        return prices

    except Exception as e:
        logger.warning(f"Failed to fetch {ticker}: {e}. Trying cache fallback.")
        if _cache_path(ticker).exists():
            df = _read_cache(ticker)
            return df["price"].rename(ticker)
        raise


# ─────────────────────────────────────────────
# FULL UNIVERSE FETCH
# ─────────────────────────────────────────────

def fetch_universe(
    tickers: list = None,
    start: str = DATA_START_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetches prices for all tickers in the universe.

    Returns a wide DataFrame:
        - Index: DatetimeIndex (business days)
        - Columns: tickers
        - Values: Adjusted Close prices

    Missing values are forward-filled then back-filled.
    Any ticker with >10% missing data after fill raises a warning.
    """
    if tickers is None:
        tickers = list(UNIVERSE.keys())

    price_dict = {}
    failed = []

    for ticker in tickers:
        try:
            s = fetch_ticker(ticker, start=start, force_refresh=force_refresh)
            price_dict[ticker] = s
        except Exception as e:
            logger.error(f"Skipping {ticker}: {e}")
            failed.append(ticker)

    if not price_dict:
        logger.warning(
            "No live price data retrieved. "
            "Falling back to synthetic data (set force_refresh=True to retry live fetch)."
        )
        from data.synthetic import fetch_universe_synthetic
        return fetch_universe_synthetic(tickers=tickers, start=start)

    prices = pd.DataFrame(price_dict)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # ── Handle missing data ───────────────────
    # Forward-fill first (most recent valid price), then back-fill for leading NaNs
    prices = prices.ffill().bfill()

    # Data quality check
    missing_pct = prices.isnull().mean() * 100
    bad_cols = missing_pct[missing_pct > 10]
    if not bad_cols.empty:
        logger.warning(
            f"High missing data (>10%) after fill:\n{bad_cols.to_string()}"
        )

    if failed:
        logger.warning(f"Failed tickers (excluded): {failed}")

    logger.info(
        f"Universe loaded: {len(prices.columns)} tickers, "
        f"{len(prices)} trading days "
        f"({prices.index[0].date()} → {prices.index[-1].date()})"
    )
    return prices


# ─────────────────────────────────────────────
# RETURN COMPUTATION
# ─────────────────────────────────────────────

def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """
    Computes daily returns from a price DataFrame.

    method:
        "log"       — log returns: ln(P_t / P_{t-1})     [preferred for risk math]
        "simple"    — simple returns: (P_t / P_{t-1}) - 1

    Log returns are additive across time; simple returns are additive across assets.
    For VaR and ES we use log returns. For P&L attribution we use simple returns.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    elif method == "simple":
        returns = prices.pct_change().dropna()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'log' or 'simple'.")

    logger.info(
        f"Returns computed ({method}): "
        f"{returns.shape[0]} observations × {returns.shape[1]} assets"
    )
    return returns


# ─────────────────────────────────────────────
# UNIVERSE METADATA HELPERS
# ─────────────────────────────────────────────

def get_asset_classes(tickers: list = None) -> pd.Series:
    """Returns a Series mapping ticker → asset class."""
    if tickers is None:
        tickers = list(UNIVERSE.keys())
    return pd.Series(
        {t: UNIVERSE[t]["asset_class"] for t in tickers if t in UNIVERSE},
        name="asset_class"
    )


def get_display_names(tickers: list = None) -> dict:
    """Returns a dict mapping ticker → human-readable name."""
    if tickers is None:
        tickers = list(UNIVERSE.keys())
    return {t: UNIVERSE[t]["name"] for t in tickers if t in UNIVERSE}


if __name__ == "__main__":
    # Quick smoke test
    prices = fetch_universe()
    print("\nPrice DataFrame:")
    print(prices.tail(3).to_string())

    log_rets = compute_returns(prices, method="log")
    print(f"\nLog return stats (annualised vol):")
    print((log_rets.std() * np.sqrt(252)).round(4).to_string())
