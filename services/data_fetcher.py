"""
DATA FETCHER - yfinance wrapper for pairs trading.

Extracted from main.py so the FastAPI backend and the backtester CLI use
the exact same, already-hardened price-fetching logic instead of each
maintaining their own copy.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_price_series(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch price series for a single ticker with robust error handling.
    Returns a pandas Series with proper index.
    """
    try:
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,  # Avoid threading issues
            prepost=False,  # Only trading hours
        )

        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.Series(dtype=float)

        price_col = None
        for col in ["Adj Close", "Close", "adjclose", "close", "Price"]:
            if col in data.columns:
                price_col = col
                break

        if price_col is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                logger.info(f"Using fallback column: {price_col} for {ticker}")
            else:
                logger.warning(f"No numeric columns found for {ticker}")
                return pd.Series(dtype=float)

        series = data[price_col]

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        series = series.dropna()

        logger.info(f"Successfully fetched {len(series)} rows for {ticker}")
        return series

    except Exception as e:
        logger.error(f"Error fetching {ticker}: {str(e)}")
        return pd.Series(dtype=float)


def fetch_pair_data(ticker1: str, ticker2: str, lookback_days: int) -> pd.DataFrame:
    """
    Fetch aligned price data for a pair of tickers.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    series1 = fetch_price_series(ticker1, start_str, end_str)
    series2 = fetch_price_series(ticker2, start_str, end_str)

    if series1.empty:
        raise ValueError(f"No data available for {ticker1}. Please check the ticker symbol.")
    if series2.empty:
        raise ValueError(f"No data available for {ticker2}. Please check the ticker symbol.")

    df = pd.DataFrame({
        ticker1: series1,
        ticker2: series2,
    }).dropna()

    if len(df) < 20:
        raise ValueError(
            f"Insufficient overlapping data: only {len(df)} rows. "
            f"Check if both tickers are active during this period."
        )

    logger.info(f"Aligned data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    return df
