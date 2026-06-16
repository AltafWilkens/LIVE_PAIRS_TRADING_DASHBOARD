"""
FASTAPI BACKEND - JSE PAIRS TRADING (ROBUST VERSION)
NWU Applied Mathematics Quant Project
Handles ALL yfinance edge cases gracefully
"""

import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import uvicorn
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JSE Pairs Trading API",
    description="Statistical arbitrage engine for South African equities",
    version="2.0.0"
)

# Add CORS middleware (allows Streamlit to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PairRequest(BaseModel):
    ticker1: str
    ticker2: str
    lookback_days: int = 252
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5

class SpreadResponse(BaseModel):
    ticker1: str
    ticker2: str
    current_spread: float
    current_zscore: float
    hedge_ratio: float
    half_life_days: float
    signal: str
    timestamp: datetime
    is_cointegrated: bool
    adf_pvalue: float
    spread_mean: float
    spread_std: float
    data_points: int

# -------------------- ROBUST DATA FETCHING --------------------
def fetch_price_series(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Fetch price series for a single ticker with robust error handling.
    Returns a pandas Series with proper index.
    """
    try:
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

        # Download with multiple attempts
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,  # Avoid threading issues
            prepost=False   # Only trading hours
        )

        # Check if data is empty
        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.Series(dtype=float)

        # Try different column names in order of preference
        price_col = None
        for col in ['Adj Close', 'Close', 'adjclose', 'close', 'Price']:
            if col in data.columns:
                price_col = col
                break

        if price_col is None:
            # If no standard column, use the first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                logger.info(f"Using fallback column: {price_col} for {ticker}")
            else:
                logger.warning(f"No numeric columns found for {ticker}")
                return pd.Series(dtype=float)

        # Extract series and ensure it's a Series with proper index
        series = data[price_col]

        # Ensure it's a Series (not DataFrame)
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        # Drop NaN values
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

    # Format dates as strings (yyyy-mm-dd)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Fetch both series
    series1 = fetch_price_series(ticker1, start_str, end_str)
    series2 = fetch_price_series(ticker2, start_str, end_str)

    # Check if we got data
    if series1.empty:
        raise ValueError(f"No data available for {ticker1}. Please check the ticker symbol.")
    if series2.empty:
        raise ValueError(f"No data available for {ticker2}. Please check the ticker symbol.")

    # Create DataFrame with aligned data
    df = pd.DataFrame({
        ticker1: series1,
        ticker2: series2
    }).dropna()

    # Ensure we have enough data
    if len(df) < 20:
        raise ValueError(
            f"Insufficient overlapping data: only {len(df)} rows. "
            f"Check if both tickers are active during this period."
        )

    logger.info(f"Aligned data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    return df

# -------------------- COINTEGRATION AND STATISTICS --------------------
def calculate_cointegration(df: pd.DataFrame, t1: str, t2: str) -> Dict[str, Any]:
    """
    Calculate cointegration statistics using Engle-Granger method.
    """
    try:
        X = df[t2]
        y = df[t1]

        # Add constant for OLS
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        hedge_ratio = model.params[t2]
        spread = y - hedge_ratio * X

        # ADF test on spread
        adf_result = adfuller(spread, autolag='AIC', regression='c')

        return {
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'is_cointegrated': adf_result[1] < 0.05,
            'adf_pvalue': adf_result[1],
            'adf_statistic': adf_result[0],
            'spread_mean': spread.mean(),
            'spread_std': spread.std()
        }
    except Exception as e:
        logger.error(f"Cointegration error: {str(e)}")
        raise

def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate half-life of mean reversion.
    """
    try:
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align by index
        common_idx = spread_lag.index.intersection(spread_diff.index)
        if len(common_idx) < 5:
            return 999.0

        spread_lag = spread_lag.loc[common_idx]
        spread_diff = spread_diff.loc[common_idx]

        # OLS regression: spread_diff = alpha + lambda * spread_lag + error
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_diff, X).fit()

        lambda_val = model.params.iloc[1] if len(model.params) > 1 else model.params[spread_lag.name]

        if lambda_val >= 0:
            return 999.0  # No mean reversion

        half_life = -np.log(2) / lambda_val
        return min(half_life, 999.0)  # Cap at 999

    except Exception as e:
        logger.error(f"Half-life error: {str(e)}")
        return 999.0

def generate_signal(zscore: float, entry: float, exit: float, is_cointegrated: bool) -> str:
    """
    Generate trading signal based on z-score and thresholds.
    """
    if not is_cointegrated:
        return "NEUTRAL"

    if zscore > entry:
        return "SHORT_SPREAD"
    elif zscore < -entry:
        return "LONG_SPREAD"
    elif abs(zscore) < exit:
        return "NEUTRAL"
    else:
        return "NEUTRAL"

# -------------------- API ENDPOINTS --------------------
@app.get("/")
def root():
    return {
        "message": "JSE Pairs Trading API v2.0",
        "status": "running",
        "endpoints": {
            "/spread": "POST - Calculate spread and z-score",
            "/health": "GET - API health check",
            "/test": "GET - Test data fetching",
            "/test_pair": "POST - Test a specific pair"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test")
def test_endpoint():
    """
    Quick test endpoint to verify data fetching works.
    """
    results = {}
    test_tickers = ["NPN.JO", "PRX.JO", "AAPL", "MSFT"]

    for ticker in test_tickers:
        try:
            series = fetch_price_series(ticker, "2025-01-01", "2026-06-01")
            results[ticker] = {
                "status": "success",
                "rows": len(series),
                "first_date": series.index[0].isoformat() if len(series) > 0 else None,
                "last_date": series.index[-1].isoformat() if len(series) > 0 else None,
                "first_price": float(series.iloc[0]) if len(series) > 0 else None,
                "last_price": float(series.iloc[-1]) if len(series) > 0 else None
            }
        except Exception as e:
            results[ticker] = {"status": "error", "message": str(e)}

    return results

@app.post("/test_pair")
def test_pair_endpoint(request: PairRequest):
    """
    Test endpoint for a specific pair - returns raw data for debugging.
    """
    try:
        df = fetch_pair_data(request.ticker1, request.ticker2, request.lookback_days)
        coint = calculate_cointegration(df, request.ticker1, request.ticker2)

        return {
            "ticker1": request.ticker1,
            "ticker2": request.ticker2,
            "data_points": len(df),
            "date_range": {
                "start": df.index[0].isoformat(),
                "end": df.index[-1].isoformat()
            },
            "hedge_ratio": float(coint['hedge_ratio']),
            "is_cointegrated": coint['is_cointegrated'],
            "adf_pvalue": float(coint['adf_pvalue']),
            "spread_mean": float(coint['spread_mean']),
            "spread_std": float(coint['spread_std']),
            "current_spread": float(coint['spread'].iloc[-1]),
            "current_zscore": float((coint['spread'].iloc[-1] - coint['spread_mean']) / coint['spread_std']),
            "sample_data": df.tail(5).to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/spread", response_model=SpreadResponse)
def calculate_spread_endpoint(request: PairRequest):
    """
    Main endpoint: Calculate current spread, z-score, and generate trade signal.
    """
    try:
        # Step 1: Fetch data
        df = fetch_pair_data(request.ticker1, request.ticker2, request.lookback_days)

        # Step 2: Calculate cointegration
        coint = calculate_cointegration(df, request.ticker1, request.ticker2)

        # Step 3: Calculate current statistics
        current_spread = coint['spread'].iloc[-1]
        current_zscore = (current_spread - coint['spread_mean']) / coint['spread_std']

        # Step 4: Calculate half-life
        half_life = calculate_half_life(coint['spread'])

        # Step 5: Generate signal
        signal = generate_signal(
            current_zscore,
            request.entry_zscore,
            request.exit_zscore,
            coint['is_cointegrated']
        )

        return SpreadResponse(
            ticker1=request.ticker1,
            ticker2=request.ticker2,
            current_spread=float(current_spread),
            current_zscore=float(current_zscore),
            hedge_ratio=float(coint['hedge_ratio']),
            half_life_days=float(half_life),
            signal=signal,
            timestamp=datetime.now(),
            is_cointegrated=coint['is_cointegrated'],
            adf_pvalue=float(coint['adf_pvalue']),
            spread_mean=float(coint['spread_mean']),
            spread_std=float(coint['spread_std']),
            data_points=len(df)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
        # reload=True removed
    )
