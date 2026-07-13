"""
FASTAPI BACKEND - JSE PAIRS TRADING (ROBUST VERSION)
NWU Applied Mathematics Quant Project
Handles ALL yfinance edge cases gracefully
"""

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import uvicorn
import logging
import warnings
warnings.filterwarnings('ignore')

from services.backtester import BacktestConfig, run_backtest
from services.data_fetcher import fetch_pair_data, fetch_price_series
from services.pair_screener import screen_pairs
from config.universe import SECTOR_GROUPS, STRUCTURAL_PAIRS

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

class ScreenedPair(BaseModel):
    ticker1: str
    ticker2: str
    sector: str
    correlation: float
    adf_pvalue: float
    is_cointegrated: bool
    half_life_days: float
    data_points: int

class ScreenResponse(BaseModel):
    pairs: list[ScreenedPair]
    pairs_tested: int
    pairs_found: int
    timestamp: datetime

class BacktestRequest(BaseModel):
    ticker1: str
    ticker2: str
    lookback_days: int = Field(default=1500, gt=0)
    coint_method: Literal["engle_granger", "johansen"] = "engle_granger"
    entry_zscore: float = Field(default=2.0, gt=0)
    exit_zscore: float = Field(default=0.5, ge=0)
    lookback_window: int = Field(default=252, gt=0)
    reestimate_every: int = Field(default=21, gt=0)
    zscore_window: int = Field(default=30, gt=0)
    require_cointegration: bool = True
    transaction_cost_bps: float = Field(default=5.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    capital: float = Field(default=100_000.0, gt=0)

class EquityPoint(BaseModel):
    date: str
    equity: float

class ZScorePoint(BaseModel):
    date: str
    zscore: Optional[float] = None

class TradeRecord(BaseModel):
    entry_date: str
    exit_date: str
    direction: int
    pnl: float

class BacktestResponse(BaseModel):
    ticker1: str
    ticker2: str
    equity_curve: list[EquityPoint]
    zscore_series: list[ZScorePoint]
    trades: list[TradeRecord]
    metrics: Dict[str, float]

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
            "/spread": "POST - Calculate spread and z-score for a given pair",
            "/screen": "GET - Scan the JSE universe for cointegrated pairs",
            "/health": "GET - API health check",
            "/test": "GET - Test data fetching",
            "/test_pair": "POST - Test a specific pair",
            "/backtest": "POST - Run a walk-forward backtest for a pair",
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/screen", response_model=ScreenResponse)
def screen_endpoint(lookback_days: int = 504, correlation_threshold: float = 0.7):
    """
    Scan the curated JSE ticker universe (config/universe.py) for
    cointegrated pairs. Runs a correlation pre-filter within each
    sector group, then Engle-Granger cointegration on the survivors.

    This can take a while on first call since it batch-downloads the
    full universe from yfinance. Results aren't cached between calls
    in this version -- re-running re-downloads.
    """
    try:
        results_df = screen_pairs(
            SECTOR_GROUPS,
            STRUCTURAL_PAIRS,
            lookback_days=lookback_days,
            correlation_threshold=correlation_threshold,
        )

        # candidate count for context, mirroring screen_pairs' internal logic
        from services.pair_screener import _candidate_pairs
        pairs_tested = len(_candidate_pairs(SECTOR_GROUPS, STRUCTURAL_PAIRS))

        pairs = [ScreenedPair(**row) for row in results_df.to_dict(orient="records")]

        return ScreenResponse(
            pairs=pairs,
            pairs_tested=pairs_tested,
            pairs_found=len(pairs),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Screening error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/backtest", response_model=BacktestResponse)
def backtest_endpoint(request: BacktestRequest):
    """
    Run a walk-forward backtest for a pair: rolling hedge-ratio
    re-estimation, z-score entry/exit, transaction costs and slippage.
    Delegates entirely to services.backtester.run_backtest.
    """
    try:
        prices = fetch_pair_data(request.ticker1, request.ticker2, request.lookback_days)

        config = BacktestConfig(
            entry_zscore=request.entry_zscore,
            exit_zscore=request.exit_zscore,
            lookback_window=request.lookback_window,
            reestimate_every=request.reestimate_every,
            zscore_window=request.zscore_window,
            coint_method=request.coint_method,
            require_cointegration=request.require_cointegration,
            transaction_cost_bps=request.transaction_cost_bps,
            slippage_bps=request.slippage_bps,
            capital=request.capital,
        )

        result = run_backtest(prices, request.ticker1, request.ticker2, config)

        equity_curve = [
            EquityPoint(date=idx.isoformat(), equity=float(value))
            for idx, value in result.equity_curve.items()
        ]
        zscore_series = [
            ZScorePoint(date=idx.isoformat(), zscore=(float(value) if pd.notna(value) else None))
            for idx, value in result.zscore_series.items()
        ]
        trades = [
            TradeRecord(
                entry_date=row.entry_date.isoformat(),
                exit_date=row.exit_date.isoformat(),
                direction=int(row.direction),
                pnl=float(row.pnl),
            )
            for row in result.trades.itertuples(index=False)
        ]

        return BacktestResponse(
            ticker1=request.ticker1,
            ticker2=request.ticker2,
            equity_curve=equity_curve,
            zscore_series=zscore_series,
            trades=trades,
            metrics=result.metrics,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
        # reload=True removed
    )
