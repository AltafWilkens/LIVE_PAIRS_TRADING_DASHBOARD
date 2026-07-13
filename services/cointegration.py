"""
COINTEGRATION - Shared statistical tests for pairs trading.

Extracted from main.py so the FastAPI backend and the pair screener use
the exact same Engle-Granger implementation. Two-step methodology:

  1. Regress y on x (OLS) to estimate the hedge ratio (beta).
  2. Run an Augmented Dickey-Fuller test on the regression residuals
     (the "spread") to test for a unit root. Rejecting the null
     (p < 0.05) means the spread is stationary, i.e. the pair is
     cointegrated.

Reference: Engle & Granger (1987), "Co-integration and Error
Correction: Representation, Estimation and Testing".
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


def engle_granger_test(df: pd.DataFrame, t1: str, t2: str) -> Dict[str, Any]:
    """
    Run the Engle-Granger two-step cointegration test on a pair of
    aligned price series.

    Parameters
    ----------
    df : DataFrame with columns [t1, t2], aligned on a common index.
    t1, t2 : column names of the two tickers. y = t1, x = t2.

    Returns
    -------
    dict with hedge_ratio, spread, is_cointegrated, adf_pvalue,
    adf_statistic, spread_mean, spread_std.
    """
    y = df[t1]
    X = df[t2]

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # Use positional indexing - robust across statsmodels/pandas versions.
    # params is [const, slope] when X had exactly one regressor.
    hedge_ratio = model.params.iloc[1]
    spread = y - hedge_ratio * X

    adf_result = adfuller(spread, autolag="AIC", regression="c")

    return {
        "hedge_ratio": float(hedge_ratio),
        "spread": spread,
        "is_cointegrated": bool(adf_result[1] < 0.05),
        "adf_pvalue": float(adf_result[1]),
        "adf_statistic": float(adf_result[0]),
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std()),
    }


def johansen_test(
    df: pd.DataFrame, t1: str, t2: str, det_order: int = 0, k_ar_diff: int = 1
) -> Dict[str, Any]:
    """
    Run the Johansen trace test for cointegration on a pair of aligned
    price series, and derive a hedge ratio from the leading cointegrating
    vector.

    Unlike Engle-Granger (which regresses one series on the other and is
    therefore direction-dependent), Johansen treats both series
    symmetrically via a VECM and can in principle detect multiple
    cointegrating relationships. For a two-asset pair there is at most one,
    so we use the eigenvector associated with the largest eigenvalue
    (``evec[:, 0]``) as the cointegrating relationship.

    Parameters
    ----------
    df : DataFrame with columns [t1, t2], aligned on a common index.
    t1, t2 : column names of the two tickers.
    det_order : deterministic trend order passed to coint_johansen
        (0 = constant term in the cointegrating relation, the standard
        choice for price-level pairs trading).
    k_ar_diff : number of lagged differences in the underlying VECM.

    Returns
    -------
    dict shaped like engle_granger_test's output (hedge_ratio, spread,
    is_cointegrated, spread_mean, spread_std) plus Johansen-specific
    trace_stat and trace_crit_90_95_99, so callers that only need the
    common fields (e.g. the backtester) can treat both tests
    interchangeably.
    """
    endog = df[[t1, t2]].to_numpy()

    result = coint_johansen(endog, det_order, k_ar_diff)

    # Leading cointegrating vector (largest eigenvalue) normalized so the
    # t1 weight is 1: y - hedge_ratio * x is stationary, matching the
    # engle_granger_test spread convention.
    evec = result.evec[:, 0]
    hedge_ratio = -evec[1] / evec[0]

    spread = df[t1] - hedge_ratio * df[t2]

    trace_stat = float(result.lr1[0])
    trace_crit_90_95_99 = tuple(float(c) for c in result.cvt[0])

    return {
        "hedge_ratio": float(hedge_ratio),
        "spread": spread,
        "is_cointegrated": bool(trace_stat > trace_crit_90_95_99[1]),  # 95% level
        "trace_stat": trace_stat,
        "trace_crit_90_95_99": trace_crit_90_95_99,
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std()),
    }


def engle_granger_pvalue_only(series1: pd.Series, series2: pd.Series) -> float:
    """
    Fast cointegration p-value using statsmodels' built-in coint() test,
    for use in the screener where we only need to rank pairs and don't
    need the full spread/hedge-ratio output yet.

    statsmodels.tsa.stattools.coint runs the same Engle-Granger procedure
    but is more convenient for batch screening since it returns the
    p-value directly without needing to manage the OLS fit ourselves.
    """
    try:
        _, pvalue, _ = coint(series1, series2)
        return float(pvalue)
    except Exception as e:
        logger.warning(f"Cointegration test failed: {e}")
        return 1.0  # treat failures as "not cointegrated"


def calculate_half_life(spread: pd.Series) -> float:
    """
    Estimate the half-life of mean reversion via discrete AR(1) proxy:

        spread_diff[t] = alpha + lambda * spread_lag[t-1] + error[t]

    half_life = -ln(2) / lambda

    This is a standard, fast approximation. For a more rigorous
    continuous-time estimate, see services/ou_process.py which fits an
    explicit Ornstein-Uhlenbeck process to the same spread.
    """
    try:
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        common_idx = spread_lag.index.intersection(spread_diff.index)
        if len(common_idx) < 5:
            return 999.0

        spread_lag = spread_lag.loc[common_idx]
        spread_diff = spread_diff.loc[common_idx]

        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_diff, X).fit()

        lambda_val = model.params.iloc[1]

        if lambda_val >= 0:
            return 999.0  # no mean reversion

        half_life = -np.log(2) / lambda_val
        return min(half_life, 999.0)

    except Exception as e:
        logger.warning(f"Half-life calculation failed: {e}")
        return 999.0
