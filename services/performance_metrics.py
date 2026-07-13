"""
PERFORMANCE METRICS - Pure functions over an equity curve / returns series.

Deliberately independent of services/backtester.py so these are trivial to
unit test in isolation against hand-computed values, and reusable for any
other equity curve (e.g. live paper-trading P&L) later.
"""

from typing import Sequence

import numpy as np
import pandas as pd


def total_return(equity_curve: pd.Series) -> float:
    """Fractional growth from the first to the last equity value."""
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate implied by the equity curve's length."""
    num_periods = len(equity_curve) - 1
    if num_periods <= 0:
        return 0.0
    growth = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if growth <= 0:
        # Equity went to zero or negative (e.g. an unbounded mark-to-market
        # loss on a still-open position) -- growth**(1/years) on a
        # non-positive base is undefined/NaN in the real numbers, so report
        # a full loss explicitly instead of silently propagating NaN.
        return -1.0
    years = num_periods / periods_per_year
    return float(growth ** (1.0 / years) - 1.0)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Worst peak-to-trough decline, expressed as a negative fraction (or 0.0)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def annualized_sharpe(
    returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> float:
    """
    Annualized Sharpe ratio of a per-period returns series.

    risk_free_rate is a per-period rate already matching the returns'
    frequency (e.g. a daily risk-free return if returns are daily).
    """
    excess = returns - risk_free_rate
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def annualized_sortino(
    returns: pd.Series, periods_per_year: int = 252, target: float = 0.0
) -> float:
    """
    Annualized Sortino ratio: like Sharpe, but only penalizes downside
    deviation below `target` rather than volatility in either direction.
    """
    excess = returns - target
    downside = np.minimum(excess, 0.0)
    downside_deviation = np.sqrt(np.mean(downside**2))
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return 0.0
    return float((excess.mean() / downside_deviation) * np.sqrt(periods_per_year))


def win_rate(trade_pnls: Sequence[float]) -> float:
    """Fraction of trades with strictly positive P&L. 0.0 if there were no trades."""
    if len(trade_pnls) == 0:
        return 0.0
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    return wins / len(trade_pnls)


def avg_trade_pnl(trade_pnls: Sequence[float]) -> float:
    """Mean P&L per trade. 0.0 if there were no trades."""
    if len(trade_pnls) == 0:
        return 0.0
    return float(np.mean(trade_pnls))
