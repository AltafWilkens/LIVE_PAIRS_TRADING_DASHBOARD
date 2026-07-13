"""
BACKTESTER - Walk-forward pairs-trading engine.

Pipeline (see run_backtest):
  1. Walk-forward hedge ratio: re-fit the cointegration relationship on a
     trailing lookback window every `reestimate_every` days. Each refit only
     ever sees data strictly before the day it starts applying to, so later
     data can never change an earlier estimate (no lookahead bias).
  2. Rolling z-score of the resulting spread, using a trailing window of the
     spread's own mean/std (never full-sample statistics, which would also
     leak future information).
  3. A flat/long-spread/short-spread state machine on that z-score, gated by
     whether the active refit window passed its cointegration test.
  4. Next-bar execution: a signal formed at the close of day t is executed
     at day t+1, matching how a real strategy driven off daily closes would
     trade.
  5. PnL simulation: hedge ratio and position size are frozen at trade entry
     (a real trader does not rehedge mid-trade every time the rolling
     regression updates); transaction costs and slippage are charged as a
     combined bps-of-notional cost on both legs, at entry and at exit.
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from services.cointegration import engle_granger_test, johansen_test
from services.performance_metrics import (
    annualized_sharpe,
    annualized_sortino,
    avg_trade_pnl,
    cagr,
    max_drawdown,
    total_return,
    win_rate,
)

CointMethod = Literal["engle_granger", "johansen"]

_COINT_TEST_FNS = {
    "engle_granger": engle_granger_test,
    "johansen": johansen_test,
}


@dataclass(frozen=True)
class BacktestConfig:
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    lookback_window: int = 252
    reestimate_every: int = 21
    zscore_window: int = 30
    coint_method: CointMethod = "engle_granger"
    require_cointegration: bool = True
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    capital: float = 100_000.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    hedge_ratio_series: pd.Series
    zscore_series: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def _rolling_hedge_ratios(
    prices: pd.DataFrame, t1: str, t2: str, config: BacktestConfig
) -> Tuple[pd.Series, pd.Series]:
    """
    Re-fit the hedge ratio and cointegration test every `reestimate_every`
    days on the trailing `lookback_window` days. Both output series are
    piecewise-constant between refit points and NaN/False before the first
    refit point (not enough history yet).

    The window used for the refit applying from day i onward is
    prices.iloc[i - lookback_window : i] -- strictly data before day i, so
    appending more rows after day i can never change the value at day i.
    """
    n = len(prices)
    test_fn = _COINT_TEST_FNS[config.coint_method]

    hedge_ratio = pd.Series(np.nan, index=prices.index)
    is_cointegrated = pd.Series(False, index=prices.index)

    for i in range(config.lookback_window, n, config.reestimate_every):
        window = prices.iloc[i - config.lookback_window : i]
        result = test_fn(window, t1, t2)
        end = min(i + config.reestimate_every, n)
        hedge_ratio.iloc[i:end] = result["hedge_ratio"]
        is_cointegrated.iloc[i:end] = result["is_cointegrated"]

    return hedge_ratio, is_cointegrated


def _spread_and_zscore(
    prices: pd.DataFrame, t1: str, t2: str, hedge_ratio: pd.Series, config: BacktestConfig
) -> Tuple[pd.Series, pd.Series]:
    """Spread from the piecewise-constant hedge ratio, and its rolling z-score."""
    spread = prices[t1] - hedge_ratio * prices[t2]
    rolling_mean = spread.rolling(config.zscore_window).mean()
    rolling_std = spread.rolling(config.zscore_window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return spread, zscore


def _generate_positions(
    zscore: pd.Series, is_cointegrated: pd.Series, config: BacktestConfig
) -> pd.Series:
    """
    Flat/long-spread(+1)/short-spread(-1) state machine.

    Enters short when z > entry_zscore, long when z < -entry_zscore; exits
    to flat when |z| < exit_zscore. Forced flat (and blocked from entering)
    whenever the active refit window failed cointegration and
    require_cointegration=True, or the z-score isn't defined yet.
    """
    positions = pd.Series(0, index=zscore.index, dtype=int)
    state = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        coint_ok = (not config.require_cointegration) or bool(is_cointegrated.iloc[i])

        if pd.isna(z) or not coint_ok:
            state = 0
        elif state == 0:
            if z > config.entry_zscore:
                state = -1
            elif z < -config.entry_zscore:
                state = 1
        elif abs(z) < config.exit_zscore:
            state = 0

        positions.iloc[i] = state

    return positions


def _simulate_pnl(
    prices: pd.DataFrame,
    t1: str,
    t2: str,
    hedge_ratio: pd.Series,
    positions: pd.Series,
    config: BacktestConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Simulate daily equity given executed positions (signals shifted one day
    for next-bar execution). Hedge ratio and leg sizing are frozen at trade
    entry; transaction costs + slippage (combined into one bps-of-notional
    rate) are charged on both legs at entry and exit.
    """
    executed = positions.shift(1).fillna(0).astype(int)
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0

    equity = pd.Series(config.capital, index=prices.index, dtype=float)
    running_equity = config.capital
    open_trade = None
    trade_records = []

    for i in range(len(prices)):
        pos = int(executed.iloc[i])
        price1 = prices[t1].iloc[i]
        price2 = prices[t2].iloc[i]

        if open_trade is None and pos != 0:
            entry_hr = hedge_ratio.iloc[i]
            units = config.capital / (price1 + abs(entry_hr) * price2)
            entry_notional = units * price1 + units * abs(entry_hr) * price2
            entry_cost = cost_rate * entry_notional
            running_equity -= entry_cost
            open_trade = {
                "entry_date": prices.index[i],
                "direction": pos,
                "hr": entry_hr,
                "units": units,
                "prev_spread": price1 - entry_hr * price2,
                "accumulated_pnl": -entry_cost,
            }
        elif open_trade is not None:
            hr = open_trade["hr"]
            spread_now = price1 - hr * price2
            pnl_today = open_trade["direction"] * open_trade["units"] * (
                spread_now - open_trade["prev_spread"]
            )
            running_equity += pnl_today
            open_trade["accumulated_pnl"] += pnl_today
            open_trade["prev_spread"] = spread_now

            if pos == 0:
                exit_notional = open_trade["units"] * price1 + open_trade["units"] * abs(hr) * price2
                exit_cost = cost_rate * exit_notional
                running_equity -= exit_cost
                open_trade["accumulated_pnl"] -= exit_cost
                trade_records.append({
                    "entry_date": open_trade["entry_date"],
                    "exit_date": prices.index[i],
                    "direction": open_trade["direction"],
                    "pnl": open_trade["accumulated_pnl"],
                })
                open_trade = None

        equity.iloc[i] = running_equity

    if open_trade is not None:
        # End-of-backtest liquidation of any still-open position.
        last_price1 = prices[t1].iloc[-1]
        last_price2 = prices[t2].iloc[-1]
        hr = open_trade["hr"]
        exit_notional = open_trade["units"] * last_price1 + open_trade["units"] * abs(hr) * last_price2
        exit_cost = cost_rate * exit_notional
        running_equity -= exit_cost
        open_trade["accumulated_pnl"] -= exit_cost
        trade_records.append({
            "entry_date": open_trade["entry_date"],
            "exit_date": prices.index[-1],
            "direction": open_trade["direction"],
            "pnl": open_trade["accumulated_pnl"],
        })
        equity.iloc[-1] = running_equity

    trades = pd.DataFrame(
        trade_records, columns=["entry_date", "exit_date", "direction", "pnl"]
    )
    return equity, trades


def run_backtest(
    prices: pd.DataFrame, t1: str, t2: str, config: BacktestConfig = BacktestConfig()
) -> BacktestResult:
    """Run the full walk-forward backtest pipeline for a pair."""
    hedge_ratio, is_cointegrated = _rolling_hedge_ratios(prices, t1, t2, config)
    spread, zscore = _spread_and_zscore(prices, t1, t2, hedge_ratio, config)
    positions = _generate_positions(zscore, is_cointegrated, config)
    equity_curve, trades = _simulate_pnl(prices, t1, t2, hedge_ratio, positions, config)

    daily_returns = equity_curve.pct_change().dropna()
    trade_pnls = trades["pnl"].tolist()

    metrics = {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve),
        "sharpe": annualized_sharpe(daily_returns),
        "sortino": annualized_sortino(daily_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trade_pnls),
        "avg_trade_pnl": avg_trade_pnl(trade_pnls),
        "num_trades": len(trade_pnls),
    }

    return BacktestResult(
        equity_curve=equity_curve,
        hedge_ratio_series=hedge_ratio,
        zscore_series=zscore,
        positions=positions,
        trades=trades,
        metrics=metrics,
    )
