"""Tests for services/performance_metrics.py against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from services.performance_metrics import (
    annualized_sharpe,
    annualized_sortino,
    avg_trade_pnl,
    cagr,
    max_drawdown,
    total_return,
    win_rate,
)


class TestTotalReturn:
    def test_computes_simple_growth(self):
        equity = pd.Series([100.0, 110.0, 121.0])
        assert total_return(equity) == pytest.approx(0.21)

    def test_zero_for_flat_equity(self):
        equity = pd.Series([100.0, 100.0, 100.0])
        assert total_return(equity) == pytest.approx(0.0)


class TestCagr:
    def test_one_year_doubling_at_252_periods_per_year(self):
        # 253 rows = 252 elapsed periods = exactly 1 year at periods_per_year=252
        equity = pd.Series([100.0] * 253)
        equity.iloc[-1] = 200.0
        assert cagr(equity, periods_per_year=252) == pytest.approx(1.0, rel=1e-6)

    def test_returns_full_loss_sentinel_for_zero_or_negative_ending_equity(self):
        # A blown-up equity curve (e.g. an unbounded mark-to-market loss) has
        # a non-positive final value; growth**(1/years) is undefined there,
        # so this must return a defined -1.0 rather than NaN.
        equity = pd.Series([100_000.0] * 10 + [-500.0])
        assert cagr(equity) == pytest.approx(-1.0)

        equity_zero = pd.Series([100_000.0] * 10 + [0.0])
        assert cagr(equity_zero) == pytest.approx(-1.0)


class TestMaxDrawdown:
    def test_computes_worst_peak_to_trough_decline(self):
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        # peak is 120 after index 1; trough 90 -> drawdown = 90/120 - 1 = -0.25
        assert max_drawdown(equity) == pytest.approx(-0.25)

    def test_zero_for_monotonically_rising_equity(self):
        equity = pd.Series([100.0, 105.0, 110.0, 120.0])
        assert max_drawdown(equity) == pytest.approx(0.0)


class TestAnnualizedSharpe:
    def test_matches_manual_annualized_calculation(self):
        returns = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
        expected = (returns.mean() / returns.std(ddof=1)) * np.sqrt(252)
        assert annualized_sharpe(returns, periods_per_year=252) == pytest.approx(expected)

    def test_zero_when_no_volatility(self):
        returns = pd.Series([0.01, 0.01, 0.01])
        assert annualized_sharpe(returns) == pytest.approx(0.0)


class TestAnnualizedSortino:
    def test_matches_hand_computed_downside_deviation(self):
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        # downside = [0, -0.01, 0, -0.02, 0]; mean(downside**2) = 0.0001 -> dd = 0.01
        # mean(returns) = 0.006 -> sortino = 0.006/0.01 * sqrt(252)
        expected = (0.006 / 0.01) * np.sqrt(252)
        assert annualized_sortino(returns, periods_per_year=252) == pytest.approx(expected)

    def test_zero_when_no_downside_periods(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        assert annualized_sortino(returns) == pytest.approx(0.0)


class TestWinRate:
    def test_computes_fraction_of_profitable_trades(self):
        assert win_rate([10.0, -5.0, 20.0, -1.0, 0.0]) == pytest.approx(0.4)

    def test_zero_for_no_trades(self):
        assert win_rate([]) == 0.0


class TestAvgTradePnl:
    def test_computes_mean_pnl(self):
        assert avg_trade_pnl([10.0, -5.0, 20.0, -1.0, 0.0]) == pytest.approx(4.8)

    def test_zero_for_no_trades(self):
        assert avg_trade_pnl([]) == 0.0
