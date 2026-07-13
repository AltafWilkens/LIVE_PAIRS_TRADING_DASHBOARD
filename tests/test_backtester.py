"""
Tests for services/backtester.py.

Covers the properties that matter most for a *realistic* walk-forward
backtest: no lookahead bias in the rolling hedge-ratio re-estimation, a
correct entry/exit state machine, transaction costs/slippage actually
biting, and the cointegration gate flattening positions when a pair
decoheres.
"""

import numpy as np
import pandas as pd
import pytest

from services.backtester import (
    BacktestConfig,
    _generate_positions,
    _rolling_hedge_ratios,
    _simulate_pnl,
    run_backtest,
)


def _cointegrated_prices(n: int, seed: int, noise_std: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    y = x + rng.normal(0, noise_std, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Y": y, "X": x}, index=dates)


class TestRollingHedgeRatios:
    def test_unaffected_by_data_appended_after_the_estimation_window(self):
        # Walk-forward correctness: a hedge ratio estimated using only data
        # up to day t must not change if more data is appended *after* t.
        prices_full = _cointegrated_prices(n=400, seed=123)
        prices_truncated = prices_full.iloc[:300]
        config = BacktestConfig(lookback_window=100, reestimate_every=20)

        hr_full, coint_full = _rolling_hedge_ratios(prices_full, "Y", "X", config)
        hr_trunc, coint_trunc = _rolling_hedge_ratios(prices_truncated, "Y", "X", config)

        pd.testing.assert_series_equal(hr_full.iloc[:300], hr_trunc, check_names=False)
        pd.testing.assert_series_equal(coint_full.iloc[:300], coint_trunc, check_names=False)

    def test_no_hedge_ratio_before_first_lookback_window_elapses(self):
        prices = _cointegrated_prices(n=150, seed=1)
        config = BacktestConfig(lookback_window=100, reestimate_every=20)
        hr, _ = _rolling_hedge_ratios(prices, "Y", "X", config)
        assert hr.iloc[:100].isna().all()
        assert hr.iloc[100:].notna().any()


class TestGeneratePositions:
    def test_entries_and_exits_fire_at_correct_thresholds(self):
        dates = pd.date_range("2024-01-01", periods=8)
        zscore = pd.Series([0.0, 0.0, 2.5, 2.2, 0.3, 0.1, -2.6, -0.2], index=dates)
        is_cointegrated = pd.Series([True] * 8, index=dates)
        config = BacktestConfig(entry_zscore=2.0, exit_zscore=0.5)

        positions = _generate_positions(zscore, is_cointegrated, config)

        assert list(positions) == [0, 0, -1, -1, 0, 0, 1, 0]

    def test_no_flip_flop_while_hovering_near_entry_threshold(self):
        dates = pd.date_range("2024-01-01", periods=4)
        zscore = pd.Series([2.5, 1.9, 2.4, 1.8], index=dates)
        is_cointegrated = pd.Series([True] * 4, index=dates)
        config = BacktestConfig(entry_zscore=2.0, exit_zscore=0.5)

        positions = _generate_positions(zscore, is_cointegrated, config)

        # Stays short throughout: dipping below the entry threshold does not
        # exit a position once open -- only crossing the exit threshold does.
        assert list(positions) == [-1, -1, -1, -1]

    def test_require_cointegration_flattens_and_blocks_new_entries(self):
        dates = pd.date_range("2024-01-01", periods=10)
        zscore = pd.Series([3.0] * 10, index=dates)
        is_cointegrated = pd.Series([True] * 5 + [False] * 5, index=dates)
        config = BacktestConfig(entry_zscore=2.0, exit_zscore=0.5, require_cointegration=True)

        positions = _generate_positions(zscore, is_cointegrated, config)

        assert (positions.iloc[:5] == -1).all()
        assert (positions.iloc[5:] == 0).all()


class TestSimulatePnl:
    def test_transaction_costs_and_slippage_reduce_net_trade_pnl(self):
        dates = pd.date_range("2024-01-01", periods=20)
        price1 = pd.Series(100 + np.linspace(0, 5, 20), index=dates)
        price2 = pd.Series(50.0, index=dates)
        prices = pd.DataFrame({"Y": price1, "X": price2})
        hedge_ratio = pd.Series(0.0, index=dates)
        positions = pd.Series(0, index=dates, dtype=int)
        positions.iloc[2:15] = 1

        no_cost = BacktestConfig(transaction_cost_bps=0.0, slippage_bps=0.0)
        with_cost = BacktestConfig(transaction_cost_bps=10.0, slippage_bps=5.0)

        _, trades_no_cost = _simulate_pnl(prices, "Y", "X", hedge_ratio, positions, no_cost)
        _, trades_with_cost = _simulate_pnl(prices, "Y", "X", hedge_ratio, positions, with_cost)

        assert len(trades_no_cost) == 1
        assert len(trades_with_cost) == 1
        assert trades_with_cost["pnl"].iloc[0] < trades_no_cost["pnl"].iloc[0]

    def test_still_open_position_is_liquidated_at_series_end(self):
        dates = pd.date_range("2024-01-01", periods=10)
        price1 = pd.Series(100 + np.linspace(0, 3, 10), index=dates)
        price2 = pd.Series(50.0, index=dates)
        prices = pd.DataFrame({"Y": price1, "X": price2})
        hedge_ratio = pd.Series(0.0, index=dates)
        positions = pd.Series(0, index=dates, dtype=int)
        positions.iloc[3:] = 1  # never exits before the series ends
        config = BacktestConfig(transaction_cost_bps=5.0, slippage_bps=2.0)

        equity, trades = _simulate_pnl(prices, "Y", "X", hedge_ratio, positions, config)

        assert len(trades) == 1
        assert trades["exit_date"].iloc[0] == dates[-1]
        assert equity.iloc[-1] == pytest.approx(config.capital + trades["pnl"].iloc[0])

    def test_flat_series_produces_no_trades(self):
        dates = pd.date_range("2024-01-01", periods=10)
        prices = pd.DataFrame({"Y": [100.0] * 10, "X": [50.0] * 10}, index=dates)
        hedge_ratio = pd.Series(1.0, index=dates)
        positions = pd.Series(0, index=dates, dtype=int)
        config = BacktestConfig()

        equity, trades = _simulate_pnl(prices, "Y", "X", hedge_ratio, positions, config)

        assert trades.empty
        assert (equity == config.capital).all()


class TestRunBacktestEndToEnd:
    def test_smoke_and_metrics_shape(self):
        prices = _cointegrated_prices(n=500, seed=99, noise_std=1.0)
        config = BacktestConfig(lookback_window=100, reestimate_every=20, zscore_window=20)

        result = run_backtest(prices, "Y", "X", config)

        assert len(result.equity_curve) == len(prices)
        assert result.equity_curve.iloc[0] == pytest.approx(config.capital)
        expected_keys = {
            "total_return", "cagr", "sharpe", "sortino",
            "max_drawdown", "win_rate", "avg_trade_pnl", "num_trades",
        }
        assert expected_keys.issubset(result.metrics.keys())

    def test_johansen_method_also_runs_end_to_end(self):
        prices = _cointegrated_prices(n=500, seed=17, noise_std=1.0)
        config = BacktestConfig(
            lookback_window=100, reestimate_every=20, zscore_window=20,
            coint_method="johansen",
        )

        result = run_backtest(prices, "Y", "X", config)

        assert len(result.equity_curve) == len(prices)
