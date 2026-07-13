"""Tests for the POST /backtest endpoint in main.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def _synthetic_cointegrated_prices(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    y = x + rng.normal(0, 0.5, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"FSR.JO": y, "SBK.JO": x}, index=dates)


class TestBacktestEndpoint:
    def test_returns_metrics_and_series_for_valid_pair(self):
        with patch("main.fetch_pair_data", return_value=_synthetic_cointegrated_prices()):
            response = client.post(
                "/backtest",
                json={
                    "ticker1": "FSR.JO",
                    "ticker2": "SBK.JO",
                    "lookback_window": 100,
                    "reestimate_every": 20,
                    "zscore_window": 20,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["ticker1"] == "FSR.JO"
        assert body["ticker2"] == "SBK.JO"
        assert len(body["equity_curve"]) == 400
        assert len(body["zscore_series"]) == 400
        expected_metric_keys = {
            "total_return", "cagr", "sharpe", "sortino",
            "max_drawdown", "win_rate", "avg_trade_pnl", "num_trades",
        }
        assert expected_metric_keys.issubset(body["metrics"].keys())

    def test_supports_johansen_method(self):
        with patch("main.fetch_pair_data", return_value=_synthetic_cointegrated_prices(seed=7)):
            response = client.post(
                "/backtest",
                json={
                    "ticker1": "FSR.JO",
                    "ticker2": "SBK.JO",
                    "coint_method": "johansen",
                    "lookback_window": 100,
                    "reestimate_every": 20,
                    "zscore_window": 20,
                },
            )
        assert response.status_code == 200

    def test_rejects_unknown_coint_method(self):
        response = client.post(
            "/backtest",
            json={"ticker1": "FSR.JO", "ticker2": "SBK.JO", "coint_method": "not_a_real_method"},
        )
        assert response.status_code == 422

    def test_missing_ticker_data_returns_400(self):
        with patch("main.fetch_pair_data", side_effect=ValueError("No data available for BAD.JO")):
            response = client.post(
                "/backtest", json={"ticker1": "BAD.JO", "ticker2": "SBK.JO"},
            )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "overrides",
        [
            {"transaction_cost_bps": -1000.0},
            {"slippage_bps": -1000.0},
            {"lookback_window": 0},
            {"reestimate_every": 0},
            {"zscore_window": 0},
            {"lookback_days": 0},
            {"capital": 0.0},
            {"entry_zscore": 0.0},
        ],
    )
    def test_rejects_out_of_range_numeric_fields(self, overrides):
        # These previously either fabricated a fake "profitable" backtest
        # (negative cost/slippage) or leaked a raw internal exception
        # message through the generic 500 handler (zero window/lookback).
        payload = {"ticker1": "FSR.JO", "ticker2": "SBK.JO", **overrides}
        response = client.post("/backtest", json=payload)
        assert response.status_code == 422
