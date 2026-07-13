"""
Tests for services/data_fetcher.py.

These mock yf.download so the tests are deterministic and don't hit the
network; they also serve as a regression check that extracting this logic
out of main.py didn't change its behavior.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from services.data_fetcher import fetch_pair_data, fetch_price_series


def _price_frame(columns_and_values: dict, n: int = 30) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(columns_and_values, index=index)


class TestFetchPriceSeries:
    def test_prefers_adj_close_over_close(self):
        data = _price_frame({"Adj Close": np.arange(30.0), "Close": np.zeros(30)})
        with patch("services.data_fetcher.yf.download", return_value=data):
            series = fetch_price_series("FSR.JO", "2024-01-01", "2024-02-01")
        assert (series == np.arange(30.0)).all()

    def test_falls_back_to_close_when_no_adj_close(self):
        data = _price_frame({"Close": np.arange(30.0)})
        with patch("services.data_fetcher.yf.download", return_value=data):
            series = fetch_price_series("FSR.JO", "2024-01-01", "2024-02-01")
        assert (series == np.arange(30.0)).all()

    def test_empty_download_returns_empty_series(self):
        with patch("services.data_fetcher.yf.download", return_value=pd.DataFrame()):
            series = fetch_price_series("BAD.JO", "2024-01-01", "2024-02-01")
        assert series.empty

    def test_download_exception_returns_empty_series(self):
        with patch("services.data_fetcher.yf.download", side_effect=RuntimeError("network down")):
            series = fetch_price_series("FSR.JO", "2024-01-01", "2024-02-01")
        assert series.empty

    def test_drops_nan_rows(self):
        values = np.arange(30.0)
        values[5] = np.nan
        data = _price_frame({"Close": values})
        with patch("services.data_fetcher.yf.download", return_value=data):
            series = fetch_price_series("FSR.JO", "2024-01-01", "2024-02-01")
        assert len(series) == 29
        assert not series.isna().any()


class TestFetchPairData:
    def test_returns_aligned_dataframe_for_valid_tickers(self):
        series1 = pd.Series(np.arange(40.0), index=pd.date_range("2024-01-01", periods=40))
        series2 = pd.Series(np.arange(40.0) * 2, index=pd.date_range("2024-01-01", periods=40))
        with patch(
            "services.data_fetcher.fetch_price_series", side_effect=[series1, series2]
        ):
            df = fetch_pair_data("FSR.JO", "SBK.JO", lookback_days=60)
        assert list(df.columns) == ["FSR.JO", "SBK.JO"]
        assert len(df) == 40

    def test_raises_when_first_ticker_has_no_data(self):
        series2 = pd.Series(np.arange(40.0), index=pd.date_range("2024-01-01", periods=40))
        with patch(
            "services.data_fetcher.fetch_price_series",
            side_effect=[pd.Series(dtype=float), series2],
        ):
            with pytest.raises(ValueError, match="BAD.JO"):
                fetch_pair_data("BAD.JO", "SBK.JO", lookback_days=60)

    def test_raises_when_overlap_too_small(self):
        series1 = pd.Series(np.arange(10.0), index=pd.date_range("2024-01-01", periods=10))
        series2 = pd.Series(np.arange(10.0), index=pd.date_range("2024-01-01", periods=10))
        with patch(
            "services.data_fetcher.fetch_price_series", side_effect=[series1, series2]
        ):
            with pytest.raises(ValueError, match="Insufficient overlapping data"):
                fetch_pair_data("FSR.JO", "SBK.JO", lookback_days=60)
