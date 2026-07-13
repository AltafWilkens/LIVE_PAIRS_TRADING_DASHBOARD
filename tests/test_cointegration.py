"""Tests for services/cointegration.py: Engle-Granger and Johansen tests."""

import numpy as np
import pandas as pd
import pytest

from services.cointegration import engle_granger_test, johansen_test


@pytest.fixture
def cointegrated_pair() -> pd.DataFrame:
    """y = x + stationary noise -> spread (y - x) is stationary by construction."""
    rng = np.random.default_rng(42)
    n = 500
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    noise = rng.normal(0, 0.5, n)
    y = x + noise
    return pd.DataFrame({"Y": y, "X": x})


def _independent_random_walk_pair(seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    y = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"Y": y, "X": x})


class TestEngleGranger:
    def test_detects_cointegrated_pair(self, cointegrated_pair):
        result = engle_granger_test(cointegrated_pair, "Y", "X")
        assert result["is_cointegrated"] is True
        assert result["hedge_ratio"] == pytest.approx(1.0, abs=0.15)

    def test_mostly_rejects_independent_random_walks(self):
        # A correctly-sized 5%-level test will still call ~1 in 20 unrelated
        # random walks "cointegrated" by chance (spurious regression), so
        # assert the rejection rate rather than a single fixed seed.
        seeds = range(1, 11)
        results = [
            engle_granger_test(_independent_random_walk_pair(seed), "Y", "X")["is_cointegrated"]
            for seed in seeds
        ]
        false_positive_rate = sum(results) / len(results)
        assert false_positive_rate <= 0.3

    def test_spread_matches_hedge_ratio(self, cointegrated_pair):
        result = engle_granger_test(cointegrated_pair, "Y", "X")
        expected_spread = cointegrated_pair["Y"] - result["hedge_ratio"] * cointegrated_pair["X"]
        pd.testing.assert_series_equal(result["spread"], expected_spread, check_names=False)


class TestJohansen:
    def test_detects_cointegrated_pair(self, cointegrated_pair):
        result = johansen_test(cointegrated_pair, "Y", "X")
        assert result["is_cointegrated"] is True
        assert result["hedge_ratio"] == pytest.approx(1.0, abs=0.15)

    def test_mostly_rejects_independent_random_walks(self):
        seeds = range(1, 11)
        results = [
            johansen_test(_independent_random_walk_pair(seed), "Y", "X")["is_cointegrated"]
            for seed in seeds
        ]
        false_positive_rate = sum(results) / len(results)
        assert false_positive_rate <= 0.3

    def test_output_shape_matches_engle_granger(self, cointegrated_pair):
        eg = engle_granger_test(cointegrated_pair, "Y", "X")
        jh = johansen_test(cointegrated_pair, "Y", "X")
        common_keys = {"hedge_ratio", "spread", "is_cointegrated", "spread_mean", "spread_std"}
        assert common_keys.issubset(eg.keys())
        assert common_keys.issubset(jh.keys())
