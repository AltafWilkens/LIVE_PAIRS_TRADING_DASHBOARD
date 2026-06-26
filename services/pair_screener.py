"""
PAIR SCREENER - Finds candidate cointegrated pairs across a universe.

Pipeline:
  1. Batch-download price history for every ticker in the universe
     (one yfinance call, not one per ticker -- avoids rate-limiting and
     is much faster).
  2. Within each sector group (plus any named structural pairs), compute
     pairwise correlation on price levels.
  3. Drop pairs below a correlation threshold (cheap pre-filter -- a
     full Engle-Granger test is much more expensive than a correlation
     check, so filtering first saves a lot of runtime).
  4. Run Engle-Granger cointegration on the surviving pairs.
  5. Rank cointegrated pairs by ADF p-value (lower = stronger evidence)
     and return a tidy results table.

Note on the correlation pre-filter: it is a standard, fast heuristic,
not a guarantee. Two series can be cointegrated without being highly
correlated (and vice versa) -- correlation measures co-movement,
cointegration measures whether their *spread* is stationary. We accept
this trade-off here for speed; widen CORRELATION_THRESHOLD downward if
the screener is missing pairs you'd expect to see.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from services.cointegration import (
    calculate_half_life,
    engle_granger_pvalue_only,
    engle_granger_test,
)

logger = logging.getLogger(__name__)

CORRELATION_THRESHOLD = 0.7  # minimum |correlation| to proceed to cointegration test
COINTEGRATION_PVALUE_THRESHOLD = 0.05  # standard 5% significance level
MIN_OVERLAPPING_OBS = 60  # require at least ~3 trading months of aligned data


@dataclass
class PairResult:
    ticker1: str
    ticker2: str
    sector: str
    correlation: float
    adf_pvalue: float
    is_cointegrated: bool
    half_life_days: float
    data_points: int


def fetch_universe_prices(
    tickers: List[str], lookback_days: int = 504
) -> pd.DataFrame:
    """
    Batch-download adjusted close prices for the full universe in a
    single yfinance call. Returns a wide DataFrame (date index, one
    column per ticker). Tickers that fail to download are dropped with
    a warning rather than failing the whole screen.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    logger.info(f"Batch downloading {len(tickers)} tickers")
    raw = yf.download(
        tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="ticker",
    )

    if raw.empty:
        raise ValueError("Batch download returned no data for any ticker in the universe.")

    prices: Dict[str, pd.Series] = {}

    # yfinance returns a flat single-level frame when only one ticker
    # succeeds, and a MultiIndex (ticker, field) frame for multiple --
    # handle both shapes.
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                col = raw[ticker]["Close"]
                series = col.dropna()
                if len(series) > 0:
                    prices[ticker] = series
                else:
                    logger.warning(f"No data for {ticker}, dropping from universe")
            except KeyError:
                logger.warning(f"{ticker} not found in download results, dropping")
    else:
        # single ticker case
        ticker = tickers[0]
        series = raw["Close"].dropna()
        if len(series) > 0:
            prices[ticker] = series

    if not prices:
        raise ValueError("No tickers in the universe returned usable data.")

    df = pd.DataFrame(prices)
    logger.info(f"Successfully fetched {df.shape[1]}/{len(tickers)} tickers, {df.shape[0]} rows")
    return df


def _candidate_pairs(
    sector_groups: Dict[str, List[str]],
    structural_pairs: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Build the list of (ticker1, ticker2, sector_label) combinations to
    test: all within-sector pairs, plus any explicitly named structural
    pairs (labeled "Structural").
    """
    pairs = []
    for sector, tickers in sector_groups.items():
        for t1, t2 in combinations(tickers, 2):
            pairs.append((t1, t2, sector))

    if structural_pairs:
        for t1, t2 in structural_pairs:
            pairs.append((t1, t2, "Structural"))

    return pairs


def screen_pairs(
    sector_groups: Dict[str, List[str]],
    structural_pairs: Optional[List[Tuple[str, str]]] = None,
    lookback_days: int = 504,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    pvalue_threshold: float = COINTEGRATION_PVALUE_THRESHOLD,
) -> pd.DataFrame:
    """
    Run the full screening pipeline and return a ranked DataFrame of
    cointegrated pairs (one row per pair, sorted by ADF p-value
    ascending -- strongest evidence of cointegration first).

    Pairs that fail the correlation pre-filter or the cointegration
    test are excluded from the returned table; pass
    return_all=False semantics are implicit (see screen_pairs_full
    for the unfiltered version with rejection reasons, if needed later).
    """
    all_tickers = sorted(set(
        t for tickers in sector_groups.values() for t in tickers
    ) | set(t for pair in (structural_pairs or []) for t in pair))

    prices = fetch_universe_prices(all_tickers, lookback_days=lookback_days)

    candidates = _candidate_pairs(sector_groups, structural_pairs)
    logger.info(f"Testing {len(candidates)} candidate pairs")

    results: List[PairResult] = []

    for t1, t2, sector in candidates:
        if t1 not in prices.columns or t2 not in prices.columns:
            logger.warning(f"Skipping {t1}/{t2}: missing price data")
            continue

        pair_df = prices[[t1, t2]].dropna()
        if len(pair_df) < MIN_OVERLAPPING_OBS:
            logger.info(
                f"Skipping {t1}/{t2}: only {len(pair_df)} overlapping "
                f"observations (need {MIN_OVERLAPPING_OBS})"
            )
            continue

        correlation = pair_df[t1].corr(pair_df[t2])

        if abs(correlation) < correlation_threshold:
            continue  # pre-filter: not worth the cointegration test

        pvalue = engle_granger_pvalue_only(pair_df[t1], pair_df[t2])
        is_cointegrated = pvalue < pvalue_threshold

        if not is_cointegrated:
            continue

        # Only compute the (more expensive) half-life for pairs that
        # actually passed cointegration.
        eg = engle_granger_test(pair_df, t1, t2)
        half_life = calculate_half_life(eg["spread"])

        results.append(PairResult(
            ticker1=t1,
            ticker2=t2,
            sector=sector,
            correlation=round(float(correlation), 4),
            adf_pvalue=round(pvalue, 5),
            is_cointegrated=is_cointegrated,
            half_life_days=round(half_life, 1),
            data_points=len(pair_df),
        ))

    if not results:
        logger.info("No cointegrated pairs found at current thresholds")
        return pd.DataFrame(columns=[
            "ticker1", "ticker2", "sector", "correlation",
            "adf_pvalue", "is_cointegrated", "half_life_days", "data_points",
        ])

    result_df = pd.DataFrame([r.__dict__ for r in results])
    result_df = result_df.sort_values("adf_pvalue").reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    # Quick manual run: python -m services.pair_screener  (from project root)
    from config.universe import SECTOR_GROUPS, STRUCTURAL_PAIRS

    logging.basicConfig(level=logging.INFO)

    results = screen_pairs(SECTOR_GROUPS, STRUCTURAL_PAIRS)
    print("\n" + "=" * 70)
    print(f"Found {len(results)} cointegrated pair(s)")
    print("=" * 70)
    if not results.empty:
        print(results.to_string(index=False))
