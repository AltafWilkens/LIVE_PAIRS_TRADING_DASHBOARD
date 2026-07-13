"""
BACKTEST RUNNER - CLI for the walk-forward pairs-trading backtester.

Standalone script: fetches historical prices for a pair, runs
services.backtester.run_backtest, prints a metrics summary, and writes the
equity curve / trade log to CSV (optionally an HTML chart via Plotly).

Example:
    python backtest_runner.py --ticker1 FSR.JO --ticker2 SBK.JO \
        --lookback-days 1500 --coint-method johansen \
        --entry-zscore 2.0 --exit-zscore 0.5 --plot
"""

import argparse
import logging

from services.backtester import BacktestConfig, run_backtest
from services.data_fetcher import fetch_pair_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward pairs-trading backtester")
    parser.add_argument("--ticker1", required=True, help="First ticker (e.g. FSR.JO)")
    parser.add_argument("--ticker2", required=True, help="Second ticker (e.g. SBK.JO)")
    parser.add_argument("--lookback-days", type=int, default=1500, help="Days of history to fetch")
    parser.add_argument(
        "--coint-method", choices=["engle_granger", "johansen"], default="engle_granger",
    )
    parser.add_argument("--entry-zscore", type=float, default=2.0)
    parser.add_argument("--exit-zscore", type=float, default=0.5)
    parser.add_argument(
        "--lookback-window", type=int, default=252,
        help="Trailing days used to re-fit the hedge ratio/cointegration test",
    )
    parser.add_argument(
        "--reestimate-every", type=int, default=21, help="Days between hedge-ratio refits",
    )
    parser.add_argument(
        "--zscore-window", type=int, default=30, help="Trailing window for the rolling z-score",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument(
        "--no-require-cointegration", action="store_true",
        help="Trade even during windows where the refit fails its cointegration test",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for equity_curve.csv / trades.csv")
    parser.add_argument("--plot", action="store_true", help="Also save an HTML chart via Plotly")
    return parser.parse_args()


def print_summary(ticker1: str, ticker2: str, metrics: dict) -> None:
    print("\n" + "=" * 60)
    print(f"BACKTEST SUMMARY: {ticker1} / {ticker2}")
    print("=" * 60)
    print(f"{'Total return':<20}{metrics['total_return']:>15.2%}")
    print(f"{'CAGR':<20}{metrics['cagr']:>15.2%}")
    print(f"{'Sharpe':<20}{metrics['sharpe']:>15.2f}")
    print(f"{'Sortino':<20}{metrics['sortino']:>15.2f}")
    print(f"{'Max drawdown':<20}{metrics['max_drawdown']:>15.2%}")
    print(f"{'Win rate':<20}{metrics['win_rate']:>15.2%}")
    print(f"{'Avg trade P&L':<20}{metrics['avg_trade_pnl']:>15.2f}")
    print(f"{'Number of trades':<20}{metrics['num_trades']:>15d}")
    print("=" * 60 + "\n")


def save_plot(result, ticker1: str, ticker2: str, output_path: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Equity Curve", "Z-Score"))
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve, name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.zscore_series.index, y=result.zscore_series, name="Z-Score"), row=2, col=1)
    fig.update_layout(title=f"Backtest: {ticker1} / {ticker2}", height=700)
    fig.write_html(output_path)
    logger.info(f"Saved chart to {output_path}")


def main() -> None:
    args = parse_args()

    logger.info(f"Fetching {args.ticker1}/{args.ticker2}, lookback={args.lookback_days} days")
    prices = fetch_pair_data(args.ticker1, args.ticker2, args.lookback_days)

    config = BacktestConfig(
        entry_zscore=args.entry_zscore,
        exit_zscore=args.exit_zscore,
        lookback_window=args.lookback_window,
        reestimate_every=args.reestimate_every,
        zscore_window=args.zscore_window,
        coint_method=args.coint_method,
        require_cointegration=not args.no_require_cointegration,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        capital=args.capital,
    )

    result = run_backtest(prices, args.ticker1, args.ticker2, config)

    print_summary(args.ticker1, args.ticker2, result.metrics)

    equity_path = f"{args.output_dir}/equity_curve.csv"
    trades_path = f"{args.output_dir}/trades.csv"
    result.equity_curve.to_csv(equity_path, header=["equity"])
    result.trades.to_csv(trades_path, index=False)
    logger.info(f"Wrote {equity_path} and {trades_path}")

    if args.plot:
        save_plot(result, args.ticker1, args.ticker2, f"{args.output_dir}/backtest_chart.html")


if __name__ == "__main__":
    main()
