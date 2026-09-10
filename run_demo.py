"""
run_demo.py
-----------
End-to-end run: load prices -> screen for cointegrated pairs -> backtest
the best pair -> report metrics -> save plots.

Works out of the box with zero setup (falls back to synthetic data if you
haven't cached real prices yet). To use real market data instead:

    python data_loader.py --tickers KO PEP XOM CVX JPM GS MSFT AAPL --start 2018-01-01
    python run_demo.py

Run on a machine with internet access for the first command.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest import BacktestConfig, run_backtest
from data_loader import load_prices
from metrics import print_summary, summarize
from pairs_selection import screen_pairs

OUT_DIR = Path(__file__).parent / "output"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    tickers = ["KO", "PEP", "XOM", "CVX", "JPM", "GS", "MSFT", "AAPL"]
    prices = load_prices(tickers)
    print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} tickers "
          f"({prices.index.min().date()} -> {prices.index.max().date()})")

    pairs = screen_pairs(prices)
    print(f"\nFound {len(pairs)} cointegrated pair(s):")
    for p in pairs:
        print(" ", p)

    if not pairs:
        print("No cointegrated pairs found in this universe/window -- try a "
              "wider ticker list or looser thresholds in pairs_selection.py")
        return

    best = pairs[0]
    print(f"\nBacktesting best pair: {best}")

    cfg = BacktestConfig(entry_z=2.0, exit_z=0.5, stop_z=4.0, lookback=30, cost_bps=5.0)
    result = run_backtest(prices[best.a], prices[best.b], best.hedge_ratio, cfg)

    # naive buy-and-hold benchmark: equal-weight the two legs
    bench_ret = prices[[best.a, best.b]].pct_change().mean(axis=1).reindex(result.daily_returns.index).fillna(0)

    stats = summarize(result.daily_returns, result.equity_curve, result.trade_pnls, bench_ret)
    print_summary(stats, title=f"Pairs Strategy: {best.a} / {best.b}")

    results_path = OUT_DIR / "results.csv"
    pd.DataFrame([stats]).to_csv(results_path, index=False)
    print(f"\nSaved metrics -> {results_path}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(result.equity_curve.index, result.equity_curve.values, label="Pairs Strategy", lw=1.4)
    bench_equity = cfg.capital * (1 + bench_ret).cumprod()
    axes[0].plot(bench_equity.index, bench_equity.values, label="Equal-weight benchmark", lw=1.0, alpha=0.7)
    axes[0].set_title(f"Equity Curve: {best.a}/{best.b}  (Sharpe {stats['Sharpe']:.2f} vs "
                       f"bench {stats['Benchmark Sharpe']:.2f})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(result.z_score.index, result.z_score.values, color="darkorange", lw=0.8)
    axes[1].axhline(cfg.entry_z, color="red", ls="--", lw=0.8)
    axes[1].axhline(-cfg.entry_z, color="red", ls="--", lw=0.8)
    axes[1].axhline(cfg.exit_z, color="green", ls="--", lw=0.8)
    axes[1].axhline(-cfg.exit_z, color="green", ls="--", lw=0.8)
    axes[1].set_title("Spread Z-Score (entry/exit bands)")
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(result.positions.index, result.positions.values, step="mid", alpha=0.5)
    axes[2].set_title("Position (+1 long spread / -1 short spread / 0 flat)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = OUT_DIR / "equity_curve.png"
    plt.savefig(fig_path, dpi=140)
    print(f"Saved chart -> {fig_path}")


if __name__ == "__main__":
    main()
