"""
metrics.py
----------
Standard performance-evaluation metrics for a daily-returns equity curve.
Kept dependency-free (just numpy/pandas) so it can be reused by any
strategy, not just the pairs trader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return np.sqrt(TRADING_DAYS) * excess.mean() / downside_std


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()


def cagr(equity_curve: pd.Series) -> float:
    n_years = len(equity_curve) / TRADING_DAYS
    if n_years <= 0 or equity_curve.iloc[0] <= 0:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1


def win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def summarize(returns: pd.Series, equity_curve: pd.Series, trade_pnls: list[float],
              benchmark_returns: pd.Series | None = None) -> dict:
    out = {
        "CAGR": cagr(equity_curve),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max Drawdown": max_drawdown(equity_curve),
        "Win Rate": win_rate(trade_pnls),
        "Num Trades": len(trade_pnls),
        "Total Return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
    }
    if benchmark_returns is not None:
        aligned = returns.align(benchmark_returns, join="inner")
        strat, bench = aligned
        if bench.std() > 0:
            out["Beta vs Benchmark"] = np.cov(strat, bench)[0, 1] / np.var(bench)
        out["Benchmark Sharpe"] = sharpe_ratio(bench)
        out["Benchmark Total Return"] = (1 + bench).prod() - 1
    return out


def print_summary(stats: dict, title: str = "Performance Summary") -> None:
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        if isinstance(v, float):
            if "Rate" in k or "Return" in k or "Drawdown" in k or "CAGR" in k:
                print(f"  {k:<24}: {v:.2%}")
            else:
                print(f"  {k:<24}: {v:.3f}")
        else:
            print(f"  {k:<24}: {v}")
