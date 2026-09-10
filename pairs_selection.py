"""
pairs_selection.py
-------------------
Screen a universe of tickers for statistically cointegrated pairs.

Why cointegration and not correlation: two prices can be highly correlated
(they both trend up) while the *spread* between them wanders off forever --
that's not tradeable. Cointegration specifically tests whether a linear
combination of the two price series is stationary (mean-reverting), which
is the property a pairs trade actually needs.

Method: Engle-Granger two-step test.
  1. Regress price_A on price_B  ->  price_A = alpha + beta * price_B + spread
  2. Run an Augmented Dickey-Fuller test on the residual `spread`.
     If we reject the null of a unit root (p-value < threshold), the
     spread is stationary => the pair is cointegrated.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class PairResult:
    a: str
    b: str
    hedge_ratio: float   # beta: how many units of B per unit of A
    coint_pvalue: float
    adf_pvalue: float
    half_life: float     # days for the spread to revert halfway to its mean

    def __repr__(self) -> str:
        return (f"{self.a}~{self.b}  beta={self.hedge_ratio:.3f}  "
                f"coint_p={self.coint_pvalue:.4f}  half_life={self.half_life:.1f}d")


def hedge_ratio_and_spread(price_a: pd.Series, price_b: pd.Series) -> tuple[float, pd.Series]:
    """OLS hedge ratio: price_a = alpha + beta * price_b + spread."""
    x = sm.add_constant(price_b.values)
    model = sm.OLS(price_a.values, x).fit()
    beta = model.params[1]
    spread = price_a - beta * price_b
    return beta, spread


def half_life_of_mean_reversion(spread: pd.Series) -> float:
    """Fit spread_t - spread_{t-1} = lambda * spread_{t-1} + eps (an OU
    discretization) and convert lambda into a half-life in days. A shorter
    half-life means the spread reverts faster -> more trade opportunities
    per unit time, all else equal.
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    spread_lag = spread_lag.loc[spread_diff.index]
    x = sm.add_constant(spread_lag.values)
    model = sm.OLS(spread_diff.values, x).fit()
    lam = model.params[1]
    if lam >= 0:
        return np.inf  # not mean reverting
    return -np.log(2) / lam


def screen_pairs(prices: pd.DataFrame, coint_pvalue_max: float = 0.05,
                  min_half_life: float = 1.0, max_half_life: float = 60.0) -> list[PairResult]:
    """Test every ticker combination in `prices` for cointegration and
    return the ones that pass, sorted by cointegration p-value (best first).
    """
    results: list[PairResult] = []
    tickers = list(prices.columns)

    for a, b in itertools.combinations(tickers, 2):
        pa, pb = prices[a].dropna(), prices[b].dropna()
        idx = pa.index.intersection(pb.index)
        pa, pb = pa.loc[idx], pb.loc[idx]
        if len(idx) < 250:
            continue

        _, coint_p, _ = coint(pa, pb)
        beta, spread = hedge_ratio_and_spread(pa, pb)
        adf_p = adfuller(spread, autolag="AIC")[1]
        hl = half_life_of_mean_reversion(spread)

        if coint_p <= coint_pvalue_max and min_half_life <= hl <= max_half_life:
            results.append(PairResult(a, b, beta, coint_p, adf_p, hl))

    return sorted(results, key=lambda r: r.coint_pvalue)


if __name__ == "__main__":
    from data_loader import load_prices

    tickers = ["KO", "PEP", "XOM", "CVX", "JPM", "GS", "MSFT", "AAPL"]
    prices = load_prices(tickers)
    pairs = screen_pairs(prices)

    print(f"\nScreened {len(tickers)} tickers -> {len(pairs)} cointegrated pairs "
          f"(p <= 0.05, half-life 1-60 days):\n")
    for p in pairs:
        print(" ", p)
