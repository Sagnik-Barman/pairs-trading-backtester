"""
data_loader.py
--------------
Single place responsible for getting daily close prices into the rest of
the pipeline. Three sources, in order of preference:

1. Local CSV cache (data/<TICKER>.csv)               -- fastest, offline
2. yfinance live pull (needs an internet connection)  -- real market data
3. Synthetic GBM-with-common-factor generator          -- for demos / when
   you have no internet handy; produces prices that are realistically
   cointegrated so the rest of the pipeline has something sensible to chew
   on.

Run this file directly on a machine WITH internet access to build a real
local cache:

    python data_loader.py --tickers KO PEP XOM CVX JPM GS --start 2018-01-01

That writes data/<TICKER>.csv for each ticker. After that, every other
script in this repo will transparently use the cached CSVs -- no internet
needed for backtesting, cointegration screening, etc.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TRADING_DAYS = 252


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker.upper()}.csv"


def fetch_and_cache(tickers: list[str], start: str, end: str | None = None) -> None:
    """Pull daily closes from yfinance and write one CSV per ticker.

    Requires internet access. If yfinance/network isn't available this will
    raise -- that's fine, it just means you run this step on your laptop
    instead of in a sandboxed environment.
    """
    import yfinance as yf

    DATA_DIR.mkdir(exist_ok=True)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0])

    for ticker in tickers:
        series = closes[ticker].dropna()
        if series.empty:
            print(f"[warn] no data returned for {ticker}, skipping")
            continue
        out = series.to_frame("close")
        out.index.name = "date"
        out.to_csv(_cache_path(ticker))
        print(f"[ok] cached {len(out)} rows -> {_cache_path(ticker)}")


def load_prices(tickers: list[str], start: str = "2018-01-01", end: str | None = None,
                 synthetic_days: int = 1500, synthetic_seed: int = 7) -> pd.DataFrame:
    """Return a DataFrame of daily close prices, columns = tickers.

    Tries the local cache first. Falls back to synthetic data (with a loud
    warning) so every other script in the repo is runnable with zero setup.
    """
    frames = {}
    missing = []
    for t in tickers:
        p = _cache_path(t)
        if p.exists():
            s = pd.read_csv(p, index_col="date", parse_dates=True)["close"]
            frames[t] = s
        else:
            missing.append(t)

    if missing:
        print(f"[data_loader] no cache for {missing} -> generating SYNTHETIC prices. "
              f"Run `python data_loader.py --tickers {' '.join(tickers)}` on a machine "
              f"with internet access to replace this with real data.")
        synth = _synthetic_cointegrated_prices(tickers, synthetic_days, synthetic_seed)
        return synth

    df = pd.DataFrame(frames).dropna(how="all")
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df.dropna()


def _synthetic_cointegrated_prices(tickers: list[str], n_days: int, seed: int) -> pd.DataFrame:
    """Generate prices for len(tickers) assets, paired up two-at-a-time.

    Each pair's first leg follows a standard GBM with realistic equity
    drift/vol. The second leg is built as `beta * leg_A * exp(spread)`
    where `spread` is a mean-reverting (Ornstein-Uhlenbeck) log-ratio.
    That construction *guarantees* log(A) and log(B) are cointegrated
    (their difference, up to the constant, is the stationary OU spread)
    without needing fragile shared random-walk factors that can blow up
    over a multi-year synthetic path. Pairs across different tickers (e.g.
    KO vs XOM) are NOT cointegrated by construction -- which is the point:
    it lets `pairs_selection.py` demonstrate that it correctly finds the
    real pairs and rejects the fake ones.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    dt = 1 / TRADING_DAYS

    prices = {}
    n = len(tickers)
    for i in range(0, n, 2):
        pair = tickers[i:i + 2]
        s0 = rng.uniform(40, 200)
        mu_annual = rng.uniform(0.04, 0.10)
        sigma_annual = rng.uniform(0.18, 0.30)

        log_returns = (mu_annual - 0.5 * sigma_annual ** 2) * dt \
            + sigma_annual * np.sqrt(dt) * rng.standard_normal(n_days)
        price_a = s0 * np.exp(np.cumsum(log_returns))
        prices[pair[0]] = price_a

        if len(pair) > 1:
            beta = rng.uniform(0.6, 1.5)
            half_life_days = rng.uniform(10, 25)
            theta = np.log(2) / half_life_days
            spread_sigma = rng.uniform(0.02, 0.05)

            spread = np.zeros(n_days)
            for t in range(1, n_days):
                spread[t] = spread[t - 1] - theta * spread[t - 1] + spread_sigma * rng.standard_normal()

            prices[pair[1]] = beta * price_a * np.exp(spread)

    df = pd.DataFrame(prices, index=dates)
    return df[tickers] if all(t in df.columns for t in tickers) else df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & cache daily close prices via yfinance.")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    fetch_and_cache(args.tickers, args.start, args.end)
