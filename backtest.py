"""
backtest.py
-----------
Turns a cointegrated pair into a z-score mean-reversion trading strategy
and simulates it day by day with transaction costs.

Strategy logic (classic, deliberately simple -- the point of this repo is
to get the *evaluation methodology* right, not to find alpha):

  spread_t = price_A,t - beta * price_B,t
  z_t      = (spread_t - rolling_mean(spread)) / rolling_std(spread)

  z_t <  -entry_z   -> spread unusually low  -> long A, short beta*B
  z_t >  +entry_z   -> spread unusually high -> short A, long beta*B
  |z_t| <  exit_z   -> close position (reverted to mean)
  |z_t| >  stop_z   -> close position (stop loss -- cointegration may have
                        broken down)

Positions are sized in dollar-neutral terms: for one "unit" of the pair
trade we go long/short $1 of A against $beta of B (approximately
dollar-neutral, standard convention for a pairs trade).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    lookback: int = 30          # rolling window for z-score
    cost_bps: float = 5.0       # round-trip transaction cost, in basis points, per leg
    capital: float = 100_000.0
    position_fraction: float = 0.5  # fraction of capital deployed on the A leg per trade


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.Series          # +1 long-spread, -1 short-spread, 0 flat
    z_score: pd.Series
    trade_pnls: list = field(default_factory=list)


def run_backtest(price_a: pd.Series, price_b: pd.Series, beta: float,
                  cfg: BacktestConfig = BacktestConfig()) -> BacktestResult:
    idx = price_a.index.intersection(price_b.index)
    a, b = price_a.loc[idx], price_b.loc[idx]

    spread = a - beta * b
    roll_mean = spread.rolling(cfg.lookback).mean()
    roll_std = spread.rolling(cfg.lookback).std()
    z = (spread - roll_mean) / roll_std

    position = pd.Series(0, index=idx, dtype=float)   # +1 / -1 / 0 direction, for plotting
    quantity = pd.Series(0.0, index=idx)               # signed shares of A actually held
    cur_pos = 0.0
    cur_qty = 0.0
    entry_price_spread = None
    trade_pnls = []

    for i, t in enumerate(idx):
        zt = z.iloc[i]
        if np.isnan(zt):
            position.iloc[i] = cur_pos
            quantity.iloc[i] = cur_qty
            continue

        if cur_pos == 0.0:
            if zt <= -cfg.entry_z or zt >= cfg.entry_z:
                direction = 1.0 if zt <= -cfg.entry_z else -1.0
                # size the trade as a fixed fraction of capital, in shares of
                # leg A at today's price -- this is what makes PnL scale
                # sensibly against `capital` instead of being priced in raw
                # spread-points-per-share.
                shares_a = (cfg.capital * cfg.position_fraction) / price_a.loc[t]
                cur_pos = direction
                cur_qty = direction * shares_a
                entry_price_spread = spread.iloc[i]
        else:
            exited = abs(zt) <= cfg.exit_z or abs(zt) >= cfg.stop_z
            if exited:
                pnl = cur_qty * (spread.iloc[i] - entry_price_spread)
                cost = abs(cur_qty) * (cfg.cost_bps / 1e4) * spread.iloc[i]
                trade_pnls.append(pnl - cost)
                cur_pos = 0.0
                cur_qty = 0.0
                entry_price_spread = None

        position.iloc[i] = cur_pos
        quantity.iloc[i] = cur_qty

    # PnL = yesterday's held quantity (shares of A, signed) times today's
    # move in the spread, minus transaction costs on days the position
    # size actually changes (entries/exits).
    spread_change = spread.diff().fillna(0.0)
    held_qty = quantity.shift(1).fillna(0.0)
    qty_change = quantity.diff().abs().fillna(0.0)
    cost = qty_change * (cfg.cost_bps / 1e4) * spread.abs()

    pnl_series = held_qty * spread_change - cost
    daily_returns = pnl_series / cfg.capital
    equity_curve = cfg.capital * (1 + daily_returns).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        positions=position,
        z_score=z,
        trade_pnls=trade_pnls,
    )
