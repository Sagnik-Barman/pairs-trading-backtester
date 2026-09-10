# Pairs Trading / Statistical Arbitrage Backtester

A market-neutral mean-reversion strategy: find two stocks whose prices move
together over the long run, trade the *spread* between them whenever it
strays too far from its historical relationship, and profit when it snaps
back — regardless of which direction the overall market moves.

## Why cointegration, not correlation

Two stocks can be highly *correlated* (they both trend upward) while the
gap between their prices wanders off to infinity — that's not tradeable,
there's no "back to normal" to bet on. What you actually need is
**cointegration**: a linear combination of the two prices that is
*stationary* (mean-reverting). This repo screens for that directly using
the **Engle-Granger two-step test**:

1. Regress `price_A` on `price_B` → `price_A = α + β · price_B + spread`
2. Run an Augmented Dickey-Fuller test on the residual `spread`. If we can
   reject the null hypothesis of a unit root, the spread is stationary and
   the pair is (statistically) cointegrated.

`β` here is the **hedge ratio**: for every 1 share of A you hold, you hold
`β` shares of B on the other side, dollar-neutral-ish, so the position's
P&L depends on the *spread* reverting, not on either stock going up.

## Strategy logic

```
z_t = (spread_t − rolling_mean(spread)) / rolling_std(spread)

z_t <= -entry_z   → spread abnormally low  → long the spread (long A, short βB)
z_t >= +entry_z   → spread abnormally high → short the spread (short A, long βB)
|z_t| <= exit_z   → close (reverted to mean)
|z_t| >= stop_z   → close (stop-loss — the cointegration relationship may
                     have broken down, don't wait for it to "come back")
```

Positions are sized as a fixed fraction of capital (`position_fraction`,
default 50%) converted to shares at entry, so P&L and Sharpe are measured
against real capital, not raw spread points.

## Project layout

| File | Purpose |
|---|---|
| `data_loader.py` | Gets daily close prices — real (yfinance, cached to CSV) or synthetic fallback |
| `pairs_selection.py` | Engle-Granger cointegration screen across a ticker universe |
| `backtest.py` | Z-score signal generation + position sizing + transaction costs |
| `metrics.py` | Sharpe, Sortino, CAGR, max drawdown, win rate |
| `run_demo.py` | Ties it all together, saves `output/results.csv` and `output/equity_curve.png` |

## Running it

```bash
pip install -r requirements.txt
python run_demo.py
```

Works with **zero setup** — if there's no cached price data it falls back
to synthetic-but-genuinely-cointegrated price series so the whole pipeline
(screening → backtest → metrics → plots) runs end to end immediately.

To use **real market data** instead (do this before you put this on your
resume — a live-data result is worth far more in an interview than a
synthetic one), run the fetch step on a machine with internet access:

```bash
python data_loader.py --tickers KO PEP XOM CVX JPM GS MSFT AAPL --start 2018-01-01
python run_demo.py
```

Every other script automatically prefers the cache once it exists.

## Reading the output

`output/equity_curve.png` has three panels: the strategy's equity curve vs.
a naive equal-weight benchmark of the two legs, the z-score with entry/exit
bands marked, and the position over time. `output/results.csv` has the
full metrics table (Sharpe, Sortino, CAGR, max drawdown, win rate, beta vs.
benchmark — that beta should be close to zero, which is the whole point of
a market-neutral pairs trade).

## Things to know before you present this in an interview

Be ready to volunteer these — a quant interviewer will ask, and knowing
the limitations of your own backtest unprompted is worth more than the
Sharpe ratio itself:

- **This is in-sample.** The strategy parameters (entry/exit z, lookback)
  were not tuned on a separate out-of-sample period. A real evaluation
  needs a walk-forward split (fit hedge ratio and thresholds on a training
  window, trade on the next window, roll forward) — see `pairs_selection.py`
  for where you'd add a rolling re-estimation of `β`.
- **Transaction costs here are just a flat bps charge.** Real costs also
  include bid-ask spread, market impact for size, and — critically for
  short legs — **borrow costs and short-sale availability**, which this
  doesn't model at all.
- **Multiple-testing risk in the screen.** If you test N tickers you run
  N·(N-1)/2 cointegration tests; at a 5% significance threshold you'd
  expect roughly 5% of *genuinely unrelated* pairs to look "significant"
  by chance, and independent trending series are well known to produce
  *spurious* cointegration/regression results even more often than that
  naive rate suggests (the classic reference is Granger & Newbold, 1974).
  Run the demo and check: some of the "significant" pairs in the printed
  list are between assets that were constructed to be totally independent.
  Tightening the p-value threshold, requiring a plausible economic
  relationship between the two names (same sector, supplier/customer,
  etc.), and out-of-sample validation are the standard fixes.
- **Cointegration is not guaranteed to persist.** Two companies can merge,
  diverge strategically, or one can get disrupted — the `stop_z` exit
  exists specifically to cut losses if the relationship breaks rather than
  assuming it always reverts.

## Extensions worth doing if you have more time

- Walk-forward re-estimation of the hedge ratio instead of a single
  full-sample OLS fit.
- Kalman-filter-based dynamic hedge ratio (lets β drift over time instead
  of being fixed).
- Portfolio version: trade several pairs simultaneously with a capital
  allocator and correlation-aware position limits.
- Sector-constrained universe (only screen pairs within the same GICS
  sector) to reduce the spurious-cointegration problem above.
