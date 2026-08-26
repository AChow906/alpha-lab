from pathlib import Path

import pandas as pd

from alphalab.backtest.engine import backtest
from alphalab.backtest.walk_forward import walk_forward_pairs
from alphalab.data.loader import load_prices
from alphalab.data.universe import PAIRS
from alphalab.plotting import plot_prices
from alphalab.strategies.cointegration import select_cointegrated
from alphalab.strategies.pairs_trading import pairs_trading

ENTRY_Z = 2.0
EXIT_Z = 0.5
COST_RATE = 0.001
P_THRESHOLD = 0.05
# Two years of daily data to judge cointegration, re-selected quarterly.
TRAIN_WINDOW = 504
REBALANCE = 63
LOOKBACK = 63
# The fixed baseline decides its universe once, on everything before this date.
FIXED_SPLIT = "2020-01-01"

tickers = sorted({ticker for pair in PAIRS for ticker in pair})
prices = load_prices(tickers, "2015-01-01", "2024-12-31")

(wf_weights, selection) = walk_forward_pairs(
    prices,
    PAIRS,
    train_window=TRAIN_WINDOW,
    rebalance=REBALANCE,
    p_threshold=P_THRESHOLD,
    lookback=LOOKBACK,
    entry_z=ENTRY_Z,
    exit_z=EXIT_Z,
)

print("Walk-forward selected set over time:")
previous: list[tuple[str, str]] | None = None
for rebalance_date, chosen in selection.items():
    marker = "" if chosen == previous else "  <- changed"
    label = ", ".join(f"{a}/{b}" for a, b in chosen) if chosen else "(none)"
    print(f"  {rebalance_date.date()}: {len(chosen):2d} pairs  {label}{marker}")
    previous = chosen

counts = pd.Series({date: len(chosen) for date, chosen in selection.items()})
print(f"\nSelected pair count: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")

fixed_train = prices[prices.index < FIXED_SPLIT]
fixed_pairs = select_cointegrated(fixed_train, PAIRS, p_threshold=P_THRESHOLD)
print(f"\nFixed (Phase 4) selection on pre-{FIXED_SPLIT}: {fixed_pairs}")
fixed_weights = pairs_trading(
    prices, fixed_pairs, lookback=LOOKBACK, entry_z=ENTRY_Z, exit_z=EXIT_Z
)

# Compare both strategies over the span the walk-forward can actually trade.
first_active = prices.index[TRAIN_WINDOW]
common = prices.loc[first_active:]
(wf_equity, wf_stats) = backtest(wf_weights.loc[common.index], common, cost_rate=COST_RATE)
(fixed_equity, fixed_stats) = backtest(fixed_weights.loc[common.index], common, cost_rate=COST_RATE)

print(f"\nStats from {first_active.date()} (entry_z={ENTRY_Z}, exit_z={EXIT_Z}, cost={COST_RATE}):")
stats = pd.DataFrame({"Walk-forward": wf_stats, "Fixed": fixed_stats})
print(stats)

curves = pd.DataFrame({"walk_forward": wf_equity, "fixed": fixed_equity})
plot_prices(
    curves,
    Path("results/pairs_trading_walkforward.png"),
    title="Pairs Trading: Walk-Forward vs Fixed Selection",
)
