from pathlib import Path

import pandas as pd

from alphalab.backtest.engine import backtest
from alphalab.backtest.splits import train_test_split
from alphalab.data.loader import load_prices
from alphalab.data.universe import PAIRS
from alphalab.plotting import plot_prices
from alphalab.strategies.cointegration import engle_granger
from alphalab.strategies.pairs_trading import pairs_trading

ENTRY_Z = 2.0
EXIT_Z = 0.5
COST_RATE = 0.001
# Only trade pairs that mean-revert in-sample. Deciding the universe on the
# training set keeps the out-of-sample test honest (no lookahead).
P_THRESHOLD = 0.05
# exit_z just below entry_z ~= the original no-hold behaviour, so the sweep
# isolates how much the exit band cuts turnover.
EXIT_Z_SWEEP = [1.9, 1.0, 0.5, 0.1]

tickers = sorted({ticker for pair in PAIRS for ticker in pair})
prices = load_prices(tickers, "2015-01-01", "2024-12-31")
(train, test) = train_test_split(prices, "2020-01-01")

print("Cointegration test (in-sample):")
cointegrated_pairs = []
for a, b in PAIRS:
    pair_prices = train[[a, b]].dropna()
    (hedge_ratio, p_value) = engle_granger(pair_prices[a], pair_prices[b])
    is_cointegrated = p_value < P_THRESHOLD
    verdict = "cointegrated" if is_cointegrated else "not cointegrated"
    print(f"  {a}/{b}: hedge={hedge_ratio:.2f} p={p_value:.3f} -> {verdict}")
    if is_cointegrated:
        cointegrated_pairs.append((a, b))

if not cointegrated_pairs:
    raise SystemExit("No cointegrated pairs in-sample; nothing to trade.")
print(f"\nTrading {len(cointegrated_pairs)} cointegrated pair(s): {cointegrated_pairs}")

weights = pairs_trading(prices, cointegrated_pairs, entry_z=ENTRY_Z, exit_z=EXIT_Z)
(train_equity, train_stats) = backtest(weights.loc[train.index], train, cost_rate=COST_RATE)
(test_equity, test_stats) = backtest(weights.loc[test.index], test, cost_rate=COST_RATE)

print(f"\nStrategy stats (entry_z={ENTRY_Z}, exit_z={EXIT_Z}, cost={COST_RATE}):")
stats = pd.DataFrame({"In-sample": train_stats, "Out-of-sample": test_stats})
print(stats)

print("\nExit-band sweep (out-of-sample):")
sweep = {}
for exit_z in EXIT_Z_SWEEP:
    sweep_weights = pairs_trading(prices, cointegrated_pairs, entry_z=ENTRY_Z, exit_z=exit_z)
    (_, sweep_stats) = backtest(sweep_weights.loc[test.index], test, cost_rate=COST_RATE)
    sweep[f"exit_z={exit_z}"] = sweep_stats
print(pd.DataFrame(sweep).loc[["turnover", "sharpe_ratio", "annualised_return"]])

curves = pd.DataFrame({"train": train_equity, "test": test_equity})
plot_prices(curves, Path("results/pairs_trading.png"), title="Pairs Trading Train vs Test")
