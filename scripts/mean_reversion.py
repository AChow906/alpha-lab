from pathlib import Path

import pandas as pd

from alphalab.backtest.engine import backtest
from alphalab.backtest.splits import train_test_split
from alphalab.data.loader import load_prices
from alphalab.data.universe import SECTOR_ETFS
from alphalab.plotting import plot_prices
from alphalab.strategies.mean_reversion import mean_reversion

prices = load_prices(SECTOR_ETFS, "2015-01-01", "2024-12-31")
(train, test) = train_test_split(prices, "2020-01-01")

mean_reversion_weights = mean_reversion(prices=prices)

(mean_reversion_train, mean_reversion_train_stats) = backtest(
    mean_reversion_weights.loc[train.index], train, cost_rate=0.001
)
(mean_reversion_test, mean_reversion_test_stats) = backtest(
    mean_reversion_weights.loc[test.index], test, cost_rate=0.001
)

curves = pd.DataFrame(
    {
        "train": mean_reversion_train,
        "test": mean_reversion_test,
    }
)

stats = pd.DataFrame(
    {"In-sample": mean_reversion_train_stats, "Out-of-sample": mean_reversion_test_stats}
)
print(stats)
plot_prices(curves, Path("results/mean_reversion.png"), title="Mean Reversion Train vs Test")
