from pathlib import Path

import pandas as pd

from alphalab.backtest.engine import backtest
from alphalab.data.loader import load_prices
from alphalab.data.universe import SECTOR_ETFS
from alphalab.plotting import plot_prices
from alphalab.strategies.ts_momentum import ts_momentum
from alphalab.strategies.xs_momentum import xs_momentum

prices = load_prices(SECTOR_ETFS, "2015-01-01", "2024-12-31")
tsmom_weights = ts_momentum(prices=prices)
xsmom_weights = xs_momentum(prices=prices)

(tsmom_equity, ts_stats) = backtest(tsmom_weights, prices, cost_rate=0.001)
(xsmom_equity, xs_stats) = backtest(xsmom_weights, prices, cost_rate=0.001)

curves = pd.DataFrame(
    {
        "TSMOM": tsmom_equity,
        "XSMOM": xsmom_equity,
    }
)
plot_prices(curves, Path("results/momentum.png"), title="Momentum Strategies")
