import pandas as pd

from alphalab.backtest.metrics import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
    turnover,
)


def backtest(
    weights: pd.DataFrame, prices: pd.DataFrame, cost_rate: float = 0.0
) -> tuple[pd.Series, dict]:
    returns = prices.pct_change()
    lagged_weights = weights.shift(1)

    portfoilio_returns = (lagged_weights * returns).sum(axis=1)
    portfoilio_returns.fillna(0)

    transaction_costs = cost_rate * weights.diff().abs().sum(axis=1)
    transaction_costs.fillna(0)

    net_returns = portfoilio_returns - transaction_costs
    equity = (1 + net_returns).cumprod()

    stats = {
        "annualised_return": annualised_return(net_returns),
        "annualised_volatility": annualised_volatility(net_returns),
        "sharpe_ratio": sharpe_ratio(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "turnover": turnover(weights),
    }

    return (equity, stats)
