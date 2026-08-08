import pandas as pd


def xs_momentum(
    prices: pd.DataFrame, lookback: int = 126, n_long: int = 3, n_short: int = 3
) -> pd.DataFrame:
    trailing = prices / prices.shift(lookback) - 1
    ranks = trailing.rank(axis=1)
    n = trailing.shape[1]

    longs = (ranks > n - n_long).astype(float)
    shorts = (ranks <= n_short).astype(float)
    weights = longs - shorts
    weights = weights.div(weights.abs().sum(axis=1), axis=0)

    return weights
