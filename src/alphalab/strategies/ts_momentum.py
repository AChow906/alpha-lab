import numpy as np
import pandas as pd


def ts_momentum(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    trailing = prices / prices.shift(lookback) - 1
    signs = np.sign(trailing)
    return signs.div(signs.abs().sum(axis=1), axis=0)
