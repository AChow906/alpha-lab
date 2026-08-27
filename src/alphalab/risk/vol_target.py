import numpy as np
import pandas as pd

from alphalab.backtest.metrics import TRADING_DAYS_PER_YEAR


def vol_target(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 63,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    returns = prices.pct_change()
    strategy_returns = (weights.shift(1) * returns).sum(axis=1)
    trailing_vol = strategy_returns.rolling(lookback).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    scale = pd.Series(0.0, index=weights.index)
    estimable = trailing_vol > 0
    scale[estimable] = (target_vol / trailing_vol[estimable]).clip(upper=max_leverage)

    return weights.mul(scale, axis=0)
