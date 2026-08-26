import numpy as np
import pandas as pd


def pairs_trading(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    lookback: int = 63,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    if exit_z >= entry_z:
        raise ValueError("exit_z must be smaller than entry_z")

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for a, b in pairs:
        pa = prices[a]
        pb = prices[b]

        hedge_ratio = pa.rolling(lookback).cov(pb) / pb.rolling(lookback).var()
        spread = pa - hedge_ratio * pb

        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        z = (spread - spread_mean) / spread_std

        target = pd.Series(np.nan, index=prices.index)
        target[z > entry_z] = -1.0
        target[z < -entry_z] = 1.0
        target[z.abs() < exit_z] = 0.0
        position = target.ffill().fillna(0.0)

        weights[a] = weights[a] + position
        weights[b] = weights[b] + (-position * hedge_ratio).fillna(0.0)

    gross = weights.abs().sum(axis=1)
    weights = weights.div(gross, axis=0).fillna(0.0)
    return weights
