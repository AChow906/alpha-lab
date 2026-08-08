import pandas as pd


def mean_reversion(prices: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()
    z = (prices - rolling_mean) / rolling_std
    raw_weights = -z
    return raw_weights.div(raw_weights.abs().sum(axis=1), axis=0)
