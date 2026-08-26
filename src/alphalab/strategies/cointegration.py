import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

MIN_OBS = 30


def engle_granger(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    hedge_ratio = np.polyfit(x, y, 1)[0]
    residuals = y - hedge_ratio * x
    p_value = adfuller(residuals)[1]

    return (hedge_ratio, p_value)


def select_cointegrated(
    prices: pd.DataFrame, pairs: list[tuple[str, str]], p_threshold: float = 0.05
) -> list[tuple[str, str]]:
    selected = []
    for a, b in pairs:
        if a not in prices.columns or b not in prices.columns:
            continue
        pair_prices = prices[[a, b]].dropna()
        if len(pair_prices) < MIN_OBS:
            continue
        (_, p_value) = engle_granger(pair_prices[a], pair_prices[b])
        if p_value < p_threshold:
            selected.append((a, b))
    return selected
