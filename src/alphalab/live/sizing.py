import math
from collections.abc import Mapping

import pandas as pd


def price_of(prices: pd.Series | Mapping[str, float], symbol: str) -> float | None:
    try:
        price = float(prices[symbol])
    except (KeyError, TypeError, ValueError):
        return None
    if math.isnan(price):
        return None
    return price


def latest_row(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[-1]


def target_shares(
    weights: pd.Series | Mapping[str, float],
    equity: float,
    prices: pd.Series | Mapping[str, float],
) -> dict[str, float]:
    if equity < 0.0:
        raise ValueError("equity must be non-negative")

    shares: dict[str, float] = {}
    for symbol, weight in weights.items():
        value = float(weight)
        if math.isnan(value):
            continue
        price = price_of(prices, symbol)
        if price is None or price <= 0.0:
            continue
        shares[str(symbol)] = float(round(value * equity / price))

    return shares


def close_atr(prices: pd.Series, lookback: int = 14) -> float:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    changes = prices.diff().abs().dropna()
    if changes.empty:
        return 0.0
    window = changes.iloc[-lookback:]
    return float(window.mean())


def risk_sized_shares(equity: float, risk_fraction: float, atr: float, price: float) -> float:
    if atr <= 0.0 or price <= 0.0 or risk_fraction <= 0.0:
        return 0.0
    budget = equity * risk_fraction
    return float(math.floor(budget / atr))
