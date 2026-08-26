import pandas as pd

from alphalab.strategies.cointegration import select_cointegrated
from alphalab.strategies.pairs_trading import pairs_trading


def walk_forward_pairs(
    prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    train_window: int,
    rebalance: int,
    p_threshold: float = 0.05,
    lookback: int = 63,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, list[tuple[str, str]]]]:
    if train_window <= 0 or rebalance <= 0:
        raise ValueError("train_window and rebalance must be positive")

    n = len(prices)
    if train_window >= n:
        raise ValueError("train_window must be shorter than the price history")

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    selection: dict[pd.Timestamp, list[tuple[str, str]]] = {}

    for t in range(train_window, n, rebalance):
        rebalance_date = prices.index[t]
        train_slice = prices.iloc[t - train_window : t]
        selected = select_cointegrated(train_slice, pairs, p_threshold)
        selection[rebalance_date] = selected

        if not selected:
            continue

        warmup_start = max(0, t - lookback)
        chunk_end = min(n, t + rebalance)
        window = prices.iloc[warmup_start:chunk_end]
        window_weights = pairs_trading(
            window, selected, lookback=lookback, entry_z=entry_z, exit_z=exit_z
        )
        chunk_index = prices.index[t:chunk_end]
        weights.loc[chunk_index] = window_weights.loc[chunk_index]

    return (weights, selection)
