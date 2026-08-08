import pandas as pd


def train_test_split(prices: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = prices[prices.index < split_date]
    test = prices[prices.index >= split_date]

    if train.empty or test.empty:
        raise ValueError("split_date results in an empty train or test set")

    return (train, test)
