import pandas as pd
import pytest

from alphalab.backtest.splits import train_test_split

dates = pd.date_range("2024-01-01", periods=5)

prices = pd.DataFrame(
    {
        "A": [100.0, 101.0, 102.0, 103.0, 104.0],
        "B": [50.0, 49.0, 48.0, 47.0, 46.0],
    },
    index=dates,
)


def test_split_in_middle():
    (train, test) = train_test_split(prices, "2024-01-03")
    assert not train.empty
    assert not test.empty


def test_split_before_all_data():
    with pytest.raises(ValueError):
        train_test_split(prices, "2023-01-01")


def test_split_after_all_data():
    with pytest.raises(ValueError):
        train_test_split(prices, "2025-01-01")
