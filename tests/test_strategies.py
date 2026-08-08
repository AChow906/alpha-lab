import pandas as pd
import pytest

from alphalab.strategies.ts_momentum import ts_momentum
from alphalab.strategies.xs_momentum import xs_momentum

dates = pd.date_range("2024-01-01", periods=5)

prices = pd.DataFrame(
    {
        "A": [100.0, 101.0, 102.0, 103.0, 104.0],
        "B": [50.0, 49.0, 48.0, 47.0, 46.0],
    },
    index=dates,
)


def test_tsmom():
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
            "B": [50.0, 49.0, 48.0, 47.0, 46.0],
        },
        index=dates,
    )

    weights = ts_momentum(prices, 3)
    assert weights["A"].iloc[3] == pytest.approx(0.5)
    assert weights["B"].iloc[3] == pytest.approx(-0.5)


def test_xsmom():
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
            "B": [50.0, 49.0, 48.0, 47.0, 46.0],
            "C": [100.0, 105.0, 110.0, 115.0, 120.0],
            "D": [50.0, 45.0, 40.0, 35.0, 30.0],
        },
        index=dates,
    )

    weights = xs_momentum(prices, lookback=3, n_long=1, n_short=1)
    assert weights["C"].iloc[3] == pytest.approx(0.5)
    assert weights["D"].iloc[3] == pytest.approx(-0.5)
    assert weights["A"].iloc[3] == pytest.approx(0.0)
    assert weights["B"].iloc[3] == pytest.approx(0.0)
