import pandas as pd
import pytest

from alphalab.backtest.engine import backtest

dates = pd.date_range("2024-01-01", periods=3)

prices = pd.DataFrame(
    {
        "A": [100.0, 102.0, 101.0],
        "B": [50.0, 51.0, 52.0],
    },
    index=dates,
)

weights = pd.DataFrame(
    {
        "A": [0.5, 0.5, 0.5],
        "B": [0.5, 0.5, 0.5],
    },
    index=dates,
)


def test_buy_and_hold_no_costs():
    equity, _ = backtest(weights, prices, cost_rate=0.0)

    day1 = 0.5 * (2 / 100) + 0.5 * (1 / 50)
    day2 = 0.5 * (-1 / 102) + 0.5 * (1 / 51)
    expected = (1 + day1) * (1 + day2)

    assert equity.iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_buy_and_hold_with_costs():
    weights_modified = pd.DataFrame(
        {
            "A": [0.5, 0.8, 0.9],
            "B": [0.5, 0.2, 0.1],
        },
        index=dates,
    )

    equity1, _ = backtest(weights, prices, cost_rate=0.0)
    equity2, _ = backtest(weights_modified, prices, cost_rate=0.1)
    assert equity1.iloc[-1] > equity2.iloc[-1]


def test_lag_prevents_lookahead():
    prices_lag = pd.DataFrame(
        {
            "A": [100.0, 110.0, 100.0],
            "B": [100.0, 100.0, 110.0],
        },
        index=dates,
    )

    weights_lag = pd.DataFrame(
        {
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 1.0],
        },
        index=dates,
    )

    equity, _ = backtest(weights_lag, prices_lag, cost_rate=0.0)

    assert equity.iloc[-1] == pytest.approx(1.21, rel=1e-6)
