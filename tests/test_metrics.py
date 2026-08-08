import numpy as np
import pandas as pd
import pytest

from alphalab.backtest.metrics import (
    annualised_return,
    annualised_volatility,
    max_drawdown,
    sharpe_ratio,
    turnover,
)


def test_annualised_return_one_year():
    daily_ret = 0.001
    returns = pd.Series([daily_ret] * 252)
    result = annualised_return(returns)
    expected = (1 + daily_ret) ** 252 - 1
    assert result == pytest.approx(expected, rel=1e-6)


def test_annualised_return_half_year():
    daily_ret = 0.001
    returns = pd.Series([daily_ret] * 126)
    result = annualised_return(returns)
    expected = ((1 + daily_ret) ** 126) ** 2 - 1
    assert result == pytest.approx(expected, rel=1e-6)


def test_annualised_return_one_year_no_returns():
    daily_ret = 0
    returns = pd.Series([daily_ret] * 126)
    result = annualised_return(returns)
    expected = ((1 + daily_ret) ** 126) ** 2 - 1
    assert result == expected


def test_annualised_volatility_constant_returns():
    daily_ret = 0.001
    returns = pd.Series([daily_ret] * 252)
    result = annualised_volatility(returns)
    assert result == pytest.approx(0.0)


def test_annualised_volatility_alternating_returns():
    daily_ret = [0.01, -0.01]
    returns = pd.Series(daily_ret * 126)
    result = annualised_volatility(returns)
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert result == pytest.approx(expected, rel=1e-6)


def test_sharpe_ratio_constant_returns():
    daily_ret = 0.001
    returns = pd.Series([daily_ret] * 252)
    result = sharpe_ratio(returns)
    assert result == float("inf")


def test_sharpe_ratio_normal():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    result = sharpe_ratio(returns)
    assert result == pytest.approx(
        annualised_return(returns) / annualised_volatility(returns), rel=1e-6
    )


def test_max_drawdown_no_drawdown():
    daily_ret = 0.01
    returns = pd.Series([daily_ret] * 100)
    result = max_drawdown(returns)
    assert result == (0.0, 0)


def test_max_drawdown():
    returns = pd.Series([0.10, -0.10, 0.10])
    dd, _ = max_drawdown(returns)
    assert dd == pytest.approx(0.10, rel=1e-6)


def test_turnover_constant_weights():
    data = {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}
    df = pd.DataFrame(data=data)
    result = turnover(df)
    assert result == pytest.approx(0.0)


def test_turnover_full_flip():
    data = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    df = pd.DataFrame(data=data)
    result = turnover(df)
    assert result == pytest.approx(2.0)


def test_turnover_partial_rebalance():
    data = {"A": [0.6, 0.5, 0.5], "B": [0.4, 0.5, 0.5]}
    df = pd.DataFrame(data=data)
    result = turnover(df)
    assert result == pytest.approx(0.1)
