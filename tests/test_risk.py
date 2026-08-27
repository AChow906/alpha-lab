import numpy as np
import pandas as pd
import pytest

from alphalab.backtest.engine import backtest
from alphalab.risk.var import monte_carlo_var, parametric_var, simulate_pnl
from alphalab.risk.vol_target import vol_target


def _synthetic_prices(n: int, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    daily = rng.normal(0.0, sigma, n)
    levels = 100.0 * np.cumprod(1.0 + daily)
    index = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": levels}, index=index)


def _unit_weights(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


def test_vol_target_hits_target():
    prices = _synthetic_prices(n=2000, sigma=0.02, seed=0)
    weights = _unit_weights(prices)

    scaled = vol_target(weights, prices, target_vol=0.10, lookback=63, max_leverage=10.0)
    (_, stats) = backtest(scaled, prices)

    assert stats["annualised_volatility"] == pytest.approx(0.10, abs=0.02)


def test_vol_target_lookahead_safe():
    prices = _synthetic_prices(n=1000, sigma=0.02, seed=1)
    weights = _unit_weights(prices)
    t = 500

    scaled = vol_target(weights, prices, target_vol=0.10, lookback=63)

    altered = prices.copy()
    altered.iloc[t + 1 :] *= 1.5
    scaled_altered = vol_target(weights, altered, target_vol=0.10, lookback=63)

    assert np.allclose(scaled.iloc[: t + 1].to_numpy(), scaled_altered.iloc[: t + 1].to_numpy())


def test_vol_target_respects_leverage_cap():
    prices = _synthetic_prices(n=500, sigma=0.001, seed=2)
    weights = _unit_weights(prices)
    max_leverage = 3.0

    scaled = vol_target(weights, prices, target_vol=0.10, lookback=63, max_leverage=max_leverage)
    gross = scaled.abs().sum(axis=1)

    assert gross.max() <= max_leverage + 1e-9
    assert gross.max() == pytest.approx(max_leverage)


def test_vol_target_handles_warmup_and_zero_vol():
    index = pd.date_range("2015-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"A": 100.0}, index=index)
    weights = _unit_weights(prices)

    scaled = vol_target(weights, prices, target_vol=0.10, lookback=63)

    assert np.isfinite(scaled.to_numpy()).all()
    assert not scaled.isna().any().any()
    assert (scaled == 0.0).all().all()


def test_vol_target_validates_inputs():
    prices = _synthetic_prices(n=200, sigma=0.01, seed=3)
    weights = _unit_weights(prices)

    with pytest.raises(ValueError):
        vol_target(weights, prices, target_vol=0.0)
    with pytest.raises(ValueError):
        vol_target(weights, prices, lookback=0)
    with pytest.raises(ValueError):
        vol_target(weights, prices, max_leverage=-1.0)


def _normal_returns(n: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, sigma, n)


def test_mc_approximates_parametric():
    returns = _normal_returns(n=5000, sigma=0.01, seed=0)

    (var_mc, es_mc) = monte_carlo_var(returns, horizon=1, n_sims=50000, level=0.95, seed=1)
    (var_p, es_p) = parametric_var(returns, horizon=1, level=0.95)

    assert var_mc == pytest.approx(var_p, rel=0.12)
    assert es_mc == pytest.approx(es_p, rel=0.12)


def test_es_ge_var():
    returns = _normal_returns(n=3000, sigma=0.01, seed=4)

    (var_mc, es_mc) = monte_carlo_var(returns, n_sims=40000, level=0.95, seed=2)
    (var_p, es_p) = parametric_var(returns, level=0.95)

    assert es_mc >= var_mc
    assert es_p >= var_p


def test_higher_confidence_larger_var():
    returns = _normal_returns(n=3000, sigma=0.01, seed=5)

    (var95_mc, _) = monte_carlo_var(returns, n_sims=40000, level=0.95, seed=3)
    (var99_mc, _) = monte_carlo_var(returns, n_sims=40000, level=0.99, seed=3)
    (var95_p, _) = parametric_var(returns, level=0.95)
    (var99_p, _) = parametric_var(returns, level=0.99)

    assert var99_mc > var95_mc
    assert var99_p > var95_p


def test_var_are_positive_losses():
    returns = _normal_returns(n=3000, sigma=0.01, seed=6)

    (var_p, es_p) = parametric_var(returns, level=0.95)

    assert var_p > 0
    assert es_p > var_p


def test_var_validates_inputs():
    returns = _normal_returns(n=100, sigma=0.01, seed=7)

    with pytest.raises(ValueError):
        monte_carlo_var(returns, level=1.0)
    with pytest.raises(ValueError):
        monte_carlo_var(returns, level=0.0)
    with pytest.raises(ValueError):
        monte_carlo_var(returns, horizon=0)
    with pytest.raises(ValueError):
        monte_carlo_var(returns, n_sims=0)
    with pytest.raises(ValueError):
        parametric_var(returns, level=1.0)
    with pytest.raises(ValueError):
        parametric_var(returns, horizon=0)


def test_simulate_pnl_shape_and_determinism():
    returns = _normal_returns(n=1000, sigma=0.01, seed=8)

    first = simulate_pnl(returns, horizon=5, n_sims=1000, seed=9)
    second = simulate_pnl(returns, horizon=5, n_sims=1000, seed=9)

    assert first.shape == (1000,)
    assert np.array_equal(first, second)
