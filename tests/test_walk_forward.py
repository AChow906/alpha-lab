import numpy as np
import pandas as pd
import pytest

from alphalab.backtest.walk_forward import walk_forward_pairs


def _walk(rng: np.random.Generator, n: int, start: float = 100.0) -> np.ndarray:
    return start + np.cumsum(rng.standard_normal(n))


def _windowed_cointegration(n: int = 800, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n // 2
    idio = 0.5

    common_first = _walk(rng, half)
    common_second = _walk(rng, n - half)

    a = np.empty(n)
    b = np.empty(n)
    c = np.empty(n)
    d = np.empty(n)

    # First half: A,B share a trend (stationary spread); C,D drift apart.
    a[:half] = common_first + rng.standard_normal(half) * idio
    b[:half] = common_first + rng.standard_normal(half) * idio
    c[:half] = _walk(rng, half)
    d[:half] = _walk(rng, half)

    # Second half: the roles swap.
    a[half:] = _walk(rng, n - half)
    b[half:] = _walk(rng, n - half)
    c[half:] = common_second + rng.standard_normal(n - half) * idio
    d[half:] = common_second + rng.standard_normal(n - half) * idio

    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=dates)


PAIRS = [("A", "B"), ("C", "D")]


def test_walk_forward_selection_tracks_regime():
    prices = _windowed_cointegration()
    (_, selection) = walk_forward_pairs(prices, PAIRS, train_window=200, rebalance=200)

    dates = prices.index
    assert ("A", "B") in selection[dates[200]]
    assert ("C", "D") not in selection[dates[200]]
    assert ("A", "B") in selection[dates[400]]
    assert ("C", "D") in selection[dates[600]]
    assert ("A", "B") not in selection[dates[600]]


def test_walk_forward_weights_span_all_dates():
    prices = _windowed_cointegration()
    (weights, _) = walk_forward_pairs(prices, PAIRS, train_window=200, rebalance=200)

    assert weights.index.equals(prices.index)
    assert list(weights.columns) == list(prices.columns)


def test_walk_forward_flat_before_first_rebalance():
    prices = _windowed_cointegration()
    (weights, _) = walk_forward_pairs(prices, PAIRS, train_window=200, rebalance=200)

    assert (weights.iloc[:200] == 0.0).all().all()


def test_walk_forward_gross_exposure_is_zero_or_one():
    prices = _windowed_cointegration()
    (weights, _) = walk_forward_pairs(prices, PAIRS, train_window=200, rebalance=200)

    gross = weights.abs().sum(axis=1)
    assert ((gross < 1e-9) | np.isclose(gross, 1.0)).all()
    assert weights.abs().to_numpy().sum() > 0.0


def test_walk_forward_zero_pairs_stays_flat():
    prices = _windowed_cointegration()
    (weights, selection) = walk_forward_pairs(
        prices, PAIRS, train_window=200, rebalance=200, p_threshold=0.0
    )

    assert (weights == 0.0).all().all()
    assert all(chosen == [] for chosen in selection.values())


def test_walk_forward_rejects_bad_params():
    prices = _windowed_cointegration(n=100)
    with pytest.raises(ValueError):
        walk_forward_pairs(prices, PAIRS, train_window=100, rebalance=50)
    with pytest.raises(ValueError):
        walk_forward_pairs(prices, PAIRS, train_window=0, rebalance=50)
    with pytest.raises(ValueError):
        walk_forward_pairs(prices, PAIRS, train_window=50, rebalance=0)
