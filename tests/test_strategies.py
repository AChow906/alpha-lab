import numpy as np
import pandas as pd
import pytest

from alphalab.strategies.cointegration import engle_granger, select_cointegrated
from alphalab.strategies.mean_reversion import mean_reversion
from alphalab.strategies.pairs_trading import pairs_trading
from alphalab.strategies.ts_momentum import ts_momentum
from alphalab.strategies.xs_momentum import xs_momentum

dates = pd.date_range("2024-01-01", periods=5)


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


def test_mean_reversion():
    prices = pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 100.0, 110.0],
        },
        index=dates,
    )

    weights = mean_reversion(prices, lookback=3)
    assert weights["A"].iloc[4] == pytest.approx(-1)


def test_engle_granger_positive():
    rng = np.random.default_rng(42)
    steps = rng.standard_normal(1000)
    x = pd.Series(np.cumsum(steps))
    noise = rng.standard_normal(1000)
    y = pd.Series(2.0 * x + noise)

    (hedge_ratio, p_value) = engle_granger(y, x)
    assert hedge_ratio == pytest.approx(2.0, rel=0.01)
    assert p_value < 0.05


def test_engle_granger_negative():
    rng = np.random.default_rng(42)
    x = pd.Series(np.cumsum(rng.standard_normal(1000)))
    y = pd.Series(np.cumsum(rng.standard_normal(1000)))
    (_, p_value) = engle_granger(y, x)
    assert p_value > 0.05


def _cointegrated_pair(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.standard_normal(n)) + 100.0
    a = common + rng.standard_normal(n) * 0.1
    b = common + rng.standard_normal(n) * 0.1
    return (a, b)


def _random_walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 100.0


def test_select_cointegrated_keeps_only_cointegrated():
    a, b = _cointegrated_pair(n=300, seed=1)
    prices = pd.DataFrame(
        {
            "A": a,
            "B": b,
            "C": _random_walk(300, seed=2),
            "D": _random_walk(300, seed=3),
        }
    )

    selected = select_cointegrated(prices, [("A", "B"), ("C", "D")])

    assert selected == [("A", "B")]


def test_select_cointegrated_threshold_controls_membership():
    a, b = _cointegrated_pair(n=300, seed=1)
    prices = pd.DataFrame({"A": a, "B": b})

    assert select_cointegrated(prices, [("A", "B")], p_threshold=0.05) == [("A", "B")]
    assert select_cointegrated(prices, [("A", "B")], p_threshold=0.0) == []


def test_select_cointegrated_skips_missing_and_short():
    a, b = _cointegrated_pair(n=300, seed=1)
    prices = pd.DataFrame({"A": a, "B": b})

    assert select_cointegrated(prices, [("A", "Z")]) == []
    assert select_cointegrated(prices.iloc[:10], [("A", "B")]) == []


def test_pairs_trading_shorts_wide_spread():
    a, b = _cointegrated_pair(n=200, seed=0)
    a[-1] += 20.0
    prices = pd.DataFrame({"A": a, "B": b})

    weights = pairs_trading(prices, [("A", "B")], lookback=63, entry_z=2.0)

    assert weights["A"].iloc[-1] < 0
    assert weights["B"].iloc[-1] > 0
    assert weights.abs().sum(axis=1).iloc[-1] == pytest.approx(1.0)


def test_pairs_trading_longs_narrow_spread():
    a, b = _cointegrated_pair(n=200, seed=0)
    a[-1] -= 20.0
    prices = pd.DataFrame({"A": a, "B": b})

    weights = pairs_trading(prices, [("A", "B")], lookback=63, entry_z=2.0)

    assert weights["A"].iloc[-1] > 0
    assert weights["B"].iloc[-1] < 0


def test_pairs_trading_flat_during_warmup():
    a, b = _cointegrated_pair(n=200, seed=0)
    prices = pd.DataFrame({"A": a, "B": b})

    weights = pairs_trading(prices, [("A", "B")], lookback=63, entry_z=2.0)

    assert weights.iloc[0].abs().sum() == pytest.approx(0.0)
    assert list(weights.columns) == ["A", "B"]


def test_pairs_trading_exit_band_holds_positions_longer():
    rng = np.random.default_rng(0)
    n = 400
    common = np.cumsum(rng.standard_normal(n)) + 100.0
    a = common + rng.standard_normal(n) * 0.3
    b = common + rng.standard_normal(n) * 0.3
    prices = pd.DataFrame({"A": a, "B": b})

    tight = pairs_trading(prices, [("A", "B")], lookback=63, entry_z=2.0, exit_z=0.1)
    loose = pairs_trading(prices, [("A", "B")], lookback=63, entry_z=2.0, exit_z=1.9)

    active_tight = (tight["A"] != 0).sum()
    active_loose = (loose["A"] != 0).sum()
    assert active_tight > active_loose


def test_pairs_trading_rejects_exit_ge_entry():
    a, b = _cointegrated_pair(n=200, seed=0)
    prices = pd.DataFrame({"A": a, "B": b})

    with pytest.raises(ValueError):
        pairs_trading(prices, [("A", "B")], entry_z=2.0, exit_z=2.0)
