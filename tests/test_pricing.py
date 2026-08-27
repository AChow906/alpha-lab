import math

import pytest

from alphalab.pricing import binomial_price, bs_greeks, bs_price

S = 100.0
K = 100.0
T = 1.0
R = 0.05
SIGMA = 0.2


def test_bs_reference_values():
    assert bs_price(S, K, T, R, SIGMA, kind="call") == pytest.approx(10.4506, abs=1e-3)
    assert bs_price(S, K, T, R, SIGMA, kind="put") == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    call = bs_price(S, K, T, R, SIGMA, kind="call")
    put = bs_price(S, K, T, R, SIGMA, kind="put")
    assert call - put == pytest.approx(S - K * math.exp(-R * T), abs=1e-10)


def test_put_call_parity_with_dividend():
    q = 0.03
    call = bs_price(S, K, T, R, SIGMA, kind="call", q=q)
    put = bs_price(S, K, T, R, SIGMA, kind="put", q=q)
    expected = S * math.exp(-q * T) - K * math.exp(-R * T)
    assert call - put == pytest.approx(expected, abs=1e-10)


def test_delta_matches_finite_difference():
    h = 1e-4
    for kind in ("call", "put"):
        up = bs_price(S + h, K, T, R, SIGMA, kind=kind)
        down = bs_price(S - h, K, T, R, SIGMA, kind=kind)
        fd = (up - down) / (2.0 * h)
        assert bs_greeks(S, K, T, R, SIGMA, kind=kind)["delta"] == pytest.approx(fd, abs=1e-6)


def test_gamma_matches_finite_difference():
    h = 0.5
    for kind in ("call", "put"):
        up = bs_price(S + h, K, T, R, SIGMA, kind=kind)
        mid = bs_price(S, K, T, R, SIGMA, kind=kind)
        down = bs_price(S - h, K, T, R, SIGMA, kind=kind)
        fd = (up - 2.0 * mid + down) / (h * h)
        assert bs_greeks(S, K, T, R, SIGMA, kind=kind)["gamma"] == pytest.approx(fd, rel=1e-3)


def test_vega_matches_finite_difference():
    h = 1e-4
    for kind in ("call", "put"):
        up = bs_price(S, K, T, R, SIGMA + h, kind=kind)
        down = bs_price(S, K, T, R, SIGMA - h, kind=kind)
        fd = (up - down) / (2.0 * h)
        assert bs_greeks(S, K, T, R, SIGMA, kind=kind)["vega"] == pytest.approx(fd, rel=1e-4)


def test_theta_matches_finite_difference():
    h = 1e-4
    for kind in ("call", "put"):
        up = bs_price(S, K, T + h, R, SIGMA, kind=kind)
        down = bs_price(S, K, T - h, R, SIGMA, kind=kind)
        fd = -(up - down) / (2.0 * h)
        assert bs_greeks(S, K, T, R, SIGMA, kind=kind)["theta"] == pytest.approx(fd, abs=1e-4)


def test_rho_matches_finite_difference():
    h = 1e-5
    for kind in ("call", "put"):
        up = bs_price(S, K, T, R + h, SIGMA, kind=kind)
        down = bs_price(S, K, T, R - h, SIGMA, kind=kind)
        fd = (up - down) / (2.0 * h)
        assert bs_greeks(S, K, T, R, SIGMA, kind=kind)["rho"] == pytest.approx(fd, rel=1e-4)


def test_binomial_converges_to_bs():
    for kind in ("call", "put"):
        bs = bs_price(S, K, T, R, SIGMA, kind=kind)
        err_coarse = abs(binomial_price(S, K, T, R, SIGMA, steps=50, kind=kind) - bs)
        err_fine = abs(binomial_price(S, K, T, R, SIGMA, steps=1000, kind=kind) - bs)
        assert err_fine < err_coarse
        assert err_fine < 5e-3


def test_american_put_at_least_european():
    european = binomial_price(S, K, T, R, SIGMA, steps=500, kind="put", exercise="european")
    american = binomial_price(S, K, T, R, SIGMA, steps=500, kind="put", exercise="american")
    assert american >= european - 1e-9


def test_american_put_premium_deep_in_the_money():
    european = binomial_price(60.0, K, T, R, SIGMA, steps=500, kind="put", exercise="european")
    american = binomial_price(60.0, K, T, R, SIGMA, steps=500, kind="put", exercise="american")
    assert american > european + 1e-4
    assert american >= K - 60.0 - 1e-6


def test_american_call_equals_european_without_dividends():
    european = binomial_price(S, K, T, R, SIGMA, steps=500, kind="call", exercise="european")
    american = binomial_price(S, K, T, R, SIGMA, steps=500, kind="call", exercise="american")
    assert american == pytest.approx(european, abs=1e-9)


def test_bs_validates_inputs():
    with pytest.raises(ValueError):
        bs_price(-1.0, K, T, R, SIGMA)
    with pytest.raises(ValueError):
        bs_price(S, -1.0, T, R, SIGMA)
    with pytest.raises(ValueError):
        bs_price(S, K, 0.0, R, SIGMA)
    with pytest.raises(ValueError):
        bs_price(S, K, T, R, 0.0)
    with pytest.raises(ValueError):
        bs_price(S, K, T, R, SIGMA, kind="straddle")
    with pytest.raises(ValueError):
        bs_greeks(S, K, -1.0, R, SIGMA)


def test_binomial_validates_inputs():
    with pytest.raises(ValueError):
        binomial_price(-1.0, K, T, R, SIGMA)
    with pytest.raises(ValueError):
        binomial_price(S, K, 0.0, R, SIGMA)
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, 0.0)
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, SIGMA, steps=0)
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, SIGMA, kind="straddle")
    with pytest.raises(ValueError):
        binomial_price(S, K, T, R, SIGMA, exercise="bermudan")
