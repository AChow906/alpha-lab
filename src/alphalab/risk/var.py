import math

import numpy as np
import pandas as pd

_SQRT_2PI = math.sqrt(2.0 * math.pi)

_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation to the inverse standard-normal CDF
    # (max relative error ~1.15e-9), used to avoid a scipy dependency.
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")

    low = 0.02425
    high = 1.0 - low

    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p <= high:
        q = p - 0.5
        r = q * q
        return (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    )


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _clean(returns: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("returns must contain at least one finite value")
    return values


def simulate_pnl(
    returns: pd.Series | np.ndarray,
    horizon: int = 1,
    n_sims: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    values = _clean(returns)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_sims, horizon), replace=True)
    return np.prod(1.0 + draws, axis=1) - 1.0


def monte_carlo_var(
    returns: pd.Series | np.ndarray,
    horizon: int = 1,
    n_sims: int = 10000,
    level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")

    losses = -simulate_pnl(returns, horizon=horizon, n_sims=n_sims, seed=seed)
    var = float(np.quantile(losses, level))
    tail = losses[losses >= var]
    es = float(tail.mean()) if tail.size else var
    return (var, es)


def parametric_var(
    returns: pd.Series | np.ndarray,
    horizon: int = 1,
    level: float = 0.95,
) -> tuple[float, float]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")

    values = _clean(returns)
    mu = float(values.mean()) * horizon
    sigma = float(values.std(ddof=1)) * math.sqrt(horizon)

    z = _norm_ppf(level)
    var = sigma * z - mu
    es = sigma * _norm_pdf(z) / (1.0 - level) - mu
    return (var, es)
