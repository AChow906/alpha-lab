import math

import numpy as np


def _validate(S: float, K: float, T: float, sigma: float, steps: int) -> None:
    if S <= 0:
        raise ValueError("S must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")


def _norm_kind(kind: str) -> str:
    k = kind.lower()
    if k not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return k


def _norm_exercise(exercise: str) -> str:
    e = exercise.lower()
    if e not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'")
    return e


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 500,
    kind: str = "call",
    exercise: str = "european",
    q: float = 0.0,
) -> float:
    _validate(S, K, T, sigma, steps)
    kind = _norm_kind(kind)
    exercise = _norm_exercise(exercise)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)

    sign = 1.0 if kind == "call" else -1.0
    american = exercise == "american"

    i = np.arange(steps + 1)
    asset = S * u**i * d ** (steps - i)
    values = np.maximum(sign * (asset - K), 0.0)

    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            i = np.arange(step + 1)
            asset = S * u**i * d ** (step - i)
            intrinsic = np.maximum(sign * (asset - K), 0.0)
            values = np.maximum(values, intrinsic)

    return float(values[0])
