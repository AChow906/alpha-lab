import math

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validate(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("S must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")


def _norm_kind(kind: str) -> str:
    k = kind.lower()
    if k not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return k


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    vol = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    return (d1, d2)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    q: float = 0.0,
) -> float:
    _validate(S, K, T, sigma)
    kind = _norm_kind(kind)

    (d1, d2) = _d1_d2(S, K, T, r, sigma, q)
    disc = math.exp(-r * T)
    carry = math.exp(-q * T)

    if kind == "call":
        return S * carry * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    return K * disc * _norm_cdf(-d2) - S * carry * _norm_cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
    q: float = 0.0,
) -> dict[str, float]:
    _validate(S, K, T, sigma)
    kind = _norm_kind(kind)

    (d1, d2) = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    disc = math.exp(-r * T)
    carry = math.exp(-q * T)
    pdf_d1 = _norm_pdf(d1)

    gamma = carry * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * carry * pdf_d1 * sqrt_t
    theta_common = -S * carry * pdf_d1 * sigma / (2.0 * sqrt_t)

    if kind == "call":
        delta = carry * _norm_cdf(d1)
        theta = theta_common - r * K * disc * _norm_cdf(d2) + q * S * carry * _norm_cdf(d1)
        rho = K * T * disc * _norm_cdf(d2)
    else:
        delta = -carry * _norm_cdf(-d1)
        theta = theta_common + r * K * disc * _norm_cdf(-d2) - q * S * carry * _norm_cdf(-d1)
        rho = -K * T * disc * _norm_cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
