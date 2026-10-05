"""Monte Carlo pricing of European options under Black-Scholes dynamics,
with antithetic and control variate variance reduction.

Risk-neutral dynamics: S_T = S0 * exp((r - q - sigma^2/2) T + sigma sqrt(T) Z), Z ~ N(0,1)
"""
import numpy as np
from scipy.stats import norm

METHODS = ("plain", "antithetic", "control", "both")


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes closed form for a European call/put."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _terminal(S, T, r, sigma, q, z):
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def _payoff(ST, K, kind):
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def mc_price(S, K, T, r, sigma, n, q=0.0, kind="call", method="plain", rng=None):
    """Monte Carlo price using n total normal draws (equal cost across methods).

    Returns (price, standard_error).

    method:
      plain       - n independent paths
      antithetic  - n/2 pairs (Z, -Z); average each pair
      control     - control variate Y = e^{-rT} S_T, known mean S0 e^{-qT}
      both        - antithetic pairs + control variate on the pair averages
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rng = np.random.default_rng() if rng is None else rng
    disc = np.exp(-r * T)
    use_anti = method in ("antithetic", "both")
    use_cv = method in ("control", "both")

    if use_anti:
        z = rng.standard_normal(n // 2)
        X = 0.5 * disc * (_payoff(_terminal(S, T, r, sigma, q, z), K, kind)
                          + _payoff(_terminal(S, T, r, sigma, q, -z), K, kind))
        Y = 0.5 * disc * (_terminal(S, T, r, sigma, q, z) + _terminal(S, T, r, sigma, q, -z))
    else:
        z = rng.standard_normal(n)
        ST = _terminal(S, T, r, sigma, q, z)
        X = disc * _payoff(ST, K, kind)
        Y = disc * ST

    if use_cv:
        EY = S * np.exp(-q * T)                      # exact E[e^{-rT} S_T] (q=0 -> S0)
        c = np.cov(X, Y)
        beta = c[0, 1] / c[1, 1]                     # estimated optimal coefficient
        X = X - beta * (Y - EY)

    return X.mean(), X.std(ddof=1) / np.sqrt(len(X))
