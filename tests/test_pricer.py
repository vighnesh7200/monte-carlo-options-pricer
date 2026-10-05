import numpy as np
from mc_pricer import bs_price, mc_price

P = dict(S=100, K=100, T=1, r=0.05, sigma=0.2)

def test_put_call_parity():
    c, p = bs_price(**P, kind="call"), bs_price(**P, kind="put")
    assert abs(c - p - (P["S"] - P["K"] * np.exp(-P["r"] * P["T"]))) < 1e-10

def test_mc_within_4_se_of_bs():
    for kind in ("call", "put"):
        for m in ("plain", "antithetic", "control", "both"):
            est, se = mc_price(**P, n=200_000, kind=kind, method=m, rng=np.random.default_rng(1))
            assert abs(est - bs_price(**P, kind=kind)) < 4 * se
