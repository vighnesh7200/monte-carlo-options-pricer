"""Reproduces all results and figures in the README."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mc_pricer import bs_price, mc_price, METHODS

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
KIND = "call"
SEED = 42
NS = [10**2, 10**3, 10**4, 10**5, 10**6]
LABELS = {"plain": "Plain MC", "antithetic": "Antithetic",
          "control": "Control variate", "both": "Antithetic + control"}

rng = np.random.default_rng(SEED)
true = bs_price(S, K, T, r, sigma, kind=KIND)
print(f"Black-Scholes {KIND}: {true:.6f}\n")

# ---- Experiment 1: error vs N, repeated runs per N ----
rmse = {m: [] for m in METHODS}
var = {m: [] for m in METHODS}
coverage = {m: [] for m in METHODS}
for n in NS:
    reps = int(max(30, min(300, 3e7 // n)))
    for m in METHODS:
        est, se = np.array([mc_price(S, K, T, r, sigma, n, kind=KIND, method=m, rng=rng)
                            for _ in range(reps)]).T
        rmse[m].append(np.sqrt(np.mean((est - true) ** 2)))
        var[m].append(est.var(ddof=1))
        coverage[m].append(np.mean(np.abs(est - true) <= 1.96 * se))

print("RMSE vs Black-Scholes")
print(f"{'N':>9} " + " ".join(f"{LABELS[m]:>22}" for m in METHODS))
for i, n in enumerate(NS):
    print(f"{n:>9} " + " ".join(f"{rmse[m][i]:>22.6f}" for m in METHODS))

print("\nLog-log slope of RMSE vs N (theory: -0.5)")
slopes = {m: np.polyfit(np.log10(NS), np.log10(rmse[m]), 1)[0] for m in METHODS}
for m in METHODS:
    print(f"  {LABELS[m]:<22} {slopes[m]:+.3f}")

print("\nVariance reduction factor vs plain (equal number of normal draws, N=1e5)")
i5 = NS.index(10**5)
vr = {m: var["plain"][i5] / var[m][i5] for m in METHODS}
for m in METHODS:
    print(f"  {LABELS[m]:<22} {vr[m]:6.1f}x   ({(1 - 1/vr[m])*100:5.1f}% variance cut)")

print("\n95% CI empirical coverage (should be ~0.95)")
for m in METHODS:
    print(f"  {LABELS[m]:<22} {np.mean(coverage[m]):.3f}")

# ---- Experiment 2: strikes, one big run each ----
print("\nPrice table, N=1e6, single run (price +/- 1.96 SE)")
print(f"{'K':>5} {'BS':>10} {'Plain MC':>22} {'Antithetic+control':>24}")
for Kx in (80, 100, 120):
    bs = bs_price(S, Kx, T, r, sigma, kind=KIND)
    p, sp = mc_price(S, Kx, T, r, sigma, 10**6, kind=KIND, method="plain", rng=rng)
    b, sb = mc_price(S, Kx, T, r, sigma, 10**6, kind=KIND, method="both", rng=rng)
    print(f"{Kx:>5} {bs:>10.4f} {p:>12.4f} +/- {1.96*sp:.4f} {b:>12.4f} +/- {1.96*sb:.5f}")

# ---- Figures ----
fig, ax = plt.subplots(figsize=(7, 5))
for m, mk in zip(METHODS, "osd^"):
    ax.loglog(NS, rmse[m], marker=mk, label=f"{LABELS[m]} (slope {slopes[m]:+.2f})")
ref = rmse["plain"][0] * (np.array(NS) / NS[0]) ** -0.5
ax.loglog(NS, ref, "k--", alpha=0.5, label=r"$1/\sqrt{N}$ reference")
ax.set(xlabel="Number of random draws N", ylabel="RMSE vs Black-Scholes price",
       title=f"Convergence of Monte Carlo European {KIND} (S=K=100, T=1, r=5%, σ=20%)")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/convergence.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4))
ms = list(METHODS)
ax.bar([LABELS[m] for m in ms], [vr[m] for m in ms], color="#4C72B0")
ax.set_yscale("log"); ax.set_ylabel("Variance reduction vs plain MC (x)")
ax.set_title("Variance reduction at N = 1e5"); plt.xticks(rotation=15, fontsize=8)
for i, m in enumerate(ms):
    ax.text(i, vr[m], f"{vr[m]:.1f}x", ha="center", va="bottom", fontsize=9)
fig.tight_layout(); fig.savefig("figures/variance_reduction.png", dpi=150)
print("\nSaved figures/convergence.png and figures/variance_reduction.png")
