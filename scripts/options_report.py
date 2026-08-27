from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from alphalab.pricing import binomial_price, bs_greeks, bs_price

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

S = 100.0
K = 100.0
T = 1.0
R = 0.05
SIGMA = 0.2
Q = 0.0
STEP_GRID = (10, 25, 50, 100, 250, 500, 1000, 2000)
SPOT_GRID = np.linspace(50.0, 150.0, 101)

print(f"Sample option: S={S}, K={K}, T={T}, r={R:.2%}, sigma={SIGMA:.0%}, q={Q:.0%}\n")

for kind in ("call", "put"):
    price = bs_price(S, K, T, R, SIGMA, kind=kind, q=Q)
    greeks = bs_greeks(S, K, T, R, SIGMA, kind=kind, q=Q)
    print(f"Black-Scholes {kind}: {price:.4f}")
    print(
        "  delta={delta:.4f}  gamma={gamma:.4f}  vega={vega:.4f}  "
        "theta={theta:.4f}  rho={rho:.4f}".format(**greeks)
    )
print("\nGreek units: vega per 1.00 change in sigma, theta per year, rho per 1.00 change in r.\n")

bs_call = bs_price(S, K, T, R, SIGMA, kind="call", q=Q)
rows = {}
for steps in STEP_GRID:
    tree = binomial_price(S, K, T, R, SIGMA, steps=steps, kind="call", exercise="european", q=Q)
    rows[steps] = {"binomial": tree, "black_scholes": bs_call, "abs_error": abs(tree - bs_call)}
convergence = pd.DataFrame(rows).T
convergence.index.name = "steps"
print("Binomial -> Black-Scholes convergence (European call):")
print(convergence.to_string(float_format=lambda x: f"{x:.6f}"))

eu_put = binomial_price(S, K, T, R, SIGMA, steps=1000, kind="put", exercise="european", q=Q)
am_put = binomial_price(S, K, T, R, SIGMA, steps=1000, kind="put", exercise="american", q=Q)
eu_call = binomial_price(S, K, T, R, SIGMA, steps=1000, kind="call", exercise="european", q=Q)
am_call = binomial_price(S, K, T, R, SIGMA, steps=1000, kind="call", exercise="american", q=Q)

print("\nAmerican vs European (1000-step CRR tree):")
print(f"  put : european={eu_put:.4f}  american={am_put:.4f}  premium={am_put - eu_put:.4f}")
print(f"  call: european={eu_call:.4f}  american={am_call:.4f}  premium={am_call - eu_call:.4f}")
print("  (American call on a non-dividend stock matches the European call.)")

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

errors = [abs(binomial_price(S, K, T, R, SIGMA, steps=n, kind="call") - bs_call) for n in STEP_GRID]
fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(STEP_GRID, errors, marker="o", color="steelblue")
ax.set_xlabel("Tree steps")
ax.set_ylabel("Absolute error vs Black-Scholes")
ax.set_title("CRR binomial convergence to Black-Scholes (European call)")
ax.grid(True, which="both")
fig.savefig(results_dir / "options_convergence.png")
plt.close(fig)

call_delta = [bs_greeks(s, K, T, R, SIGMA, kind="call")["delta"] for s in SPOT_GRID]
put_delta = [bs_greeks(s, K, T, R, SIGMA, kind="put")["delta"] for s in SPOT_GRID]
gamma = [bs_greeks(s, K, T, R, SIGMA, kind="call")["gamma"] for s in SPOT_GRID]

fig, (ax_delta, ax_gamma) = plt.subplots(1, 2, figsize=(14, 6))
ax_delta.plot(SPOT_GRID, call_delta, label="call delta", color="steelblue")
ax_delta.plot(SPOT_GRID, put_delta, label="put delta", color="crimson")
ax_delta.axvline(K, color="grey", linestyle="--", alpha=0.6)
ax_delta.set_xlabel("Spot")
ax_delta.set_ylabel("Delta")
ax_delta.set_title("Delta vs spot")
ax_delta.legend()
ax_delta.grid(True)

ax_gamma.plot(SPOT_GRID, gamma, color="darkorange")
ax_gamma.axvline(K, color="grey", linestyle="--", alpha=0.6)
ax_gamma.set_xlabel("Spot")
ax_gamma.set_ylabel("Gamma")
ax_gamma.set_title("Gamma vs spot")
ax_gamma.grid(True)

fig.savefig(results_dir / "options_greeks.png")
plt.close(fig)

print("\nSaved results/options_convergence.png and results/options_greeks.png")
