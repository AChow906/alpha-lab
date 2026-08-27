from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from alphalab.backtest.engine import backtest
from alphalab.data.loader import load_prices
from alphalab.data.universe import SECTOR_ETFS
from alphalab.plotting import plot_prices
from alphalab.risk.var import monte_carlo_var, parametric_var, simulate_pnl
from alphalab.risk.vol_target import vol_target
from alphalab.strategies.ts_momentum import ts_momentum

matplotlib.use("Agg")

START = "2015-01-01"
END = "2024-12-31"
MOM_LOOKBACK = 126
COST_RATE = 0.001
TARGET_VOL = 0.10
VOL_LOOKBACK = 63
MAX_LEVERAGE = 3.0
HORIZON = 10
N_SIMS = 100_000
SEED = 7
LEVELS = (0.95, 0.99)

prices = load_prices(SECTOR_ETFS, START, END)
raw_weights = ts_momentum(prices, lookback=MOM_LOOKBACK)
targeted_weights = vol_target(
    raw_weights, prices, target_vol=TARGET_VOL, lookback=VOL_LOOKBACK, max_leverage=MAX_LEVERAGE
)

(raw_equity, raw_stats) = backtest(raw_weights, prices, cost_rate=COST_RATE)
(targeted_equity, targeted_stats) = backtest(targeted_weights, prices, cost_rate=COST_RATE)

print(f"TSMOM before vs after vol targeting (target={TARGET_VOL:.0%}, cost={COST_RATE}):")
comparison = pd.DataFrame({"Raw": raw_stats, "Vol-targeted": targeted_stats})
print(comparison)

targeted_returns = targeted_equity.pct_change().dropna()

print(f"\nVaR / Expected Shortfall on vol-targeted returns ({HORIZON}-day horizon):")
rows = {}
for level in LEVELS:
    (mc_var, mc_es) = monte_carlo_var(
        targeted_returns, horizon=HORIZON, n_sims=N_SIMS, level=level, seed=SEED
    )
    (p_var, p_es) = parametric_var(targeted_returns, horizon=HORIZON, level=level)
    rows[f"{level:.0%}"] = {
        "MC VaR": mc_var,
        "MC ES": mc_es,
        "Normal VaR": p_var,
        "Normal ES": p_es,
    }
print(pd.DataFrame(rows))

plot_prices(
    pd.DataFrame({"raw": raw_equity, "vol_targeted": targeted_equity}),
    Path("results/risk_report_equity.png"),
    title="TSMOM: Raw vs Vol-Targeted Equity",
)

pnl = simulate_pnl(targeted_returns, horizon=HORIZON, n_sims=N_SIMS, seed=SEED)
(hist_var, hist_es) = monte_carlo_var(
    targeted_returns, horizon=HORIZON, n_sims=N_SIMS, level=0.95, seed=SEED
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(pnl, bins=100, color="steelblue", alpha=0.8)
ax.axvline(-hist_var, color="darkorange", linestyle="--", label=f"95% VaR = {hist_var:.2%}")
ax.axvline(-hist_es, color="crimson", linestyle="--", label=f"95% ES = {hist_es:.2%}")
ax.set_xlabel(f"{HORIZON}-day P&L (return fraction)")
ax.set_ylabel("Simulated paths")
ax.set_title("Monte Carlo P&L Distribution (vol-targeted TSMOM)")
ax.legend()
ax.grid(True)

output = Path("results/risk_report_pnl.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output)
plt.close(fig)
