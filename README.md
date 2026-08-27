# Alpha Lab — Quantitative Strategy Research & Backtesting

A Python platform for researching and backtesting systematic trading strategies — momentum, mean-reversion, and statistical arbitrage — alongside options pricing and Monte Carlo risk modelling, built with research rigour and reproducibility as first-class concerns.

> This is the **participant's research view** of markets. Its companion project — a C++20 low-latency trading engine — builds the **exchange infrastructure** (matching engine, limit order book, market-data distribution). The two are deliberately kept separate so each stays focused: this repo is statistics- and research-heavy; the engine is systems- and performance-heavy.

## Setup

**Requirements:** [uv](https://docs.astral.sh/uv/), Python 3.12 (uv installs it automatically from `.python-version`).

All dependencies are pinned in `uv.lock` and resolved from public PyPI. A small Makefile wraps the common tasks.

```bash
make sync       # create the virtualenv and install the locked dependencies
```

## Docker

```bash
docker build -t alphalab .
docker run alphalab              # runs make check inside the container
docker run -it alphalab bash     # interactive shell for development
```

## Usage

```bash
make test       # run the unit tests (pytest)
make lint       # static checks (ruff)
make fmt        # auto-format (ruff)
make check      # full CI gate: lockfile integrity + lint + format + tests
```

Each phase ships a script under `scripts/` that regenerates its figures and tables into `results/`:

```bash
uv run python scripts/<report>.py
```

## Data

Daily US equity and sector-ETF prices from free sources (Yahoo Finance via `yfinance`, with Stooq as a fallback), cached locally as Parquet under `data/` (gitignored). Caching keeps results reproducible and avoids repeated API calls — once fetched, a price series is frozen, insulating results from silent provider restatements. Prices are **dividend/split-adjusted** so that day-to-day changes reflect true returns.

## Project Structure

```
src/alphalab/
  data/            # Data ingestion, parquet caching, and the ticker universe
  backtest/        # Vectorised backtest engine, performance metrics, cost models
  strategies/      # Signal generators: momentum, mean-reversion, pairs
  risk/            # Volatility targeting, Monte Carlo VaR / CVaR
  pricing/         # Options pricing: Black-Scholes, binomial trees, Greeks
tests/             # Unit tests (pytest)
scripts/           # Reproducible report scripts (regenerate figures into results/)
results/           # Generated figures and tables
data/              # Local parquet cache (gitignored)
```

## Roadmap

- [x] Phase 0 — Foundation — reproducible `uv` env, CI gate, Docker dev container, and the data pipeline (loader + cache + ticker universe)
- [x] Phase 1 — Backtest engine + performance metrics (Sharpe, max drawdown, turnover)
- [x] Phase 2 — Momentum strategies (time-series + cross-sectional) + runner script
- [x] Phase 3 — Mean-reversion + validation discipline: transaction costs, train/test splits, in- vs out-of-sample comparison
- [x] Phase 4 — Statistical arbitrage / pairs trading (cointegration, spread z-score) with walk-forward pair selection over an expanded universe
- [x] Phase 5 — Risk modelling: volatility targeting and Monte Carlo VaR / CVaR
- [ ] Phase 6 — Options pricing (Black-Scholes, binomial trees, Greeks) + documentation polish
- [ ] **Phase 7 — Paper-trading execution** *(stretch)* — the strategy core driven live against the Alpaca **paper** API: a daily scheduled rebalancer, ATR/volatility position sizing, trade + P&L logging, and Discord briefings
- [ ] **Phase 8 — Live trading** *(stretch, personal use — not a project deliverable)* — the same runner pointed at a funded account; paper-validated strategies only, with hard per-trade stops and a portfolio drawdown circuit-breaker

### Research discipline

The backtester is designed to avoid the classic ways strategy research lies to you:

- **No lookahead bias** — signals computed at the close of day *t* are applied to day *t+1* returns.
- **Out-of-sample by default** — parameters are tuned on a training window and reported on held-out data; in- vs out-of-sample results are shown side by side.
- **Realistic costs** — transaction costs and turnover are charged, not ignored.
- **Honest metrics** — full-sample equity curves and drawdowns, no cherry-picked periods.

### Phases 7–8 — From research to live execution (stretch goals)

Phases 0–6 build the *research* core; Phases 7–8 add *execution*, on a single principle: a backtester and a trading bot are two drivers of the **same strategy code**. Strategies are pure functions from price history to target positions — the backtest engine feeds them all of history at once; the live runner feeds them the latest bar. Same signals, different driver, so the bot is an adapter rather than a rewrite — and the no-lookahead discipline that keeps the backtest honest is exactly what makes live behaviour match it.

**Phase 7 — Paper trading.** A daily scheduled job (not a 24/7 loop — daily, equities-only strategies only need to run once at the close): pull the latest bars through `data/loader.py`, compute target weights, reconcile against current positions, and rebalance via the Alpaca **paper** API behind a thin, swappable broker interface. Position sizing reuses the Phase 5 risk module (volatility/ATR-based, constant risk per trade). Trades and daily P&L are logged; a Discord webhook posts a morning/evening briefing. Secrets (API keys, webhook URL) live in a gitignored `.env`.

**Phase 8 — Live trading.** The identical runner pointed at a funded account, gated behind a successful paper-trading period, with non-negotiable risk limits: a hard stop on every trade, a portfolio-level drawdown circuit-breaker that flattens and halts, and small position sizes. This phase is **personal-use and explicitly not a project deliverable** — it involves real capital and real risk, and nothing in this repository is financial advice.

**One core, three venues.** The same strategy modules can ultimately drive three execution targets: the backtester (historical), Alpaca (real market), and — closing the loop with the companion project — the C++ engine's gateway (a self-hosted venue). Those are precisely the "strategy agents" that engine's Phase 10 anticipates.

### Scope

Low-latency systems engineering and exchange infrastructure (matching engine, order book, networking) are intentionally **out of scope** — they live in the companion C++ trading-engine project. For the **research core (Phases 0–6)**, service deployment (containers, dashboards, cloud) is also out of scope: reproducibility is handled by the pinned `uv` lockfile and CI on a clean checkout, not runtime infrastructure. The execution **stretch goals (Phases 7–8)** deliberately reintroduce a minimal slice of that — a scheduled runner and secrets management — kept lean by staying daily and equities-only. Live trading (Phase 8) is personal-use, not a deliverable, and nothing here is financial advice.
