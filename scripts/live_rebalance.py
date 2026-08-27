"""Daily paper-trading rebalancer (Phase 7 — PAPER ONLY).

Intended to run once per day, after the US close, via cron on the execution
machine. This script performs live network I/O (market data + broker) only when
run directly; importing it does nothing. It is not part of the offline test suite.

Requirements on the execution machine:
  - pip install alpaca-py       (the SDK is imported lazily; not a repo dependency)
  - a populated .env            (copy .env.example, then fill in ALPACA_* / DISCORD_*)

Safety:
  - paper endpoint only: AlpacaBroker refuses any base_url that is not the Alpaca
    paper endpoint.
  - ALPHALAB_DRY_RUN=true (the default) computes, logs, and reports orders WITHOUT
    submitting them to the broker.

Example cron entry (weekdays, 16:10 America/New_York):
  10 16 * * 1-5  cd /path/to/alpha-lab && .venv/bin/python scripts/live_rebalance.py
"""

import datetime as dt

from alphalab.data.loader import load_prices
from alphalab.live.broker import AlpacaBroker, Broker, FakeBroker
from alphalab.live.config import LiveConfig, load_config
from alphalab.live.journal import Journal
from alphalab.live.notifier import DiscordNotifier, Notifier, NullNotifier, format_briefing
from alphalab.live.rebalancer import Rebalancer
from alphalab.live.sizing import latest_row, target_shares
from alphalab.risk.vol_target import vol_target
from alphalab.strategies.mean_reversion import mean_reversion
from alphalab.strategies.ts_momentum import ts_momentum
from alphalab.strategies.xs_momentum import xs_momentum

STRATEGIES = {
    "ts_momentum": ts_momentum,
    "xs_momentum": xs_momentum,
    "mean_reversion": mean_reversion,
}


def build_broker(config: LiveConfig) -> Broker:
    if config.use_fake_broker:
        return FakeBroker()

    missing = [
        name
        for name, value in (
            ("ALPACA_API_KEY", config.alpaca_api_key),
            ("ALPACA_SECRET_KEY", config.alpaca_secret_key),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing required secrets: {', '.join(missing)}. "
            "Populate .env (see .env.example) or set ALPHALAB_USE_FAKE_BROKER=true."
        )

    return AlpacaBroker(config.alpaca_api_key, config.alpaca_secret_key, config.alpaca_base_url)


def build_notifier(config: LiveConfig) -> Notifier:
    if config.discord_webhook_url:
        return DiscordNotifier(config.discord_webhook_url)
    return NullNotifier()


def main() -> None:
    config = load_config()
    if config.strategy not in STRATEGIES:
        raise SystemExit(f"Unknown strategy {config.strategy!r}; choose from {sorted(STRATEGIES)}")

    broker = build_broker(config)
    notifier = build_notifier(config)
    journal = Journal()

    end = dt.date.today()
    start = end - dt.timedelta(days=config.history_days)
    prices = load_prices(list(config.universe), start.isoformat(), end.isoformat())

    strategy = STRATEGIES[config.strategy]
    weights = strategy(prices, **config.strategy_params)
    targeted = vol_target(
        weights,
        prices,
        target_vol=config.target_vol,
        lookback=config.vol_lookback,
        max_leverage=config.max_leverage,
    )

    equity = broker.get_equity()
    shares = target_shares(latest_row(targeted), equity, latest_row(prices))

    rebalancer = Rebalancer(
        broker,
        no_trade_band=config.no_trade_band,
        max_order_notional=config.max_order_notional,
        dry_run=config.dry_run,
    )
    orders = rebalancer.rebalance(shares, latest_row(prices))

    journal.record_orders(orders)
    journal.record_equity(equity)

    message = format_briefing(equity, broker.get_positions(), orders, dry_run=config.dry_run)
    notifier.send(message)
    print(message)


if __name__ == "__main__":
    main()
