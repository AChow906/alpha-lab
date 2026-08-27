from pathlib import Path

import pandas as pd
import pytest

from alphalab.live.broker import AlpacaBroker, FakeBroker, Order
from alphalab.live.config import load_config
from alphalab.live.journal import Journal
from alphalab.live.notifier import NullNotifier, format_briefing
from alphalab.live.rebalancer import Rebalancer, compute_orders
from alphalab.live.sizing import target_shares


def _prices(mapping: dict[str, float]) -> pd.Series:
    return pd.Series(mapping)


def test_rebalance_flat_account_buys_to_target():
    broker = FakeBroker(equity=100_000.0, positions={})
    prices = _prices({"AAA": 100.0, "BBB": 50.0})
    shares = {"AAA": 10.0, "BBB": 20.0}

    rebalancer = Rebalancer(broker, dry_run=False)
    orders = rebalancer.rebalance(shares, prices)

    assert orders == [Order("AAA", 10.0), Order("BBB", 20.0)]
    assert broker.orders == orders
    assert broker.get_positions() == {"AAA": 10.0, "BBB": 20.0}


def test_rebalance_at_target_is_idempotent():
    broker = FakeBroker(positions={"AAA": 10.0})
    prices = _prices({"AAA": 100.0})

    orders = Rebalancer(broker, dry_run=False).rebalance({"AAA": 10.0}, prices)

    assert orders == []
    assert broker.orders == []


def test_rebalance_overweight_sells():
    broker = FakeBroker(positions={"AAA": 30.0})
    prices = _prices({"AAA": 100.0})

    orders = Rebalancer(broker, dry_run=False).rebalance({"AAA": 10.0}, prices)

    assert orders == [Order("AAA", -20.0)]
    assert broker.get_positions()["AAA"] == 10.0


def test_no_trade_band_suppresses_small_diffs():
    broker = FakeBroker(positions={"AAA": 100.0})
    prices = _prices({"AAA": 50.0})

    orders = Rebalancer(broker, no_trade_band=5.0, dry_run=False).rebalance({"AAA": 103.0}, prices)

    assert orders == []
    assert broker.orders == []


def test_dry_run_returns_orders_without_submitting():
    broker = FakeBroker(equity=100_000.0, positions={})
    prices = _prices({"AAA": 100.0})

    orders = Rebalancer(broker, dry_run=True).rebalance({"AAA": 10.0}, prices)

    assert orders == [Order("AAA", 10.0)]
    assert broker.orders == []
    assert broker.get_positions() == {}


def test_safety_cap_clips_oversized_order():
    broker = FakeBroker(positions={})
    prices = _prices({"AAA": 100.0})

    orders = Rebalancer(broker, max_order_notional=1000.0, dry_run=False).rebalance(
        {"AAA": 50.0}, prices
    )

    assert orders == [Order("AAA", 10.0)]
    assert broker.get_positions() == {"AAA": 10.0}


def test_safety_cap_can_reject():
    prices = _prices({"AAA": 100.0})

    with pytest.raises(ValueError):
        compute_orders({"AAA": 50.0}, {}, prices, max_order_notional=1000.0, reject_over_cap=True)


def test_target_shares_whole_share_quantities():
    weights = pd.Series({"AAA": 0.5, "BBB": -0.25})
    prices = pd.Series({"AAA": 100.0, "BBB": 40.0})

    shares = target_shares(weights, equity=100_000.0, prices=prices)

    assert shares == {"AAA": 500.0, "BBB": -625.0}


def test_target_shares_skips_nan_and_bad_prices():
    weights = pd.Series({"AAA": float("nan"), "BBB": 0.5, "CCC": 0.5})
    prices = pd.Series({"AAA": 100.0, "BBB": 0.0, "CCC": 50.0})

    shares = target_shares(weights, 10_000.0, prices)

    assert shares == {"CCC": 100.0}


def test_format_briefing_contains_key_fields():
    orders = [Order("AAA", 10.0), Order("BBB", -5.0)]
    positions = {"AAA": 10.0}

    message = format_briefing(12_345.67, positions, orders, pnl=250.0, dry_run=True)

    assert "dry run" in message.lower()
    assert "BUY" in message
    assert "SELL" in message
    assert "AAA" in message
    assert "BBB" in message
    assert "12,345.67" in message


def test_null_notifier_send_is_noop():
    assert NullNotifier().send("anything") is None


def test_journal_round_trip(tmp_path):
    journal = Journal(log_dir=tmp_path / "logs")
    orders = [Order("AAA", 10.0), Order("BBB", -5.0)]

    journal.record_orders(orders, timestamp="2026-08-27T16:00:00")
    journal.record_equity(100_000.0, pnl=250.0, timestamp="2026-08-27T16:00:00")

    trades = journal.read_trades()
    equity = journal.read_equity()

    assert list(trades["symbol"]) == ["AAA", "BBB"]
    assert list(trades["qty"]) == [10.0, -5.0]
    assert equity["equity"].iloc[0] == 100_000.0
    assert equity["pnl"].iloc[0] == 250.0


def test_load_config_from_environment():
    environ = {
        "ALPACA_API_KEY": "key",
        "ALPACA_SECRET_KEY": "secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "DISCORD_WEBHOOK_URL": "https://discord.example/webhook",
        "ALPHALAB_STRATEGY": "xs_momentum",
        "ALPHALAB_TARGET_VOL": "0.15",
        "ALPHALAB_NO_TRADE_BAND": "2",
        "ALPHALAB_MAX_ORDER_NOTIONAL": "5000",
        "ALPHALAB_DRY_RUN": "false",
        "ALPHALAB_UNIVERSE": "XLK,XLF,XLE",
    }

    config = load_config(env_path=Path("/nonexistent/.env"), environ=environ)

    assert config.alpaca_api_key == "key"
    assert config.alpaca_secret_key == "secret"
    assert config.strategy == "xs_momentum"
    assert config.target_vol == 0.15
    assert config.no_trade_band == 2.0
    assert config.max_order_notional == 5000.0
    assert config.dry_run is False
    assert config.universe == ("XLK", "XLF", "XLE")


def test_load_config_defaults_without_secrets():
    config = load_config(env_path=Path("/nonexistent/.env"), environ={})

    assert config.alpaca_api_key == ""
    assert config.strategy == "ts_momentum"
    assert config.dry_run is True
    assert "paper" in config.alpaca_base_url


def test_alpaca_broker_rejects_non_paper_url():
    with pytest.raises(ValueError):
        AlpacaBroker("key", "secret", "https://api.alpaca.markets")
