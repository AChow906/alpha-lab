import json
import urllib.request
from collections.abc import Mapping
from typing import Protocol

from alphalab.live.broker import Order


class Notifier(Protocol):
    def send(self, message: str) -> None: ...


class NullNotifier:
    def send(self, message: str) -> None:
        return None


class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, message: str) -> None:
        payload = json.dumps({"content": message}).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "alphalab-bot/1.0"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=10)


def format_briefing(
    equity: float,
    positions: Mapping[str, float],
    orders: list[Order],
    pnl: float | None = None,
    dry_run: bool = False,
    label: str = "Alpha Lab",
) -> str:
    lines = [f"{label} — paper rebalance"]
    if dry_run:
        lines.append("(dry run — no orders submitted)")
    lines.append(f"Equity: ${equity:,.2f}")
    if pnl is not None:
        lines.append(f"Daily P&L: ${pnl:,.2f}")

    if orders:
        lines.append(f"Orders ({len(orders)}):")
        for order in orders:
            side = "BUY" if order.qty > 0 else "SELL"
            lines.append(f"  {side} {abs(order.qty):g} {order.symbol}")
    else:
        lines.append("Orders: none")

    if positions:
        lines.append("Positions:")
        for symbol in sorted(positions):
            lines.append(f"  {symbol}: {positions[symbol]:g}")

    return "\n".join(lines)
