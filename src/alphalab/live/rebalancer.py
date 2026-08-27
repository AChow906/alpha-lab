import math
from collections.abc import Mapping

import pandas as pd

from alphalab.live.broker import Broker, Order
from alphalab.live.sizing import price_of


def compute_orders(
    target_shares: Mapping[str, float],
    current_positions: Mapping[str, float],
    prices: pd.Series | Mapping[str, float],
    no_trade_band: float = 0.0,
    max_order_notional: float | None = None,
    reject_over_cap: bool = False,
) -> list[Order]:
    symbols = sorted(set(target_shares) | set(current_positions))
    orders: list[Order] = []

    for symbol in symbols:
        target = float(target_shares.get(symbol, 0.0))
        current = float(current_positions.get(symbol, 0.0))
        diff = target - current

        if abs(diff) < no_trade_band:
            continue

        qty = float(round(diff))
        if qty == 0.0:
            continue

        if max_order_notional is not None:
            price = price_of(prices, symbol)
            if price is not None and price > 0.0 and abs(qty) * price > max_order_notional:
                if reject_over_cap:
                    raise ValueError(
                        f"order for {symbol} notional {abs(qty) * price:.2f} exceeds cap "
                        f"{max_order_notional:.2f}"
                    )
                capped = math.floor(max_order_notional / price)
                if capped == 0:
                    continue
                qty = math.copysign(capped, qty)

        orders.append(Order(symbol, qty))

    return orders


class Rebalancer:
    def __init__(
        self,
        broker: Broker,
        no_trade_band: float = 0.0,
        max_order_notional: float | None = None,
        reject_over_cap: bool = False,
        dry_run: bool = True,
    ):
        self.broker = broker
        self.no_trade_band = no_trade_band
        self.max_order_notional = max_order_notional
        self.reject_over_cap = reject_over_cap
        self.dry_run = dry_run

    def rebalance(
        self,
        target_shares: Mapping[str, float],
        prices: pd.Series | Mapping[str, float],
    ) -> list[Order]:
        current = self.broker.get_positions()
        orders = compute_orders(
            target_shares,
            current,
            prices,
            no_trade_band=self.no_trade_band,
            max_order_notional=self.max_order_notional,
            reject_over_cap=self.reject_over_cap,
        )

        if not self.dry_run:
            for order in orders:
                self.broker.submit_order(order.symbol, order.qty)

        return orders
