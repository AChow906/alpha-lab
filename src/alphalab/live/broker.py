from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    symbol: str
    qty: float


class Broker(ABC):
    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_positions(self) -> dict[str, float]: ...

    @abstractmethod
    def submit_order(self, symbol: str, qty: float) -> None: ...


class FakeBroker(Broker):
    def __init__(self, equity: float = 100_000.0, positions: dict[str, float] | None = None):
        self._equity = equity
        self._positions = dict(positions or {})
        self.orders: list[Order] = []

    def get_equity(self) -> float:
        return self._equity

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def submit_order(self, symbol: str, qty: float) -> None:
        self.orders.append(Order(symbol, qty))
        self._positions[symbol] = self._positions.get(symbol, 0.0) + qty


class AlpacaBroker(Broker):
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        if "paper" not in base_url:
            raise ValueError(
                "AlpacaBroker is paper-only (Phase 7); base_url must be the Alpaca paper endpoint"
            )
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(api_key, secret_key, paper=True, url_override=base_url)

    def get_equity(self) -> float:
        account = self._client.get_account()
        return float(account.equity)

    def get_positions(self) -> dict[str, float]:
        positions = self._client.get_all_positions()
        return {position.symbol: float(position.qty) for position in positions}

    def submit_order(self, symbol: str, qty: float) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        self._client.submit_order(request)
