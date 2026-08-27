import csv
import datetime as dt
from pathlib import Path

import pandas as pd

from alphalab.live.broker import Order

DEFAULT_LOG_DIR = Path("logs")
TRADE_FIELDS = ["timestamp", "symbol", "qty"]
EQUITY_FIELDS = ["timestamp", "equity", "pnl"]


class Journal:
    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR):
        self.log_dir = Path(log_dir)
        self.trades_path = self.log_dir / "trades.csv"
        self.equity_path = self.log_dir / "equity.csv"

    def _now(self) -> str:
        return dt.datetime.now(dt.UTC).isoformat()

    def record_orders(self, orders: list[Order], timestamp: str | None = None) -> None:
        stamp = timestamp or self._now()
        for order in orders:
            self._append(
                self.trades_path,
                TRADE_FIELDS,
                {"timestamp": stamp, "symbol": order.symbol, "qty": order.qty},
            )

    def record_equity(
        self, equity: float, pnl: float | None = None, timestamp: str | None = None
    ) -> None:
        stamp = timestamp or self._now()
        self._append(
            self.equity_path,
            EQUITY_FIELDS,
            {"timestamp": stamp, "equity": equity, "pnl": "" if pnl is None else pnl},
        )

    def _append(self, path: Path, fields: list[str], row: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def read_trades(self) -> pd.DataFrame:
        return pd.read_csv(self.trades_path)

    def read_equity(self) -> pd.DataFrame:
        return pd.read_csv(self.equity_path)
