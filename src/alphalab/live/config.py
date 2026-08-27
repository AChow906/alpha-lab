import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from alphalab.data.universe import SECTOR_ETFS

DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_ENV_PATH = Path(".env")


@dataclass(frozen=True)
class LiveConfig:
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = DEFAULT_BASE_URL
    discord_webhook_url: str = ""
    universe: tuple[str, ...] = field(default_factory=lambda: tuple(SECTOR_ETFS))
    strategy: str = "ts_momentum"
    strategy_params: dict[str, int] = field(default_factory=dict)
    target_vol: float = 0.10
    vol_lookback: int = 63
    max_leverage: float = 3.0
    no_trade_band: float = 1.0
    max_order_notional: float = 10_000.0
    history_days: int = 400
    dry_run: bool = True
    use_fake_broker: bool = False


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> LiveConfig:
    environ = os.environ if environ is None else environ
    file_values = _parse_env_file(env_path if env_path is not None else DEFAULT_ENV_PATH)

    def get(key: str, default: str) -> str:
        if environ.get(key):
            return environ[key]
        if file_values.get(key):
            return file_values[key]
        return default

    universe_raw = get("ALPHALAB_UNIVERSE", "")
    universe = tuple(item.strip() for item in universe_raw.split(",") if item.strip())
    if not universe:
        universe = tuple(SECTOR_ETFS)

    lookback_raw = get("ALPHALAB_STRATEGY_LOOKBACK", "")
    strategy_params = {"lookback": int(lookback_raw)} if lookback_raw else {}

    return LiveConfig(
        alpaca_api_key=get("ALPACA_API_KEY", ""),
        alpaca_secret_key=get("ALPACA_SECRET_KEY", ""),
        alpaca_base_url=get("ALPACA_BASE_URL", DEFAULT_BASE_URL),
        discord_webhook_url=get("DISCORD_WEBHOOK_URL", ""),
        universe=universe,
        strategy=get("ALPHALAB_STRATEGY", "ts_momentum"),
        strategy_params=strategy_params,
        target_vol=float(get("ALPHALAB_TARGET_VOL", "0.10")),
        vol_lookback=int(get("ALPHALAB_VOL_LOOKBACK", "63")),
        max_leverage=float(get("ALPHALAB_MAX_LEVERAGE", "3.0")),
        no_trade_band=float(get("ALPHALAB_NO_TRADE_BAND", "1.0")),
        max_order_notional=float(get("ALPHALAB_MAX_ORDER_NOTIONAL", "10000")),
        history_days=int(get("ALPHALAB_HISTORY_DAYS", "400")),
        dry_run=_parse_bool(get("ALPHALAB_DRY_RUN", "true")),
        use_fake_broker=_parse_bool(get("ALPHALAB_USE_FAKE_BROKER", "false")),
    )
