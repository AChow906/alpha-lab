from pathlib import Path

from alphalab.data.loader import load_prices
from alphalab.data.universe import SECTOR_ETFS
from alphalab.plotting import plot_prices


def main() -> None:
    prices = load_prices(SECTOR_ETFS, "2019-01-01", "2024-12-31")
    rebased = prices / prices.iloc[0] * 100
    out = Path("results/prices.png")
    plot_prices(rebased, out, title="Sector ETFs (rebased to 100)")
    print(f"Wrote {out} — {prices.shape[0]} rows x {prices.shape[1]} tickers")


if __name__ == "__main__":
    main()
