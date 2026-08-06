from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_prices(prices: pd.DataFrame, path: Path, title: str = "Prices") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    prices.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.grid(True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
