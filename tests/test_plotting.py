import pandas as pd

from alphalab.plotting import plot_prices


def test_plot_prices_writes_file(tmp_path):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 199.0, 201.0]},
        index=dates,
    )
    out = tmp_path / "prices.png"

    plot_prices(prices, out)

    assert out.exists()
