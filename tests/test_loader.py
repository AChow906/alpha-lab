import pandas as pd
import pytest

from alphalab.data import loader


def test_reads_from_cache_without_downloading(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("should not hit the network on a cache hit")

    monkeypatch.setattr(loader.yf, "download", boom)

    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates).to_parquet(
        tmp_path / "AAPL_2024-01-01_2024-01-31.parquet"
    )

    result = loader.load_prices(["AAPL"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)

    assert list(result.columns) == ["AAPL"]
    assert result["AAPL"].tolist() == [100.0, 101.0]


def test_raises_on_empty_download(tmp_path, monkeypatch):
    monkeypatch.setattr(loader.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError):
        loader.load_prices(["AAPL"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)


def test_sorts_and_dedupes_date(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("should not hit the network on a cache hit")

    monkeypatch.setattr(loader.yf, "download", boom)

    dates = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"])
    pd.DataFrame({"AAPL": [100.0, 101.0, 101.0]}, index=dates).to_parquet(
        tmp_path / "AAPL_2024-01-01_2024-01-31.parquet"
    )

    result = loader.load_prices(["AAPL"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)

    assert list(result.columns) == ["AAPL"]

    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
