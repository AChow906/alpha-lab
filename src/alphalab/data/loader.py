from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data")


def _load_one(ticker: str, start: str, end: str, cache_dir: Path) -> pd.Series:
    cache_path = cache_dir / f"{ticker}_{start}_{end}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)[ticker]

    res = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if res.empty:
        raise ValueError(f"No data for {ticker}")

    close = res["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    series = close.rename(ticker)

    cache_dir.mkdir(parents=True, exist_ok=True)
    series.to_frame().to_parquet(cache_path)
    return series


def load_prices(
    tickers: list[str], start: str, end: str, cache_dir: Path = CACHE_DIR
) -> pd.DataFrame:
    series = [_load_one(ticker, start, end, cache_dir) for ticker in tickers]
    df = pd.concat(series, axis=1)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(how="all")
    return df
