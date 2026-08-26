SECTOR_ETFS: list[str] = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),
    ("PG", "CL"),
    ("CL", "KMB"),
    ("WMT", "TGT"),
    ("MO", "PM"),
    ("MA", "V"),
    ("JPM", "BAC"),
    ("GS", "MS"),
    ("WFC", "C"),
    ("CVX", "XOM"),
    ("XLE", "XOP"),
    ("COP", "EOG"),
    ("MSFT", "AAPL"),
    ("GOOGL", "META"),
    ("NVDA", "AMD"),
    ("HD", "LOW"),
    ("LEN", "DHI"),
    ("GLD", "GDX"),
    ("GDX", "GDXJ"),
    ("SPY", "XLK"),
]


def all_tickers() -> list[str]:
    tickers = set(SECTOR_ETFS)

    for first, second in PAIRS:
        tickers.add(first)
        tickers.add(second)

    return sorted(tickers)
