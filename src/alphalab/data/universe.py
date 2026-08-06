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
    ("MA", "V"),
    ("HD", "LOW"),
    ("XLE", "XOP"),
    ("GLD", "GDX"),
]


def all_tickers() -> list[str]:
    tickers = set(SECTOR_ETFS)

    for first, second in PAIRS:
        tickers.add(first)
        tickers.add(second)

    return sorted(tickers)
