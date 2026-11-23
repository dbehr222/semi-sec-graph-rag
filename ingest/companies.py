"""Company universe loader. Everything downstream is keyed on CIK, never ticker."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COMPANIES_CSV = ROOT / "config" / "companies.csv"


def pad_cik(cik: int | str) -> str:
    """10-digit zero-padded CIK, the form EDGAR uses in data.sec.gov URLs."""
    return str(int(cik)).zfill(10)


@dataclass(frozen=True)
class CompanyRow:
    cik: str  # zero-padded, 10 digits
    ticker: str
    name: str
    role: str


def load_companies(path: Path = COMPANIES_CSV, tickers: list[str] | None = None) -> list[CompanyRow]:
    # dtype=str so tickers like "Q" or "ON" are never coerced into something else
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = [
        CompanyRow(cik=pad_cik(r.cik), ticker=r.ticker, name=r.name, role=r.role)
        for r in df.itertuples()
    ]
    if tickers:
        wanted = {t.upper() for t in tickers}
        unknown = wanted - {r.ticker.upper() for r in rows}
        if unknown:
            raise SystemExit(f"Tickers not in {path.name}: {', '.join(sorted(unknown))}")
        rows = [r for r in rows if r.ticker.upper() in wanted]
    return rows
