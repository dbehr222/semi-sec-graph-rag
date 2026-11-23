"""Build the company universe for the semiconductor supply-chain graph.

Steps
1. List every SIC 3674 (semiconductors) CIK from EDGAR's browse-by-SIC feed.
2. Pull the 10-K filing index for the last N years and keep SIC 3674 CIKs
   that filed one (this drops defunct companies without per-company calls).
3. Build Company(filing.cik) for each survivor, read revenue, rank, take top N.
4. Add supply-chain companies SIC 3674 misses (equipment makers, big customers).
5. Write data/universe.csv.

Run:  python -m ingest.universe
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from edgar import Company, get_filings
from edgar.httprequests import download_text

load_dotenv()
log = logging.getLogger(__name__)

SIC_CODE = 3674
YEARS_BACK = 3
TOP_N = 35
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.csv"

# SIC 3674 misses these, but the graph needs them
EXTRA_COMPANIES = {
    "equipment": ["AMAT", "LRCX", "KLAC"],
    "customer": ["MSFT", "AMZN", "GOOGL", "AAPL", "META"],
}

# SIC 3674 also covers solar makers, which aren't part of the chip supply chain.
# (TYGO is here too: its revenue is mis-scaled in XBRL.)
EXCLUDE_TICKERS = {"FSLR", "ENPH", "SEDG", "ARRY", "NXT", "TYGO", "TE", "OSIS", "VIAV", "IPGP", "SHLS"}

# SIC 3674 lumps suppliers in with chip designers; relabel so "supplier of" queries stay clean.
ROLE_OVERRIDES = {
    "UCTT": "equipment",  # Ultra Clean: subsystems for fab tools
    "ICHR": "equipment",  # Ichor: fluid delivery subsystems
    "KLIC": "equipment",  # Kulicke & Soffa: assembly/packaging tools
    "FORM": "equipment",  # FormFactor: probe cards and test
    "PLAB": "materials",  # Photronics: photomasks
    "Q": "materials",  # Qnity (DuPont electronics): CMP pads etc.
    "PENG": "systems",  # Penguin: memory modules / AI systems; suppliers named in 10-K, customers anonymized
}

_BROWSE_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic}"
    "&owner=include&count=100&start={start}&output=atom"
)


def sic_ciks(sic: int) -> set[int]:
    """All CIKs EDGAR has ever classified under `sic` (includes defunct filers)."""
    ciks: set[int] = set()
    start = 0
    while True:
        root = ET.fromstring(download_text(_BROWSE_URL.format(sic=sic, start=start)))
        found = {int(el.text) for el in root.findall(".//{*}cik") if el.text}
        if not found or found <= ciks:
            break
        ciks.update(found)
        start += 100
    return ciks


def recent_10k_filings(years_back: int) -> pd.DataFrame:
    """Latest 10-K per CIK across the last `years_back` calendar years."""
    this_year = date.today().year
    years = list(range(this_year - years_back + 1, this_year + 1))
    df = get_filings(year=years, form="10-K").to_pandas()
    df["cik"] = df["cik"].astype(int)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df.sort_values("filing_date").drop_duplicates("cik", keep="last")


def describe(company: Company, role: str, latest_10k=None) -> dict:
    facts = company.get_facts()
    return {
        "cik": int(company.cik),
        "ticker": company.tickers[0] if company.tickers else None,
        "name": company.name,
        "sic": company.sic,
        "role": role,
        "revenue": facts.get_revenue(),
        "public_float": facts.public_float,
        "latest_10k": None if latest_10k is None else latest_10k.date().isoformat(),
    }


def build_universe() -> pd.DataFrame:
    log.info("Listing SIC %s companies", SIC_CODE)
    semi = sic_ciks(SIC_CODE)
    log.info("%d CIKs ever classified SIC %s", len(semi), SIC_CODE)

    filings = recent_10k_filings(YEARS_BACK)
    active = filings[filings["cik"].isin(semi)]
    log.info("%d of them filed a 10-K in the last %d years", len(active), YEARS_BACK)

    rows = []
    for filing in active.itertuples():
        try:
            company = Company(filing.cik)
            if str(company.sic) != str(SIC_CODE) or not company.tickers:
                continue
            if company.tickers[0] in EXCLUDE_TICKERS:
                continue
            rows.append(describe(company, "core", filing.filing_date))
        except Exception as exc:  # one bad filer shouldn't kill the run
            log.warning("skipping CIK %s: %s", filing.cik, exc)

    core = pd.DataFrame(rows)
    core = core.dropna(subset=["revenue"]).sort_values("revenue", ascending=False).head(TOP_N)

    extras = []
    for role, tickers in EXTRA_COMPANIES.items():
        for ticker in tickers:
            if ticker in set(core["ticker"]):  # already SIC 3674 (e.g. AMAT): relabel
                core.loc[core["ticker"] == ticker, "role"] = role
                continue
            company = Company(ticker)
            latest = company.get_filings(form="10-K").latest()
            extras.append(describe(company, role, pd.Timestamp(latest.filing_date)))

    universe = pd.concat([core, pd.DataFrame(extras)], ignore_index=True)
    universe["role"] = universe["ticker"].map(ROLE_OVERRIDES).fillna(universe["role"])
    return universe.drop_duplicates("cik")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = build_universe()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(result[["ticker", "name", "role", "revenue", "latest_10k"]].to_string())
    print(f"\nWrote {len(result)} companies to {OUT_PATH}")
