"""Download the most recent N filings of a form for each company in config/companies.csv.

Layout:
    data/raw/{cik10}/{accession}/10-K.htm
    data/raw/{cik10}/{accession}/metadata.json

Idempotent: a filing whose file is on disk and matches the sha256 in its
metadata.json is skipped without touching the network for the document.

Run:  python -m ingest.download --tickers NVDA DIOD AMAT --years 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

from ingest.companies import CompanyRow, load_companies
from ingest.edgar_client import EdgarClient

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


@dataclass(frozen=True)
class FilingRef:
    """One row from the submissions index."""
    cik: str  # 10-digit
    accession: str  # 0001045810-26-000021
    form: str
    filing_date: str
    period_of_report: str
    primary_document: str


# -- form registry -------------------------------------------------------------
# To add a form, add a FormSpec. `resolve` returns (source_url, saved_filename).
# 10-K405 and DEF 14A can reuse primary_document; EX-21 needs a resolver that
# reads the filing index ({acc}-index.json) and picks the EX-21 attachment.

def primary_document(ref: FilingRef, client: EdgarClient) -> tuple[str, str]:
    url = ARCHIVE_URL.format(
        cik=int(ref.cik), acc_nodash=ref.accession.replace("-", ""), doc=ref.primary_document
    )
    return url, f"{ref.form.replace(' ', '').replace('/', '-')}.htm"


@dataclass(frozen=True)
class FormSpec:
    forms: tuple[str, ...]  # exact form types to match in the index (no amendments)
    resolve: Callable[[FilingRef, EdgarClient], tuple[str, str]]


FORMS: dict[str, FormSpec] = {
    "10-K": FormSpec(forms=("10-K",), resolve=primary_document),
}


# -- helpers -------------------------------------------------------------------

def fiscal_year(period_of_report: str) -> int:
    """Year of the period end, minus 1 if it ends Jan 1-7 (52/53-week filers).

    e.g. period 2025-01-03 -> FY2024, but NVDA's 2026-01-25 -> FY2026.
    """
    d = date.fromisoformat(period_of_report)
    return d.year - 1 if (d.month == 1 and d.day <= 7) else d.year


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def list_filings(client: EdgarClient, cik: str, spec: FormSpec, limit: int) -> list[FilingRef]:
    data = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    recent = data["filings"]["recent"]
    refs = []
    for i, form in enumerate(recent["form"]):
        if form not in spec.forms:
            continue
        refs.append(
            FilingRef(
                cik=cik,
                accession=recent["accessionNumber"][i],
                form=form,
                filing_date=recent["filingDate"][i],
                period_of_report=recent["reportDate"][i],
                primary_document=recent["primaryDocument"][i],
            )
        )
    refs.sort(key=lambda r: r.filing_date, reverse=True)
    # `recent` covers the last ~1000 filings, far more than 3 years for these filers
    return refs[:limit]


def already_downloaded(filing_dir: Path) -> bool:
    meta_path = filing_dir / "metadata.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    doc = filing_dir / meta["file"]
    return doc.exists() and sha256_file(doc) == meta["sha256"]


def download_filing(client: EdgarClient, company: CompanyRow, ref: FilingRef,
                    spec: FormSpec, raw_dir: Path = RAW_DIR) -> str:
    """Returns 'skipped' or 'downloaded'."""
    filing_dir = raw_dir / company.cik / ref.accession
    if already_downloaded(filing_dir):
        return "skipped"

    url, filename = spec.resolve(ref, client)
    content = client.get_bytes(url)
    filing_dir.mkdir(parents=True, exist_ok=True)
    doc_path = filing_dir / filename
    doc_path.write_bytes(content)

    meta = {
        "cik": company.cik,
        "ticker": company.ticker,
        "accession": ref.accession,
        "form": ref.form,
        "filing_date": ref.filing_date,
        "period_of_report": ref.period_of_report,
        "fiscal_year": fiscal_year(ref.period_of_report),
        "source_url": url,
        "file": filename,
        "sha256": sha256_file(doc_path),
    }
    (filing_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return "downloaded"


def run(tickers: list[str] | None, years: int, form: str,
        client: EdgarClient | None = None, raw_dir: Path = RAW_DIR,
        companies_csv: Path | None = None) -> dict[str, int]:
    spec = FORMS[form]
    client = client or EdgarClient()
    kwargs = {"tickers": tickers}
    if companies_csv:
        kwargs["path"] = companies_csv
    companies = load_companies(**kwargs)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for company in companies:
        try:
            refs = list_filings(client, company.cik, spec, limit=years)
        except Exception as exc:
            log.error("%s (CIK %s): could not list filings: %s", company.ticker, company.cik, exc)
            counts["failed"] += 1
            continue
        if len(refs) < years:
            log.warning("%s: only %d %s filings found (wanted %d)", company.ticker, len(refs), form, years)
        for ref in refs:
            try:
                status = download_filing(client, company, ref, spec, raw_dir)
            except Exception as exc:
                log.error("%s %s: download failed: %s", company.ticker, ref.accession, exc)
                counts["failed"] += 1
                continue
            counts[status] += 1
            log.info("%-6s %s %s FY%s  %s", company.ticker, ref.accession, ref.form,
                     fiscal_year(ref.period_of_report), status)
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*", help="subset of tickers (default: all in config/companies.csv)")
    p.add_argument("--years", type=int, default=3, help="number of most recent filings per company")
    p.add_argument("--form", default="10-K", choices=sorted(FORMS))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = run(args.tickers, args.years, args.form)
    print(f"downloaded={counts['downloaded']} skipped={counts['skipped']} failed={counts['failed']}")


if __name__ == "__main__":
    main()
