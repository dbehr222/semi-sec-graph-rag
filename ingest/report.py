"""Summarise sections and chunks, and estimate extraction input cost.

The price is a CLI argument on purpose: no model prices are hardcoded.
Cost here covers chunk text sent as input only; prompt instructions,
few-shot examples and output tokens come on top.

Run:  python -m ingest.report --price-per-mtok 3.00 [--tickers NVDA DIOD AMAT]
      [--extrapolate-companies 42]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingest.chunk import CHUNKS_DIR, SECTIONS_DIR
from ingest.companies import load_companies
from ingest.sections import FAILURES_CSV

SECTION_ORDER = ["Item1", "Item1A", "Item1C", "Item7", "Item7A"]


def collect(tickers: list[str] | None, sections_dir: Path = SECTIONS_DIR,
            chunks_dir: Path = CHUNKS_DIR) -> list[dict]:
    ciks = {c.cik for c in load_companies(tickers=tickers)}
    rows = []
    for path in sorted(sections_dir.glob("*.json")):
        rec = json.loads(path.read_text())
        if rec["cik"] not in ciks:
            continue
        chunk_path = chunks_dir / f"{path.stem}.jsonl"
        chunks = [json.loads(l) for l in chunk_path.open()] if chunk_path.exists() else []
        rows.append({
            "filing": path.stem,
            "cik": rec["cik"],
            "accession": rec["accession"],
            "chars": {k: v["chars"] for k, v in rec["sections"].items()},
            "methods": {k: v["method"] for k, v in rec["sections"].items()},
            "chunks": len(chunks),
            "tokens": sum(c["token_count"] for c in chunks),
        })
    return rows


def render(rows: list[dict], price_per_mtok: float, extrapolate: int | None) -> str:
    lines = []
    header = f"{'filing':<14}" + "".join(f"{s:>9}" for s in SECTION_ORDER) + f"{'chunks':>8}{'tokens':>10}"
    lines += ["Section sizes in characters ('-' = absent, * = regex fallback)", header, "-" * len(header)]
    for r in rows:
        cells = ""
        for s in SECTION_ORDER:
            if s in r["chars"]:
                mark = "*" if r["methods"][s] == "regex" else ""
                cells += f"{r['chars'][s]:>8,}{mark or ' '}"
            else:
                cells += f"{'-':>8} "
        lines.append(f"{r['filing']:<14}{cells}{r['chunks']:>8}{r['tokens']:>10,}")

    n = len(rows)
    total_tokens = sum(r["tokens"] for r in rows)
    total_chunks = sum(r["chunks"] for r in rows)
    lines += [
        "",
        f"Filings: {n}   Chunks: {total_chunks:,}   Tokens: {total_tokens:,}"
        + (f"   (avg {total_tokens // n:,} tokens/filing)" if n else ""),
        f"Estimated extraction input cost @ ${price_per_mtok:.2f}/MTok: "
        f"${total_tokens / 1e6 * price_per_mtok:,.2f}",
    ]
    if extrapolate and n:
        companies = len({r["cik"] for r in rows})
        per_company = total_tokens / companies
        est = per_company * extrapolate
        lines.append(
            f"Extrapolated to {extrapolate} companies (same filings/company): "
            f"~{est / 1e6:,.1f}M tokens, ~${est / 1e6 * price_per_mtok:,.2f}"
        )
    if FAILURES_CSV.exists():
        failures = FAILURES_CSV.read_text().strip().splitlines()[1:]
        lines += ["", f"Parse failures ({len(failures)}): {FAILURES_CSV}"] + [f"  {f}" for f in failures]
    lines.append("Note: tokens counted with tiktoken cl100k_base as a proxy; Claude's tokenizer differs.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--price-per-mtok", type=float, required=True,
                   help="USD per million input tokens for the extraction model")
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--extrapolate-companies", type=int,
                   help="also estimate cost for this many companies at the pilot's average")
    args = p.parse_args()
    print(render(collect(args.tickers), args.price_per_mtok, args.extrapolate_companies))


if __name__ == "__main__":
    main()
