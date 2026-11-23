"""Split downloaded 10-Ks into Items.

Primary path: edgartools' 10-K parser (edgar.documents.parse_html on the local
file; no network). Fallback: a regex splitter over a plain-text rendering,
used per item when edgartools misses an item or returns too little text.

Output:
    data/processed/sections/{ticker}_{fy}.json
    data/processed/parse_failures.csv

Run:  python -m ingest.sections --tickers NVDA DIOD AMAT
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # inline XBRL is XHTML

from ingest.companies import load_companies
from ingest.textutil import paragraph_spans, squash

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SECTIONS_DIR = PROCESSED_DIR / "sections"
FAILURES_CSV = PROCESSED_DIR / "parse_failures.csv"

# item id -> (section key used downstream, title)
ITEMS = {
    "1": ("Item1", "Business"),
    "1A": ("Item1A", "Risk Factors"),
    "1C": ("Item1C", "Cybersecurity"),
    "7": ("Item7", "Management's Discussion and Analysis"),
    "7A": ("Item7A", "Quantitative and Qualitative Disclosures About Market Risk"),
}
REQUIRED_ITEMS = ("1", "1A")
MIN_REQUIRED_CHARS = 2000


# -- text normalisation --------------------------------------------------------

# Page furniture that survives HTML->text: bare page numbers ("17", "- 17 -",
# "Page 17", "F-3") and "Table of Contents" back-links repeated on every page.
_PAGE_FURNITURE = re.compile(
    r"^[ \t]*(?:"
    r"[-–—]?[ \t]*(?:page[ \t]+)?(?:[A-Z]-)?\d{1,3}[ \t]*[-–—]?"
    r"|table[ \t]+of[ \t]+contents"
    r")[ \t]*$",
    re.I | re.M,
)


def normalize(text: str) -> str:
    text = text.replace("\xa0", " ").replace("​", "")
    text = _PAGE_FURNITURE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]*(?:\n[ \t]*)+", "\n\n", text)
    # a page break left mid-sentence: "... claims from their\n\ncustomers) ..." -> rejoin
    text = re.sub(r"(?<=[a-z,;])[ \t]*\n\n[ \t]*(?=[a-z(])", " ", text)
    return text.strip()


_BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"]


def html_to_text(html: str) -> str:
    """Plain text with blank lines between blocks and tables flattened to one row per line."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    # inline-XBRL filings carry a hidden header block
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(NavigableString("\n\n" + "\n".join(rows) + "\n\n"))
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before(NavigableString("\n\n"))
        tag.insert_after(NavigableString("\n\n"))
    text = soup.get_text()
    text = "\n".join(re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines())
    return normalize(text)


# -- headings (from bold / italic runs in the HTML) ------------------------------
# Filers mark risk-factor headings with emphasis, but differently: NVDA bold,
# DIOD italic, AMAT bold+italic, some as a run-in lead ("<i>Heading.</i> Body").
# We collect the leading emphasized run of every block element, then mark a
# paragraph of the section text as a heading if it starts with one of them.

_EMPH_TAGS_BOLD = {"b", "strong"}
_EMPH_TAGS_ITALIC = {"i", "em"}
_BLOCKS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
_WEIGHT = re.compile(r"font-weight\s*:\s*([a-z0-9]+)", re.I)
_FSTYLE = re.compile(r"font-style\s*:\s*([a-z]+)", re.I)
MIN_HEADING_WORDS = 4


def _emphasized(node) -> bool:
    """Is this text node bold or italic? Nearest declaration wins for each property."""
    bold = italic = None
    el = node.parent
    while el is not None and getattr(el, "name", None) not in (None, "[document]", "body"):
        style = el.get("style") or ""
        if bold is None:
            m = _WEIGHT.search(style)
            if m:
                w = m.group(1).lower()
                bold = w in ("bold", "bolder") or (w.isdigit() and int(w) >= 600)
            elif el.name in _EMPH_TAGS_BOLD:
                bold = True
        if italic is None:
            m = _FSTYLE.search(style)
            if m:
                italic = m.group(1).lower() in ("italic", "oblique")
            elif el.name in _EMPH_TAGS_ITALIC:
                italic = True
        if el.name in _BLOCKS:
            break
        el = el.parent
    return bool(bold) or bool(italic)


def emphasized_leads(html: str) -> set[str]:
    """Leading bold/italic run of every block element, whitespace-squashed."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    blocks: dict[int, list] = {}
    order: list[int] = []
    for node in soup.find_all(string=True):
        if not node.strip() or node.parent.name in ("script", "style", "title"):
            continue
        block = node.find_parent(list(_BLOCKS))
        if block is None:
            continue
        key = id(block)
        if key not in blocks:
            blocks[key] = []
            order.append(key)
        blocks[key].append(node)

    leads = set()
    for key in order:
        run = []
        for node in blocks[key]:
            if not _emphasized(node):
                break
            run.append(str(node))
        lead = squash(" ".join(run))
        if len(lead.split()) >= MIN_HEADING_WORDS:
            leads.add(lead)
    return leads


def find_headings(text: str, leads: set[str]) -> list[dict]:
    """Paragraphs of `text` that start with an emphasized run.

    kind = "risk" for sentence-style headings (end in . or ?), else "group"
    (e.g. "Risks Related to Our Industry and Markets").
    """
    index: dict[str, list[str]] = {}
    for lead in leads:
        index.setdefault(lead[:24], []).append(lead)
    headings = []
    for start, end in paragraph_spans(text):
        para = squash(text[start:end])
        for lead in index.get(para[:24], ()):
            if para.startswith(lead):
                kind = "risk" if lead.rstrip("”\"’) ").endswith((".", "?")) else "group"
                headings.append({"start": start, "kind": kind, "text": lead[:300]})
                break
    return headings


# -- regex fallback ------------------------------------------------------------

_HEADING = re.compile(r"^item\s*(\d{1,2}[a-c]?)\b\s*[.:\-–—]?", re.I | re.M)


def regex_sections(text: str) -> dict[str, str]:
    """Find each item's body in plain text, ignoring the table of contents.

    Every line starting "Item N" is a candidate heading; an item's body runs to
    the next candidate for a *different* item. TOC entries (and cross-reference
    lines) have tiny bodies because the next item follows on the next line, so
    for each item we keep the candidate with the longest body.
    """
    heads = [(m.start(), m.group(1).upper()) for m in _HEADING.finditer(text)]
    best: dict[str, tuple[int, int]] = {}
    for i, (start, item) in enumerate(heads):
        end = len(text)
        for nxt_start, nxt_item in heads[i + 1:]:
            if nxt_item != item:
                end = nxt_start
                break
        if item not in best or (end - start) > (best[item][1] - best[item][0]):
            best[item] = (start, end)
    return {item: text[s:e].strip() for item, (s, e) in best.items() if item in ITEMS}


# -- edgartools path -----------------------------------------------------------

def edgartools_sections(html: str) -> dict[str, str]:
    from edgar.documents import ParserConfig, parse_html

    doc = parse_html(html, ParserConfig(form="10-K"))
    out: dict[str, str] = {}
    for section in doc.sections.values():
        item = (getattr(section, "item", None) or "").upper()
        if item in ITEMS and item not in out:
            out[item] = normalize(section.text() or "")
    return out


# -- orchestration -------------------------------------------------------------

def split_sections(html: str) -> dict[str, dict]:
    """Returns {item_id: {"text":..., "method":...}} for the items we keep."""
    try:
        primary = edgartools_sections(html)
    except Exception as exc:  # parser bugs shouldn't stop the run; fall back
        log.warning("edgartools parse failed (%s); using regex fallback", exc)
        primary = {}

    fallback: dict[str, str] | None = None
    result: dict[str, dict] = {}
    for item in ITEMS:
        text = primary.get(item, "")
        method = "edgartools"
        if len(text) < MIN_REQUIRED_CHARS:
            if fallback is None:
                fallback = regex_sections(html_to_text(html))
            alt = fallback.get(item, "")
            if len(alt) > len(text):
                text, method = alt, "regex"
        if text:
            result[item] = {"text": text, "method": method}
    return result


def process_filing(filing_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((filing_dir / "metadata.json").read_text())
    html = (filing_dir / meta["file"]).read_text(encoding="utf-8", errors="replace")
    items = split_sections(html)
    leads = emphasized_leads(html)

    failures = []
    for item in REQUIRED_ITEMS:
        n = len(items.get(item, {}).get("text", ""))
        if n < MIN_REQUIRED_CHARS:
            issue = "missing" if n == 0 else f"short ({n} chars)"
            log.warning("%s %s: Item %s %s", meta["ticker"], meta["accession"], item, issue)
            failures.append({**_ids(meta), "item": item, "issue": issue})

    record = {
        **_ids(meta),
        "form": meta["form"],
        "filing_date": meta["filing_date"],
        "period_of_report": meta["period_of_report"],
        "source_url": meta["source_url"],
        "sections": {
            ITEMS[item][0]: {
                "item": item,
                "title": ITEMS[item][1],
                "method": data["method"],
                "chars": len(data["text"]),
                "headings": find_headings(data["text"], leads),
                "text": data["text"],
            }
            for item, data in items.items()
        },
    }
    return record, failures


def _ids(meta: dict) -> dict:
    return {k: meta[k] for k in ("cik", "ticker", "accession", "fiscal_year")}


def run(tickers: list[str] | None, raw_dir: Path = RAW_DIR, out_dir: Path = SECTIONS_DIR,
        failures_csv: Path = FAILURES_CSV) -> list[dict]:
    ciks = {c.cik for c in load_companies(tickers=tickers)}
    out_dir.mkdir(parents=True, exist_ok=True)
    all_failures: list[dict] = []
    written: dict[str, str] = {}

    for meta_path in sorted(raw_dir.glob("*/*/metadata.json")):
        if meta_path.parent.parent.name not in ciks:
            continue
        filing_dir = meta_path.parent
        try:
            record, failures = process_filing(filing_dir)
        except Exception as exc:
            meta = json.loads(meta_path.read_text())
            log.error("%s %s: section split failed: %s", meta["ticker"], meta["accession"], exc)
            all_failures.append({**_ids(meta), "item": "*", "issue": f"error: {exc}"})
            continue
        all_failures.extend(failures)

        name = f"{record['ticker']}_{record['fiscal_year']}"
        if name in written and written[name] != record["accession"]:
            log.warning("%s: two filings map to FY%s (%s, %s); suffixing with accession",
                        record["ticker"], record["fiscal_year"], written[name], record["accession"])
            name = f"{name}_{record['accession']}"
        written[name] = record["accession"]
        (out_dir / f"{name}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
        summary = ", ".join(f"{k}={v['chars']:,}({v['method'][0]})" for k, v in record["sections"].items())
        log.info("%-6s FY%s  %s", record["ticker"], record["fiscal_year"], summary)

    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    with failures_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cik", "ticker", "accession", "fiscal_year", "item", "issue"])
        w.writeheader()
        w.writerows(all_failures)
    return all_failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("edgar").setLevel(logging.WARNING)  # parser is chatty at INFO
    failures = run(args.tickers)
    print(f"{len(failures)} parse failures -> {FAILURES_CSV}")


if __name__ == "__main__":
    main()
