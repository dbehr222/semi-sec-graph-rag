"""Chunk section text into ~800-token passages with ~100-token overlap.

Rules
- Chunks never cross a section boundary: each section is chunked on its own,
  and offsets are character offsets into that section's text.
- A chunk is always a contiguous slice: text == section_text[start:end].
- Breaks fall on paragraph boundaries; a paragraph longer than the target is
  split on sentences (then lines, then raw characters as a last resort).
- Overlap: the next chunk starts at the earliest sentence/line boundary inside
  the previous chunk such that the overlap is <= OVERLAP_TOKENS.
- Item 1A is heading-aware: sections.py records where each risk-factor
  heading starts. Each risk (heading + body) is a group; whole groups are
  packed into chunks and a break always falls on a heading, so a chunk never
  holds part of one risk plus part of the next. A risk longer than the target
  is split inside itself (with overlap) and those chunks stay within that risk.
  Every chunk carries the text of the heading it falls under.
- Tables arrive already flattened to text by sections.py. Numeric-heavy blocks
  (financial tables) are cut out of the flow, and any chunk that is still
  mostly numeric is dropped; numbers will come from XBRL.
- chunk_id = sha1(f"{accession}|{section}|{start}")[:16], so IDs are stable
  across runs and shared by Neo4j and pgvector.

Tokens are counted with tiktoken's cl100k_base as a proxy; Claude's tokenizer
differs, so treat counts as estimates.

Output: data/processed/chunks/{ticker}_{fy}.jsonl
Run:    python -m ingest.chunk --tickers NVDA DIOD AMAT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ingest.companies import load_companies
from ingest.textutil import PARA as _PARA
from ingest.textutil import spans as _spans

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SECTIONS_DIR = ROOT / "data" / "processed" / "sections"
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"

TARGET_TOKENS = 800
OVERLAP_TOKENS = 100
NUMERIC_THRESHOLD = 0.5  # fraction of numeric-looking words that makes text "mostly numeric"
MIN_NUMERIC_BLOCK_TOKENS = 20  # don't cut tiny numeric lines (e.g. a "2025" heading) out of the flow
MIN_CHUNK_TOKENS = 30  # drops orphaned table captions ("Net revenue by segment were as follows:")
HEADING_BREAK_SECTIONS = {"Item1A"}  # sections chunked on risk-factor headings


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


_NUMERIC_WORD = re.compile(r"^[\$€£(]*[-+–—]?[\d.,]+[%)]*$|^[\$%()\-–—|]+$")


def is_mostly_numeric(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    numeric = sum(1 for w in words if _NUMERIC_WORD.match(w))
    return numeric / len(words) >= NUMERIC_THRESHOLD


def make_chunk_id(accession: str, section: str, start: int) -> str:
    return hashlib.sha1(f"{accession}|{section}|{start}".encode()).hexdigest()[:16]


# -- units: paragraphs, split further only when too big -------------------------

@dataclass
class Unit:
    start: int
    end: int
    tokens: int
    numeric: bool


_SENTENCE = re.compile(r"(?<=[.!?;])[\"”’)]*\s+")
_LINE = re.compile(r"\n")


def _split_big(text: str, start: int, end: int, level: int = 0) -> list[tuple[int, int]]:
    """Split one oversized span into pieces under TARGET_TOKENS."""
    if count_tokens(text[start:end]) <= TARGET_TOKENS:
        return [(start, end)]
    if level < 2:
        sep = _SENTENCE if level == 0 else _LINE
        pieces = _spans(text, start, end, sep)
        if len(pieces) > 1:
            out = []
            for s, e in pieces:
                out.extend(_split_big(text, s, e, level + 1) if level == 0 else _split_big(text, s, e, 2))
            return out
        return _split_big(text, start, end, level + 1)
    # last resort: fixed character windows (~4 chars/token)
    step = TARGET_TOKENS * 3
    return [(s, min(s + step, end)) for s in range(start, end, step)]


def make_units(text: str) -> list[Unit]:
    units = []
    for ps, pe in _spans(text, 0, len(text), _PARA):
        para = text[ps:pe]
        numeric = is_mostly_numeric(para)
        for s, e in _split_big(text, ps, pe):
            units.append(Unit(s, e, count_tokens(text[s:e]), numeric))
    return units


def _overlap_start(text: str, chunk_start: int, chunk_end: int) -> int | None:
    """Earliest sentence/line boundary in the chunk whose tail is <= OVERLAP_TOKENS."""
    # look back only a bounded window: OVERLAP_TOKENS tokens is well under 1,000 chars
    window_start = max(chunk_start + 1, chunk_end - OVERLAP_TOKENS * 8)
    for m in re.finditer(r"(?<=[.!?;])[\"”’)]*\s+|\n+", text[window_start:chunk_end]):
        s = window_start + m.end()
        if s >= chunk_end:
            break
        if count_tokens(text[s:chunk_end]) <= OVERLAP_TOKENS:
            return s
    return None


# -- chunking -----------------------------------------------------------------

def _pack(text: str, units: list[Unit]) -> list[tuple[int, int]]:
    """Greedy paragraph packing with sentence-level overlap; numeric tables cut out."""
    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    cur_tokens = 0

    def emit(carry_overlap: bool) -> None:
        nonlocal cur_start, cur_end, cur_tokens
        if cur_start is None:
            return
        spans.append((cur_start, cur_end))
        ov = _overlap_start(text, cur_start, cur_end) if carry_overlap else None
        if ov is not None:
            cur_start, cur_tokens = ov, count_tokens(text[ov:cur_end])
        else:
            cur_start, cur_tokens = None, 0

    for u in units:
        if u.numeric and u.tokens >= MIN_NUMERIC_BLOCK_TOKENS:
            # financial table: close the current chunk, skip the table, restart after it
            emit(carry_overlap=False)
            continue
        if cur_start is not None and cur_tokens + u.tokens > TARGET_TOKENS and cur_end > cur_start:
            emit(carry_overlap=True)
        if cur_start is None:
            cur_start = u.start
        cur_end = u.end
        cur_tokens += u.tokens
    if cur_start is not None and (not spans or spans[-1][1] < cur_end):
        spans.append((cur_start, cur_end))

    return spans


def _is_table(u: Unit) -> bool:
    return u.numeric and u.tokens >= MIN_NUMERIC_BLOCK_TOKENS


def _heading_groups(units: list[Unit], heading_starts: set[int]) -> list[list[Unit]]:
    """Split units at headings. Consecutive headings (e.g. a "Risks Related to X"
    group title followed by the first risk) stay in one group."""
    groups: list[list[Unit]] = []
    for u in units:
        starts_new = u.start in heading_starts and groups and not all(
            g.start in heading_starts for g in groups[-1])
        if not groups or starts_new:
            groups.append([u])
        else:
            groups[-1].append(u)
    return groups


def _pack_groups(text: str, groups: list[list[Unit]]) -> list[tuple[int, int]]:
    """Pack whole groups into chunks; oversized groups are split on their own."""
    spans: list[tuple[int, int]] = []
    cur: list[Unit] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            spans.append((cur[0].start, cur[-1].end))
        cur, cur_tokens = [], 0

    for g in groups:
        g_tokens = sum(u.tokens for u in g)
        if g_tokens > TARGET_TOKENS or any(_is_table(u) for u in g):
            flush()
            spans.extend(_pack(text, g))
            continue
        if cur and cur_tokens + g_tokens > TARGET_TOKENS:
            flush()
        cur.extend(g)
        cur_tokens += g_tokens
    flush()
    return spans


def chunk_section(text: str, heading_starts: set[int] | None = None) -> list[tuple[int, int]]:
    """Return (start, end) spans of kept chunks for one section's text.

    With `heading_starts`, chunk breaks fall on those offsets (see module doc).
    """
    units = make_units(text)
    if heading_starts:
        spans = _pack_groups(text, _heading_groups(units, heading_starts))
    else:
        spans = _pack(text, units)
    return [
        (s, e) for s, e in spans
        if not is_mostly_numeric(text[s:e]) and count_tokens(text[s:e]) >= MIN_CHUNK_TOKENS
    ]


def _heading_at(headings: list[dict], offset: int) -> str | None:
    """Text of the last heading starting at or before `offset`."""
    current = None
    for h in headings:
        if h["start"] > offset:
            break
        current = h["text"]
    return current


def chunk_filing(record: dict) -> list[dict]:
    chunks = []
    for section, data in record["sections"].items():
        text = data["text"]
        headings = sorted(data.get("headings", []), key=lambda h: h["start"])
        breaks = {h["start"] for h in headings} if section in HEADING_BREAK_SECTIONS else None
        for start, end in chunk_section(text, breaks):
            body = text[start:end]
            chunks.append({
                "chunk_id": make_chunk_id(record["accession"], section, start),
                "cik": record["cik"],
                "ticker": record["ticker"],
                "accession": record["accession"],
                "fiscal_year": record["fiscal_year"],
                "section_path": f"{record['form']}/{record['fiscal_year']}/{section}",
                "start_char": start,
                "end_char": end,
                "token_count": count_tokens(body),
                "heading": _heading_at(headings, start),
                "text": body,
            })
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), f"duplicate chunk ids in {record['accession']}"
    return chunks


def run(tickers: list[str] | None, sections_dir: Path = SECTIONS_DIR,
        out_dir: Path = CHUNKS_DIR) -> dict[str, int]:
    ciks = {c.cik for c in load_companies(tickers=tickers)}
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"filings": 0, "chunks": 0, "tokens": 0}
    for path in sorted(sections_dir.glob("*.json")):
        record = json.loads(path.read_text())
        if record["cik"] not in ciks:
            continue
        chunks = chunk_filing(record)
        with (out_dir / f"{path.stem}.jsonl").open("w") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        tokens = sum(c["token_count"] for c in chunks)
        totals["filings"] += 1
        totals["chunks"] += len(chunks)
        totals["tokens"] += tokens
        log.info("%-14s %4d chunks %8d tokens", path.stem, len(chunks), tokens)
    return totals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t = run(args.tickers)
    print(f"{t['filings']} filings, {t['chunks']} chunks, {t['tokens']:,} tokens -> {CHUNKS_DIR}")


if __name__ == "__main__":
    main()
