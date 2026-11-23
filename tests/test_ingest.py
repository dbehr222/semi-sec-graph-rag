import json
from pathlib import Path

import pytest

from ingest import chunk, download, sections

FIXTURES = Path(__file__).parent / "fixtures"


# -- helpers -------------------------------------------------------------------

def _paragraphs(word: str, n: int) -> str:
    """n distinct paragraphs of ~120 tokens built from one marker word."""
    return "\n\n".join(
        f"Paragraph {i} about {word}. " + " ".join(f"{word}{i}x{j} is discussed here." for j in range(30))
        for i in range(n)
    )


def _record() -> dict:
    return {
        "cik": "0000000001",
        "ticker": "TEST",
        "accession": "0000000001-25-000001",
        "fiscal_year": 2025,
        "form": "10-K",
        "sections": {
            "Item1": {"text": _paragraphs("alpha", 25)},
            "Item1A": {"text": _paragraphs("beta", 25)},
        },
    }


# -- chunking ------------------------------------------------------------------

def test_chunk_ids_are_deterministic():
    first = chunk.chunk_filing(_record())
    second = chunk.chunk_filing(_record())
    assert len(first) > 4
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert first == second
    # id is exactly the documented formula
    c = first[0]
    section = c["section_path"].rsplit("/", 1)[1]
    assert c["chunk_id"] == chunk.make_chunk_id(c["accession"], section, c["start_char"])


def test_chunks_never_cross_section_boundaries():
    rec = _record()
    chunks = chunk.chunk_filing(rec)
    by_section = {"Item1": "alpha", "Item1A": "beta"}
    for c in chunks:
        section = c["section_path"].rsplit("/", 1)[1]
        source = rec["sections"][section]["text"]
        assert 0 <= c["start_char"] < c["end_char"] <= len(source)
        assert c["text"] == source[c["start_char"]:c["end_char"]]
        other = "beta" if by_section[section] == "alpha" else "alpha"
        assert other not in c["text"]
    assert {c["section_path"] for c in chunks} == {"10-K/2025/Item1", "10-K/2025/Item1A"}


def test_chunks_respect_size_and_overlap():
    chunks = [c for c in chunk.chunk_filing(_record()) if c["section_path"].endswith("Item1")]
    assert all(c["token_count"] <= chunk.TARGET_TOKENS + chunk.OVERLAP_TOKENS for c in chunks)
    # consecutive chunks overlap (next starts before previous ends) and always advance
    for a, b in zip(chunks, chunks[1:]):
        assert a["start_char"] < b["start_char"] < a["end_char"]


def test_numeric_tables_are_skipped():
    table = "\n".join(f"Line {i}  $ {i*1000:,}  $ {i*900:,}  {i}%" for i in range(40))
    text = _paragraphs("gamma", 3) + "\n\n" + table + "\n\n" + _paragraphs("delta", 3)
    spans = chunk.chunk_section(text)
    assert spans
    for s, e in spans:
        assert "$ 1,000" not in text[s:e]


# -- download ------------------------------------------------------------------

class FakeClient:
    def __init__(self):
        self.calls = []

    def get_json(self, url):
        self.calls.append(url)
        return {"filings": {"recent": {
            "form": ["10-K", "8-K", "10-K/A", "10-K"],
            "accessionNumber": ["0000000001-25-000010", "x", "y", "0000000001-24-000010"],
            "filingDate": ["2025-02-20", "2025-03-01", "2025-04-01", "2024-02-20"],
            "reportDate": ["2025-01-26", "", "", "2024-01-05"],
            "primaryDocument": ["doc25.htm", "x.htm", "y.htm", "doc24.htm"],
        }}}

    def get_bytes(self, url):
        self.calls.append(url)
        return f"<html>{url}</html>".encode()


@pytest.fixture
def companies_csv(tmp_path):
    p = tmp_path / "companies.csv"
    p.write_text("cik,ticker,name,sic,role,revenue,public_float,latest_10k\n"
                 "1,TEST,Test Corp,3674,core,1,1,2025-02-20\n")
    return p


def test_rerunning_download_is_a_noop(tmp_path, companies_csv):
    raw = tmp_path / "raw"
    client = FakeClient()
    first = download.run(None, years=3, form="10-K", client=client, raw_dir=raw, companies_csv=companies_csv)
    assert first == {"downloaded": 2, "skipped": 0, "failed": 0}  # 8-K and 10-K/A ignored

    files = sorted(raw.rglob("*"))
    snapshot = {p: p.stat().st_mtime_ns for p in files if p.is_file()}
    doc_fetches = sum(1 for u in client.calls if "/Archives/" in u)

    second = download.run(None, years=3, form="10-K", client=client, raw_dir=raw, companies_csv=companies_csv)
    assert second == {"downloaded": 0, "skipped": 2, "failed": 0}
    assert sum(1 for u in client.calls if "/Archives/" in u) == doc_fetches  # no document re-fetch
    assert {p: p.stat().st_mtime_ns for p in snapshot} == snapshot  # nothing rewritten

    meta = json.loads((raw / "0000000001" / "0000000001-24-000010" / "metadata.json").read_text())
    assert meta["fiscal_year"] == 2023  # period ends Jan 5 -> 52/53-week rule
    assert meta["cik"] == "0000000001"


def test_redownloads_when_sha_mismatches(tmp_path, companies_csv):
    raw = tmp_path / "raw"
    client = FakeClient()
    download.run(None, 3, "10-K", client=client, raw_dir=raw, companies_csv=companies_csv)
    doc = raw / "0000000001" / "0000000001-25-000010" / "10-K.htm"
    doc.write_text("corrupted")
    again = download.run(None, 3, "10-K", client=client, raw_dir=raw, companies_csv=companies_csv)
    assert again["downloaded"] == 1 and again["skipped"] == 1


@pytest.mark.parametrize("period,expected", [
    ("2026-01-25", 2026),  # NVDA: late January stays
    ("2025-01-03", 2024),  # 52/53-week filer ending first week of January
    ("2025-01-07", 2024),
    ("2025-01-08", 2025),
    ("2025-10-26", 2025),
    ("2025-12-31", 2025),
])
def test_fiscal_year(period, expected):
    assert download.fiscal_year(period) == expected


# -- sections ------------------------------------------------------------------

def _fixture_html() -> str:
    return (FIXTURES / "toc_10k.html").read_text()


def test_regex_splitter_ignores_table_of_contents():
    found = sections.regex_sections(sections.html_to_text(_fixture_html()))
    assert found["1"].startswith("Item 1. Business")
    assert "BUSINESS-BODY-MARKER" in found["1"]
    assert "RISK-BODY-MARKER" in found["1A"]
    assert "Page" not in found["1A"] and "| 12" not in found["1A"]
    assert "CYBER-BODY-MARKER" in found["1C"]
    assert "MDA-BODY-MARKER" in found["7"]
    assert "MARKET-RISK-MARKER" in found["7A"]
    # bodies stop at the next item
    assert "RISK-BODY-MARKER" not in found["1"]
    assert "MARKET-RISK-MARKER" not in found["7"]


def test_split_sections_ignores_table_of_contents():
    """Full path (edgartools first, regex fallback) must also land on the bodies."""
    found = sections.split_sections(_fixture_html())
    assert "RISK-BODY-MARKER" in found["1A"]["text"]
    assert "BUSINESS-BODY-MARKER" in found["1"]["text"]
    assert "Unresolved Staff Comments" not in found["1A"]["text"]


# -- page furniture ------------------------------------------------------------

def test_normalize_strips_page_numbers_and_rejoins_sentences():
    raw = ("Damages against our customers from their\n\n  17\n\n  customers), we may face claims.\n\n"
           "Table of Contents\n\nNext paragraph.\n\n- 18 -\n\nPage 19\n\nFinal paragraph.")
    out = sections.normalize(raw)
    assert "17" not in out and "18" not in out and "19" not in out
    assert "Table of Contents" not in out
    assert "from their customers), we may face claims." in out
    assert "Next paragraph.\n\nFinal paragraph." in out


def test_normalize_keeps_numbers_inside_text():
    out = sections.normalize("Revenue was $17 million.\n\nIn 2025 we grew.\n\n2025  2024  Change")
    assert out == "Revenue was $17 million.\n\nIn 2025 we grew.\n\n2025  2024  Change"


# -- risk-factor headings ------------------------------------------------------

def test_headings_detected_from_emphasis_on_fixture():
    html = _fixture_html()
    item1a = sections.split_sections(html)["1A"]["text"]
    found = sections.find_headings(item1a, sections.emphasized_leads(html))
    texts = [h["text"] for h in found]
    assert "We depend on a limited number of third-party foundries." in texts  # run-in <i>
    assert "Export controls could limit our sales." in texts
    assert all(item1a[h["start"]:].startswith(h["text"][:20]) for h in found)
    assert not any("RISK-BODY-MARKER" in t for t in texts)  # plain body text is not a heading


def _risk_section(n_risks: int = 12) -> tuple[str, set[int], list[tuple[int, int]]]:
    """Section text with risks of varied length; returns text, heading offsets, group spans."""
    parts, starts = [], []
    pos = 0
    for i in range(n_risks):
        heading = f"Risk heading number {i} could adversely affect our business."
        n_paras = [1, 2, 9, 1, 3][i % 5]  # i%5==2 gives a risk longer than the target
        body = "\n\n".join(
            " ".join(f"risk{i}word{j}p{k} matters." for j in range(25)) for k in range(n_paras))
        block = heading + "\n\n" + body
        starts.append(pos)
        parts.append(block)
        pos += len(block) + 2
    text = "\n\n".join(parts)
    groups = [(s, (starts[i + 1] - 2) if i + 1 < len(starts) else len(text)) for i, s in enumerate(starts)]
    return text, set(starts), groups


def test_risk_chunks_break_on_headings():
    text, heads, groups = _risk_section()
    spans = chunk.chunk_section(text, heads)
    assert spans
    for s, e in spans:
        for gs, ge in groups:
            overlap = min(e, ge) - max(s, gs)
            if overlap <= 0:
                continue
            # a chunk either contains a whole risk, or lies entirely inside one risk
            assert (s <= gs and e >= ge) or (s >= gs and e <= ge), (s, e, gs, ge)
    # every chunk that isn't a continuation of an oversized risk starts on a heading
    for s, e in spans:
        if s in heads:
            continue
        assert any(gs < s < ge for gs, ge in groups)
    # oversized risks are split and their pieces overlap
    long_pieces = [(s, e) for s, e in spans if groups[2][0] <= s < groups[2][1]]
    assert len(long_pieces) >= 2
    assert long_pieces[1][0] < long_pieces[0][1]


def test_chunk_records_carry_heading():
    text, heads, _ = _risk_section(5)
    headings = [{"start": s, "kind": "risk", "text": text[s:text.index("\n", s)]} for s in sorted(heads)]
    rec = _record()
    rec["sections"] = {"Item1A": {"text": text, "headings": headings}}
    chunks = chunk.chunk_filing(rec)
    assert all(c["heading"] and c["heading"].startswith("Risk heading number") for c in chunks)
    assert chunk.chunk_filing(rec) == chunks  # still deterministic
