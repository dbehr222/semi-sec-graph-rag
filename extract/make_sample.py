"""Pick an information-rich pilot sample of chunk_ids -> data/pilot_sample.txt.

Chunks are scored by how much extractable content they carry: distinct named
companies (universe + common supply-chain names) and relationship cue phrases
(sole source, % of revenue, compete, acquired, foundry, export controls...).
The top chunks per ticker (with a floor of Item 1A chunks) are taken from each
ticker's latest fiscal year only, because 10-K text repeats almost verbatim year to year.

Note: an eval on hand-picked rich chunks measures extraction quality where
there is something to extract; it overstates quality on an average chunk.

Run:  python -m extract.make_sample
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
COMPANIES_CSV = ROOT / "config" / "companies.csv"
OUT = ROOT / "data" / "pilot_sample.txt"

PER_TICKER = 10          # 3 tickers x 10 = 30
MIN_RISK_CHUNKS = 3      # keep some Item 1A so risk categories get exercised
MIN_SCORE = 3            # at least one named company, or three relationship cues
SECTIONS = ("Item1", "Item1A", "Item7")

# Surface forms that are the same company; counted once.
ALIASES = {
    "advanced micro devices": "amd", "micron technology": "micron", "taiwan semiconductor": "tsmc",
    "hon hai": "foxconn", "alphabet": "google", "on semiconductor": "onsemi",
    "marvell technology": "marvell", "siliconware": "ase",
}

# Names that matter for a semiconductor supply chain but aren't in the universe CSV.
EXTRA_NAMES = [
    "TSMC", "Taiwan Semiconductor", "Samsung", "SK hynix", "Kioxia", "GlobalFoundries",
    "UMC", "SMIC", "Foxconn", "Hon Hai", "Wistron", "Quanta", "ASE", "Siliconware", "JCET",
    "Arrow", "Avnet", "WPG", "Digi-Key", "Mouser", "ASML", "Tokyo Electron", "Synopsys",
    "Cadence", "Arm", "Dell", "HP", "Lenovo", "Super Micro", "Tesla", "Oracle", "CoreWeave",
    "OpenAI", "Mellanox", "Xilinx", "Lite-On", "BYD", "Bosch", "Continental", "Huawei",
]

CUES = [
    r"sole[- ]source", r"single[- ]source", r"limited number of (?:suppliers|customers)",
    r"accounted for (?:approximately )?\d+%", r"\d+% of (?:our )?(?:net )?(?:revenue|sales)",
    r"compet\w+ (?:with|include)", r"competitors include", r"acquir\w+", r"joint venture",
    r"distributors?", r"foundr\w+", r"wafers?", r"assembly and test", r"subcontract\w+",
    r"export control", r"\bChina\b", r"\bTaiwan\b", r"\bKorea\b", r"tariff",
    r"customer concentration", r"largest customers?", r"headquarter\w*", r"manufacturing facilit\w+",
]


def name_patterns() -> list[tuple[str, re.Pattern]]:
    df = pd.read_csv(COMPANIES_CSV, dtype=str, keep_default_na=False)
    names = []
    for raw in df["name"]:
        # "ADVANCED MICRO DEVICES INC" -> "Advanced Micro Devices"
        core = re.sub(r"[,.]|\b(INC|CORP|CORPORATION|LTD|HOLDINGS|N\.?V|LLC|PLC|CO)\b\.?|/\w+/", " ", raw, flags=re.I)
        core = re.sub(r"\s+", " ", core).strip()
        if len(core) >= 3:
            names.append(core)
    # tickers that are also distinctive words in prose
    names += ["NVIDIA", "Intel", "AMD", "Micron", "Broadcom", "Qualcomm", "Microsoft", "Amazon",
              "Google", "Alphabet", "Apple", "Meta", "Applied Materials", "Lam Research", "KLA"]
    names += EXTRA_NAMES
    seen, out = set(), []
    for n in names:
        if n.lower() in seen:
            continue
        seen.add(n.lower())
        # case-insensitive body (CSV names are ALL CAPS) but must start with a capital
        # in the text, so the word "cadence" doesn't count as Cadence Design Systems
        out.append((n, re.compile(rf"\b(?=(?-i:[A-Z]))(?i:{re.escape(n)})\b")))
    return out


def score(text: str, filer_words: set[str], names: list[tuple[str, re.Pattern]]) -> tuple[float, list[str], list[str]]:
    hits: dict[str, str] = {}
    for n, pat in names:
        key = ALIASES.get(n.lower(), n.lower())
        if key in filer_words or key in hits:
            continue
        if pat.search(text):
            hits[key] = n
    hit_names = list(hits.values())
    hit_cues = [c for c in CUES if re.search(c, text, re.I)]
    return 3 * len(hit_names) + len(hit_cues), hit_names, hit_cues


# the filer mentioning itself doesn't count as information
FILER_SELF_NAMES = {"NVDA": "nvidia", "DIOD": "diodes", "AMAT": "applied materials"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="*", default=["NVDA", "DIOD", "AMAT"])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    names = name_patterns()

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for line in path.open():
            c = json.loads(line)
            if c["ticker"] in args.tickers:
                by_ticker[c["ticker"]].append(c)

    picked = []
    for ticker in args.tickers:
        chunks = by_ticker[ticker]
        latest = max(c["fiscal_year"] for c in chunks)
        filer_words = {ticker.lower(), FILER_SELF_NAMES.get(ticker, "")}
        scored = []
        for c in chunks:
            section = c["section_path"].rsplit("/", 1)[1]
            if c["fiscal_year"] != latest or section not in SECTIONS:
                continue
            s, hn, hc = score(c["text"], filer_words, names)
            if s >= MIN_SCORE:
                scored.append((s, section, c["start_char"], c, hn, hc))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))  # deterministic
        risks = [t for t in scored if t[1] == "Item1A"][:MIN_RISK_CHUNKS]
        rest = [t for t in scored if t not in risks][: PER_TICKER - len(risks)]
        chosen = sorted(risks + rest, key=lambda t: (-t[0], t[1], t[2]))
        if len(chosen) < PER_TICKER:
            print(f"warning: only {len(chosen)} chunks for {ticker} reach score {MIN_SCORE}")
        picked.extend((t[0], t[2], t[3], t[4], t[5]) for t in chosen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# pilot sample: {len(picked)} information-rich chunks (latest FY), tickers={','.join(args.tickers)}\n")
        for s, _, c, hn, hc in picked:
            f.write(f"{c['chunk_id']}  # {c['ticker']} {c['section_path']} score={s} names={','.join(hn[:8])}\n")
    for s, _, c, hn, hc in picked:
        print(f"{c['chunk_id']} {c['ticker']:4} {c['section_path']:18} score={s:3} names={', '.join(hn[:7])}")
    print(f"\nWrote {len(picked)} chunk ids to {args.out}")


if __name__ == "__main__":
    main()
