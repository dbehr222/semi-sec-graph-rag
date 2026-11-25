"""Extract entities and relationships from chunks with Claude, validated against extract/ontology.py.

Per chunk:
  1. System prompt = build_ontology_prompt() (cached) + the filer's name.
  2. One tool, `record_extraction`, whose input_schema is LLMExtraction.model_json_schema().
     Models that accept a forced tool_choice get one. Claude Opus 5.5 / Sonnet 5.5 /
     Fable 5.1+ reject forced tool_choice (400), so for those the prompt names the
     tool and a reply without a tool call counts as a failed attempt.
  3. Validate with LLMExtraction.model_validate; on failure send the validation error
     back as an error tool_result and retry (up to --max-retries times).
  4. check_evidence per edge; edges whose quote isn't in the chunk are dropped and logged.

API results are cached in data/cache/extraction/ by sha256(chunk text + prompt_version + model),
so re-runs are free. Outputs: data/processed/extractions/{model}/{chunk_id}.json and failures.jsonl.

Run:
  python -m extract.run_extraction --chunks 'data/processed/chunks/*.jsonl' \\
      --sample-file data/pilot_sample.txt --model claude-sonnet-5-5 --prompt-version v1 \\
      --input-price 2.00 --output-price 10.00
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from extract.ontology import (
    ChunkRef,
    LLMExtraction,
    RelType,
    build_ontology_prompt,
    check_evidence,
    finalize,
    normalize_name,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "extraction"
OUT_ROOT = ROOT / "data" / "processed" / "extractions"
COMPANIES_CSV = ROOT / "config" / "companies.csv"

TOOL_NAME = "record_extraction"
DEFAULT_MODEL = "claude-sonnet-5-5"  # project ceiling is Sonnet tier; Haiku is the cheaper option
MAX_TOKENS = 16000

# Models that return 400 on tool_choice {"type": "tool"} / {"type": "any"}.
NO_FORCED_TOOL = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2",
                  "claude-mythos-5-1", "claude-mythos-5-2")
# Models that support the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1",
                   "claude-fable-5-2")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Prompt-cache pricing relative to the base input price (5-minute TTL).
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


# -- request building ----------------------------------------------------------

# Extra prompt rules appended after build_ontology_prompt(), keyed by --prompt-version.
# Versions not listed here (e.g. "v1") use the ontology prompt unchanged. extract/ontology.py
# stays the single source of truth for the schema; these only tighten granularity.
PROMPT_ADDENDA = {
    "v2": """
ADDITIONAL RULES:
- OPERATES_IN: the target must be a country or a broad region ('Taiwan', 'China', 'United States', 'Europe', 'Asia'). Never a city, state, province, county or site name. If the text names a city or site, use its country instead. Emit at most one OPERATES_IN edge per (country, activity) in a chunk.
- OFFERS: name product families or brands the filer sells ('Blackwell', 'GeForce', 'MOSFETs'). Do not list every variant, part type or generic category. Prefer the most general name the text uses, and emit at most 5 OFFERS edges per chunk.
- ACQUIRED: only when the text states that the source acquired the target. Competitors, partners, suppliers and customers are not acquisitions.
""",
}

def tool_definition() -> dict:
    return {
        "name": TOOL_NAME,
        "description": (
            "Record every entity and relationship stated in the chunk, following the ontology "
            "in the system prompt. Call exactly once. Use empty lists if nothing fits."
        ),
        "input_schema": LLMExtraction.model_json_schema(),
    }


def system_blocks(filer: dict, prompt_version: str = "v1") -> list[dict]:
    return [
        # identical for every chunk and every filer -> cacheable prefix (tools + this block)
        {"type": "text", "text": build_ontology_prompt() + PROMPT_ADDENDA.get(prompt_version, ""),
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": (
            f"FILER: {filer['name']} (ticker {filer['ticker']}, CIK {filer['cik']}). "
            f"Whenever the text says 'we', 'us', 'our' or 'the Company', the entity is "
            f"{filer['name']}; use that exact canonical_name."
        )},
    ]


def user_message(chunk: dict) -> str:
    return (
        f"Section: {chunk['section_path']}\n\n<chunk>\n{chunk['text']}\n</chunk>\n\n"
        f"Extract the entities and relationships this chunk states, then call the "
        f"{TOOL_NAME} tool once with the result."
    )


def request_kwargs(model: str, effort: str | None, use_fallback: bool) -> dict:
    kw: dict = {"model": model, "max_tokens": MAX_TOKENS, "tools": [tool_definition()]}
    if model.startswith(NO_FORCED_TOOL):
        kw["tool_choice"] = {"type": "auto"}
    else:
        kw["tool_choice"] = {"type": "tool", "name": TOOL_NAME}
    if effort:
        kw["output_config"] = {"effort": effort}
    if use_fallback and model.startswith(FALLBACK_MODELS):
        kw["betas"] = [FALLBACK_BETA]
        kw["fallbacks"] = "default"
    return kw


# -- one chunk -----------------------------------------------------------------

def cache_key(text: str, prompt_version: str, model: str) -> str:
    return hashlib.sha256(f"{text}\x1f{prompt_version}\x1f{model}".encode()).hexdigest()


def _usage_dict(u) -> dict:
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


def filer_key(name: str) -> str:
    """Normalized filer name; also strips SEC state tags ('DIODES INC /DEL/' -> 'diodes')."""
    return normalize_name(re.sub(r"/[A-Za-z]{2,4}/?", " ", name))


def reversed_competes(extraction: LLMExtraction, filer: dict) -> list:
    """COMPETES_WITH edges that point INTO the filer. A filing names competitors of the filer,
    so the edge must be filer -> competitor; the other direction is a model mistake."""
    fk = filer_key(filer["name"])
    return [r for r in extraction.relationships
            if r.rel_type == RelType.COMPETES_WITH and filer_key(r.target) == fk and filer_key(r.source) != fk]


def direction_error_text(edges: list, filer: dict) -> str:
    lines = [f"- You wrote '{r.source} COMPETES_WITH {r.target}'. The target is the filer, but COMPETES_WITH "
             f"must run from the filer to the competitor it names. Write '{filer['name']} COMPETES_WITH {r.source}'."
             for r in edges]
    return "Direction error in COMPETES_WITH edges:\n" + "\n".join(lines)


def call_model(client, chunk: dict, filer: dict, model: str, effort: str | None,
               max_retries: int, use_fallback: bool, prompt_version: str = "v1") -> dict:
    """Run the extract -> validate -> retry loop. Returns a JSON-serialisable record."""
    kw = request_kwargs(model, effort, use_fallback)
    create = client.beta.messages.create if "betas" in kw else client.messages.create
    messages: list = [{"role": "user", "content": user_message(chunk)}]
    usage = Counter()
    attempts = []
    extraction = None
    direction_dropped: list[dict] = []

    for attempt in range(max_retries + 1):
        resp = create(system=system_blocks(filer, prompt_version), messages=messages, **kw)
        usage.update(_usage_dict(resp.usage))
        served_by = getattr(resp, "model", model)

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            attempts.append({"ok": False, "stop_reason": "refusal",
                             "error": f"refusal: {getattr(details, 'category', None)}"})
            break

        tool_use = next((b for b in resp.content if b.type == "tool_use" and b.name == TOOL_NAME), None)
        if tool_use is None:
            error = f"No {TOOL_NAME} call in the response (stop_reason={resp.stop_reason})."
            attempts.append({"ok": False, "stop_reason": resp.stop_reason, "error": error})
            messages += [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": f"{error} Call the {TOOL_NAME} tool now with your extraction."},
            ]
            continue

        try:
            if resp.stop_reason == "max_tokens":
                raise ValueError("Response hit max_tokens; the tool input is truncated. Return fewer, shorter items.")
            extraction = LLMExtraction.model_validate(tool_use.input)
        except (ValidationError, ValueError) as exc:
            error = str(exc)[:4000]
            attempts.append({"ok": False, "stop_reason": resp.stop_reason, "error": error})
            messages += [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": tool_use.id, "is_error": True,
                    "content": f"Schema validation failed:\n{error}\n\nFix these problems and call {TOOL_NAME} again with the complete corrected extraction.",
                }]},
            ]
            continue

        flipped = reversed_competes(extraction, filer)
        if flipped:
            error = direction_error_text(flipped, filer)
            attempts.append({"ok": False, "kind": "direction", "stop_reason": resp.stop_reason, "error": error})
            if attempt < max_retries:
                messages += [
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": tool_use.id, "is_error": True,
                        "content": f"{error}\n\nCall {TOOL_NAME} again with the complete corrected extraction.",
                    }]},
                ]
                extraction = None
                continue
            # out of retries: drop the reversed edges, keep everything else
            direction_dropped = [r.model_dump(mode="json") for r in flipped]
            extraction = LLMExtraction.model_construct(
                entities=extraction.entities,
                relationships=[r for r in extraction.relationships if r not in flipped])
            break

        attempts.append({"ok": True, "stop_reason": resp.stop_reason, "served_by": served_by})
        break

    return {
        "model": model,
        "attempts": attempts,
        # "first try" = the first response passed the schema (a direction error is a separate check)
        "first_try_valid": bool(attempts) and (attempts[0]["ok"] or attempts[0].get("kind") == "direction"),
        "valid": extraction is not None,
        "extraction": extraction.model_dump(mode="json") if extraction else None,
        "direction_checked": True,
        "direction_retries": sum(1 for a in attempts if a.get("kind") == "direction"),
        "direction_dropped": direction_dropped,
        "usage": dict(usage),
    }


def split_by_evidence(result: LLMExtraction, text: str) -> tuple[list, list[dict]]:
    kept, dropped = [], []
    for r in result.relationships:
        problems = check_evidence(LLMExtraction.model_construct(entities=result.entities, relationships=[r]), text)
        if problems:
            dropped.append({"edge": r.model_dump(mode="json"), "reason": problems[0]})
        else:
            kept.append(r)
    return kept, dropped


def load_cached(chunk: dict, filer: dict, prompt_version: str, model: str) -> dict | None:
    """Cached API result, or None if absent or stale. Entries written before the direction
    check existed are stale only if they contain a reversed COMPETES_WITH edge; clean ones
    are exactly what the check would have accepted, so they stay valid."""
    path = CACHE_DIR / f"{cache_key(chunk['text'], prompt_version, model)}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if raw.get("valid") and not raw.get("direction_checked"):
        if reversed_competes(LLMExtraction.model_validate(raw["extraction"]), filer):
            log.info("%s: cached result has reversed COMPETES_WITH edges; re-calling", chunk["chunk_id"])
            return None
    return raw


def process_chunk(client, chunk: dict, filer: dict, args) -> dict:
    key = cache_key(chunk["text"], args.prompt_version, args.model)
    cache_path = CACHE_DIR / f"{key}.json"
    raw = load_cached(chunk, filer, args.prompt_version, args.model)
    cached = raw is not None
    if not cached:
        raw = call_model(client, chunk, filer, args.model, args.effort, args.max_retries,
                         not args.no_fallback, args.prompt_version)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw))
        tmp.replace(cache_path)

    kept, dropped, stored = [], [], []
    entities = []
    if raw["valid"]:
        result = LLMExtraction.model_validate(raw["extraction"])
        entities = [e.model_dump(mode="json") for e in result.entities]
        kept, dropped = split_by_evidence(result, chunk["text"])
        dropped = [{"edge": e, "reason": "direction: COMPETES_WITH points into the filer"}
                   for e in raw.get("direction_dropped", [])] + dropped
        ref = ChunkRef(chunk_id=chunk["chunk_id"], cik=chunk["cik"], accession=chunk["accession"],
                       fiscal_year=chunk["fiscal_year"], section_path=chunk["section_path"], text=chunk["text"])
        stored = [s.model_dump(mode="json") for s in
                  finalize(LLMExtraction.model_construct(entities=result.entities, relationships=kept), ref)]

    return {
        "chunk_id": chunk["chunk_id"],
        "cik": chunk["cik"],
        "ticker": chunk["ticker"],
        "accession": chunk["accession"],
        "fiscal_year": chunk["fiscal_year"],
        "section_path": chunk["section_path"],
        "model": args.model,
        "prompt_version": args.prompt_version,
        "cache_key": key,
        "cached": cached,
        "first_try_valid": raw["first_try_valid"],
        "direction_retries": raw.get("direction_retries", 0),
        "valid": raw["valid"],
        "attempts": raw["attempts"],
        "entities": entities,
        "relationships": stored,
        "dropped_edges": dropped,
        "usage": raw["usage"],
    }


# -- inputs --------------------------------------------------------------------

def load_chunks(pattern: str) -> dict[str, dict]:
    paths = sorted(glob.glob(pattern)) or ([pattern] if Path(pattern).exists() else [])
    if not paths:
        raise SystemExit(f"No chunk files match {pattern!r}")
    chunks = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                c = json.loads(line)
                chunks[c["chunk_id"]] = c
    return chunks


def load_sample(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def load_filers() -> dict[str, dict]:
    df = pd.read_csv(COMPANIES_CSV, dtype=str, keep_default_na=False)
    return {str(int(r.cik)).zfill(10): {"cik": str(int(r.cik)).zfill(10), "ticker": r.ticker, "name": r.name}
            for r in df.itertuples()}


# -- summary -------------------------------------------------------------------

def cost(usage: dict, price_in: float, price_out: float) -> float:
    return (usage.get("input_tokens", 0) * price_in
            + usage.get("cache_creation_input_tokens", 0) * price_in * CACHE_WRITE_MULT
            + usage.get("cache_read_input_tokens", 0) * price_in * CACHE_READ_MULT
            + usage.get("output_tokens", 0) * price_out) / 1e6


def summarize(records: list[dict], price_in: float, price_out: float) -> str:
    n = len(records)
    if not n:
        return "No chunks processed."
    first = sum(r["first_try_valid"] for r in records)
    valid = sum(r["valid"] for r in records)
    kept = sum(len(r["relationships"]) for r in records)
    dropped = sum(1 for r in records for d in r["dropped_edges"] if not d["reason"].startswith("direction"))
    dir_dropped = sum(1 for r in records for d in r["dropped_edges"] if d["reason"].startswith("direction"))
    dir_retried = sum(1 for r in records if r.get("direction_retries", 0) > 0)
    rel_counts = Counter(e["rel_type"] for r in records for e in r["relationships"])
    risk_counts = Counter(e["target"] for r in records for e in r["relationships"]
                          if e["rel_type"] == RelType.EXPOSED_TO_RISK.value)
    total_usage, run_usage = Counter(), Counter()
    for r in records:
        total_usage.update(r["usage"])
        if not r["cached"]:
            run_usage.update(r["usage"])
    cached = sum(r["cached"] for r in records)

    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"

    lines = [
        f"Chunks: {n}  (from cache: {cached}, API calls this run: {n - cached})",
        f"Schema pass, first try:     {pct(first, n)}",
        f"Schema pass, after retries: {pct(valid, n)}",
        f"Evidence pass (edges kept): {pct(kept, kept + dropped)}",
        f"Direction check: {dir_retried} chunks had reversed COMPETES_WITH edges and were retried; "
        f"{dir_dropped} edges dropped after retries",
        f"Edges per valid chunk:      {kept / valid:.1f}" if valid else "Edges per valid chunk: n/a",
        "",
        "Relationships kept by type:",
        *[f"  {k:<18} {v}" for k, v in rel_counts.most_common()],
        "",
        "EXPOSED_TO_RISK by category:",
        *([f"  {k:<28} {v}" for k, v in risk_counts.most_common()] or ["  (none)"]),
        "",
        "Tokens (all chunks, incl. cached results' original cost):",
        f"  input {total_usage['input_tokens']:,}  cache-write {total_usage['cache_creation_input_tokens']:,}"
        f"  cache-read {total_usage['cache_read_input_tokens']:,}  output {total_usage['output_tokens']:,}",
        f"Cost @ ${price_in:.2f} in / ${price_out:.2f} out per MTok "
        f"(cache write x{CACHE_WRITE_MULT}, read x{CACHE_READ_MULT} of input price):",
        f"  this run ${cost(run_usage, price_in, price_out):.4f}   all records ${cost(total_usage, price_in, price_out):.4f}"
        f"   per chunk ${cost(total_usage, price_in, price_out) / n:.4f}",
        "  (output tokens include thinking tokens, which are billed as output)",
    ]
    return "\n".join(lines)


# -- main ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks", required=True, help="chunk .jsonl path or glob")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--sample-file", type=Path, help="text file of chunk_ids to run (# comments allowed)")
    p.add_argument("--prompt-version", required=True, help="bump when the prompt or ontology changes; part of the cache key")
    p.add_argument("--input-price", type=float, required=True, help="USD per million input tokens")
    p.add_argument("--output-price", type=float, required=True, help="USD per million output tokens")
    p.add_argument("--effort", default="medium", help="output_config.effort; 'none' to omit")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-fallback", action="store_true", help="disable the server-side refusal fallback")
    args = p.parse_args()
    args.effort = None if args.effort == "none" else args.effort
    if args.prompt_version not in PROMPT_ADDENDA and args.prompt_version != "v1":
        log.warning("prompt version %r has no entry in PROMPT_ADDENDA; using the base ontology prompt "
                    "(known versions: v1, %s)", args.prompt_version, ", ".join(PROMPT_ADDENDA))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    load_dotenv()
    chunks = load_chunks(args.chunks)
    ids = load_sample(args.sample_file) if args.sample_file else sorted(chunks)
    missing = [i for i in ids if i not in chunks]
    if missing:
        raise SystemExit(f"{len(missing)} sample ids not found in {args.chunks}: {missing[:5]}")
    filers = load_filers()

    client = None
    need_api = any(load_cached(chunks[i], filers[chunks[i]["cik"]], args.prompt_version, args.model) is None
                   for i in ids)
    if need_api:
        # preflight (no tokens billed): catches missing credentials and bad model ids
        # before any worker thread starts
        try:
            client = anthropic.Anthropic()
            client.models.retrieve(args.model)
        except (TypeError, anthropic.AuthenticationError) as exc:
            raise SystemExit(f"No usable Anthropic credentials ({exc}).\n"
                             "Add ANTHROPIC_API_KEY=... to .env (gitignored) or export it in your shell.")
        except anthropic.NotFoundError:
            raise SystemExit(f"Model {args.model!r} not found for this account.")

    # v1 keeps the plain {model} folder; other prompt versions get {model}__{version}
    # so a new prompt never overwrites an earlier version's results
    out_dir = OUT_ROOT / (args.model if args.prompt_version == "v1" else f"{args.model}__{args.prompt_version}")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir / "failures.jsonl"
    run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    records, failures = [], []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_chunk, client, chunks[i], filers[chunks[i]["cik"]], args): i for i in ids}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                rec = fut.result()
            except anthropic.APIStatusError as exc:
                log.error("%s: API error %s: %s", cid, exc.status_code, exc.message)
                failures.append({"chunk_id": cid, "kind": "api_error", "detail": f"{exc.status_code} {exc.message}"})
                continue
            except anthropic.APIConnectionError as exc:
                log.error("%s: connection error: %s", cid, exc)
                failures.append({"chunk_id": cid, "kind": "api_error", "detail": str(exc)})
                continue
            records.append(rec)
            (out_dir / f"{cid}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            if not rec["valid"]:
                failures.append({"chunk_id": cid, "kind": "schema", "attempts": rec["attempts"]})
            for d in rec["dropped_edges"]:
                kind = "direction" if d["reason"].startswith("direction") else "evidence"
                log.warning("%s: dropped edge (%s): %s", cid, kind, d["reason"])
                failures.append({"chunk_id": cid, "kind": kind, **d})
            log.info("%s %-4s %-18s valid=%s first_try=%s edges=%d dropped=%d%s", cid, rec["ticker"],
                     rec["section_path"], rec["valid"], rec["first_try_valid"], len(rec["relationships"]),
                     len(rec["dropped_edges"]), " (cached)" if rec["cached"] else "")

    with failures_path.open("a") as f:
        for item in failures:
            f.write(json.dumps({"run_at": run_at, "prompt_version": args.prompt_version, **item}) + "\n")

    print()
    print(summarize(records, args.input_price, args.output_price))
    api_errors = sum(1 for f in failures if f["kind"] == "api_error")
    if api_errors:
        print(f"\n{api_errors} chunks failed with API errors (not cached; re-run to retry).")
    print(f"\nOutputs: {out_dir}   failures: {failures_path}   wall time {time.time() - started:.0f}s")
    if api_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
