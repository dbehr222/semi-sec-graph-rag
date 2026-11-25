"""Score extractions against a hand-labeled gold file: precision and recall on edges.

Gold file (JSONL), one line per labeled chunk:
  {"chunk_id": "b5c4f5ca62b4d3ea",
   "edges": [
     {"source": "NVIDIA Corp", "rel_type": "COMPETES_WITH", "target": "Intel"},
     {"source": ["TSMC", "Taiwan Semiconductor Manufacturing Company"],
      "rel_type": "SUPPLIER_OF", "target": "NVIDIA Corp"}
   ]}
A source/target may be a list of acceptable names. Extra keys (text, notes) are ignored.
A labeled chunk with "edges": [] means "nothing should be extracted here".

Matching: (source, rel_type, target) after ontology.normalize_name. A predicted
entity also matches through its mentioned_as aliases, so gold can say "TSMC"
even when the model's canonical_name is the full legal name. Each gold edge
matches at most one predicted edge.

Only chunks present in the gold file are scored. Precision = matched predicted /
all predicted; recall = matched gold / all gold (micro-averaged).

Run:
  python -m extract.eval_gold --gold data/gold/pilot_gold.jsonl --model claude-sonnet-5-5
  python -m extract.eval_gold --write-template data/gold/pilot_gold.template.jsonl \\
      --sample-file data/pilot_sample.txt --chunks 'data/processed/chunks/*.jsonl'
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import re

from extract.ontology import normalize_name as _ontology_normalize

ROOT = Path(__file__).resolve().parent.parent
PRED_ROOT = ROOT / "data" / "processed" / "extractions"


def normalize_name(name: str) -> str:
    """ontology.normalize_name plus removal of SEC state tags like '/DEL/' or '/DE',
    so 'DIODES INC /DEL/' matches 'Diodes Incorporated'. (ontology.py itself is untouched;
    entity resolution will need the same cleanup.)"""
    return _ontology_normalize(re.sub(r"/[A-Za-z]{2,4}/?", " ", name))


def _names(value) -> set[str]:
    values = value if isinstance(value, list) else [value]
    return {normalize_name(v) for v in values if v}


def load_gold(path: Path) -> dict[str, list[dict]]:
    gold = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        edges = []
        for e in row.get("edges", []):
            missing = {"source", "rel_type", "target"} - set(e)
            if missing:
                raise SystemExit(f"{path}:{n}: edge missing {sorted(missing)}")
            edges.append({"source": _names(e["source"]), "rel_type": e["rel_type"],
                          "target": _names(e["target"]), "raw": e})
        gold[row["chunk_id"]] = edges
    return gold


def predicted_edges(record: dict) -> list[dict]:
    aliases: dict[str, set[str]] = {}
    for ent in record.get("entities", []):
        names = {normalize_name(ent["canonical_name"])} | {normalize_name(m) for m in ent.get("mentioned_as", [])}
        aliases[normalize_name(ent["canonical_name"])] = names
    out = []
    for r in record.get("relationships", []):
        s, t = normalize_name(r["source"]), normalize_name(r["target"])
        out.append({"source": aliases.get(s, {s}), "rel_type": r["rel_type"],
                    "target": aliases.get(t, {t}), "raw": r})
    return out


def _match(pred: dict, gold: dict) -> bool:
    return (pred["rel_type"] == gold["rel_type"]
            and bool(pred["source"] & gold["source"])
            and bool(pred["target"] & gold["target"]))


def score(gold: dict[str, list[dict]], pred_dir: Path, verbose: bool) -> None:
    tp_pred = n_pred = tp_gold = n_gold = 0
    by_type = Counter()  # (rel_type, "tp"/"pred"/"gold")
    missing = []
    for chunk_id, gold_edges in sorted(gold.items()):
        path = pred_dir / f"{chunk_id}.json"
        if not path.exists():
            missing.append(chunk_id)
            continue
        preds = predicted_edges(json.loads(path.read_text()))
        used = set()
        matched_pred = set()
        for gi, g in enumerate(gold_edges):
            for pi, p in enumerate(preds):
                if pi not in matched_pred and _match(p, g):
                    matched_pred.add(pi)
                    used.add(gi)
                    break
        n_pred += len(preds)
        n_gold += len(gold_edges)
        tp_pred += len(matched_pred)
        tp_gold += len(used)
        for p in preds:
            by_type[(p["rel_type"], "pred")] += 1
        for g in gold_edges:
            by_type[(g["rel_type"], "gold")] += 1
        for gi in used:
            by_type[(gold_edges[gi]["rel_type"], "tp")] += 1
        if verbose:
            for pi, p in enumerate(preds):
                if pi not in matched_pred:
                    r = p["raw"]
                    print(f"  FP {chunk_id}: {r['source']} -{r['rel_type']}-> {r['target']}")
            for gi, g in enumerate(gold_edges):
                if gi not in used:
                    r = g["raw"]
                    print(f"  FN {chunk_id}: {r['source']} -{r['rel_type']}-> {r['target']}")

    scored = len(gold) - len(missing)
    precision = tp_pred / n_pred if n_pred else float("nan")
    recall = tp_gold / n_gold if n_gold else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    print(f"Chunks scored: {scored} of {len(gold)} gold chunks" + (f"  (no prediction for {len(missing)})" if missing else ""))
    print(f"Precision: {precision:.3f}  ({tp_pred}/{n_pred} predicted edges correct)")
    print(f"Recall:    {recall:.3f}  ({tp_gold}/{n_gold} gold edges found)")
    print(f"F1:        {f1:.3f}")
    types = sorted({k[0] for k in by_type})
    if types:
        print(f"\n{'rel_type':<18}{'gold':>6}{'pred':>6}{'match':>7}{'P':>7}{'R':>7}")
        for t in types:
            g, pr, tp = by_type[(t, "gold")], by_type[(t, "pred")], by_type[(t, "tp")]
            p_ = f"{tp / pr:.2f}" if pr else "-"
            r_ = f"{tp / g:.2f}" if g else "-"
            print(f"{t:<18}{g:>6}{pr:>6}{tp:>7}{p_:>7}{r_:>7}")
    if missing:
        print("\nNo predictions for: " + ", ".join(missing))


def write_template(out: Path, sample_file: Path, chunks_glob: str) -> None:
    from extract.run_extraction import load_chunks, load_sample

    chunks = load_chunks(chunks_glob)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for cid in load_sample(sample_file):
            c = chunks[cid]
            f.write(json.dumps({"chunk_id": cid, "ticker": c["ticker"], "section_path": c["section_path"],
                                "text": c["text"], "edges": []}, ensure_ascii=False) + "\n")
    print(f"Wrote label template for {len(load_sample(sample_file))} chunks to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path)
    p.add_argument("--model", help="read predictions from data/processed/extractions/{model}/")
    p.add_argument("--pred-dir", type=Path, help="or point at a predictions directory directly")
    p.add_argument("-v", "--verbose", action="store_true", help="list false positives / negatives")
    p.add_argument("--write-template", type=Path, help="write a blank gold file to label, then exit")
    p.add_argument("--sample-file", type=Path)
    p.add_argument("--chunks")
    args = p.parse_args()

    if args.write_template:
        if not (args.sample_file and args.chunks):
            p.error("--write-template needs --sample-file and --chunks")
        write_template(args.write_template, args.sample_file, args.chunks)
        return
    if not args.gold or not (args.model or args.pred_dir):
        p.error("--gold and one of --model / --pred-dir are required")
    score(load_gold(args.gold), args.pred_dir or PRED_ROOT / args.model, args.verbose)


if __name__ == "__main__":
    main()
