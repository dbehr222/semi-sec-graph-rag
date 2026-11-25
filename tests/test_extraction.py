import json
from types import SimpleNamespace as NS

import pytest

from extract import eval_gold, run_extraction as rx

CHUNK = {
    "chunk_id": "abc123abc123abc1",
    "cik": "0001045810",
    "ticker": "NVDA",
    "accession": "0001045810-26-000021",
    "fiscal_year": 2026,
    "section_path": "10-K/2026/Item1",
    "text": "We utilize foundries, such as Taiwan Semiconductor Manufacturing Company Limited, "
            "or TSMC, to produce our semiconductor wafers. Our competitors include Intel.",
}
FILER = {"cik": "0001045810", "ticker": "NVDA", "name": "NVIDIA CORP"}

GOOD = {
    "entities": [
        {"type": "COMPANY", "canonical_name": "NVIDIA CORP", "mentioned_as": ["we"]},
        {"type": "COMPANY", "canonical_name": "Taiwan Semiconductor Manufacturing Company Limited",
         "mentioned_as": ["TSMC"]},
        {"type": "COMPANY", "canonical_name": "Intel Corporation", "mentioned_as": ["Intel"]},
    ],
    "relationships": [
        {"source": "Taiwan Semiconductor Manufacturing Company Limited", "source_type": "COMPANY",
         "rel_type": "SUPPLIER_OF", "target": "NVIDIA CORP", "target_type": "COMPANY",
         "evidence": "We utilize foundries, such as Taiwan Semiconductor Manufacturing Company Limited",
         "confidence": 0.95, "supply_type": "FOUNDRY"},
        {"source": "NVIDIA CORP", "source_type": "COMPANY", "rel_type": "COMPETES_WITH",
         "target": "Intel Corporation", "target_type": "COMPANY",
         "evidence": "Our competitors include Intel and Qualcomm.",  # not verbatim -> dropped
         "confidence": 0.9},
    ],
}
BAD = {**GOOD, "relationships": [{**GOOD["relationships"][0], "supply_type": None}]}  # SUPPLIER_OF needs supply_type


def _resp(payload=None, stop="tool_use", text_only=False):
    content = [NS(type="thinking", thinking="")]
    if text_only:
        content.append(NS(type="text", text="Here is the extraction..."))
    else:
        content.append(NS(type="tool_use", name=rx.TOOL_NAME, id="toolu_1", input=payload))
    return NS(stop_reason=stop, content=content, model="claude-opus-5-5", stop_details=None,
              usage=NS(input_tokens=100, output_tokens=50, cache_creation_input_tokens=0,
                       cache_read_input_tokens=2000))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=lambda o: "<obj>")))
        return self.responses.pop(0)


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(rx, "CACHE_DIR", tmp_path / "cache")
    return NS(model="claude-opus-5-5", prompt_version="test", effort="medium", max_retries=2, no_fallback=False)


def test_retries_with_validation_error_then_succeeds(args):
    client = FakeClient([_resp(BAD), _resp(GOOD)])
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert rec["valid"] and not rec["first_try_valid"]
    assert len(client.calls) == 2
    retry_msg = client.calls[1]["messages"][-1]["content"][0]
    assert retry_msg["type"] == "tool_result" and retry_msg["is_error"] is True
    assert "supply_type" in retry_msg["content"]
    assert rec["usage"]["input_tokens"] == 200


def test_gives_up_after_max_retries(args):
    client = FakeClient([_resp(BAD)] * 3)
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert not rec["valid"] and len(rec["attempts"]) == 3 and len(client.calls) == 3


def test_missing_tool_call_is_retried(args):
    client = FakeClient([_resp(stop="end_turn", text_only=True), _resp(GOOD)])
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert rec["valid"] and not rec["first_try_valid"]
    assert "Call the record_extraction tool" in client.calls[1]["messages"][-1]["content"]


def test_evidence_failures_are_dropped_and_provenance_added(args):
    rec = rx.process_chunk(FakeClient([_resp(GOOD)]), CHUNK, FILER, args)
    assert [r["rel_type"] for r in rec["relationships"]] == ["SUPPLIER_OF"]
    assert len(rec["dropped_edges"]) == 1 and rec["dropped_edges"][0]["edge"]["rel_type"] == "COMPETES_WITH"
    kept = rec["relationships"][0]
    assert kept["source_chunk_id"] == CHUNK["chunk_id"] and kept["fiscal_year"] == 2026


def test_cache_hit_makes_no_api_call(args):
    rx.process_chunk(FakeClient([_resp(GOOD)]), CHUNK, FILER, args)
    client = FakeClient([])
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert rec["cached"] and client.calls == [] and rec["valid"]
    # a different prompt version is a different cache entry
    args.prompt_version = "test2"
    client = FakeClient([_resp(GOOD)])
    assert not rx.process_chunk(client, CHUNK, FILER, args)["cached"]


def test_request_shape_per_model():
    opus55 = rx.request_kwargs("claude-opus-5-5", "medium", True)
    assert opus55["tool_choice"] == {"type": "auto"}  # forced tool_choice 400s on this model
    assert opus55["fallbacks"] == "default" and opus55["betas"] == [rx.FALLBACK_BETA]
    old = rx.request_kwargs("claude-opus-5", "medium", False)
    assert old["tool_choice"] == {"type": "tool", "name": rx.TOOL_NAME} and "fallbacks" not in old
    schema = opus55["tools"][0]["input_schema"]
    assert schema == rx.LLMExtraction.model_json_schema()
    sys_blocks = rx.system_blocks(FILER)
    assert "cache_control" in sys_blocks[0] and "NVIDIA CORP" in sys_blocks[1]["text"]
    other = rx.system_blocks({"cik": "0000029002", "ticker": "DIOD", "name": "DIODES INC /DEL/"})
    assert other[0] == sys_blocks[0]  # cached block identical for every filer
    assert other[1] != sys_blocks[1]


def test_eval_precision_recall(tmp_path):
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    rec = {
        "entities": GOOD["entities"],
        "relationships": [GOOD["relationships"][0],
                          {**GOOD["relationships"][1], "evidence": "Our competitors include Intel."}],
    }
    (pred_dir / f"{CHUNK['chunk_id']}.json").write_text(json.dumps(rec))
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps({"chunk_id": CHUNK["chunk_id"], "edges": [
        {"source": "TSMC", "rel_type": "SUPPLIER_OF", "target": "NVIDIA Corporation"},  # alias + suffix
        {"source": "NVIDIA", "rel_type": "COMPETES_WITH", "target": "AMD"},              # missed
    ]}) + "\n")
    gold = eval_gold.load_gold(gold_path)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eval_gold.score(gold, pred_dir, verbose=False)
    out = buf.getvalue()
    assert "Precision: 0.500" in out and "Recall:    0.500" in out


def test_prompt_addendum_is_versioned_and_changes_cache_key():
    v1 = rx.system_blocks(FILER, "v1")[0]["text"]
    v2 = rx.system_blocks(FILER, "v2")[0]["text"]
    assert v2.startswith(v1) and "country or a broad region" in v2 and v1 != v2
    assert rx.system_blocks(FILER)[0]["text"] == v1  # default is the base prompt
    assert rx.cache_key("t", "v1", "m") != rx.cache_key("t", "v2", "m")


# -- direction check -----------------------------------------------------------

def _competes(source, target):
    return {"source": source, "source_type": "COMPANY", "rel_type": "COMPETES_WITH", "target": target,
            "target_type": "COMPANY", "evidence": "Our competitors include Intel.", "confidence": 0.9}


FLIPPED = {**GOOD, "relationships": [GOOD["relationships"][0], _competes("Intel Corporation", "NVIDIA CORP")]}
FIXED = {**GOOD, "relationships": [GOOD["relationships"][0], _competes("NVIDIA CORP", "Intel Corporation")]}


def test_filer_key_ignores_sec_state_tags_and_suffixes():
    assert rx.filer_key("DIODES INC /DEL/") == rx.filer_key("Diodes Incorporated")
    assert rx.filer_key("APPLIED MATERIALS INC /DE") == rx.filer_key("Applied Materials, Inc.")
    assert rx.filer_key("NVIDIA CORP") == rx.filer_key("NVIDIA Corporation")


def test_reversed_competes_detects_only_edges_into_the_filer():
    ext = rx.LLMExtraction.model_validate
    assert len(rx.reversed_competes(ext(FLIPPED), FILER)) == 1
    assert rx.reversed_competes(ext(FIXED), FILER) == []


def test_reversed_competes_is_retried_with_the_fix_spelled_out(args):
    client = FakeClient([_resp(FLIPPED), _resp(FIXED)])
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert rec["valid"] and rec["first_try_valid"]  # schema was fine; direction is a separate check
    assert rec["direction_retries"] == 1 and len(client.calls) == 2
    retry = client.calls[1]["messages"][-1]["content"][0]
    assert retry["is_error"] is True
    assert "NVIDIA CORP COMPETES_WITH Intel Corporation" in retry["content"]
    assert [(r["source"], r["target"]) for r in rec["relationships"] if r["rel_type"] == "COMPETES_WITH"] \
        == [("NVIDIA CORP", "Intel Corporation")]


def test_reversed_edges_are_dropped_when_retries_run_out(args):
    client = FakeClient([_resp(FLIPPED)] * 3)
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert len(client.calls) == 3 and rec["valid"]
    assert [r["rel_type"] for r in rec["relationships"]] == ["SUPPLIER_OF"]  # rest of the chunk survives
    assert [d["reason"] for d in rec["dropped_edges"]] == ["direction: COMPETES_WITH points into the filer"]


def _write_cache(args, payload, **extra):
    key = rx.cache_key(CHUNK["text"], args.prompt_version, args.model)
    rx.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = {"model": args.model, "attempts": [{"ok": True}], "first_try_valid": True, "valid": True,
           "extraction": rx.LLMExtraction.model_validate(payload).model_dump(mode="json"),
           "usage": {"input_tokens": 1, "output_tokens": 1}, **extra}
    (rx.CACHE_DIR / f"{key}.json").write_text(json.dumps(raw))


def test_stale_cache_entry_with_reversed_edge_is_recalled(args):
    _write_cache(args, FLIPPED)  # written before the check existed: no direction_checked flag
    client = FakeClient([_resp(FIXED)])
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert len(client.calls) == 1 and not rec["cached"]
    assert [(r["source"], r["target"]) for r in rec["relationships"] if r["rel_type"] == "COMPETES_WITH"] \
        == [("NVIDIA CORP", "Intel Corporation")]
    # and the corrected result replaced the stale cache entry
    again = rx.process_chunk(FakeClient([]), CHUNK, FILER, args)
    assert again["cached"]


def test_clean_legacy_cache_entry_is_still_used(args):
    _write_cache(args, FIXED)  # no flag, but nothing reversed: the check would have accepted it
    client = FakeClient([])
    rec = rx.process_chunk(client, CHUNK, FILER, args)
    assert rec["cached"] and client.calls == []
