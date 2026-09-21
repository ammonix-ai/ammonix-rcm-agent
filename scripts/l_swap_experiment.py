"""Swap the language model in the L seat and measure what moves.

The decision path makes no language-model call. L enters only when an
execute Skill fires: it fills the paperwork from the case record, and the
agent's scrubber accepts the form or sends it back with the reason, up to
twice; a third failure hands the claim to a person.

This script replays the 200 demonstration claims with one writer in the L
seat (--model qwen | ornith | ornith_ft | opus5), records every trace, and
(--compare) diffs the writers state by state.

Usage:
  .venv/Scripts/python.exe scripts/l_swap_experiment.py --model qwen
  .venv/Scripts/python.exe scripts/l_swap_experiment.py --compare
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

REPORTS = ROOT / "runs" / "reports"
OPUS_MODEL = "claude-opus-5"
_OFFLINE = ("the local model is not running and this request is not in the "
            "cache; start the vLLM container to compute it")


def _model_online() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2):
            return True
    except OSError:
        return False


class _OfflineText:
    model_id = "cache-only"

    def generate(self, prompt: str) -> str:
        raise RuntimeError(_OFFLINE)


def _cache_only_m1():
    from ammonix_core.hashing import sha256_text
    from cardessa.harness import M1Provider

    class CacheOnlyM1(M1Provider):
        def __post_init__(self) -> None:
            self.model_id = "cache-only"

        def generate(self, prompt: str, schema: dict) -> str:
            key = sha256_text(prompt + json.dumps(schema, sort_keys=True))
            cached = self.cache_dir / f"{key}.json"
            if cached.is_file():
                return cached.read_text(encoding="utf-8")
            raise RuntimeError(_OFFLINE)

    return CacheOnlyM1(cache_dir=ROOT / "data" / "m1_cache")


_UNSUPPORTED_DECODE_KEYS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minItems", "maxItems", "uniqueItems", "minLength", "maxLength",
    "pattern", "format", "minProperties", "maxProperties",
}


def _decode_schema(schema):
    """Copy of the schema without numeric-bound keywords, which Anthropic's
    structured-output decoder rejects. The verifiable check validates the
    produced payload against the FULL schema afterwards, so the constraint
    is enforced, only later."""
    if isinstance(schema, dict):
        return {k: _decode_schema(v) for k, v in schema.items()
                if k not in _UNSUPPORTED_DECODE_KEYS}
    if isinstance(schema, list):
        return [_decode_schema(v) for v in schema]
    return schema


class OpusNoThink:
    """Anthropic client, thinking disabled, schema-constrained output, own
    cache (key = model + tag + messages + schema). Same contract as
    ui.opus_llm.OpusAgentLLM.complete."""

    CONFIG_TAG = "nothink-v1"

    def __init__(self, cache_dir: Path, max_tokens: int = 4000) -> None:
        import threading

        self.cache_dir = cache_dir
        self.max_tokens = max_tokens
        self.model_id = OPUS_MODEL
        self.calls_to_provider = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()
        self._client = None  # created on the first cache miss (keyless replay works)

    def _anthropic(self):
        if self._client is None:
            import anthropic
            from ui.opus_llm import _load_api_key
            self._client = anthropic.Anthropic(
                api_key=_load_api_key(self.cache_dir.parent))
        return self._client

    def complete(self, messages: list[dict], schema: dict) -> str:
        import hashlib

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256((
            self.model_id + self.CONFIG_TAG + json.dumps(messages, sort_keys=True)
            + json.dumps(schema, sort_keys=True)
        ).encode("utf-8")).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.is_file():
            entry = json.loads(cached.read_text(encoding="utf-8"))
            with self._lock:
                self.cache_hits += 1
                self.prompt_tokens += entry["usage"]["prompt_tokens"]
                self.completion_tokens += entry["usage"]["completion_tokens"]
            return entry["content"]
        response = self._anthropic().messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            messages=messages,
            thinking={"type": "disabled"},
            output_config={"format": {"type": "json_schema",
                                      "schema": _decode_schema(schema)}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        content = next(b.text for b in response.content if b.type == "text").strip()
        usage = {"prompt_tokens": response.usage.input_tokens,
                 "completion_tokens": response.usage.output_tokens}
        cached.write_text(json.dumps({"content": content, "usage": usage}),
                          encoding="utf-8")
        with self._lock:
            self.calls_to_provider += 1
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]
        return content

    def stats(self) -> dict:
        return {"model_id": self.model_id, "thinking": "disabled",
                "calls_to_provider": self.calls_to_provider,
                "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


ORNITH_MODEL = "ornith-ai/Ornith-1.5-9B"
# Hub revision the measurements used (the repo's main moved on 2026-08-23 to a
# re-uploaded shard 4 that decodes to noise; vLLM is pinned with --revision)
ORNITH_REVISION = "c927ad73b7eb20f00aafcaa0a11a9d58ed5487bc"
ORNITH_URL = "http://127.0.0.1:8002/v1"
# the same model after one round of rejection-sampling fine-tuning on the
# agent's own success signal (scripts/l_finetune_ornith.py), served merged
ORNITH_FT_MODEL = "local/Ornith-1.5-9B-rcm-r2"


class LocalJsonLLM:
    """OpenAI-compatible vLLM client for an open-weights model: temperature
    0, seed 0, JSON-schema-constrained decoding, thinking disabled via the
    chat template where supported and any <think> block stripped. Same
    complete(messages, schema) contract and cache shape as OpusNoThink."""

    CONFIG_TAG = "nothink-v1"

    def __init__(self, cache_dir: Path, model_id: str = ORNITH_MODEL,
                 base_url: str = ORNITH_URL, max_tokens: int = 1500) -> None:
        import threading
        import urllib.request

        self.cache_dir = cache_dir
        self.model_id = model_id
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.calls_to_provider = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()
        self._served_id = None  # resolved on the first cache miss (keyless replay works)

    @property
    def served_id(self) -> str:
        if self._served_id is None:
            import urllib.request
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as r:
                    served = [m["id"] for m in json.loads(r.read())["data"]]
            except OSError as e:
                raise RuntimeError(_OFFLINE) from e
            tail = self.model_id.split("/")[-1]
            # exact match: "Ornith-1.5-9B" must not bind to "Ornith-1.5-9B-rcm-r1"
            if tail not in served:
                raise RuntimeError(f"{self.base_url} serves {served}, not {self.model_id}")
            self._served_id = tail
        return self._served_id

    def complete(self, messages: list[dict], schema: dict) -> str:
        import hashlib
        import urllib.request

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256((
            self.model_id + self.CONFIG_TAG + json.dumps(messages, sort_keys=True)
            + json.dumps(schema, sort_keys=True)
        ).encode("utf-8")).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.is_file():
            entry = json.loads(cached.read_text(encoding="utf-8"))
            with self._lock:
                self.cache_hits += 1
                self.prompt_tokens += entry["usage"]["prompt_tokens"]
                self.completion_tokens += entry["usage"]["completion_tokens"]
            return entry["content"]
        payload = json.dumps({
            "model": self.served_id, "messages": messages,
            "temperature": 0.0, "seed": 0, "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "payload", "schema": schema}},
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
        body = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    body = json.loads(r.read())
                break
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
        msg = body["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1].strip()
        usage = {"prompt_tokens": body["usage"]["prompt_tokens"],
                 "completion_tokens": body["usage"]["completion_tokens"]}
        cached.write_text(json.dumps({"content": content, "usage": usage}),
                          encoding="utf-8")
        with self._lock:
            self.calls_to_provider += 1
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]
        return content

    def stats(self) -> dict:
        return {"model_id": self.model_id, "served_id": self._served_id or "cache-only",
                "thinking": "disabled (template kwarg; <think> stripped)",
                "calls_to_provider": self.calls_to_provider,
                "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


def build_demo():
    import numpy as np
    import polars as pl
    from ammonix_core.schema import HarnessArtefacts, Skill
    from cardessa import MASTER_SEED
    from cardessa.features_kept import build_kept_features
    from cardessa.harness import BasisRuntime
    from cardessa.livecases import generate_demo_cases
    from cardessa.textgen import CachedTextGenerator, VllmProvider

    rt = BasisRuntime.load(ROOT)
    harness = HarnessArtefacts.model_validate_json(
        (ROOT / "runs" / "manifests" / "harness.json").read_text(encoding="utf-8")
    )
    skills = [
        Skill.model_validate(s)
        for s in json.loads(
            (ROOT / "runs" / "manifests" / "skills.json").read_text(encoding="utf-8")
        )["skills"]
    ]
    text_provider = VllmProvider() if _model_online() else _OfflineText()
    text = CachedTextGenerator(provider=text_provider,
                               cache_dir=ROOT / "data" / "text_cache")
    _episodes, states, _world, _engine = generate_demo_cases(ROOT, MASTER_SEED, text)
    working = pl.read_parquet(ROOT / "data" / "working" / "cardessa_sim" / "states.parquet")
    frame, _, _, _ = build_kept_features(
        ROOT, pl.concat([working, states.select(working.columns)], how="vertical_relaxed")
    )
    names = rt.feature_names
    ids = sorted(states["state_id"].to_list())
    rows = {r["state_id"]: r for r in states.to_dicts()}
    feats = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame.filter(pl.col("state_id").is_in(ids)).to_dicts()
    }
    scaled = rt.scaler.transform(np.array([[feats[s][n] for n in names] for s in ids]))
    scaled_of = dict(zip(ids, scaled, strict=True))
    return rt, harness, skills, ids, rows, feats, scaled_of


WRITERS = {
    "qwen": None,                 # the pinned 27B through M1Provider
    "ornith": ORNITH_MODEL,       # local 9B, base weights
    "ornith_ft": ORNITH_FT_MODEL,  # local 9B after fine-tuning on the agent's reward
    "opus5": OPUS_MODEL,          # frontier model through the API
}


def make_provider(model: str):
    """The writer in the L seat. The check is the agent's own deterministic
    rule set whichever model writes. Returns (provider, stats callable)."""
    from cardessa.harness import M1Provider

    if model == "qwen":
        provider = M1Provider(cache_dir=ROOT / "data" / "m1_cache") if _model_online()             else _cache_only_m1()
        return provider, lambda: {"model_id": provider.model_id,
                                  "calls_to_provider": provider.calls_to_provider}
    mid = WRITERS[model]
    if model == "opus5":
        llm = OpusNoThink(cache_dir=ROOT / "data" / "m1_opus5")
    else:
        llm = LocalJsonLLM(cache_dir=ROOT / "data" / f"m1_{model}", model_id=mid)

    class Writer:
        model_id = mid

        def generate(self, prompt: str, schema: dict) -> str:
            return llm.complete([{"role": "user", "content": prompt}], schema)

    return Writer(), lambda: {"m1": llm.stats()}


def _payer_facts(row: dict) -> dict:
    return {
        "cpt": row["cpt"],
        "dx_codes": [d for d in (row.get("dx_codes") or "").split(",") if d],
        "carc": row.get("carc") or None,
        "patient_share": None,
        "balance": row.get("balance"),
        "clinical_indication_text": row.get("clinical_indication_text") or "",
        "payer_correspondence_text": row.get("payer_correspondence_text") or "",
    }


def run(model: str, workers: int) -> None:
    from run_harness_eval import resolve_and_run
    from cardessa.payer_reader import PayerReader, payer_receives

    rt, harness, skills, ids, rows, feats, scaled_of = build_demo()
    provider, stats = make_provider(model)
    reader = PayerReader(cache_dir=ROOT / "data" / "payer_reader")
    print(f"L = {model}: {len(ids)} demonstration states, {workers} workers")

    def one(state_id):
        t0 = time.perf_counter()
        try:
            tr = resolve_and_run(rt, skills, harness, provider, rows[state_id],
                                 feats[state_id], scaled_of[state_id])
            err = None
        except Exception as exc:  # noqa: BLE001 - record, keep going
            tr, err = None, repr(exc)
        dt = time.perf_counter() - t0
        if tr is None:
            return state_id, {"error": err, "seconds": round(dt, 2)}
        its = tr.iterations
        payer_reason = None
        if tr.status == "executed" and tr.final is not None:
            try:
                payer_reason = payer_receives(tr.retrieval.recommended_action_id,
                                              tr.final.payload, _payer_facts(rows[state_id]),
                                              reader)
            except Exception as exc:  # noqa: BLE001
                payer_reason = f"reviewer unavailable: {exc!r}"
        return state_id, {
            "status": tr.status,
            "escalation_reason": tr.escalation.reason if tr.escalation else None,
            "recommended_action": tr.retrieval.recommended_action_id,
            "skill_id": tr.retrieval.skill_id,
            "n_iterations": len(its),
            "first_check_passed": bool(its[0].check.passed) if its else None,
            "final_failures": its[-1].check.failures if its else [],
            "payer_rejected": payer_reason is not None,
            "payer_reason": payer_reason,
            "balance": float(rows[state_id].get("balance") or 0.0),
            "seconds": round(dt, 2),
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        traces = dict(pool.map(one, ids))
    wall = time.perf_counter() - started

    ok = {k: v for k, v in traces.items() if "error" not in v}
    paperwork = {k: v for k, v in ok.items() if v["n_iterations"] > 0}
    summary = {
        "model": model,
        "n_states": len(ids),
        "errors": sum(1 for v in traces.values() if "error" in v),
        "status": dict(Counter(v["status"] for v in ok.values())),
        "escalation_reasons": dict(Counter(
            v["escalation_reason"] for v in ok.values() if v["escalation_reason"])),
        "paperwork_states": len(paperwork),
        "paperwork_first_pass_rate": round(
            sum(1 for v in paperwork.values() if v["first_check_passed"]) / max(1, len(paperwork)), 4),
        "paperwork_retries": sum(1 for v in paperwork.values() if v["n_iterations"] > 1),
        "paperwork_escalations": sum(
            1 for v in ok.values() if v["escalation_reason"] == "iteration_cap"),
        "balance_on_paperwork_escalations": round(sum(
            v["balance"] for v in ok.values() if v["escalation_reason"] == "iteration_cap"), 2),
        "payer_rejected": sum(1 for v in ok.values() if v.get("payer_rejected")),
        "balance_rejected": round(sum(
            v["balance"] for v in ok.values() if v.get("payer_rejected")), 2),
        "payer_reasons": dict(Counter(
            (v["payer_reason"] or "").split(",")[0][:60] for v in ok.values() if v.get("payer_rejected"))),
        "reader": {"calls": reader.calls_to_provider, "cache_hits": reader.cache_hits},
        "mean_seconds_per_state": round(sum(v["seconds"] for v in ok.values()) / max(1, len(ok)), 3),
        "wall_seconds": round(wall, 1),
        "llm": stats(),
    }
    out = REPORTS / f"l_swap_{model}_demo.json"
    out.write_text(json.dumps({"summary": summary, "states": traces}, indent=1),
                   encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))
    print("written:", out)


def compare() -> None:
    """Diff every measured writer against the pinned one, state by state."""
    reports = {}
    for name in WRITERS:
        path = REPORTS / f"l_swap_{name}_demo.json"
        if path.is_file():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    ref = reports["qwen"]["states"]
    mismatches = {}
    for name, rep in reports.items():
        st = rep["states"]
        mismatches[name] = [
            s for s in sorted(set(ref) & set(st))
            if "error" not in ref[s] and "error" not in st[s]
            and ref[s]["recommended_action"] != st[s]["recommended_action"]
        ]
    out = {
        "decisions_identical": all(not v for v in mismatches.values()),
        "decision_mismatches": mismatches,
        "summaries": {name: rep["summary"] for name, rep in reports.items()},
    }
    path = REPORTS / "l_swap_comparison_demo.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print("decisions identical:", out["decisions_identical"])
    keys = ("status", "paperwork_first_pass_rate", "paperwork_retries",
            "paperwork_escalations", "balance_on_paperwork_escalations",
            "payer_rejected", "balance_rejected")
    for k in keys:
        print(f"{k:28} " + "  ".join(f"{n}={rep['summary'].get(k)!s}" for n, rep in reports.items()))
    print("written:", path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(WRITERS))
    p.add_argument("--compare", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.compare:
        compare()
    elif a.model:
        run(a.model, a.workers)
    else:
        p.error("give --model or --compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
