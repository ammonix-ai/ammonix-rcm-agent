"""Fine-tune the 9B open model in the L seat on the agent's own success signal.

Method (expert iteration / rejection-sampling SFT): for every working-corpus
state where the agent's decision path fires an execute Skill, the writer
drafts the paperwork K times at temperature 0.8. Each draft goes through the
agent's scrubber (the deterministic rules) and then to the payer
(cardessa.payer_reader.payer_receives), whose reviewer reads appeal letters
against the records it holds. The reward of a draft is the money the claim
collected at the end of its episode (`paid_amount_final`) if the payer
processed the paperwork, else 0. Per state the best draft becomes one
(prompt -> JSON) training pair; a LoRA adapter is trained on the pairs; the
tuned model is then measured on the demonstration claims exactly as the
other writers in the L seat are.

The writer never judges itself: the scrubber is code and the payer's
reviewer is one pinned model outside the agent. Only the reviewer's verdict
is used; no model's generated text is a training target.

Temporal firewall: `paid_amount_final` is a post-hoc quarantine column. It
is read here ONLY as the reward; it never enters a prompt, a feature, or the
training pair.

Stages (run in order; every stage is resumable):
  prompts   replay the decision path on the working corpus, capture the
            first M1 prompt + schema + rules of every execute state
  sample    K drafts per state from the served base model (vLLM on :8002)
  score     scrubber, then payer; attach reward and advantage
  select    best draft per state -> runs/sft/ornith_r2/pairs.jsonl
  train     QLoRA (refuses while any model server answers on the GPU)
  merge     merge the adapter into the BF16 weights for serving

Training venv: any venv with torch (cu128) + trl; point ORNITH_TRAIN_PY at it.
Everything before `train` runs in the factory .venv.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

WORK = ROOT / "runs" / "sft" / "ornith_r2"
PROMPTS = WORK / "prompts.jsonl"
SAMPLES = WORK / "samples"
SCORES = WORK / "scores.jsonl"
PAIRS = WORK / "pairs.jsonl"
REPORT = ROOT / "runs" / "reports" / "l_finetune_ornith.json"

ORNITH_MODEL = "ornith-ai/Ornith-1.5-9B"
ORNITH_URL = os.environ.get("ORNITH_URL", "http://127.0.0.1:8002/v1")
HF_DIR = Path(os.environ.get("ORNITH_HF_DIR", str(Path.home() / "models" / "Ornith-1.5-9B-hf")))
MAX_TOKENS = 400          # the agent's own M1 budget (cardessa.harness.M1Provider)
SEED = 20260823
TIEBREAK_NOTE = ("reward = paid_amount_final if the scrubber and the payer both accept the draft, "
                 "else 0; ties: the shorter draft")


# --------------------------------------------------------------------------
# stage 1: prompts
# --------------------------------------------------------------------------
class _Captured(Exception):
    pass


class _CaptureProvider:
    """Records the first M1 prompt/schema run_case asks for, then aborts."""

    model_id = "capture"

    def __init__(self):
        self.last = None

    def generate(self, prompt: str, schema: dict) -> str:
        self.last = (prompt, schema)
        raise _Captured()


def stage_prompts(per_skill_cap: int) -> None:
    import numpy as np
    import polars as pl
    from ammonix_core.schema import HarnessArtefacts, Skill
    from cardessa.features_kept import build_kept_features
    from cardessa.harness import BasisRuntime
    import run_harness_eval as rhe
    from run_harness_eval import resolve_and_run

    rt = BasisRuntime.load(ROOT)
    harness = HarnessArtefacts.model_validate_json(
        (ROOT / "runs" / "manifests" / "harness.json").read_text(encoding="utf-8"))
    skills = [Skill.model_validate(s) for s in json.loads(
        (ROOT / "runs" / "manifests" / "skills.json").read_text(encoding="utf-8"))["skills"]]
    by_id = {s.skill_id: s for s in skills}
    working = pl.read_parquet(ROOT / "data" / "working" / "cardessa_sim" / "states.parquet")
    episodes = pl.read_parquet(ROOT / "data" / "working" / "cardessa_sim" / "episodes.parquet")
    paid = dict(zip(episodes["episode_id"].to_list(),
                    episodes["paid_amount_final"].to_list(), strict=True))
    success = dict(zip(episodes["episode_id"].to_list(),
                       episodes["success"].to_list(), strict=True))
    folds = json.loads((ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8"))["assignment"]
    frame, _, _, _ = build_kept_features(ROOT, working)
    names = rt.feature_names
    rows = {r["state_id"]: r for r in working.to_dicts()}
    feats = {r["state_id"]: {n: r[n] for n in names} for r in frame.to_dicts()}
    ids = sorted(rows)
    scaled = rt.scaler.transform(np.array([[feats[s][n] for n in names] for s in ids]))
    scaled_of = dict(zip(ids, scaled, strict=True))

    # the check must never be invoked here (capture aborts before it)
    rhe._EVALUATOR = lambda rule, payload, case_row: (True, "capture")
    prov = _CaptureProvider()
    captured, skipped = [], Counter()
    t0 = time.perf_counter()
    for i, sid in enumerate(ids):
        prov.last = None
        try:
            tr = resolve_and_run(rt, skills, harness, prov, rows[sid], feats[sid], scaled_of[sid])
            skipped[tr.status + ":" + (tr.escalation.reason if tr.escalation else "-")] += 1
            continue
        except _Captured:
            pass
        prompt, schema = prov.last
        # the M1 prompt names the action; execute skills map 1:1 to actions
        m = re.search(r"^Action to execute: (\S+)$", prompt, flags=re.M)
        by_action = {s.scope.ref: s for s in skills if s.kind == "execute" and s.scope.level == "cluster"}
        sk = by_action.get(m.group(1)) if m else None
        if sk is None:
            skipped["no_skill_match"] += 1
            continue
        ep = rows[sid]["episode_id"]
        captured.append({
            "state_id": sid, "episode_id": ep, "fold": folds.get(ep),
            "skill_id": sk.skill_id, "action": sk.scope.ref,
            "rules": list(sk.expected_result.rules or []),
            "schema": schema, "prompt": prompt,
            "paid_amount_final": float(paid[ep]), "episode_success": bool(success[ep]),
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(ids)} states routed, {len(captured)} execute prompts, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    print("routing outcomes (non-execute):", dict(skipped))
    print("execute prompts captured:", len(captured),
          dict(Counter(c["skill_id"] for c in captured)))
    # reward is known to be zero for episodes that collected nothing: keep
    # them out of the sample budget (they cannot produce a pair)
    with_money = [c for c in captured if c["paid_amount_final"] > 0]
    print("with paid_amount_final > 0:", len(with_money))
    rng = random.Random(SEED)
    chosen = []
    per = defaultdict(list)
    for c in with_money:
        per[c["skill_id"]].append(c)
    for skill_id, items in sorted(per.items()):
        rng.shuffle(items)
        chosen.extend(items[:per_skill_cap])
    chosen.sort(key=lambda c: c["state_id"])
    WORK.mkdir(parents=True, exist_ok=True)
    with PROMPTS.open("w", encoding="utf-8", newline="\n") as fh:
        for c in chosen:
            c.pop("_first_line", None)
            fh.write(json.dumps(c) + "\n")
    print("sampled for training:", len(chosen), dict(Counter(c["skill_id"] for c in chosen)))
    print("written:", PROMPTS)


# --------------------------------------------------------------------------
# stage 2: sample
# --------------------------------------------------------------------------
def _served_id(model_substring: str) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{ORNITH_URL}/models", timeout=10) as r:
        served = [m["id"] for m in json.loads(r.read())["data"]]
    m = [s for s in served if model_substring in s]
    if not m:
        raise RuntimeError(f"{ORNITH_URL} serves {served}, not *{model_substring}*")
    return m[0]


def _chat(served: str, prompt: str, schema: dict, n: int, temperature: float, seed: int,
          max_tokens: int = MAX_TOKENS) -> list[str]:
    import urllib.request

    payload = json.dumps({
        "model": served, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "seed": seed, "n": n, "max_tokens": max_tokens,
        # the model card's sampling settings (top-p 0.95, top-k 20, min-p 0);
        # with the full 248k vocabulary open, T=0.8 degenerates into noise
        "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "payload", "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(f"{ORNITH_URL}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    body = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                body = json.loads(r.read())
            break
        except (TimeoutError, OSError):
            if attempt == 2:
                raise
    out = []
    for ch in body["choices"]:
        c = (ch["message"].get("content") or "").strip()
        if "</think>" in c:
            c = c.rsplit("</think>", 1)[1].strip()
        out.append(c)
    return out


def stage_sample(k: int, temperature: float, workers: int) -> None:
    served = _served_id("Ornith-1.5-9B")
    items = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    SAMPLES.mkdir(parents=True, exist_ok=True)
    todo = [c for c in items if not (SAMPLES / f"{c['state_id']}.json").is_file()]
    print(f"sampling K={k} T={temperature} from {served}: {len(todo)} of {len(items)} states to do")

    def one(c):
        seed = int(hashlib.sha256(c["state_id"].encode()).hexdigest()[:8], 16)
        drafts = _chat(served, c["prompt"], c["schema"], k, temperature, seed)
        (SAMPLES / f"{c['state_id']}.json").write_text(
            json.dumps({"state_id": c["state_id"], "model": served, "k": k,
                        "temperature": temperature, "seed": seed, "drafts": drafts}),
            encoding="utf-8")
        return c["state_id"]

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(one, todo):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(todo)} states, {time.perf_counter() - t0:.0f}s", flush=True)
    print("sampling done:", time.perf_counter() - t0, "s")


# --------------------------------------------------------------------------
# stage 3: score
# --------------------------------------------------------------------------
def _make_check(*_args, **_kwargs):
    """The agent's own M2 evaluator: the deterministic rules."""
    from cardessa.harness import make_rule_evaluator

    return make_rule_evaluator()


def _payer_facts(case_row: dict) -> dict:
    """What the payer holds about a corpus state (for payer_receives)."""
    return {
        "cpt": case_row["cpt"],
        "dx_codes": [d for d in (case_row.get("dx_codes") or "").split(",") if d],
        "carc": case_row.get("carc") or None,
        "patient_share": None,  # the corpus state does not carry the pending share
        "balance": case_row.get("balance"),
        "clinical_indication_text": case_row.get("clinical_indication_text") or "",
        "payer_correspondence_text": case_row.get("payer_correspondence_text") or "",
    }


def _check_draft(evaluate, validate_payload, c: dict, draft: str, case_row: dict,
                 reader=None) -> dict:
    """The scrubber, then the payer. `scrubbed` is the scrubber's verdict;
    `passed` means the payer processed the paperwork."""
    from cardessa.payer_reader import payer_receives

    try:
        payload = json.loads(draft)
    except (ValueError, TypeError):
        return {"passed": False, "scrubbed": False, "failures": ["$: output was not valid JSON"],
                "payer": None}
    failures = validate_payload(c["schema"], payload)
    for rule in c["rules"]:
        ok, detail = evaluate(rule, payload, case_row)
        if not ok:
            failures.append(f"rule {rule}: {detail}")
    if failures:
        return {"passed": False, "scrubbed": False, "failures": failures, "payer": None}
    reason = payer_receives(c["action"], payload, _payer_facts(case_row), reader)
    return {"passed": reason is None, "scrubbed": True,
            "failures": [f"payer: {reason}"] if reason else [], "payer": reason}


def stage_score(workers: int) -> None:
    import polars as pl
    from ammonix_core.runtime import validate_payload

    from cardessa.payer_reader import PayerReader

    evaluate = _make_check()
    reader = PayerReader(cache_dir=ROOT / "data" / "payer_reader")
    working = pl.read_parquet(ROOT / "data" / "working" / "cardessa_sim" / "states.parquet")
    rows = {r["state_id"]: r for r in working.to_dicts()}
    items = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines()]
    items = [c for c in items if (SAMPLES / f"{c['state_id']}.json").is_file()]
    print("scoring", len(items), "states")

    def one(c):
        s = json.loads((SAMPLES / f"{c['state_id']}.json").read_text(encoding="utf-8"))
        results = []
        for d in s["drafts"]:
            r = _check_draft(evaluate, validate_payload, c, d, rows[c["state_id"]], reader)
            r["reward"] = c["paid_amount_final"] if r["passed"] else 0.0
            r["draft"] = d
            results.append(r)
        rewards = [r["reward"] for r in results]
        mean = sum(rewards) / len(rewards)
        spread = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
        for r in results:
            r["advantage"] = (r["reward"] - mean) / spread if spread > 0 else 0.0
        return {"state_id": c["state_id"], "skill_id": c["skill_id"], "fold": c["fold"],
                "paid_amount_final": c["paid_amount_final"],
                "n_pass": sum(r["passed"] for r in results),
                "n_scrubbed": sum(r["scrubbed"] for r in results), "k": len(results),
                "drafts": results}

    t0 = time.perf_counter()
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(one, items)):
            out.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(items)}, {time.perf_counter() - t0:.0f}s", flush=True)
    with SCORES.open("w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    pass_rate = sum(r["n_pass"] for r in out) / max(1, sum(r["k"] for r in out))
    any_pass = sum(1 for r in out if r["n_pass"] > 0)
    all_pass = sum(1 for r in out if r["n_pass"] == r["k"])
    per_skill = defaultdict(lambda: [0, 0])
    for r in out:
        per_skill[r["skill_id"]][0] += r["n_pass"]
        per_skill[r["skill_id"]][1] += r["k"]
    scrub_rate = sum(r["n_scrubbed"] for r in out) / max(1, sum(r["k"] for r in out))
    print(json.dumps({
        "states": len(out), "draft_scrubber_pass_rate": round(scrub_rate, 4),
        "draft_payer_accept_rate": round(pass_rate, 4),
        "reader_calls": reader.calls_to_provider, "reader_cache_hits": reader.cache_hits,
        "states_any_pass": any_pass, "states_all_pass": all_pass,
        "per_skill_draft_pass_rate": {k: round(v[0] / v[1], 3) for k, v in sorted(per_skill.items())},
    }, indent=1))
    print("written:", SCORES)


# --------------------------------------------------------------------------
# stage 4: select
# --------------------------------------------------------------------------
def _render_prompt(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def stage_select(holdout_fold: int | None) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    prompts = {json.loads(l)["state_id"]: json.loads(l)
               for l in PROMPTS.read_text(encoding="utf-8").splitlines()}
    scored = [json.loads(l) for l in SCORES.read_text(encoding="utf-8").splitlines()]
    pairs, dropped = [], Counter()
    for r in scored:
        if holdout_fold is not None and r["fold"] == holdout_fold:
            dropped["holdout_fold"] += 1
            continue
        winners = [d for d in r["drafts"] if d["passed"] and d["reward"] > 0]
        if not winners:
            dropped["no_passing_draft"] += 1
            continue
        # ties on reward are the rule (reward is per state): the shorter draft
        best = max(winners, key=lambda d: (d["reward"], -len(d["draft"])))
        # normalize the JSON text so training targets are canonical
        canon = json.dumps(json.loads(best["draft"]), ensure_ascii=False)
        pairs.append({
            "state_id": r["state_id"], "skill_id": r["skill_id"], "fold": r["fold"],
            "prompt": _render_prompt(tok, prompts[r["state_id"]]["prompt"]),
            "completion": canon + tok.eos_token,
            "reward": best["reward"], "advantage": best["advantage"],
            "n_pass": r["n_pass"], "k": r["k"],
        })
    with PAIRS.open("w", encoding="utf-8", newline="\n") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("pairs:", len(pairs), dict(Counter(p["skill_id"] for p in pairs)))
    print("dropped:", dict(dropped))
    print("pairs with contrast (some draft failed):", sum(1 for p in pairs if p["n_pass"] < p["k"]))
    print("eos:", repr(tok.eos_token))
    print("written:", PAIRS)


# --------------------------------------------------------------------------
# stage 5: train (training venv)
# --------------------------------------------------------------------------
def _gpu_server_answers() -> list[str]:
    import urllib.request

    up = []
    for port in (8000, 8001, 8002):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                up.append(str(port))
        except OSError:
            pass
    return up


def stage_train(tag: str, epochs: int, max_length: int) -> None:
    up = _gpu_server_answers()
    if up:
        raise SystemExit(f"refusing to train: a model server answers on port(s) {up}. "
                         "Stop it first (trap 1: training beside a server crashed the machine).")
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    assert torch.cuda.is_available(), "CUDA torch required"
    out_dir = WORK / "train" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(HF_DIR), quantization_config=bnb,
                                                 dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    lora = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
                        "out_proj", "gate_proj", "up_proj", "down_proj"],
    )
    ds = load_dataset("json", data_files=str(PAIRS), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("prompt", "completion")])
    cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_steps=0.03, logging_steps=1, save_strategy="no", bf16=True,
        optim="adamw_torch", max_length=max_length, packing=False,
        completion_only_loss=True, gradient_checkpointing=True, report_to=[],
        seed=SEED, dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=lora)
    n_train = trainer.model.get_nb_trainable_parameters()
    print("trainable params:", n_train, flush=True)
    t0 = time.perf_counter()
    result = trainer.train()
    wall = time.perf_counter() - t0
    trainer.model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    log = [h for h in trainer.state.log_history if "loss" in h]
    (out_dir / "train_log.json").write_text(json.dumps({
        "tag": tag, "pairs": len(ds), "epochs": epochs, "steps": trainer.state.global_step,
        "wall_seconds": round(wall, 1), "first_loss": log[0]["loss"] if log else None,
        "last_loss": log[-1]["loss"] if log else None, "log": log,
        "train_result": result.metrics, "lora": lora.to_dict() | {"target_modules": sorted(lora.target_modules)},
        "max_length": max_length, "optimizer": "adamw_torch", "quantization": "nf4 (QLoRA)",
    }, indent=1, default=str), encoding="utf-8")
    print(f"trained {trainer.state.global_step} steps in {wall / 60:.1f} min; "
          f"loss {log[0]['loss'] if log else '?'} -> {log[-1]['loss'] if log else '?'}")


def stage_merge(tag: str, out_name: str) -> None:
    import shutil

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = WORK / "train" / tag / "adapter"
    out = HF_DIR.parent / out_name
    model = AutoModelForCausalLM.from_pretrained(str(HF_DIR), dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(str(HF_DIR)).save_pretrained(str(out))
    # vLLM loads the checkpoint as the full (vision + text) model: add the
    # untouched vision tensors and the original config and processor files
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_index = json.loads((HF_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
    out_index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
    missing = [k for k in src_index["weight_map"] if k not in out_index["weight_map"]]
    by_shard: dict[str, list[str]] = {}
    for k in missing:
        by_shard.setdefault(src_index["weight_map"][k], []).append(k)
    tensors = {}
    for shard, keys in by_shard.items():
        with safe_open(HF_DIR / shard, "pt") as f:
            for k in keys:
                tensors[k] = f.get_tensor(k)
    if tensors:
        save_file(tensors, out / "model-vision.safetensors", metadata={"format": "pt"})
        for k in missing:
            out_index["weight_map"][k] = "model-vision.safetensors"
        out_index["metadata"] = src_index.get("metadata", {})
        (out / "model.safetensors.index.json").write_text(json.dumps(out_index, indent=2), encoding="utf-8")
    for name in ("config.json", "generation_config.json", "chat_template.jinja",
                 "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"):
        if (HF_DIR / name).is_file():
            shutil.copy(HF_DIR / name, out / name)
    print(f"merged model written: {out} ({len(missing)} vision tensors carried over)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prompts"); s.add_argument("--per-skill-cap", type=int, default=200)
    s = sub.add_parser("sample"); s.add_argument("--k", type=int, default=8)
    s.add_argument("--temperature", type=float, default=0.8); s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score"); s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("select"); s.add_argument("--holdout-fold", type=int, default=None)
    s = sub.add_parser("train"); s.add_argument("--tag", default="r1")
    s.add_argument("--epochs", type=int, default=2); s.add_argument("--max-length", type=int, default=4096)
    s = sub.add_parser("merge"); s.add_argument("--tag", default="r1")
    s.add_argument("--out-name", default="Ornith-1.5-9B-rcm-r1")
    a = p.parse_args()
    if a.cmd == "prompts":
        stage_prompts(a.per_skill_cap)
    elif a.cmd == "sample":
        stage_sample(a.k, a.temperature, a.workers)
    elif a.cmd == "score":
        stage_score(a.workers)
    elif a.cmd == "select":
        stage_select(a.holdout_fold)
    elif a.cmd == "train":
        stage_train(a.tag, a.epochs, a.max_length)
    elif a.cmd == "merge":
        stage_merge(a.tag, a.out_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
