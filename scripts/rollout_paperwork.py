"""Head-to-head rollout in which every arm writes its paperwork.

Each arm decides the action at every touch of the same fresh episodes
(identical engine randomness). The decision is then turned into paperwork by
ONE shared writer (the model in the L seat) through the agent's own
paperwork loop: the scrubber (deterministic rules) accepts the form or sends
it back with the reason, up to twice; a third failure hands the touch to
the simulated biller. The
payer receives the paperwork and adjudicates it against the records it holds
(cardessa.payer_reader.payer_receives); an appeal letter is read by the
payer's reviewer. Rejected paperwork costs the touch; the claim continues.

Arms: personas (simulated billers, correct forms by construction), ammonix
(the decision path of the shipped agent), agentic / agentic_rag (the LLM
agent on the pinned 27B), claude (the same agent loop on claude-sonnet-5),
sol_tuned / sol_rag_tuned (the tuned GPT configuration), gpt6_tuned /
gpt6_rag_tuned (the same tuned configuration on gpt-6-astra).

One GPU serves one model at a time, so every step runs in phases and the
driver swaps containers itself: the 27B (state texts, agentic decisions),
the writer, then the payer's reviewer through the API. Every model call is
cached by prompt hash; an interrupted run resumes from the caches.

Usage:
  .venv/Scripts/python.exe scripts/rollout_paperwork.py --tranche 99 \\
      --arms personas,ammonix,agentic,agentic_rag,claude,sol_tuned,sol_rag_tuned \\
      --writer ornith_ft --report comparison_paperwork_t99.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "agentic_baseline"
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts"), str(BASELINE)):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.schema import HarnessArtefacts, RetrievalResult, Skill  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import (  # noqa: E402
    MISTAKE_PENALTY,
    CachedTextGenerator,
    CardessaEnvironment,
    compute_mistakes,
    state_tabular,
    state_text,
)
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.harness import BasisRuntime, make_rule_evaluator, prompt_fields  # noqa: E402
from cardessa.livecases import training_world  # noqa: E402
from cardessa.payer_reader import PayerReader  # noqa: E402
from cardessa.personas import CaseView, choose_action  # noqa: E402
from cardessa.textgen import VllmProvider  # noqa: E402

from agentic.fallback import persona_action  # noqa: E402

N_EPISODES = 300
ALLOCATION = {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}
MAX_STEPS = 8
LLM_WORKERS = 6
WRITE_WORKERS = 6

SERVERS = {
    # name: (container, url, served-model substring)
    "qwen": ("ammonix-m1-27b", "http://127.0.0.1:8000/v1", "Qwen"),
    "ornith": ("ammonix-ornith-9b", "http://127.0.0.1:8002/v1", "Ornith-1.5-9B"),
    "ornith_ft": ("ammonix-ornith-ft", "http://127.0.0.1:8002/v1", "Ornith-1.5-9B-rcm"),
}
WRITER_MODEL_ID = {
    "qwen": None,
    "ornith": "ornith-ai/Ornith-1.5-9B",
    "ornith_ft": "local/Ornith-1.5-9B-rcm-r2",
    "opus5": "claude-opus-5",
    "gpt6": "gpt-6-astra",
}
GPT6_MODEL_ID = "gpt-6-astra"


# --------------------------------------------------------------------------
# servers
# --------------------------------------------------------------------------
def _served(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=3) as r:
            return [m["id"] for m in json.loads(r.read())["data"]]
    except OSError:
        return []


def ensure_server(name: str) -> None:
    """Stop every other model container and bring `name` up."""
    container, url, tag = SERVERS[name]
    if any(tag in m for m in _served(url)):
        return
    for other, (c, _, _) in SERVERS.items():
        if other != name:
            subprocess.run(["docker", "stop", c], capture_output=True)
    subprocess.run(["docker", "start", container], capture_output=True, check=True)
    t0 = time.perf_counter()
    # 40 min: under heavy disk-cache pressure a swap can take well over the
    # old 15-minute budget (observed 2026-09-08, ~1 min per checkpoint shard)
    while time.perf_counter() - t0 < 2400:
        time.sleep(10)
        if any(tag in m for m in _served(url)):
            print(f"  [server] {name} up after {time.perf_counter() - t0:.0f}s", flush=True)
            return
    raise RuntimeError(f"{container} did not come up")


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def case_view(case, payer):
    return CaseView(
        payer=payer, touch_seq=case.touch_seq, submitted_once=case.submitted_once,
        carc=case.carc,
        underpaid=(case.patient_share_pending > 0 and case.carc is None
                   and case.collected_payer > 0),
        auth_required=case.study["cpt"] in payer.auth_required_cpts,
        auth_on_file=case.auth_ref is not None,
        days_since_service=case.days_since_service,
        balance=case.allowed - case.collected_payer - case.collected_patient,
        has_secondary=case.coverage["secondary_payer_id"] is not None,
        doc_quality=case.clinic["doc_quality"],
        resubmitted_unchanged_already=case.resubmitted_unchanged,
        appealed_already=case.appealed,
        retro_requested_already=case.retro_requested,
    )


def clerk_action(env, case, payer):
    return persona_action(
        choose_action, case.persona, case_view(case, payer), MASTER_SEED,
        case.episode_id, case.patient_share_pending, env.legal_actions(case),
    )


def make_baseline(arm: str):
    """The LLM-agent policies, exactly as the registered comparison built them."""
    from agentic.llm import AgentLLM
    from agentic.policy import AgenticPolicy

    if arm == "agentic":
        return AgenticPolicy(AgentLLM(cache_dir=BASELINE / "cache" / "decisions"))
    if arm == "agentic_rag":
        from agentic.rag_policy import RagPolicy

        return RagPolicy(AgentLLM(cache_dir=BASELINE / "cache" / "decisions_rag"), ROOT)
    if arm == "claude":
        from agentic.llm_claude import ClaudeLLM

        key_file = ROOT / "data" / "anthropic_key.txt"
        if key_file.is_file() and not os.environ.get("ANTHROPIC_API_KEY"):
            os.environ["ANTHROPIC_API_KEY"] = key_file.read_text(encoding="utf-8-sig").strip()
        return AgenticPolicy(ClaudeLLM(cache_dir=BASELINE / "cache" / "decisions_claude"))
    if arm in ("sol_tuned", "sol_rag_tuned"):
        from agentic.briefing import TUNED_SYSTEM_PROMPT
        from ui.openai_llm import SolAgentLLM

        llm = SolAgentLLM(cache_dir=ROOT / "data" / f"sol_decisions_{arm}")
        if arm == "sol_tuned":
            return AgenticPolicy(llm, system_prompt=TUNED_SYSTEM_PROMPT)
        from agentic.rag_policy import RagPolicy

        return RagPolicy(llm, ROOT, k=25, system_prompt=TUNED_SYSTEM_PROMPT)
    if arm in ("gpt6_tuned", "gpt6_rag_tuned"):
        # GPT-6 on the same tuned configuration the Sol arms registered
        # (prompt P2, k=25); no new tuning was run for GPT-6
        from agentic.briefing import TUNED_SYSTEM_PROMPT
        from ui.openai_llm import SolAgentLLM

        llm = SolAgentLLM(cache_dir=ROOT / "data" / f"gpt6_decisions_{arm}",
                          model_id=GPT6_MODEL_ID)
        if arm == "gpt6_tuned":
            return AgenticPolicy(llm, system_prompt=TUNED_SYSTEM_PROMPT)
        from agentic.rag_policy import RagPolicy

        return RagPolicy(llm, ROOT, k=25, system_prompt=TUNED_SYSTEM_PROMPT)
    raise ValueError(arm)


NEEDS_QWEN = {"agentic", "agentic_rag"}


class OracleDecider:
    """Upper bound, not a competitor: reads the engine's true resolution
    probabilities (evaluation-only numbers no policy sees) and greedily
    takes the best action."""

    def __init__(self, env, engine):
        sys.path.insert(0, str(BASELINE / "scripts"))
        import run_oracle_arm as roa

        self.roa = roa
        self.env = env
        self.engine = engine

    def decide(self, case, row):
        return self.roa.oracle_action(self.env, self.engine, case, row)


class ImitationDecider:
    """Behavior cloning: the classifier trained to copy the clerks' actions
    (no outcome information). Predicted-but-illegal falls back to the
    highest-probability legal action; no legal action hands the touch off."""

    def __init__(self, env):
        import pickle

        sys.path.insert(0, str(BASELINE / "scripts"))
        import run_imitation_arm as ria

        self.ria = ria
        self.env = env
        with open(BASELINE / "cache" / "imitation" / "clerk_bc.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        self.model = bundle["model"]
        self.vocab = bundle["vocab"]
        self.col = {a: i for i, a in enumerate(self.model.classes_)}

    def decide(self, case, payer):
        return self.ria.bc_action(self.model, self.vocab, self.col, case, payer,
                                  self.env.legal_actions(case))


class _Captured(Exception):
    pass


class _CaptureProvider:
    """Records the M1 prompt the decision path asks for, then stops."""

    model_id = "capture"

    def __init__(self):
        self.last = None

    def generate(self, prompt: str, schema: dict) -> str:
        self.last = (prompt, schema)
        raise _Captured()


class AblatedDecider:
    """The memory gates off: the arm always acts on the masked calibrated
    argmax. Kept (hard rules): the applicability mask, the forced CO-16
    responsive action, the blown success clock (empty scores -> the clerk).
    Disabled (memory gates): unknown-payer escalation, the confidence floor
    with its close-out, the ambiguity ask, the case-based fallback."""

    def __init__(self, base: "AmmonixDecider"):
        self.base = base

    def decide(self, snap_rows: list[dict]) -> dict[str, str | None]:
        from cardessa.harness import route_case

        d = self.base
        snaps = pl.DataFrame([{c: r.get(c) for c in d.working.columns} for r in snap_rows],
                             schema_overrides=d.working.schema)
        frame, _, _, _ = build_kept_features(
            ROOT, pl.concat([d.working, snaps], how="vertical_relaxed"))
        names = d.rt.feature_names
        feats = {r["state_id"]: {n: r[n] for n in names}
                 for r in frame.filter(pl.col("state_id").is_in(snaps["state_id"].to_list())).to_dicts()}
        out = {}
        for row in snap_rows:
            scores, _known, forced, _amb = route_case(d.rt, row, feats[row["state_id"]])
            if forced is not None:
                out[row["state_id"]] = forced
            elif scores:
                out[row["state_id"]] = max(scores.items(), key=lambda kv: kv[1])[0]
            else:  # blown clock / empty mask: the hard rule stands
                out[row["state_id"]] = None
        return out


class AmmonixDecider:
    """The shipped decision path (route, floor, close-out, Skills), with the
    writer deferred to its own phase."""

    def __init__(self):
        import run_harness_eval as rhe

        self.rt = BasisRuntime.load(ROOT)
        self.harness = HarnessArtefacts.model_validate_json(
            (ROOT / "runs/manifests/harness.json").read_text(encoding="utf-8"))
        self.skills = [Skill.model_validate(s) for s in json.loads(
            (ROOT / "runs/manifests/skills.json").read_text(encoding="utf-8"))["skills"]]
        self.working = pl.read_parquet(ROOT / "data/working/cardessa_sim/states.parquet")
        self.resolve_and_run = rhe.resolve_and_run
        rhe._EVALUATOR = make_rule_evaluator()

    def decide(self, snap_rows: list[dict]) -> dict[str, str | None]:
        """state_id -> action to write paperwork for, or None (hand-off)."""
        snaps = pl.DataFrame([{c: r.get(c) for c in self.working.columns} for r in snap_rows],
                             schema_overrides=self.working.schema)
        frame, _, _, _ = build_kept_features(
            ROOT, pl.concat([self.working, snaps], how="vertical_relaxed"))
        names = self.rt.feature_names
        feats = {r["state_id"]: {n: r[n] for n in names}
                 for r in frame.filter(pl.col("state_id").is_in(snaps["state_id"].to_list())).to_dicts()}
        ids = [r["state_id"] for r in snap_rows]
        scaled = self.rt.scaler.transform(np.array([[feats[s][n] for n in names] for s in ids]))
        out = {}
        for row, x in zip(snap_rows, scaled, strict=True):
            prov = _CaptureProvider()
            try:
                trace = self.resolve_and_run(self.rt, self.skills, self.harness, prov,
                                             row, feats[row["state_id"]], x)
                out[row["state_id"]] = None  # escalated before any paperwork
                _ = trace
            except _Captured:
                m = re.search(r"^Action to execute: (\S+)$", prov.last[0], flags=re.M)
                out[row["state_id"]] = m.group(1)
        return out


# --------------------------------------------------------------------------
# the shared writer
# --------------------------------------------------------------------------
class Writer:
    def __init__(self, name: str):
        import l_swap_experiment as lse

        self.name = name
        if name == "qwen":
            from cardessa.harness import M1Provider

            self.llm = M1Provider(cache_dir=ROOT / "data" / "m1_cache")
            self.generate = self.llm.generate
        elif name == "opus5":
            self.llm = lse.OpusNoThink(cache_dir=ROOT / "data" / "m1_opus5")
            self.generate = lambda p, s: self.llm.complete([{"role": "user", "content": p}], s)
        elif name == "gpt6":
            from ui.openai_llm import SolAgentLLM

            self.llm = SolAgentLLM(cache_dir=ROOT / "data" / "m1_gpt6",
                                   model_id=WRITER_MODEL_ID[name])
            self.generate = lambda p, s: self.llm.complete([{"role": "user", "content": p}], s)
        else:
            self.llm = lse.LocalJsonLLM(cache_dir=ROOT / "data" / f"m1_{name}",
                                        model_id=WRITER_MODEL_ID[name])
            self.generate = lambda p, s: self.llm.complete([{"role": "user", "content": p}], s)


def write_paperwork(decider: AmmonixDecider, writer: Writer, action: str, row: dict):
    """The agent's paperwork loop for `action` on `row`. Returns
    (payload or None, trace)."""
    from ammonix_core.runtime import run_case

    skill = next(s for s in decider.skills
                 if s.scope.level == "cluster" and s.scope.ref == action and s.kind == "execute")
    retrieval = RetrievalResult(
        case_id=row["state_id"], features={}, scores_cal={}, neighbours=[],
        tribe_id=f"{action}::cluster", recommended_action_id=action, ambiguous=False,
        skill_id=skill.skill_id, expected_result=skill.expected_result.model_dump(),
    )
    trace = run_case(
        retrieval, skill, row, decider.harness.m1_prompts[skill.skill_id],
        decider.harness.m2_prompt, decider.harness.llm, writer.generate,
        make_rule_evaluator(), decider.rt.manifest.basis_id, decider.harness.harness_id,
        prompt_fields=prompt_fields,
    )
    return (trace.final.payload if trace.status == "executed" else None), trace


# --------------------------------------------------------------------------
# the rollout
# --------------------------------------------------------------------------
def outcome_v2(env, case, pending):
    outcome = env.outcome(case)
    mistakes = compute_mistakes(pending, case, outcome.success)
    n = sum(mistakes.values())
    return outcome.success, max(0.0, outcome.score - MISTAKE_PENALTY * n), n


def snapshot_row(case, payer, texts):
    row = state_tabular(case, payer, 0.0)
    row.update({"episode_id": case.episode_id, "state_id": f"{case.episode_id}-r{case.touch_seq}",
                "touch_seq": case.touch_seq, "action_raw": "", **texts})
    return row


def play(arms, writer_name, world, engine, indices, text, reader, swap):
    # the 27B must answer before anything is built: the baseline clients and
    # the text generator ping it at construction
    if swap:
        ensure_server("qwen")
    env = CardessaEnvironment(world, engine, MASTER_SEED, reader=reader)
    decider = AmmonixDecider()
    ablated = AblatedDecider(decider)
    imitation = ImitationDecider(env) if "imitation" in arms else None
    oracle = OracleDecider(env, engine) if "oracle" in arms else None
    policies = {a: make_baseline(a) for a in arms
                if a not in ("personas", "ammonix", "ammonix_ablated", "imitation", "oracle")}
    writer = Writer(writer_name)
    live = {a: {} for a in arms}
    for a in arms:
        for index in indices:
            case = env.reset(index)
            live[a][case.episode_id] = {"case": case, "pending": [], "handed_off": 0,
                                        "rejected": 0, "written": 0}
    results = {a: {} for a in arms}

    for step in range(1, MAX_STEPS + 1):
        if not any(live[a] for a in arms):
            break
        n_live = sum(len(live[a]) for a in arms)
        print(f"step {step}: {n_live} live claims", flush=True)

        # -- phase 1: texts and decisions (the 27B serves both) --------------
        if swap:
            ensure_server("qwen")
        rows = {}  # (arm, episode) -> snapshot row with texts
        for a in arms:
            for ep, slot in live[a].items():
                case = slot["case"]
                payer = engine.payers[case.payer_id]
                rows[(a, ep)] = snapshot_row(case, payer, state_text(case, payer, text, MASTER_SEED))
        chosen = {}  # (arm, episode) -> action or None (hand-off)
        for a in arms:
            arm_rows = [rows[(a, ep)] for ep in live[a]]
            if a == "personas":
                for ep in live[a]:
                    chosen[(a, ep)] = None
            elif a == "imitation":
                for ep in live[a]:
                    case = live[a][ep]["case"]
                    chosen[(a, ep)] = imitation.decide(case, engine.payers[case.payer_id])
            elif a == "oracle":
                for ep in live[a]:
                    chosen[(a, ep)] = oracle.decide(live[a][ep]["case"], rows[(a, ep)])
            elif a in ("ammonix", "ammonix_ablated"):
                # an arm can run out of live claims before the others do; an
                # empty snapshot list would build a zero-column frame and
                # crash the concat inside decide
                picks = ((decider if a == "ammonix" else ablated).decide(arm_rows)
                         if arm_rows else {})
                for ep in live[a]:
                    chosen[(a, ep)] = picks[rows[(a, ep)]["state_id"]]
            else:
                pol = policies[a]
                pol.prepare(arm_rows)

                def decide(row, a=a, pol=pol):
                    case = live[a][row["episode_id"]]["case"]
                    try:
                        return row["episode_id"], pol.decide(row, env.legal_actions(case)).action
                    except Exception as err:  # noqa: BLE001 - a dead call hands the touch off
                        print(f"  decision failed {a} {row['state_id']}: {err!r}", file=sys.stderr)
                        pol.decision_errors += 1
                        return row["episode_id"], None

                with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
                    for ep, action in pool.map(decide, arm_rows):
                        chosen[(a, ep)] = action
                if pol.decision_errors:
                    raise RuntimeError(
                        f"{pol.decision_errors} failed decision call(s) in arm {a}: "
                        "a failed call would hand the touch to the clerk and contaminate "
                        "the arm; fix the provider and rerun (everything else is cached)")

        # -- phase 2: the shared writer fills the paperwork ------------------
        to_write = [(k, v) for k, v in chosen.items() if v is not None]
        if to_write and swap and writer_name in SERVERS:
            ensure_server(writer_name)
        payloads = {}

        def write(item):
            (a, ep), action = item
            try:
                payload, trace = write_paperwork(decider, writer, action, rows[(a, ep)])
            except Exception as err:  # noqa: BLE001
                print(f"  writer failed {a} {ep} {action}: {err!r}", file=sys.stderr)
                payload = None
            return (a, ep), payload

        with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as pool:
            for key, payload in pool.map(write, to_write):
                payloads[key] = payload

        # -- phase 3: the payer receives it (reviewer through the API) -------
        finished = []
        for a in arms:
            for ep, slot in list(live[a].items()):
                case = slot["case"]
                payer = engine.payers[case.payer_id]
                action = chosen[(a, ep)]
                payload = payloads.get((a, ep))
                if action is None or (action is not None and payload is None):
                    # no decision, or paperwork the scrubber refused at the cap:
                    # the simulated biller works the touch with a correct form
                    if a != "personas":
                        slot["handed_off"] += 1
                    action = clerk_action(env, case, payer)
                    payload = None
                else:
                    slot["written"] += 1
                before = case.paperwork_rejections
                slot["pending"].append((state_tabular(case, payer, 0.0), action))
                texts = {k: rows[(a, ep)][k] for k in ("clinical_indication_text",
                                                        "payer_correspondence_text")}
                case = env.apply(case, action, payload=payload, records=texts)
                slot["rejected"] += case.paperwork_rejections - before
                slot["case"] = case
                if case.terminal:
                    success, score, n_mistakes = outcome_v2(env, case, slot["pending"])
                    results[a][ep] = {
                        "success": success, "score": score, "mistakes": n_mistakes,
                        "touches": len(case.action_history),
                        "collected_payer": case.collected_payer,
                        "collected_patient": case.collected_patient,
                        "allowed": case.allowed, "resolution": case.resolution,
                        "handed_off": slot["handed_off"], "written": slot["written"],
                        "paperwork_rejected": slot["rejected"],
                        "rejection_log": list(case.paperwork_rejection_log),
                        "actions": list(case.action_history),
                    }
                    finished.append((a, ep))
        for a, ep in finished:
            del live[a][ep]
    # claims still open after the step cap: written off, as in the registered protocol
    for a in arms:
        for ep, slot in live[a].items():
            case = slot["case"]
            while not case.terminal:
                case = env.apply(case, "write_off")
            success, score, n_mistakes = outcome_v2(env, case, slot["pending"])
            results[a][ep] = {
                "success": success, "score": score, "mistakes": n_mistakes,
                "touches": len(case.action_history), "collected_payer": case.collected_payer,
                "collected_patient": case.collected_patient, "allowed": case.allowed,
                "resolution": case.resolution, "handed_off": slot["handed_off"],
                "written": slot["written"], "paperwork_rejected": slot["rejected"],
                "rejection_log": list(case.paperwork_rejection_log),
                "actions": list(case.action_history), "step_cap": True,
            }
    stats = {a: {**policies[a].stats(), "llm": policies[a].llm.stats()}
             for a in policies}
    return results, stats, {"writer": getattr(writer.llm, "stats", lambda: {})(),
                            "reader_calls": reader.calls_to_provider,
                            "reader_cache_hits": reader.cache_hits}


def summary(results):
    n = len(results)
    rej_eps = [r for r in results.values() if r["paperwork_rejected"] > 0]
    return {
        "episodes": n,
        "success_rate_v2": round(sum(r["success"] for r in results.values()) / n, 4),
        "mean_score": round(float(np.mean([r["score"] for r in results.values()])), 4),
        "payer_collected_capped": round(
            sum(min(r["collected_payer"], r["allowed"]) for r in results.values()), 2),
        "patient_collected_total": round(sum(r["collected_patient"] for r in results.values()), 2),
        "mean_touches": round(float(np.mean([r["touches"] for r in results.values()])), 3),
        "episodes_with_mistakes": sum(1 for r in results.values() if r["mistakes"] > 0),
        "touches_written": sum(r["written"] for r in results.values()),
        "touches_handed_off": sum(r["handed_off"] for r in results.values()),
        "paperwork_rejected": sum(r["paperwork_rejected"] for r in results.values()),
        "episodes_with_rejection": len(rej_eps),
        # money the claims with rejected paperwork ended without
        "balance_rejected": round(sum(
            max(0.0, r["allowed"] - r["collected_payer"] - r["collected_patient"])
            for r in rej_eps), 2),
        "resolutions": dict(Counter(r["resolution"] for r in results.values())),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="personas,ammonix,agentic,agentic_rag,claude,sol_tuned,sol_rag_tuned")
    p.add_argument("--writer", default="ornith_ft", choices=list(WRITER_MODEL_ID))
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--tranche", type=int, required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--no-swap", action="store_true", help="never touch docker (everything cached)")
    a = p.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    world_ext, engine = training_world(ROOT, MASTER_SEED)
    start = world_ext.studies.height
    world = extend_world(world_ext, expansion_studies(world_ext, MASTER_SEED, a.tranche, ALLOCATION, start))
    indices = list(range(start, start + a.episodes))
    text = CachedTextGenerator(provider=_LazyQwen(), cache_dir=ROOT / "data" / "text_cache")
    reader = PayerReader(cache_dir=ROOT / "data" / "payer_reader")
    print(f"{arms} on {a.episodes} fresh episodes of tranche {a.tranche}; writer {a.writer}")
    t0 = time.perf_counter()
    results, stats, llm = play(arms, a.writer, world, engine, indices, text, reader, swap=not a.no_swap)
    report = {
        "analysis": "every arm writes its paperwork through one shared writer; the payer adjudicates it",
        "tranche": a.tranche, "n_episodes": a.episodes, "writer": a.writer,
        "reviewer": reader.model_id, "wall_seconds": round(time.perf_counter() - t0, 1),
        "llm": llm, "policies": stats,
    }
    for arm in arms:
        report[arm] = summary(results[arm])
        print(f"[{arm}] {json.dumps(report[arm])}")
        (ROOT / "runs/reports" / f"episodes_{arm}_{a.report}").write_text(
            json.dumps(results[arm], indent=1), encoding="utf-8", newline="\n")
    out = ROOT / "runs/reports" / a.report
    out.write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    print("written:", out)
    return 0


class _LazyQwen:
    """The 27B text provider, connected on first use (the server may not be
    up when the driver starts)."""

    model_id = "lazy"

    def __init__(self):
        self._p = None

    def generate(self, prompt: str) -> str:
        if self._p is None:
            self._p = VllmProvider()
            self.model_id = self._p.model_id
        return self._p.generate(prompt)


if __name__ == "__main__":
    raise SystemExit(main())
