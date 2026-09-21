"""Demo UI server (agent-UI spec Section 4): three screens over the runtime.

Everything is precomputed at startup from the fresh demo tranche (all M1 and
text generations hit the P9a caches, so startup makes ~no LLM calls). The
execute endpoint runs a LIVE payer-engine adjudication round trip.
"""

import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_text  # noqa: E402
from ammonix_core.runtime import run_case  # noqa: E402
from ammonix_core.schema import HarnessArtefacts, RetrievalResult, Skill  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from run_harness_eval import resolve_and_run  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.engine import Claim  # noqa: E402
from cardessa.forms import TOOLS  # noqa: E402
from cardessa.harness import (  # noqa: E402
    BasisRuntime,
    M1Provider,
    close_out_choice,
    expected_values,
    label_coordinate,
    make_rule_evaluator,
    prompt_fields,
    route_case,
)
from cardessa.impossible import (  # noqa: E402
    MISSING_DOC_LETTER,
    holdout_cases,
    missing_document_cases,
)
from cardessa.livecases import bundle_for_state, generate_demo_cases  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402

app = FastAPI(title="Ammonix demo")
STATE: dict = {}


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Local demo: browsers must never serve stale pages - repeated
    cached-JS confusion (owner saw pre-fix behaviour after fixes shipped)."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

TRIAGE = {"executed": "green", "ambiguous": "amber", "escalated": "red"}


def triage_of(trace) -> str:
    if trace.status == "escalated":
        return "red"
    return "amber" if trace.retrieval.ambiguous else "green"


_OFFLINE_MESSAGE = (
    "The local model is not running and this request is not in the shipped "
    "cache. Start the vLLM container with the pinned model to compute it."
)


class _OfflineTextProvider:
    """Stands in for VllmProvider when no model is served. The disk cache in
    front of it still answers everything already generated; only a genuine
    cache miss reaches this and fails with a clear message."""

    model_id = "cache-only"

    def generate(self, prompt: str) -> str:
        raise RuntimeError(_OFFLINE_MESSAGE)


class _CacheOnlyM1(M1Provider):
    """M1 decisions from the shipped cache only (same key as M1Provider)."""

    def __post_init__(self) -> None:
        self.model_id = "cache-only"

    def generate(self, prompt: str, schema: dict) -> str:
        key = sha256_text(prompt + json.dumps(schema, sort_keys=True))
        cached = self.cache_dir / f"{key}.json"
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        raise RuntimeError(_OFFLINE_MESSAGE)


WRITER_MODEL = "local/Ornith-1.5-9B-rcm-r2"
WRITER_URL = "http://127.0.0.1:8002/v1"

# Optional hosted backend for the operator dialog only (public demo hosting):
# an OpenAI-compatible endpoint serving stock Qwen. Off by default - without
# these variables the dialog keeps its local wiring and the rest of the demo
# (caches, live run, replay) is unaffected either way. When a remote backend
# is configured, the chat widget discloses the served model to the visitor.
DIALOG_BASE = os.getenv("DEMO_DIALOG_BASE", "http://127.0.0.1:8000/v1")
DIALOG_MODEL = os.getenv("DEMO_DIALOG_MODEL", "")
DIALOG_KEY = os.getenv("DEMO_DIALOG_KEY", "")
# Extra request fields for gateway endpoints whose control fields differ from
# the local server's (e.g. OpenRouter: reasoning off + provider pin). JSON.
DIALOG_EXTRA = json.loads(os.getenv("DEMO_DIALOG_EXTRA", "{}"))


class _TrainedWriter:
    """The shipped writer: the trained 9B, served on :8002, cached by the
    same key scheme as its evaluation runs (data/m1_ornith_ft). Falls back
    to the cache when the server is not running."""

    def __init__(self, cache_dir):
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "scripts"))
        from l_swap_experiment import LocalJsonLLM

        self.cache_dir = cache_dir
        self._llm = LocalJsonLLM(cache_dir=cache_dir, model_id=WRITER_MODEL,
                                 base_url=WRITER_URL)
        self.calls_to_provider = 0
        self._meter_lock = threading.Lock()
        self._tls = threading.local()

    @property
    def model_id(self):
        return WRITER_MODEL

    # per-case usage capture, same bracket contract as SolAgentLLM. The
    # underlying client only keeps aggregate counters, so each call reads
    # the delta under a lock; cache hits are the common path and cheap.

    def begin_call_capture(self) -> None:
        self._tls.capture = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def take_call_capture(self) -> dict | None:
        return self._tls.__dict__.pop("capture", None)

    def generate(self, prompt: str, schema: dict) -> str:
        with self._meter_lock:
            p0, c0 = self._llm.prompt_tokens, self._llm.completion_tokens
            try:
                out = self._llm.complete([{"role": "user", "content": prompt}], schema)
            except RuntimeError as exc:
                raise RuntimeError(_OFFLINE_MESSAGE) from exc
            dp = self._llm.prompt_tokens - p0
            dc = self._llm.completion_tokens - c0
        self.calls_to_provider = self._llm.calls_to_provider
        cap = getattr(self._tls, "capture", None)
        if cap is not None:
            cap["prompt_tokens"] += dp
            cap["completion_tokens"] += dc
            cap["calls"] += 1
        return out


@app.on_event("startup")
def load() -> None:
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
    # the writer seat: the trained 9B, its own server and shipped cache
    provider = _TrainedWriter(cache_dir=ROOT / "data" / "m1_ornith_ft")
    # the dialog panel and the world's text channel: the pinned Qwen 27B
    try:
        text_provider = VllmProvider()
    except OSError as exc:
        print(
            f"no Qwen 27B at http://127.0.0.1:8000 ({exc}); texts and "
            "dialog run from the shipped caches",
            flush=True,
        )
        text_provider = _OfflineTextProvider()
    import run_harness_eval as _rhe

    _rhe._EVALUATOR = _rhe.make_rule_evaluator()
    text = CachedTextGenerator(
        provider=text_provider, cache_dir=ROOT / "data" / "text_cache"
    )
    episodes, states, world, engine = generate_demo_cases(ROOT, MASTER_SEED, text)

    working = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    special_rows = holdout_cases(world, engine, MASTER_SEED, 2) + (
        missing_document_cases(
            working.filter(
                (pl.col("payer_id") == "meridian") & (pl.col("touch_seq") == 0)
            ).to_dicts()[0],
            1,
        )
    )
    special = pl.DataFrame(
        [{c: r.get(c) for c in working.columns} for r in special_rows],
        schema_overrides=working.schema,
    )
    all_states = pl.concat(
        [states.select(working.columns), special], how="vertical_relaxed"
    )
    from cardessa.features_kept import build_kept_features
    frame, _kept_names, _, _ = build_kept_features(
        ROOT, pl.concat([working, all_states], how="vertical_relaxed")
    )
    names = rt.feature_names
    demo_ids = sorted(all_states["state_id"].to_list())
    feats = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame.filter(pl.col("state_id").is_in(demo_ids)).to_dicts()
    }
    rows = {r["state_id"]: r for r in all_states.to_dicts()}
    scaled = rt.scaler.transform(
        np.array([[feats[s][n] for n in names] for s in demo_ids])
    )
    scaled_of = dict(zip(demo_ids, scaled, strict=True))
    episode_of = {e["episode_id"]: e for e in episodes.to_dicts()}

    traces, artifacts, bundles = {}, {}, {}

    # cases resolve in a background thread so the UI serves immediately;
    # cached cases land in seconds, uncached ones trickle in as the live
    # model answers them (each answer is cached for the next boot)
    def compute_one_case(state_id: str) -> None:
        try:
            provider.begin_call_capture()
            trace = resolve_and_run(
                rt, skills, harness, provider, rows[state_id],
                feats[state_id], scaled_of[state_id],
            )
            usage = provider.take_call_capture()
            if usage and usage["calls"]:
                STATE.setdefault("writer_usage", {})[state_id] = usage
            row = rows[state_id]
            if row["episode_id"] in episode_of:
                bundle = bundle_for_state(
                    world, engine, row, episode_of[row["episode_id"]]
                )
                bundles[state_id] = bundle
                if (
                    trace.status == "executed" and trace.final
                    and trace.final.action_id in TOOLS
                ):
                    artifacts[state_id] = TOOLS[trace.final.action_id](
                        trace.final.payload, bundle
                    )
            # trace assigned last: a case never appears without its artifact
            traces[state_id] = trace
        except Exception as exc:  # noqa: BLE001 - keep serving the rest
            STATE.setdefault("failed", {})[state_id] = repr(exc)

    def compute_cases() -> None:
        # 6-wide like the exam pipeline: cached cases land instantly either
        # way; fresh ones stop being a single-file queue
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(compute_one_case, demo_ids))

    # neighbour lookups: a small fact index over the historical working states
    working_index = {
        r["state_id"]: {
            "payer_id": r["payer_id"], "carc": r["carc"] or None, "cpt": r["cpt"],
            "balance": r["balance"], "touch": r["touch_seq"],
            "days_since_service": r["days_since_service"],
        }
        for r in working.select(
            "state_id", "payer_id", "carc", "cpt", "balance", "touch_seq",
            "days_since_service",
        ).to_dicts()
    }

    universe = pl.read_parquet(ROOT / "basis" / "universe.parquet")
    STATE.update(
        rt=rt, harness=harness, skills=skills, engine=engine, world=world,
        rows=rows, feats=feats, traces=traces, artifacts=artifacts,
        bundles=bundles, tribes={t.tribe_id: t for t in rt.tribes},
        universe=universe,
        adjudications={}, human_queue={}, working_index=working_index,
        working_columns=working.columns, working_schema=working.schema,
        textgen=text, episode_of=episode_of, provider=provider,
        universe_map=build_universe_map(
            rt, universe, working_index, demo_ids,
            {sid: label_coordinate(
                rt, route_case(rt, rows[sid], feats[sid])[0] or {}
            ) for sid in demo_ids},
        ),
    )

    def compute_all() -> None:
        compute_cases()
        try:
            compute_rollouts(world, engine, rt, working)
        except Exception as exc:  # noqa: BLE001 - scoreboard is optional
            STATE["rollout_error"] = repr(exc)
        # llm lane BEFORE replays: the claims page completes sooner; replay
        # moments are deep-dive material and can fill last
        try:
            compute_llm_lane(world, engine)
        except Exception as exc:  # noqa: BLE001 - the llm lane is optional
            STATE["llm_error"] = repr(exc)
        try:
            compute_llm_paperwork(world, engine)
        except Exception as exc:  # noqa: BLE001 - lane paperwork is optional
            STATE["llm_paperwork_error"] = repr(exc)
        try:
            compute_replay_traces(world, engine, rt, working)
            _install_system_tokens()
        except Exception as exc:  # noqa: BLE001 - replay pages are optional
            STATE["replay_error"] = repr(exc)

    threading.Thread(target=compute_all, daemon=True).start()


def _before_facts(case) -> dict:
    """env.apply MUTATES the case in place - capture the comparison facts
    before applying or every before/after diff is vacuous."""
    return {
        "touch": case.touch_seq,
        "collected_payer": case.collected_payer,
        "collected_patient": case.collected_patient,
        "carc": case.carc,
        "auth_ref": case.auth_ref,
    }


_NO_CHANGE_ANSWER = {
    "request_retro_auth": "authorization denied",
    "appeal_with_necessity": "appeal denied",
    "request_peer_to_peer": "denial upheld",
    "provide_requested_info": "info still missing",
}


def _step_note(before: dict, after, action: str) -> str:
    """One short line: what the payer's world answered to this move."""
    if after.collected_payer > before["collected_payer"]:
        who = "secondary paid" if action == "bill_secondary" else "payer paid"
        return f"{who} ${after.collected_payer - before['collected_payer']:,.2f}"
    if after.collected_patient > before["collected_patient"]:
        paid = after.collected_patient - before["collected_patient"]
        # money past the contractual share never counts (outcome v2):
        # balance-billing a failed claim must not read as a green win
        if after.collected_patient > after.contractual_share + 0.005:
            return f"balance-billed ${paid:,.2f} - not counted"
        return f"patient paid ${paid:,.2f}"
    if after.auth_ref and not before["auth_ref"]:
        return "authorization granted"
    # bill_secondary never produces a denial in this world; a standing CARC
    # on it would mislabel a paid-with-secondary close (audit F1, latent)
    if action in (
        "submit_clean", "submit_with_records", "correct_and_resubmit",
    ) and after.carc:
        return f"denied {after.carc}"
    if (after.carc or None) != (before["carc"] or None) and after.carc:
        return f"denied {after.carc}"
    if before["carc"] and not after.carc:
        return "denial cleared"
    if action == "bill_patient":
        # reached only when no patient money arrived: say so plainly
        return "patient didn't pay"
    if after.terminal:
        if (after.resolution == "paid_with_secondary"
                and after.collected_payer <= before["collected_payer"]):
            return "secondary had nothing to pay - claim was still denied"
        return (after.resolution or "closed").replace("_", " ")
    return _NO_CHANGE_ANSWER.get(action, "no response")


def compute_rollouts(world, engine, rt, working) -> None:
    """Play every demo claim TWICE with identical engine randomness - once
    under the simulated-human (persona) policy, once under the system policy
    (escalations fall back to the human, as in the v0.3 rollout) - and score
    both with the signed-off outcome-v2 measurement: dollars collected,
    ledger mistakes, touches. Cached on disk; the rollout is deterministic."""
    from policy_rollout import CONFIDENCE_FLOOR, case_view, outcome_v2
    from cardessa.corpus import CardessaEnvironment, state_tabular
    from cardessa.features_kept import build_kept_features
    from cardessa.harness import route_case
    from cardessa.personas import choose_action

    episode_ids = sorted(
        {r["episode_id"] for r in STATE["rows"].values()
         if r["episode_id"].startswith("ep-")}
    )
    # v0.6: the scoreboard depends on the decision field. The kernel field
    # keeps its own cache so the v0.4 (model) scoreboard is never overwritten
    # and a stale cache can never be served for the other field.
    field_tag = getattr(STATE["rt"], "decision_field", "model")
    cache = ROOT / "data" / (
        "ui_rollout.json" if field_tag == "model" else f"ui_rollout_{field_tag}.json"
    )
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if (
            data.get("metric") == "v2-cap-total-4"
            and data.get("decision_field", "model") == field_tag
            and sorted(data.get("claims", {})) == episode_ids
        ):
            STATE["scoreboard"] = data["scoreboard"]
            STATE["claims"] = data["claims"]
            return

    indices = [int(e.removeprefix("ep-")) for e in episode_ids]
    lanes: dict[str, dict] = {}
    for policy in ("human", "system"):
        env = CardessaEnvironment(world, engine, MASTER_SEED)
        live = {}
        for index in indices:
            case = env.reset(index)
            live[case.episode_id] = {"case": case, "pending": [], "steps": []}
        step = 0
        while live and step < 8:
            step += 1
            snap_rows = []
            for episode_id, slot in live.items():
                case = slot["case"]
                payer = engine.payers[case.payer_id]
                row = state_tabular(case, payer, 0.0)
                row.update({
                    "episode_id": episode_id,
                    "state_id": f"{episode_id}-r{case.touch_seq}",
                    "touch_seq": case.touch_seq, "action_raw": "",
                    "clinical_indication_text": "",
                    "payer_correspondence_text": "",
                })
                snap_rows.append(row)
            feats = {}
            if policy == "system":
                snaps = pl.DataFrame(
                    [{c: r.get(c) for c in working.columns} for r in snap_rows],
                    schema_overrides=working.schema,
                )
                frame, _kn, _, _ = build_kept_features(
                    ROOT, pl.concat([working, snaps], how="vertical_relaxed")
                )
                names = rt.feature_names
                feats = {
                    r["state_id"]: {n: r[n] for n in names}
                    for r in frame.filter(
                        pl.col("state_id").is_in(snaps["state_id"].to_list())
                    ).to_dicts()
                }
            finished = []
            for row in snap_rows:
                episode_id = row["episode_id"]
                slot = live[episode_id]
                case = slot["case"]
                payer = engine.payers[case.payer_id]
                by = "human"
                detail = None
                if policy == "system":
                    scores, known, forced, _amb = route_case(
                        rt, row, feats[row["state_id"]]
                    )
                    action = forced
                    by = "system"
                    if action is None and known and scores:
                        best_action, best = max(scores.items(), key=lambda kv: kv[1])
                        if best >= CONFIDENCE_FLOOR:
                            action = best_action
                        else:  # v0.4: close out by expected value
                            action = close_out_choice(
                                expected_values(row, scores), scores
                            )
                    if action is None:  # escalated: the human works this touch
                        by = "handed to human"
                        action = choose_action(
                            case.persona, case_view(case, payer),
                            MASTER_SEED, case.episode_id,
                        )
                    # what Ammonix saw at this moment - lets the UI show the
                    # same "details" for its moves as for the human's
                    detail = {
                        "scores": {
                            a: round(v, 4)
                            for a, v in sorted(
                                scores.items(), key=lambda kv: -kv[1]
                            )
                        } if scores else {},
                        "known_payer": bool(known),
                        "forced": forced,
                        "facts": {
                            "carc": row["carc"] or None,
                            "days_since_service": row["days_since_service"],
                            "days_to_filing_deadline": row[
                                "days_to_filing_deadline"
                            ],
                            "auth_status": row["auth_status"],
                            "balance": row["balance"],
                        },
                    }
                else:
                    action = choose_action(
                        case.persona, case_view(case, payer),
                        MASTER_SEED, case.episode_id,
                    )
                snapshot = state_tabular(case, payer, 0.0)
                slot["pending"].append((snapshot, action))
                before = _before_facts(case)
                after = env.apply(case, action)
                slot["steps"].append({
                    "touch": before["touch"], "action": action, "by": by,
                    "note": _step_note(before, after, action),
                    **({"detail": detail} if detail else {}),
                })
                slot["case"] = after
                if after.terminal:
                    success, score, n_mistakes = outcome_v2(
                        env, after, slot["pending"]
                    )
                    # honest money: patient dollars count up to the contractual
                    # share (balance-billing must not look good); payer dollars
                    # up to whatever of the allowed amount the patient did not
                    # cover - a SECONDARY payer legitimately pays the patient's
                    # share (ep-05004 bug: the old allowed-minus-share cap
                    # zeroed real secondary payments), while total counted can
                    # never exceed the claim's allowed amount, so the engine's
                    # duplicate-payment flaw still cannot decide verdicts.
                    patient_valid = round(
                        min(after.collected_patient, after.contractual_share), 2
                    )
                    payer_valid = round(
                        min(after.collected_payer, after.allowed - patient_valid), 2
                    )
                    lanes.setdefault(episode_id, {})[policy] = {
                        "steps": slot["steps"],
                        "resolution": (after.resolution or "closed"),
                        "collected": round(payer_valid + patient_valid, 2),
                        "payer_collected": round(after.collected_payer, 2),
                        "payer_counted": payer_valid,
                        "patient_collected": round(after.collected_patient, 2),
                        "patient_counted": patient_valid,
                        "touches": len(after.action_history),
                        "mistakes": n_mistakes,
                        "success": bool(success),
                        "autonomous": policy == "system"
                        and all(st["by"] == "system" for st in slot["steps"]),
                    }
                    finished.append(episode_id)
            for episode_id in finished:
                del live[episode_id]

    claims = {}
    for eid in episode_ids:
        h, s = lanes[eid]["human"], lanes[eid]["system"]
        delta = round(s["collected"] - h["collected"], 2)
        # money decides; equal money is a win only on fewer process
        # MISTAKES (a real quality gap). A shorter road to the same
        # dollars - e.g. writing off sooner - is not a win (owner rule).
        if abs(delta) > 0.005:
            verdict = "ammonix" if delta > 0 else "human"
        elif s["mistakes"] != h["mistakes"]:
            verdict = "ammonix" if s["mistakes"] < h["mistakes"] else "human"
        else:
            verdict = "same"
        claims[eid] = {"human": h, "system": s, "verdict": verdict, "delta": delta}

    n = len(claims)
    scoreboard = {
        "claims": n,
        "system_collected": round(sum(c["system"]["collected"] for c in claims.values()), 2),
        "human_collected": round(sum(c["human"]["collected"] for c in claims.values()), 2),
        "ammonix_wins": sum(1 for c in claims.values() if c["verdict"] == "ammonix"),
        "human_wins": sum(1 for c in claims.values() if c["verdict"] == "human"),
        "same": sum(1 for c in claims.values() if c["verdict"] == "same"),
        "system_mistake_claims": sum(
            1 for c in claims.values() if c["system"]["mistakes"] > 0
        ),
        "human_mistake_claims": sum(
            1 for c in claims.values() if c["human"]["mistakes"] > 0
        ),
        "system_autonomous": sum(
            1 for c in claims.values() if c["system"]["autonomous"]
        ),
        "note": (
            "identical claims, identical payer randomness; system escalations "
            "are worked by the human policy; dollars = payer + patient up to "
            "the contractual share, payer capped at allowed minus that share "
            "(outcome v2; duplicate-payment exploit dollars do not count)"
        ),
    }
    cache.write_text(
        json.dumps(
            {"metric": "v2-cap-total-4", "decision_field": field_tag,
             "claims": claims, "scoreboard": scoreboard}
        ),
        encoding="utf-8", newline="\n",
    )
    STATE["scoreboard"] = scoreboard
    STATE["claims"] = claims



def _capped_lane(after, steps, pending, env, by_key) -> dict:
    """Assemble a lane dict with the shared outcome-v2 capped money rule."""
    from policy_rollout import outcome_v2

    success, _score, n_mistakes = outcome_v2(env, after, pending)
    patient_valid = round(min(after.collected_patient, after.contractual_share), 2)
    payer_valid = round(min(after.collected_payer, after.allowed - patient_valid), 2)
    return {
        "steps": steps,
        "resolution": (after.resolution or "closed"),
        "collected": round(payer_valid + patient_valid, 2),
        "payer_collected": round(after.collected_payer, 2),
        "payer_counted": payer_valid,
        "patient_collected": round(after.collected_patient, 2),
        "patient_counted": patient_valid,
        "touches": len(after.action_history),
        "mistakes": n_mistakes,
        "success": bool(success),
        "autonomous": all(st["by"] == by_key for st in steps),
    }


def compute_llm_lane(world, engine) -> None:
    """Third comparison arm: the Agentic_Baseline LLM agent (same pinned 27B,
    briefing + guardrails + one retry + explicit escalate; escalations worked
    by the persona policy) playing the SAME 200 demo claims with identical
    dice. Reuses the baseline repo's policy verbatim - architecture
    comparison, not model comparison. Cached on disk; deterministic."""
    from concurrent.futures import ThreadPoolExecutor

    from policy_rollout import case_view
    from cardessa.corpus import CardessaEnvironment, state_tabular
    from cardessa.personas import choose_action

    claims = STATE.get("claims") or {}
    if not claims:
        return
    episode_ids = sorted(claims)
    # AMMONIX_LLM_LANE swaps the lane's model (demo only): "opus" = Claude
    # Opus 4.8, "sol" = GPT-5.6 Sol, "kimi" = Kimi K3, default "qwen" = the
    # pinned 27B. Each
    # variant has its own lane cache + decision cache, so the pinned-Qwen lane
    # and its artefacts stay untouched and switching is instant once computed.
    # With opus/sol the lane is a model+architecture comparison, unlike the
    # paper's pre-registered same-model one.
    lane_model = os.environ.get("AMMONIX_LLM_LANE", "qwen").lower()
    _frontier_note = ("A frontier model given the whole job through a prompt: "
                      "it reads each claim and decides every move itself.")
    if lane_model == "opus":
        # AMMONIX_OPUS_MODEL selects the Claude model (default claude-opus-4-8);
        # each model keeps its own lane cache and tag so nothing is shared
        _opus = os.environ.get("AMMONIX_OPUS_MODEL", "claude-opus-4-8")
        _suffix = "" if _opus == "claude-opus-4-8" else "_" + _opus.replace("claude-", "").replace("-", "")
        cache_file = ROOT / "data" / f"ui_llm_lane_opus{_suffix}.json"
        tag = f"llm-lane-opus-2{_suffix}"  # strong config: adaptive thinking, 2 retries
        STATE["llm_model_label"] = {"claude-opus-4-8": "Opus 4.8", "claude-opus-5": "Opus 5"}.get(_opus, _opus)
        STATE["llm_lane_note"] = _frontier_note
    elif lane_model == "sol":
        cache_file = ROOT / "data" / "ui_llm_lane_sol.json"
        tag = "llm-lane-sol-1"  # strong config: default reasoning, 2 retries
        STATE["llm_model_label"] = "GPT-5.6 Sol"
        STATE["llm_lane_note"] = _frontier_note
    elif lane_model == "kimi":
        cache_file = ROOT / "data" / "ui_llm_lane_kimi.json"
        tag = "llm-lane-kimi-1"  # strong config: default (max) reasoning, 2 retries
        STATE["llm_model_label"] = "Kimi K3"
        STATE["llm_lane_note"] = _frontier_note
    elif lane_model == "gpt6":
        cache_file = ROOT / "data" / "ui_llm_lane_gpt6.json"
        # -3: per-move reasoning + usage recorded, lane totals nested as
        # {decide, write} (replays from the decision cache, so the bump
        # costs no API calls and changes no move)
        tag = "llm-lane-gpt6-3"  # strong config: default reasoning, 2 retries
        STATE["llm_model_label"] = "GPT-6"
        STATE["llm_lane_note"] = _frontier_note
    else:
        cache_file = ROOT / "data" / "ui_llm_lane.json"
        tag = "llm-lane-1"
        STATE["llm_model_label"] = "Qwen 27B"
        STATE["llm_lane_note"] = (
            "The same language model Ammonix uses to write its paperwork, "
            "here doing the whole job alone: it reads each claim and decides "
            "every move from a prompt. Same model, different architecture.")
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if data.get("tag") == tag and sorted(data["lanes"]) == episode_ids:
            _install_llm_lane(data["lanes"])
            return

    # the baseline repo lives inside this repo since the merge; the external
    # Agentic_Baseline location is kept for historical checkouts
    agentic_root = next(
        (c for c in ([ROOT / "agentic_baseline"]
                     + [p / "Agentic_Baseline" for p in ROOT.parents])
         if c.is_dir()),
        ROOT.parents[2] / "Agentic_Baseline",
    )
    if not agentic_root.is_dir():
        STATE["llm_error"] = f"baseline repo not found at {agentic_root}"
        return
    if str(agentic_root) not in sys.path:
        sys.path.insert(0, str(agentic_root))
    if lane_model == "opus":
        from ui.opus_llm import OpusAgentLLM, build_strong_policy

        policy = build_strong_policy(
            OpusAgentLLM(cache_dir=ROOT / "data" / "opus_decisions")
        )
    elif lane_model == "sol":
        from ui.openai_llm import SolAgentLLM
        from ui.opus_llm import build_strong_policy

        policy = build_strong_policy(
            SolAgentLLM(cache_dir=ROOT / "data" / "sol_decisions")
        )
    elif lane_model == "kimi":
        from ui.kimi_llm import KimiAgentLLM
        from ui.opus_llm import build_strong_policy

        policy = build_strong_policy(
            KimiAgentLLM(cache_dir=ROOT / "data" / "kimi_decisions")
        )
    elif lane_model == "gpt6":
        from ui.openai_llm import SolAgentLLM
        from ui.opus_llm import build_strong_policy

        policy = build_strong_policy(
            SolAgentLLM(cache_dir=ROOT / "data" / "gpt6_ui_decisions",
                        model_id="gpt-6-astra")
        )
    else:
        from agentic.llm import AgentLLM
        from agentic.policy import AgenticPolicy

        policy = AgenticPolicy(
            AgentLLM(cache_dir=agentic_root / "cache" / "decisions")
        )
    indices = [int(e.removeprefix("ep-")) for e in episode_ids]
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    live = {}
    for index in indices:
        case = env.reset(index)
        live[case.episode_id] = {"case": case, "pending": [], "steps": []}
    STATE["llm_progress"] = {"done": 0, "total": len(episode_ids)}
    lanes: dict[str, dict] = {}
    step = 0
    while live and step < 8:
        step += 1
        snap_rows = []
        for episode_id, slot in live.items():
            case = slot["case"]
            payer = engine.payers[case.payer_id]
            row = state_tabular(case, payer, 0.0)
            row.update({
                "episode_id": episode_id,
                "state_id": f"{episode_id}-llm{case.touch_seq}",
                "touch_seq": case.touch_seq, "action_raw": "",
                "clinical_indication_text": "", "payer_correspondence_text": "",
            })
            snap_rows.append(row)

        def decide(row):
            case = live[row["episode_id"]]["case"]
            legal = env.legal_actions(case)
            # per-move metering: bracket the decide so every completion it
            # issues (retries included, cache hits included) lands on this move
            begin = getattr(policy.llm, "begin_call_capture", None)
            if begin:
                begin()
            try:
                d = policy.decide(row, legal)
                action, reasoning = d.action, d.reasoning
            except Exception:  # noqa: BLE001 - dead call: human works the touch
                policy.decision_errors += 1
                action, reasoning = None, ""
            usage = policy.llm.take_call_capture() if begin else None
            return row["episode_id"], (action, reasoning, usage)

        with ThreadPoolExecutor(max_workers=6) as pool:
            chosen = dict(pool.map(decide, snap_rows))

        finished = []
        for row in snap_rows:
            episode_id = row["episode_id"]
            slot = live[episode_id]
            case = slot["case"]
            payer = engine.payers[case.payer_id]
            action, reasoning, usage = chosen[episode_id]
            by = "llm"
            if action is None:
                by = "handed to human"
                action = choose_action(
                    case.persona, case_view(case, payer),
                    MASTER_SEED, case.episode_id,
                )
            slot["pending"].append((state_tabular(case, payer, 0.0), action))
            before = _before_facts(case)
            after = env.apply(case, action)
            slot["steps"].append({
                "touch": before["touch"], "action": action, "by": by,
                "note": _step_note(before, after, action),
                # what the model said, and what saying it cost - the demo's
                # per-move answer to "what does deciding cost in tokens"
                **({"reasoning": reasoning[:1200]} if reasoning else {}),
                **({"tokens": usage} if usage and usage.get("calls") else {}),
            })
            slot["case"] = after
            if after.terminal:
                lanes[episode_id] = _capped_lane(
                    after, slot["steps"], slot["pending"], env, "llm"
                )
                totals = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
                for st in slot["steps"]:
                    for k in totals:
                        totals[k] += (st.get("tokens") or {}).get(k, 0)
                if totals["calls"]:
                    # decide is metered here; write is filled in by
                    # compute_llm_paperwork once the lane's filings exist
                    lanes[episode_id]["tokens"] = {"decide": totals}
                finished.append(episode_id)
                STATE["llm_progress"]["done"] += 1
        for episode_id in finished:
            del live[episode_id]

    cache_file.write_text(
        json.dumps({"tag": tag, "lanes": lanes}),
        encoding="utf-8", newline="\n",
    )
    _install_llm_lane(lanes)


def _install_llm_lane(lanes: dict) -> None:
    claims = STATE["claims"]
    for eid, lane in lanes.items():
        if eid in claims:
            claims[eid]["llm"] = lane
    # scoreboard: totals + the substance-rule comparison vs Ammonix
    a_wins = l_wins = same = 0
    for eid, c in claims.items():
        lane = c.get("llm")
        if not lane:
            continue
        delta = round(c["system"]["collected"] - lane["collected"], 2)
        if abs(delta) > 0.005:
            winner = "a" if delta > 0 else "l"
        elif c["system"]["mistakes"] != lane["mistakes"]:
            winner = "a" if c["system"]["mistakes"] < lane["mistakes"] else "l"
        else:
            winner = "s"
        if winner == "a":
            a_wins += 1
        elif winner == "l":
            l_wins += 1
        else:
            same += 1
    tok = _llm_token_totals(claims)
    STATE["scoreboard"] = {
        **STATE["scoreboard"],
        "llm_collected": round(
            sum(c["llm"]["collected"] for c in claims.values() if c.get("llm")), 2
        ),
        "llm_mistake_claims": sum(
            1 for c in claims.values() if c.get("llm") and c["llm"]["mistakes"] > 0
        ),
        "ammonix_vs_llm": {"ammonix": a_wins, "llm": l_wins, "same": same},
        "llm_model_label": STATE.get("llm_model_label", "Qwen 27B"),
        "llm_lane_note": STATE.get("llm_lane_note", ""),
        **({"llm_tokens": tok} if tok["claims"] else {}),
    }
    STATE["llm_done"] = True


def _llm_token_totals(claims: dict) -> dict:
    """Scoreboard roll-up of the LLM lane's bill: deciding and writing
    summed separately over the claims that carry per-move usage (older
    lane caches predate the metering and simply do not show a number)."""
    tok = {
        "decide": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "write": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0},
        "claims": 0,
    }
    for c in claims.values():
        t = (c.get("llm") or {}).get("tokens")
        if not t:
            continue
        tok["claims"] += 1
        for part in ("decide", "write"):
            for k in ("prompt_tokens", "completion_tokens", "calls"):
                tok[part][k] += (t.get(part) or {}).get(k, 0)
    return tok


def compute_llm_paperwork(world, engine) -> None:
    """The lane's model files its own paperwork. For every move the GPT-6
    lane decided, the SAME writer harness Ammonix uses (M1 prompt, M2
    checks, provenance-sourced forms) runs with GPT-6 in the writer seat -
    the t103 protocol, on the demo claims. Every writer call is metered per
    move; results cached on disk so the public server replays keylessly."""
    from concurrent.futures import ThreadPoolExecutor

    from cardessa.corpus import CardessaEnvironment, state_tabular, state_text

    claims = STATE.get("claims") or {}
    if not claims or not STATE.get("llm_done"):
        return
    if os.environ.get("AMMONIX_LLM_LANE", "qwen").lower() != "gpt6":
        return  # only the GPT-6 lane has a writer seat wired for this
    episode_ids = sorted(e for e in claims if claims[e].get("llm"))
    if not episode_ids:
        return
    cache_file = ROOT / "data" / "ui_llm_paperwork_gpt6.json"
    tag = "llm-paperwork-gpt6-1"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if data.get("tag") == tag and sorted(data.get("episodes", [])) == episode_ids:
            _install_llm_paperwork(data["moves"])
            return

    from ui.openai_llm import SolAgentLLM

    # the same cache the t103 writer runs use: identical prompts replay free
    writer = SolAgentLLM(cache_dir=ROOT / "data" / "m1_gpt6",
                         model_id="gpt-6-astra")
    harness, skills, rt = STATE["harness"], STATE["skills"], STATE["rt"]
    evaluator = make_rule_evaluator()
    text = STATE["textgen"]
    episode_of = STATE["episode_of"]
    exec_skill = {
        s.scope.ref: s for s in skills
        if s.scope.level == "cluster" and s.kind == "execute"
    }
    STATE["llm_paperwork_progress"] = {"done": 0, "total": len(episode_ids)}
    moves: dict[str, dict] = {}

    def write_move(row: dict, action: str) -> dict | None:
        skill = exec_skill.get(action)
        if skill is None:
            return None  # close-out moves file nothing, same as the agent
        retrieval = RetrievalResult(
            case_id=row["state_id"], features={}, scores_cal={}, neighbours=[],
            tribe_id=f"{action}::cluster", recommended_action_id=action,
            ambiguous=False, skill_id=skill.skill_id,
            expected_result=skill.expected_result.model_dump(),
        )
        writer.begin_call_capture()
        trace = run_case(
            retrieval, skill, row, harness.m1_prompts[skill.skill_id],
            harness.m2_prompt, harness.llm,
            lambda p, s: writer.complete([{"role": "user", "content": p}], s),
            evaluator, rt.manifest.basis_id, harness.harness_id,
            prompt_fields=prompt_fields,
        )
        usage = writer.take_call_capture()
        entry: dict = {
            "action": action,
            "status": trace.status,
            "m2": [
                {"n": it.n, "passed": it.check.passed,
                 "failures": it.check.failures}
                for it in trace.iterations
            ],
            **({"tokens": usage} if usage and usage["calls"] else {}),
        }
        if trace.status == "executed" and trace.final:
            ep = episode_of.get(row["episode_id"])
            if action in TOOLS and ep is not None:
                art = TOOLS[action](
                    trace.final.payload, bundle_for_state(world, engine, row, ep)
                )
                entry["artifact"] = {
                    "kind": art["artifact"], "field_map": art["field_map"],
                    "gaps": art["gaps"],
                }
            else:  # no printable form for this action: show the filed fields
                entry["payload"] = trace.final.payload
        return entry

    def one_episode(episode_id: str) -> None:
        try:
            steps = claims[episode_id]["llm"]["steps"]
            env = CardessaEnvironment(world, engine, MASTER_SEED)
            case = env.reset(int(episode_id.removeprefix("ep-")))
            for st in steps:
                action = st["action"]
                if st.get("by") == "llm":
                    payer = engine.payers[case.payer_id]
                    row = state_tabular(case, payer, 0.0)
                    try:
                        texts = state_text(case, payer, text, MASTER_SEED)
                    except Exception:  # noqa: BLE001 - model offline: write from facts
                        texts = {}
                    row.update({
                        "episode_id": episode_id,
                        "state_id": f"{episode_id}-llm{case.touch_seq}",
                        "touch_seq": case.touch_seq, "action_raw": action,
                        "clinical_indication_text":
                            texts.get("clinical_indication_text") or "",
                        "payer_correspondence_text":
                            texts.get("payer_correspondence_text") or "",
                    })
                    entry = write_move(row, action)
                    if entry is not None:
                        moves[row["state_id"]] = entry
                case = env.apply(case, action)
        except Exception as exc:  # noqa: BLE001 - keep the rest of the lane
            STATE.setdefault("failed", {})[f"{episode_id}-llmpw"] = repr(exc)
        STATE["llm_paperwork_progress"]["done"] += 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one_episode, episode_ids))

    print(f"llm paperwork writer stats: {writer.stats()}", flush=True)
    cache_file.write_text(
        json.dumps({"tag": tag, "episodes": episode_ids, "moves": moves}),
        encoding="utf-8", newline="\n",
    )
    _install_llm_paperwork(moves)


def _install_llm_paperwork(moves: dict) -> None:
    claims = STATE["claims"]
    for eid, c in claims.items():
        lane = c.get("llm")
        if not lane:
            continue
        wt = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        wrote = False
        for st in lane["steps"]:
            m = moves.get(f"{eid}-llm{st['touch']}")
            if not m:
                continue
            if m.get("artifact"):
                st["paperwork"] = m["artifact"]
            if m.get("payload") is not None:
                st["filed_payload"] = m["payload"]
            if m.get("m2"):
                st["checks"] = m["m2"]
            st["write_status"] = m.get("status")
            t = m.get("tokens")
            if t:
                wrote = True
                st["write_tokens"] = t
                for k in wt:
                    wt[k] += t.get(k, 0)
        if wrote:
            lane.setdefault("tokens", {})["write"] = wt
    STATE["llm_paperwork"] = moves
    if STATE.get("scoreboard"):
        STATE["scoreboard"] = {
            **STATE["scoreboard"], "llm_tokens": _llm_token_totals(claims),
        }


def _install_system_tokens() -> None:
    """Ammonix's bill: the local writer's tokens over each claim's replay
    moments. Deciding is the calibrated model - there is no LLM call to
    meter, which is the point the scoreboard makes."""
    claims = STATE.get("claims") or {}
    usage = STATE.get("writer_usage") or {}
    total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    n = 0
    for eid, c in claims.items():
        wt = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        found = False
        for st in c["system"]["steps"]:
            u = usage.get(f"{eid}-r{st['touch']}")
            if u:
                found = True
                st["write_tokens"] = u
                for k in wt:
                    wt[k] += u.get(k, 0)
        if found:
            c["system"]["tokens"] = {"write": wt}
            n += 1
            for k in total:
                total[k] += wt[k]
    if n and STATE.get("scoreboard"):
        STATE["scoreboard"] = {
            **STATE["scoreboard"],
            "system_tokens": {"write": total, "claims": n,
                              "writer_model": WRITER_MODEL},
        }


def compute_replay_traces(world, engine, rt, working) -> None:
    """Full trace pipeline for Ammonix's counterfactual moments. Each system-
    lane replay state is rebuilt with the same dice, its letters generated,
    and the real resolve pipeline (M1, checks, paperwork) run - so case.html
    works for Ammonix moves exactly like for recorded moments. Registered
    progressively; M1/text caches make later boots near-instant."""
    from concurrent.futures import ThreadPoolExecutor

    from cardessa.corpus import CardessaEnvironment, state_tabular, state_text
    from cardessa.features_kept import build_kept_features

    claims = STATE.get("claims") or {}
    if not claims:
        return
    text = STATE["textgen"]

    def replay_rows(episode_id: str) -> list[dict]:
        steps = claims[episode_id]["system"]["steps"]
        env = CardessaEnvironment(world, engine, MASTER_SEED)
        case = env.reset(int(episode_id.removeprefix("ep-")))
        out = []
        for st in steps:
            action = st["action"]
            payer = engine.payers[case.payer_id]
            row = state_tabular(case, payer, 0.0)
            try:
                texts = state_text(case, payer, text, MASTER_SEED)
            except Exception:  # noqa: BLE001 - model offline: facts still shown
                texts = {}
            row.update({
                "episode_id": episode_id,
                "state_id": f"{episode_id}-r{case.touch_seq}",
                "touch_seq": case.touch_seq, "action_raw": action,
                "_by": st.get("by"),
                "clinical_indication_text":
                    texts.get("clinical_indication_text") or "",
                "payer_correspondence_text":
                    texts.get("payer_correspondence_text") or "",
                "_replay": True,
            })
            out.append(row)
            case = env.apply(case, action)
        return out

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rows_ in pool.map(replay_rows, sorted(claims)):
            all_rows.extend(rows_)
    todo = [r for r in all_rows if r["state_id"] not in STATE["traces"]]
    STATE["replay_progress"] = {"done": 0, "total": len(todo)}
    if not todo:
        return

    cols = STATE["working_columns"]
    snaps = pl.DataFrame(
        [{c: r.get(c) for c in cols} for r in todo],
        schema_overrides=STATE["working_schema"],
    )
    frame, _n, _, _ = build_kept_features(
        ROOT, pl.concat([working, snaps], how="vertical_relaxed")
    )
    names = rt.feature_names
    feats_of = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame.filter(
            pl.col("state_id").is_in(snaps["state_id"].to_list())
        ).to_dicts()
    }
    episode_of = STATE["episode_of"]

    def one_replay(row: dict) -> None:
        sid = row["state_id"]
        try:
            f = feats_of[sid]
            scaled = rt.scaler.transform(np.array([[f[n] for n in names]]))[0]
            STATE["provider"].begin_call_capture()
            trace = resolve_and_run(
                rt, STATE["skills"], STATE["harness"], STATE["provider"],
                row, f, scaled,
            )
            usage = STATE["provider"].take_call_capture()
            if usage and usage["calls"]:
                STATE.setdefault("writer_usage", {})[sid] = usage
            STATE["rows"][sid] = row
            STATE["feats"][sid] = f
            ep = episode_of.get(row["episode_id"])
            if ep is not None:
                bundle = bundle_for_state(world, engine, row, ep)
                STATE["bundles"][sid] = bundle
                if (
                    trace.status == "executed" and trace.final
                    and trace.final.action_id in TOOLS
                ):
                    STATE["artifacts"][sid] = TOOLS[trace.final.action_id](
                        trace.final.payload, bundle
                    )
            STATE["traces"][sid] = trace
        except Exception as exc:  # noqa: BLE001 - keep going, page per page
            STATE.setdefault("failed", {})[sid] = repr(exc)
        STATE["replay_progress"]["done"] += 1

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(one_replay, todo))


ACTION_FAMILY = {
    "submit_clean": "submit", "submit_with_records": "submit",
    "correct_and_resubmit": "submit", "bill_secondary": "submit",
    "request_retro_auth": "contest", "appeal_with_necessity": "contest",
    "request_peer_to_peer": "contest", "provide_requested_info": "contest",
    "bill_patient": "close", "write_off": "close",
}


def build_universe_map(rt, universe, working_index, demo_ids, u_of):
    """3D map of the basis index space. UMAP embedding (nonlinear, so real
    clusters separate into islands), computed once and cached on disk; tribe
    centroids and live demo cases are placed at the weighted centre of their
    nearest historical neighbours in the original feature space."""
    fit_x = rt.index._fit_X  # noqa: SLF001 - the scaled basis vectors
    ids_list = list(rt.index_state_ids)
    cache = ROOT / "data" / "universe_embed3d.parquet"
    emb = None
    if cache.is_file():
        df = pl.read_parquet(cache)
        if df["state_id"].to_list() == ids_list:
            emb = df.select("x", "y", "z").to_numpy()
    if emb is None:
        from umap import UMAP

        emb = UMAP(
            n_components=3, n_neighbors=12, min_dist=0.9, spread=1.2,
            random_state=0
        ).fit_transform(fit_x)
        pl.DataFrame(
            {"state_id": ids_list, "x": emb[:, 0], "y": emb[:, 1], "z": emb[:, 2]}
        ).write_parquet(cache)
    pos = {
        sid: (float(emb[i][0]), float(emb[i][1]), float(emb[i][2]))
        for i, sid in enumerate(ids_list)
    }

    def place(vec):
        dist, nn = rt.index.kneighbors(np.array([vec]), n_neighbors=8)
        w = 1.0 / (dist[0] + 1e-6)
        p = (emb[nn[0]] * w[:, None]).sum(axis=0) / w.sum()
        return float(p[0]), float(p[1]), float(p[2])

    STATE["universe_place"] = place  # reused to place Ammonix replay moments

    tribe_pts = []
    tribe_index: dict = {}       # tribe_id -> index into tribe_pts
    for t in rt.tribes:
        try:
            centroid = (
                [t.centroid[a] for a in (rt.u_order or sorted(t.centroid))]
                if isinstance(t.centroid, dict) else list(t.centroid)
            )
            cx, cy, cz = place(centroid)
        except Exception:  # noqa: BLE001 - centroid space mismatch: skip label
            continue
        tribe_index[t.tribe_id] = len(tribe_pts)
        tribe_pts.append({
            "tribe_id": t.tribe_id, "action": t.action_id,
            "x": round(cx, 2), "y": round(cy, 2), "z": round(cz, 2),
            "members": t.member_count,
            "success_rate": round(t.stats.success_rate, 2),
        })

    actions = sorted(universe["true_action_id"].unique().to_list())
    a_idx = {a: i for i, a in enumerate(actions)}
    payers, carcs, p_idx, c_idx = [], [], {}, {}
    ids, pts = [], []
    for r in universe.select(
        "state_id", "true_action_id", "outcome_success", "tribe_id"
    ).to_dicts():
        sid = r["state_id"]
        if sid not in pos:
            continue
        facts = working_index.get(sid) or {}
        payer = facts.get("payer_id") or "?"
        carc = facts.get("carc") or ""
        if payer not in p_idx:
            p_idx[payer] = len(payers)
            payers.append(payer)
        if carc not in c_idx:
            c_idx[carc] = len(carcs)
            carcs.append(carc)
        x, y, z = pos[sid]
        ids.append(sid)
        pts.append([
            round(x, 2), round(y, 2), round(z, 2), a_idx[r["true_action_id"]],
            1 if r["outcome_success"] else 0, p_idx[payer], c_idx[carc],
            # the record's REAL neighborhood, straight from the universe
            tribe_index.get(r["tribe_id"], -1),
        ])

    live = [
        {"id": sid, "x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}
        for sid, (x, y, z) in ((sid, place(u_of[sid])) for sid in demo_ids)
    ]
    return {
        "actions": actions, "families": ACTION_FAMILY,
        "payers": payers, "carcs": carcs,
        "ids": ids, "points": pts, "tribes": tribe_pts, "live": live,
    }


def pick_story_cases(rows, traces) -> list[dict]:
    def find(predicate, title, note, expected):
        step = {"title": title, "claim": note, "expected": expected}
        for state_id, trace in traces.items():
            # counterfactual replay states are not recorded story material
            if rows[state_id].get("_replay"):
                continue
            if predicate(rows[state_id], trace):
                return {"state_id": state_id, **step}
        return {"state_id": None, **step}

    return [
        find(
            lambda r, t: r["payer_id"] == "meridian" and t.status == "executed"
            and not r["carc"] and t.retrieval.recommended_action_id == "submit_clean"
            and not t.retrieval.ambiguous,
            "Straight through",
            "A clean claim: filled form, checks green, sent.",
            "executed",
        ),
        find(
            lambda r, t: r["payer_id"] == "meridian" and r["auth_required"]
            and r["auth_status"] == "missing"
            and r["days_since_service"] <= r["retro_window_days"]
            and t.retrieval.recommended_action_id == "request_retro_auth",
            "Knows the payer's rules",
            "No authorization on file, window still open: it requests retro "
            "auth first instead of submitting into a certain denial.",
            "recommend:request_retro_auth",
        ),
        find(
            lambda r, t: r["payer_id"] == "silverbridge" and r["carc"] == "CO-197"
            and t.retrieval.recommended_action_id != "request_retro_auth",
            "Same denial, different payer",
            "This payer never grants retro auth — so it appeals instead.",
            "recommend:not_retro",
        ),
        find(
            lambda r, t: r["payer_id"] == "cornerstone" and r["carc"] == "CO-97"
            and t.retrieval.ambiguous,
            "Asks when it's a coin flip",
            "Two moves nearly tied: it asks instead of guessing.",
            "ask",
        ),
        find(
            lambda r, t: r["payer_id"] in ("granite", "pelican")
            and t.status == "escalated",
            "Admits what it hasn't seen",
            "A payer that was never in training: it hands off instead of pretending.",
            "escalated",
        ),
        find(
            lambda r, t: (r.get("payer_correspondence_text") or "")
            == MISSING_DOC_LETTER and t.status == "escalated",
            "Won't invent paperwork",
            "The letter demands a document that doesn't exist: it refuses to invent one.",
            "escalated",
        ),
    ]


def case_summary(state_id: str) -> dict:
    row = STATE["rows"][state_id]
    trace = STATE["traces"][state_id]
    adjudicated = STATE["adjudications"].get(state_id)
    if adjudicated:
        status = "adjudicated"
    elif state_id in STATE["human_queue"]:
        status = "sent_to_human"
    else:
        status = trace.status
    scores = sorted(trace.retrieval.scores_cal.values(), reverse=True)
    payer = STATE["engine"].payers.get(row["payer_id"])
    return {
        "state_id": state_id,
        "episode_id": row["episode_id"],
        "payer_id": row["payer_id"],
        "payer_name": payer.name if payer else row["payer_id"],
        "cpt": row["cpt"],
        "carc": row["carc"] or None,
        "balance": row["balance"],
        "attempt": int(row["touches_so_far"]),
        "persona": row["persona_id"],
        "triage": triage_of(trace),
        "skill_id": trace.retrieval.skill_id,
        "has_artifact": state_id in STATE["artifacts"],
        "status": status,
        "recommended": trace.retrieval.recommended_action_id,
        "ambiguous": trace.retrieval.ambiguous,
        "margin": round(scores[0] - scores[1], 4) if len(scores) >= 2 else None,
        "escalation_reason": trace.escalation.reason if trace.escalation else None,
    }


@app.get("/api/cases")
def cases() -> list[dict]:
    # only computed, recorded cases; Ammonix replay moments are reached
    # from their claim story, not this list
    return [
        case_summary(s) for s in sorted(STATE["rows"])
        if s in STATE["traces"] and not STATE["rows"][s].get("_replay")
    ]


def round_tried_features() -> dict[int, list[str]]:
    """The concrete feature names each tokenizer round PROPOSED, computed
    from the real builders (round deltas), never hardcoded."""
    from cardessa.extract import INDICATION_BOOLEANS, LETTER_BOOLEANS
    from cardessa.features import build_round1_features
    from cardessa.features_rounds import build_with_blocks

    sample = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    ).head(500)
    _, s1 = build_round1_features(sample)
    n1 = [s.name for s in s1]
    _, s3 = build_with_blocks(sample, {"derived"})
    n3 = [s.name for s in s3]
    _, s4 = build_with_blocks(sample, {"derived", "last_event"})
    n4 = [s.name for s in s4]
    return {
        1: n1,
        2: [*LETTER_BOOLEANS, *INDICATION_BOOLEANS],
        3: [n for n in n3 if n not in set(n1)],
        4: [n for n in n4 if n not in set(n3)],
    }


@app.get("/api/metrics")
def metrics() -> dict:
    """Model-quality numbers straight from the pinned build manifests."""
    swarm_state = json.loads(
        (ROOT / "runs" / "state" / "swarm_rounds.json").read_text(encoding="utf-8")
    )
    kept = [r["round"] for r in swarm_state if r["kept"]][-1]
    swarm = json.loads(
        (ROOT / "runs" / "manifests" / f"swarm_round{kept}.json").read_text(
            encoding="utf-8"
        )
    )
    tokenizer = json.loads(
        (ROOT / "runs" / "state" / "tokenizer_rounds.json").read_text(
            encoding="utf-8"
        )
    )
    if "round_features" not in STATE:
        STATE["round_features"] = round_tried_features()
    tried = STATE["round_features"]
    return {
        "per_action_auroc": swarm["pooled_auroc"],
        "constant_prior": swarm["constant_prior"],
        "tokenizer_rounds": [
            {
                "round": r["round"], "proposal": r["proposal"],
                "auroc": r["mean_oof_auroc"], "kept": r["kept"],
                "gain": r["gain"],
                "n_features": r["n_features"],
                "tried": tried.get(r["round"], []),
            }
            for r in tokenizer
        ],
    }


def _ammonix_paths():
    """Per-claim trajectories of AMMONIX's decision moments (replay lane),
    placed in the universe embedding and connected in move order. Built once
    after the replay finishes; the recorded human states are NOT shown."""
    if "universe_paths" in STATE:
        return STATE["universe_paths"]
    prog = STATE.get("replay_progress")
    claims = STATE.get("claims")
    place = STATE.get("universe_place")
    if not claims or place is None or prog is None or (
        prog["total"] and prog["done"] < prog["total"]
    ):
        return None
    rt = STATE["rt"]
    paths = []
    for eid in sorted(claims):
        pts = []
        for st in claims[eid]["system"]["steps"]:
            sid = f"{eid}-r{st['touch']}"
            f = STATE["feats"].get(sid)
            if f is None:
                continue
            # v0.4 s.3b: the index lives in LABEL space - place the moment by
            # its calibrated score coordinate, not its feature vector
            scores = route_case(rt, STATE["rows"][sid], f)[0]
            x, y, z = place(label_coordinate(rt, scores or {}))
            trace = STATE["traces"].get(sid)
            nn = (
                [n.state_id for n in trace.retrieval.neighbours[:5]]
                if trace else []
            )
            pts.append({
                "id": sid, "x": round(x, 2), "y": round(y, 2),
                "z": round(z, 2), "action": st["action"],
                "move": st["touch"] + 1, "nn": nn,
            })
        if pts:
            summ = claim_summary(eid)
            paths.append({
                "episode": eid,
                "points": pts,
                "summary": {
                    "payer_name": summ["payer_name"], "cpt": summ["cpt"],
                    "value": summ["balance"], "verdict": summ["verdict"],
                    "delta": summ["delta"],
                    "resolution": summ["system"]["resolution"],
                    "collected": summ["system"]["collected"],
                    "mistakes": summ["system"]["mistakes"],
                    "moves": summ["system"]["touches"],
                },
            })
    STATE["universe_paths"] = paths
    return paths


@app.get("/api/universe")
def universe_map() -> dict:
    # only Ammonix's decision moments overlay the universe (owner rule);
    # the recorded human-simulation states are not shown
    paths = _ammonix_paths()
    base = {k: v for k, v in STATE["universe_map"].items() if k != "live"}
    return {**base, "live_paths": paths or [], "live_ready": paths is not None}


@app.get("/api/curve")
def curve() -> dict:
    """The learning-curve points (scripts/learning_curve.py output) plus the
    LLM flat line from the live scoreboard when that lane has finished."""
    # v0.6: like the scoreboard, the curve depends on the decision field; the
    # kernel field keeps its own file so the v0.4 curve is never mistaken for it.
    field_tag = getattr(STATE.get("rt"), "decision_field", "model")
    path = ROOT / "data" / (
        "ui_curve.json" if field_tag == "model" else f"ui_curve_{field_tag}.json"
    )
    if not path.exists():
        raise HTTPException(status_code=425, detail="curve not computed yet")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["decision_field"] = field_tag
    sb = STATE.get("scoreboard") or {}
    doc["llm_collected"] = sb.get("llm_collected")
    doc["llm_mistake_claims"] = sb.get("llm_mistake_claims")
    doc["llm_model_label"] = STATE.get("llm_model_label", "Qwen 27B")
    full = [p for p in doc.get("points", []) if p["n"] == 4750]
    doc["complete"] = bool(full)
    return doc


@app.get("/api/headtohead")
def headtohead() -> dict:
    """The pre-registered tranche-98 three-way comparison plus the earlier
    tranche-92 reference (LLM tied v0.3, beat v0.2 - surfaced, not hidden).
    Both reports are checked-in copies from the Agentic_Baseline repo; the
    _provenance field inside each names the source file and, for tranche 98,
    the pre-registration (tie band, amendment)."""
    reports = ROOT / "runs" / "reports"
    t98 = json.loads(
        (reports / "comparison_v4_tranche98.json").read_text(encoding="utf-8")
    )
    t92 = json.loads(
        (reports / "comparison_t92_reference.json").read_text(encoding="utf-8")
    )
    prov = t98["_provenance"]

    def arm(a: dict) -> dict:
        return {
            "success_rate": a["success_rate_v2"],
            "collected_capped": a["payer_collected_capped"],
            "mistake_episodes": a["episodes_with_mistakes"],
            "wall_seconds": a["wall_seconds"],
        }

    margin = round(
        t98["ammonix_v4"]["success_rate_v2"] - t98["agentic"]["success_rate_v2"], 4
    )
    return {
        "n_episodes": t98["n_episodes"],
        "tranche": t98["tranche"],
        "run_date": prov["run_date"],
        "design": prov["design"],
        "amendment": prov["amendment"],
        "tie_band": prov["tie_band_success_pts"],
        "margin": margin,
        "outside_band": abs(margin) > prov["tie_band_success_pts"],
        "arms": {
            "ammonix": arm(t98["ammonix_v4"]),
            "llm": {**arm(t98["agentic"]), "cost": t98["agentic"]["llm"]},
            "personas": arm(t98["personas"]),
        },
        "oracle": (
            {
                "success_rate": t98["oracle_greedy"]["success_rate_v2"],
                "collected_capped": t98["oracle_greedy"]["payer_collected_capped"],
            }
            if "oracle_greedy" in t98
            else None
        ),
        "t92": {
            "run_date": t92["_provenance"]["run_date"],
            "note": t92["_provenance"]["note"],
            "llm_success": t92["agentic"]["success_rate_v2"],
            "v3_success": t92["system_v3"]["success_rate_v2"],
            "v2_success": t92["system"]["success_rate_v2"],
        },
    }


@app.get("/api/ablation")
def ablation() -> dict:
    """The episodic-memory ablation (v0.4 with its memory gates disabled, same
    tranche-98 episodes and dice; post-hoc analysis, NOT pre-registered) plus
    the showcase claim ep-05179 - all read from the checked-in report copy;
    its _provenance names the source files and the gate-classification table."""
    doc = json.loads(
        (ROOT / "runs" / "reports" / "ablation_v4_tranche98.json").read_text(
            encoding="utf-8"
        )
    )

    def arm(a: dict) -> dict:
        return {
            "success_rate": a["success_rate_v2"],
            "collected_capped": a["payer_collected_capped"],
            "mistake_episodes": a["episodes_with_mistakes"],
            "escalated_touches": a["escalated_touches"],
        }

    div = doc["divergences"]
    eps = div["episodes"]
    flips_saved = sum(
        1 for e in eps if e["verdict"] == "memory_saved"
        and e["full_outcome"]["success"] and not e["ablated_outcome"]["success"]
    )
    flips_cost = sum(
        1 for e in eps if e["verdict"] == "memory_cost"
        and e["ablated_outcome"]["success"] and not e["full_outcome"]["success"]
    )
    case = json.loads(
        (ROOT / "runs" / "reports" / "memory_case_v4t98.json").read_text(
            encoding="utf-8"
        )
    )

    return {
        "date": doc["_provenance"]["date"],
        "n_episodes": doc["aggregate"]["ammonix_v4_full"]["episodes"],
        "full": arm(doc["aggregate"]["ammonix_v4_full"]),
        "ablated": arm(doc["aggregate"]["ammonix_v4_ablated"]),
        "divergent": div["n_divergent_episodes"],
        "memory_saved": div["memory_saved"],
        "memory_cost": div["memory_cost"],
        "neutral": div["neutral"],
        "success_flips_saved": flips_saved,
        "success_flips_cost": flips_cost,
        # what actually fired on this tranche (decisions_by_gate: only the
        # confidence floor plus the hard-rule blown clock; no OOD, no ambiguity)
        "floor_escalations": doc["decisions_by_gate"]["full"]["escalate_floor"],
        # showcase: a claim Ammonix wins unaided (zero escalation) from its
        # learned playbook while both baselines fail - see the file's
        # _provenance for selection criteria
        "showcase": {
            "episode_id": case["episode_id"],
            "allowed": case["allowed"],
            "payer_paid_primary": case["payer_paid_primary"],
            "pattern": case["pattern"],
            "lanes": case["lanes"],
        },
    }


@app.get("/api/progress")
def progress() -> dict:
    # recorded cases only: replay states are reported separately below
    return {
        "done": sum(
            1 for s in STATE["traces"]
            if not STATE["rows"][s].get("_replay")
        ),
        "total": sum(
            1 for r in STATE["rows"].values() if not r.get("_replay")
        ),
        "failed": STATE.get("failed", {}),
        "rollout_done": "claims" in STATE,
        "rollout_error": STATE.get("rollout_error"),
        "replay": STATE.get("replay_progress"),
        "replay_error": STATE.get("replay_error"),
        "llm": STATE.get("llm_progress"),
        "llm_done": STATE.get("llm_done", False),
        "llm_error": STATE.get("llm_error"),
    }


def claim_summary(episode_id: str) -> dict:
    c = STATE["claims"][episode_id]
    s0 = STATE["rows"].get(f"{episode_id}-s0") or next(
        r for r in STATE["rows"].values()
        if r["episode_id"] == episode_id and not r.get("_replay")
    )
    payer = STATE["engine"].payers.get(s0["payer_id"])
    lane = {
        p: {k: c[p].get(k) for k in (
            "resolution", "collected", "touches", "mistakes", "success",
            "patient_collected", "patient_counted", "autonomous",
            "payer_collected", "payer_counted", "tokens",
        )}
        for p in ("human", "system", *((["llm"]) if c.get("llm") else []))
    }
    return {
        "episode_id": episode_id,
        "payer_id": s0["payer_id"],
        "payer_name": payer.name if payer else s0["payer_id"],
        "cpt": s0["cpt"],
        "balance": s0["allowed_amount"],
        "persona": s0["persona_id"],
        **lane,
        "verdict": c["verdict"],
        "delta": c["delta"],
    }


@app.get("/api/claims")
def claims_list() -> dict:
    if "claims" not in STATE:
        raise HTTPException(425, "rollout still computing - retry shortly")
    return {
        "scoreboard": STATE["scoreboard"],
        "claims": [claim_summary(e) for e in sorted(STATE["claims"])],
        "specials": [
            case_summary(s) for s in sorted(STATE["rows"])
            if not STATE["rows"][s]["episode_id"].startswith("ep-")
            and s in STATE["traces"]
        ],
    }


@app.get("/api/claim/{episode_id}")
def claim_detail(episode_id: str) -> dict:
    if "claims" not in STATE:
        raise HTTPException(425, "rollout still computing - retry shortly")
    if episode_id not in STATE["claims"]:
        raise HTTPException(404)
    c = STATE["claims"][episode_id]
    moments = sorted(
        s for s, r in STATE["rows"].items()
        if r["episode_id"] == episode_id and not r.get("_replay")
    )
    system_moments = sorted(
        s for s, r in STATE["rows"].items()
        if r["episode_id"] == episode_id and r.get("_replay")
        and s in STATE["traces"]
    )
    # the Ammonix lane's filings, inline: each replay moment's rendered
    # document, check log and writing bill ride on the step, so the claim
    # page can open them in the same details block as the LLM lane's
    system_steps = []
    for st in c["system"]["steps"]:
        sid = f"{episode_id}-r{st['touch']}"
        extra: dict = {}
        art = STATE["artifacts"].get(sid)
        if art:
            extra["paperwork"] = {
                "kind": art["artifact"], "field_map": art["field_map"],
                "gaps": art["gaps"],
            }
        trace = STATE["traces"].get(sid)
        if trace is not None:
            extra["write_status"] = trace.status
            if trace.iterations:
                extra["checks"] = [
                    {"n": it.n, "passed": it.check.passed,
                     "failures": it.check.failures}
                    for it in trace.iterations
                ]
            if art is None and trace.status == "executed" and trace.final:
                extra["filed_payload"] = trace.final.payload
        system_steps.append({**st, **extra})
    return {
        **claim_summary(episode_id),
        "human_steps": c["human"]["steps"],
        "system_steps": system_steps,
        "llm_steps": (c.get("llm") or {}).get("steps"),
        "llm_model_label": STATE.get("llm_model_label", "Qwen 27B"),
        "human_detail": c["human"],
        "system_detail": c["system"],
        "llm_detail": c.get("llm"),
        "moments": moments,
        "system_moments": system_moments,
    }


def _play_snapshot_row(case) -> dict:
    from cardessa.corpus import state_tabular

    payer = STATE["engine"].payers[case.payer_id]
    row = state_tabular(case, payer, 0.0)
    row.update({
        "episode_id": case.episode_id,
        "state_id": f"{case.episode_id}-live{case.touch_seq}",
        "touch_seq": case.touch_seq, "action_raw": "",
        "clinical_indication_text": "", "payer_correspondence_text": "",
    })
    return row


_ALL_ACTIONS = [
    "submit_clean", "submit_with_records", "correct_and_resubmit",
    "bill_secondary", "request_retro_auth", "appeal_with_necessity",
    "request_peer_to_peer", "provide_requested_info", "bill_patient",
    "write_off",
]


def _unavailable_reasons(row: dict, applicable: set[str]) -> list[dict]:
    """Why each masked action is off the menu, mirroring applicable_actions
    branch by branch - so the live run can SHOW the rulebook, not hide it."""
    carc = row.get("carc") or ""
    prior = (row.get("prior_actions") or "").split(",")
    already_paid = row["balance"] < row["allowed_amount"] - 0.005
    out = []
    for a in _ALL_ACTIONS:
        if a in applicable:
            continue
        if a == "request_retro_auth":
            if not row["auth_required"]:
                reason = "no authorization needed"
            elif row["auth_status"] == "on_file":
                reason = "authorization already on file"
            elif "request_retro_auth" in prior:
                reason = "already asked - same answer guaranteed"
            else:
                reason = (
                    f"retro window closed (day {int(row['days_since_service'])}"
                    f" of {int(row['retro_window_days'])})"
                )
        elif a in ("appeal_with_necessity", "request_peer_to_peer"):
            if not carc:
                reason = "no denial standing"
            elif a in prior:
                reason = "already tried - same answer guaranteed"
            else:
                reason = "payer offers no peer-to-peer"
        elif a in ("submit_clean", "submit_with_records"):
            if already_paid:
                reason = "payer already paid - duplicate billing"
            elif row["days_to_filing_deadline"] <= 0:
                reason = "filing deadline passed"
            elif not row["pairing_valid"]:
                reason = "code pairing not legal"
            elif not row["cob_position_ok"]:
                reason = "wrong insurer order - correct it instead"
            else:
                reason = "unchanged resubmission - same denial guaranteed"
        elif a == "correct_and_resubmit":
            if already_paid:
                reason = "payer already paid"
            elif row["days_to_filing_deadline"] <= 0:
                reason = "filing deadline passed"
            else:
                reason = "insurer order already correct - nothing to fix"
        elif a == "bill_patient":
            reason = (
                "payer hasn't adjudicated yet"
                if row["touches_so_far"] < 1 else "no balance left"
            )
        elif a == "bill_secondary":
            if not row["has_secondary"]:
                reason = "no secondary coverage"
            elif carc:
                reason = "denial standing - resolve it first"
            elif row["touches_so_far"] < 1:
                reason = "payer hasn't adjudicated yet"
            else:
                reason = "no balance left"
        elif a == "provide_requested_info":
            reason = (
                "already sent - same answer guaranteed"
                if "provide_requested_info" in prior else "nothing was requested"
            )
        else:
            reason = "not applicable"
        out.append({"action": a, "reason": reason})
    return out


def _play_situation(sess) -> dict:
    """Current live state + a fresh recommendation computed ON that state."""
    from policy_rollout import CONFIDENCE_FLOOR
    from cardessa.features_kept import build_kept_features
    from cardessa.harness import applicable_actions, route_case

    case = sess["case"]
    row = _play_snapshot_row(case)
    snaps = pl.DataFrame(
        [{c: row.get(c) for c in STATE["working_columns"]}],
        schema_overrides=STATE["working_schema"],
    )
    frame, _n, _, _ = build_kept_features(ROOT, snaps)
    r = frame.to_dicts()[0]
    # one-hot categories absent from a single-row frame are exactly 0
    feats = {n: float(r.get(n, 0.0)) for n in STATE["rt"].feature_names}
    scores, known, forced, _amb = route_case(STATE["rt"], row, feats)
    recommended = forced
    if recommended is None and known and scores:
        best_action, best = max(scores.items(), key=lambda kv: kv[1])
        if best >= CONFIDENCE_FLOOR:
            recommended = best_action
        else:  # v0.4: close out by expected value
            recommended = close_out_choice(expected_values(row, scores), scores)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    # on a handoff the system lane falls back to the persona policy; expose
    # that same pick so a live run can reproduce ANY lane, not just the
    # fully-autonomous ones (audit finding)
    human_policy = None
    if recommended is None:
        from policy_rollout import case_view
        from cardessa.personas import choose_action

        payer = STATE["engine"].payers[case.payer_id]
        human_policy = choose_action(
            case.persona, case_view(case, payer), MASTER_SEED, case.episode_id
        )
    return {
        "situation": {
            "move": case.touch_seq + 1,
            "carc": case.carc,
            "days_since_service": case.days_since_service,
            "collected": round(case.collected_payer + case.collected_patient, 2),
            "outstanding": round(
                case.allowed - case.collected_payer - case.collected_patient, 2
            ),
        },
        "scores": [{"action": a, "p_success": round(v, 4)} for a, v in ranked],
        "recommended": recommended,
        "handed_to_human": recommended is None,
        "human_policy": human_policy,
        "known_payer": known,
        "applicable": [a for a, _ in ranked] or None,
        "unavailable": _unavailable_reasons(row, applicable_actions(row)),
    }


@app.post("/api/claim/{episode_id}/play/start")
def play_start(episode_id: str) -> dict:
    """Live play: a fresh copy of this claim, same engine dice as the lanes."""
    from cardessa.corpus import CardessaEnvironment

    if "claims" not in STATE or episode_id not in STATE["claims"]:
        raise HTTPException(404)
    env = CardessaEnvironment(STATE["world"], STATE["engine"], MASTER_SEED)
    case = env.reset(int(episode_id.removeprefix("ep-")))
    sess = {"env": env, "case": case, "pending": [], "steps": [], "done": False}
    STATE.setdefault("play", {})[episode_id] = sess
    return {"steps": [], "done": False, **_play_situation(sess)}


@app.post("/api/claim/{episode_id}/play/step")
def play_step(episode_id: str, body: dict | None = None) -> dict:
    """Execute one move on the live claim; the next recommendation is
    computed on the state this execution produces."""
    from policy_rollout import outcome_v2
    from cardessa.corpus import state_tabular

    sess = STATE.get("play", {}).get(episode_id)
    if sess is None:
        raise HTTPException(400, "no live run - start one first")
    if sess["done"]:
        raise HTTPException(400, "this run is finished - start again to replay")
    payload = _play_situation(sess)
    chosen = (body or {}).get("action")
    action = chosen or payload["recommended"]
    if action is None:
        raise HTTPException(400, "Ammonix handed this move to you - pick an action")
    if payload["applicable"] and action not in payload["applicable"]:
        raise HTTPException(400, "action not applicable to this claim right now")
    case = sess["case"]
    payer = STATE["engine"].payers[case.payer_id]
    sess["pending"].append((state_tabular(case, payer, 0.0), action))
    before = _before_facts(case)
    after = sess["env"].apply(case, action)
    by = "ammonix" if (not chosen or chosen == payload["recommended"]) \
        and not payload["handed_to_human"] else "you"
    # the payer's ANSWER to this move: money/denial facts, plus the actual
    # letter when the payer wrote one (cached; live model if the run
    # diverged from recorded history; degrades to facts-only if offline)
    response = {
        "headline": _step_note(before, after, action),
        "carc": after.carc,
        "paid": round(after.collected_payer - before["collected_payer"], 2),
        "patient_paid": round(
            after.collected_patient - before["collected_patient"], 2
        ),
        "letter": None,
    }
    if after.correspondence_kind is not None:
        try:
            from cardessa.corpus import state_text
            response["letter"] = state_text(
                after, payer, STATE["textgen"], MASTER_SEED
            )["payer_correspondence_text"] or None
        except Exception:  # noqa: BLE001 - model offline: facts still shown
            response["letter_unavailable"] = True
    sess["steps"].append({
        "touch": before["touch"], "action": action, "by": by,
        "note": response["headline"], "response": response,
    })
    sess["case"] = after
    out: dict = {"steps": sess["steps"], "done": bool(after.terminal)}
    if after.terminal:
        sess["done"] = True
        success, _score, n_mistakes = outcome_v2(sess["env"], after, sess["pending"])
        # same money rule as the lanes: patient up to the contractual share,
        # payer up to the rest of the allowed amount (secondary payers
        # legitimately cover the patient's share; total never exceeds allowed)
        patient_valid = round(
            min(after.collected_patient, after.contractual_share), 2
        )
        payer_valid = round(
            min(after.collected_payer, after.allowed - patient_valid), 2
        )
        out["outcome"] = {
            "resolution": (after.resolution or "closed"),
            "collected": round(payer_valid + patient_valid, 2),
            "touches": len(after.action_history),
            "mistakes": n_mistakes,
            "success": bool(success),
            "days": int(after.days_elapsed),
        }
        # v0.6 consolidation: the claim is terminal and its outcome is now
        # adjudicated, so every touch of this episode enters the kernel
        # field as a record (Lambda places it; the success label is the
        # episode's). Never before terminal (temporal firewall). Only when
        # the kernel field is the active decision field.
        out["consolidated"] = _consolidate_episode(sess, episode_id, bool(success))
    else:
        out.update(_play_situation(sess))
    return out


def _consolidate_episode(sess, episode_id: str, success: bool) -> int:
    """Insert each (state, action) touch of a finished, adjudicated episode
    into the live kernel field. Returns the number of records consolidated
    (0 when the field is inactive or the touches cannot be placed)."""
    rt = STATE["rt"]
    if getattr(rt, "decision_field", "model") != "kernel" or rt.kernel_field is None:
        return 0
    from cardessa.consolidation import ConsolidationLog, consolidate

    log = STATE.setdefault("consolidation_log",
                           ConsolidationLog(len(rt.kernel_field.state_ids)))
    n = 0
    for touch, (snapshot, action) in enumerate(sess["pending"]):
        if action not in rt.trained_actions:
            continue  # constant-prior actions carry no learned field entry
        try:
            feats = {k: float(snapshot.get(k, 0.0)) for k in rt.feature_names}
            x = np.array([[feats[k] for k in rt.feature_names]])
            u = rt.kernel_field.live_u(rt, x)[0]
            consolidate(rt.kernel_field, log, u_row=u, action=action,
                        success=success, state_id=f"{episode_id}-live{touch}",
                        terminal=True, adjudicated=True)
            n += 1
        except Exception:  # noqa: BLE001 - consolidation must never break a reply
            continue
    if n:
        STATE["consolidated_records"] = STATE.get("consolidated_records", 0) + n
    return n


@app.post("/api/kernel/snapshot")
def kernel_snapshot() -> dict:
    """Pin the current kernel field (basis records + consolidated ones) as a
    content-hashed snapshot in basis/kernel_snapshots/, for deployment."""
    rt = STATE["rt"]
    if getattr(rt, "decision_field", "model") != "kernel" or rt.kernel_field is None:
        raise HTTPException(400, "kernel decision field is not active")
    from cardessa.consolidation import ConsolidationLog, snapshot

    log = STATE.setdefault("consolidation_log",
                           ConsolidationLog(len(rt.kernel_field.state_ids)))
    entry = snapshot(rt.kernel_field, log, ROOT, "v0.6")
    return {"content_hash": entry["content_hash"], "n_records": entry["n_records"],
            "n_consolidated": entry["n_consolidated"], "stamp": entry["stamp"]}


@app.get("/api/case/{state_id}")
def case_detail(state_id: str) -> dict:
    if state_id not in STATE["rows"]:
        raise HTTPException(404)
    if state_id not in STATE["traces"]:
        raise HTTPException(425, "case is still being computed - retry shortly")
    row = STATE["rows"][state_id]
    trace = STATE["traces"][state_id]
    retrieval = trace.retrieval
    tribe = STATE["tribes"].get(retrieval.tribe_id)
    universe = STATE["universe"]
    episode_states = sorted(
        s for s, r in STATE["rows"].items()
        if r["episode_id"] == row["episode_id"]
        and r.get("_replay") == row.get("_replay")
    )
    neighbours = []
    for n in retrieval.neighbours[:8]:
        urow = universe.filter(pl.col("state_id") == n.state_id)
        if urow.height:
            u = urow.to_dicts()[0]
            neighbours.append(
                {
                    "state_id": n.state_id,
                    "distance": round(n.distance, 3),
                    "true_action": u["true_action_id"],
                    "outcome_success": u["outcome_success"],
                    "facts": STATE["working_index"].get(n.state_id),
                }
            )
    scores = sorted(retrieval.scores_cal.items(), key=lambda kv: -kv[1])
    rival = scores[1] if retrieval.ambiguous and len(scores) > 1 else None
    artifact = STATE["artifacts"].get(state_id)
    question = None
    if retrieval.ambiguous and len(scores) > 1:
        question = (
            f"Calibrated success is nearly tied: {scores[0][0]} "
            f"({scores[0][1]:.2f}) vs {scores[1][0]} ({scores[1][1]:.2f}). "
            "Which should be executed?"
        )
    skill = next(
        (s for s in STATE["skills"] if s.skill_id == retrieval.skill_id), None
    )
    return {
        **case_summary(state_id),
        "episode_states": episode_states,
        "recorded_action": row.get("action_raw"),
        "taken_by": row.get("_by"),
        "skill": {
            "skill_id": skill.skill_id,
            "kind": skill.kind,
            "scope_level": skill.scope.level,
            "scope_ref": skill.scope.ref,
            "rules": skill.expected_result.rules or [],
            "m1_prompt_ref": skill.m1_prompt_ref or None,
        }
        if skill
        else None,
        "facts": {
            k: row[k]
            for k in (
                "dx_codes", "days_since_service", "days_to_filing_deadline",
                "auth_status", "retro_window_days", "touches_so_far",
                "prior_actions", "prior_carcs", "allowed_amount",
            )
        },
        "correspondence": row.get("payer_correspondence_text") or "",
        "indication": row.get("clinical_indication_text") or "",
        "scores": [
            {"action": a, "p_success": round(v, 4)} for a, v in scores
        ],
        "rival": {"action": rival[0], "p_success": round(rival[1], 4)} if rival else None,
        "tribe": {
            "tribe_id": retrieval.tribe_id,
            "description": (
                f"{tribe.member_count} similar cases · "
                f"{tribe.stats.success_rate:.0%} collected"
                if tribe
                else "thin precedent"
            ),
            "out_of_tribe_distance": round(
                min((n.distance for n in retrieval.neighbours), default=-1), 3
            ),
        },
        "neighbours": neighbours,
        "question": question,
        "artifact": {
            "kind": artifact["artifact"],
            "html": artifact["html"],
            "field_map": artifact["field_map"],
            "gaps": artifact["gaps"],
        }
        if artifact
        else None,
        "m2_log": [
            {
                "n": it.n,
                "passed": it.check.passed,
                "failures": it.check.failures,
                "adjustment": it.adjustment,
            }
            for it in trace.iterations
        ],
        "adjudication": STATE["adjudications"].get(state_id),
    }


@app.post("/api/case/{state_id}/execute")
def execute(state_id: str, body: dict | None = None) -> dict:
    """Approve & execute: LIVE payer-engine adjudication round trip.

    Ambiguous cases may pass {"action": ...} to execute the rival instead of
    the recommendation; any action the runtime scored as applicable is legal.
    """
    trace = STATE["traces"].get(state_id)
    bundle = STATE["bundles"].get(state_id)
    if trace is None or trace.status != "executed" or bundle is None:
        raise HTTPException(400, "case is not executable")
    if state_id in STATE["human_queue"]:
        raise HTTPException(400, "case was sent to a human reviewer")
    row = STATE["rows"][state_id]
    action = trace.final.action_id
    chosen = (body or {}).get("action")
    if chosen and chosen != action:
        if chosen not in trace.retrieval.scores_cal:
            raise HTTPException(400, "action not applicable to this case")
        action = chosen
    engine = STATE["engine"]
    # rework requests trigger the payer's DECISION process, not a claim
    # submission; each maps to its own engine round trip
    if action in (
        "request_retro_auth", "appeal_with_necessity", "request_peer_to_peer",
        "provide_requested_info",
    ):
        if action == "request_retro_auth":
            granted = engine.retro_auth_granted(
                row["payer_id"], row["days_since_service"], state_id
            )
            status = "retro_auth_granted" if granted else "retro_auth_denied"
        elif action == "appeal_with_necessity":
            granted = engine.appeal_granted(
                row["payer_id"], row["clinic_doc_quality"] >= 0.6, state_id
            )
            status = "appeal_granted" if granted else "appeal_denied"
        elif action == "request_peer_to_peer":
            granted = engine.p2p_reversed(row["payer_id"], state_id)
            status = "p2p_reversed" if granted else "p2p_upheld"
        else:
            granted = engine.info_request_resolved(row["payer_id"], state_id)
            status = "info_accepted" if granted else "info_still_missing"
        outcome = {
            "action": action, "engine_status": status, "carc": row["carc"] or None,
            "paid_amount": 0.0, "patient_responsibility": 0.0, "delay_days": 8,
        }
        STATE["adjudications"][state_id] = outcome
        return outcome
    # internal actions never reach the payer: simulate their own semantics
    # (the corpus constants) instead of a bogus claim submission
    if action in ("bill_patient", "write_off"):
        from cardessa.engine import stable_seed

        if action == "write_off":
            outcome = {
                "action": action, "engine_status": "closed_written_off",
                "carc": None, "paid_amount": 0.0,
                "patient_responsibility": 0.0, "delay_days": 0,
            }
        else:
            rng = np.random.default_rng(
                stable_seed(MASTER_SEED, "ui_patient_pay", state_id)
            )
            amount = float(row["balance"])
            pay_prob = 0.7 if amount <= 300 else 0.4
            paid = rng.random() < pay_prob
            outcome = {
                "action": action,
                "engine_status": "patient_paid" if paid else "patient_did_not_pay",
                "carc": None,
                "paid_amount": amount if paid else 0.0,
                "patient_responsibility": amount,
                "delay_days": int(rng.integers(20, 40)),
            }
        STATE["adjudications"][state_id] = outcome
        return outcome
    with_records = action in ("submit_with_records", "correct_and_resubmit")
    # a WON appeal or peer-to-peer clears the necessity/auth objection: the
    # payer's own approval letter stands in for the authorization. Detectable
    # in the record as a contest action that produced no new denial code.
    prior_acts = [a for a in (row["prior_actions"] or "").split(",") if a]
    prior_carcs = [c for c in (row["prior_carcs"] or "").split(",") if c]
    contest_won = (
        any(a in ("appeal_with_necessity", "request_peer_to_peer") for a in prior_acts)
        and len(prior_carcs) < len(prior_acts)
        and not row["carc"]
    )
    if row["auth_status"] == "on_file":
        auth_ref = "AUTH-PRE"
    elif contest_won:
        auth_ref = "AUTH-APPEAL"
    else:
        auth_ref = None
    claim = Claim(
        claim_id=f"{state_id}-live",
        payer_id=row["payer_id"],
        plan_variant=bundle["coverage"].get("plan_variant", "standard"),
        cpt=row["cpt"],
        icd_codes=(row["dx_codes"] or "").split(","),
        member_id=bundle["coverage"]["member_id"],
        auth_ref=auth_ref,
        attachments=["clinical_notes"] if with_records else [],
        days_since_service=row["days_since_service"],
        ordering_npi_medicaid_enrolled=bool(
            bundle["clinics"].get("medicaid_enrolled", True)
        ),
        cob_position_correct=bool(row["cob_position_ok"])
        or action == "correct_and_resubmit",
        has_medicare_primary=bool(bundle["coverage"].get("has_medicare_primary")),
        coverage_active=row["eligibility_status"] == "active",
        prior_holter_within_days=None,
        deductible_remaining=float(row["deductible_remaining"]),
        coinsurance_pct=float(row["coinsurance_pct"]),
        billed_amount=float(row["allowed_amount"]),
    )
    result = STATE["engine"].adjudicate(claim)
    outcome = {
        "action": action,
        "engine_status": result.status,
        "carc": result.carc,
        "paid_amount": result.paid_amount,
        "patient_responsibility": result.patient_responsibility,
        "delay_days": result.delay_days,
    }
    STATE["adjudications"][state_id] = outcome
    return outcome


@app.post("/api/case/{state_id}/escalate")
def escalate(state_id: str) -> dict:
    """Hand the case to a human reviewer instead of executing."""
    if state_id not in STATE["rows"]:
        raise HTTPException(404)
    if state_id in STATE["adjudications"]:
        raise HTTPException(400, "case already adjudicated")
    STATE["human_queue"][state_id] = "sent to human review by the operator"
    return {"status": "sent_to_human"}


@app.get("/api/story")
def story() -> list[dict]:
    # recomputed per request: steps fill in as background cases complete
    out = []
    for i, step in enumerate(pick_story_cases(STATE["rows"], STATE["traces"]), start=1):
        state_id = step["state_id"]
        summary = case_summary(state_id) if state_id else {}
        trace = STATE["traces"].get(state_id)
        actual = None
        if trace is not None:
            if step["expected"] == "executed":
                actual = trace.status
            elif step["expected"].startswith("recommend:"):
                rec = trace.retrieval.recommended_action_id
                actual = (
                    f"recommend:{rec}"
                    if step["expected"] == f"recommend:{rec}"
                    else (
                        "recommend:not_retro"
                        if step["expected"] == "recommend:not_retro"
                        and rec != "request_retro_auth"
                        else f"recommend:{rec}"
                    )
                )
            elif step["expected"] == "ask":
                actual = "ask" if trace.retrieval.ambiguous else trace.status
            else:
                actual = trace.status
        out.append(
            {
                "step": i,
                **step,
                "summary": summary,
                "actual": actual,
                "reached": actual == step["expected"],
            }
        )
    return out



_DIALOG_SYSTEM = (
    "You are L, the operator-facing rationale layer of the Ammonix claims "
    "system (architecture paper: 'a frozen local language model used for "
    "operator-facing rationale and dialog'). You explain THIS claim decision "
    "to the human operator. HARD RULES: answer ONLY from the CASE RECORD "
    "JSON below - the facts, letter, scores, tribe, neighbours and check "
    "log. If the record does not contain the answer, say exactly that. "
    "Never invent codes, amounts, dates or policy. Do not give medical or "
    "coding advice beyond the record. Plain language, at most 120 words. "
    "CASE RECORD: "
)


@app.post("/api/case/{state_id}/ask")
def ask_case(state_id: str, body: dict | None = None) -> dict:
    """Operator dialog (paper's L): grounded strictly in this case's record."""
    question = ((body or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(400, "empty question")
    if state_id not in STATE.get("traces", {}):
        raise HTTPException(425, "case not computed yet")
    detail = case_detail(state_id)
    ctx = {
        k: detail.get(k)
        for k in (
            "state_id", "payer_name", "cpt", "carc", "balance", "status",
            "recommended", "ambiguous", "escalation_reason", "facts",
            "correspondence", "indication", "scores", "tribe", "neighbours",
            "m2_log", "recorded_action",
        )
    }
    history = [
        {"role": m.get("role"), "content": str(m.get("content"))[:600]}
        for m in ((body or {}).get("history") or [])[-6:]
        if m.get("role") in ("user", "assistant")
    ]
    payload = json.dumps({
        "model": DIALOG_MODEL or STATE["provider"].model_id,
        "messages": [
            {"role": "system",
             "content": _DIALOG_SYSTEM + json.dumps(ctx, default=str)},
            *history,
            {"role": "user", "content": question[:600]},
        ],
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 350,
        "chat_template_kwargs": {"enable_thinking": False},
        **DIALOG_EXTRA,
    }).encode("utf-8")
    import urllib.request
    headers = {"Content-Type": "application/json"}
    if DIALOG_KEY:
        headers["Authorization"] = f"Bearer {DIALOG_KEY}"
    request = urllib.request.Request(
        f"{DIALOG_BASE}/chat/completions", data=payload, headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content = json.loads(response.read())["choices"][0]["message"]["content"]
    except OSError as exc:
        raise HTTPException(503, f"model offline: {exc}") from exc
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    return {"answer": content.strip()}


@app.get("/api/llm_health")
def llm_health() -> dict:
    """Is the dialog model answering? The chat widget hides itself if not.

    A remote backend is disclosed (model id) so the page can label the chat
    precisely: hosted stock Qwen is not the paper's pinned local model."""
    import urllib.request

    remote = not DIALOG_BASE.startswith(("http://127.0.0.1", "http://localhost"))
    request = urllib.request.Request(
        f"{DIALOG_BASE}/models",
        headers={"Authorization": f"Bearer {DIALOG_KEY}"} if DIALOG_KEY else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            return {"available": True, "remote": remote,
                    "model": DIALOG_MODEL or None}
    except OSError:
        return {"available": False, "remote": remote, "model": None}


@app.get("/healthz", response_class=HTMLResponse)
def healthz() -> str:
    return "ok"


# the catch-all static mount must be registered LAST or it shadows the routes
app.mount(
    "/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static"
)
