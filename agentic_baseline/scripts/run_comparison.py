"""Three-arm rollout: agentic baseline vs the Ammonix system vs the personas,
on the SAME 300 fresh episodes as the factory's policy-rollout analysis
(tranche 92, appended after the full training lineage; never sealed data,
never training data). Identical engine randomness per case; outcome v2 with
the process-mistake ledger; escalated touches fall back to the persona
(human) policy in every arm.

Run with the factory worktree's venv, the pinned 27B serving:
  <v2-agent>\\.venv\\Scripts\\python.exe scripts\\run_comparison.py
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FACTORY_ROOT = Path(
    os.environ.get(
        "AMMONIX_FACTORY_ROOT",
        BASE.parent,  # co-located default; set AMMONIX_FACTORY_ROOT for historical builds
    )
)
for entry in (str(BASE), str(FACTORY_ROOT), str(FACTORY_ROOT / "ammonix_core"),
              str(FACTORY_ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import (  # noqa: E402
    MISTAKE_PENALTY,
    CardessaEnvironment,
    compute_mistakes,
    state_tabular,
)
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.livecases import training_world  # noqa: E402
from cardessa.personas import CaseView, choose_action  # noqa: E402

from agentic.fallback import persona_action  # noqa: E402
from agentic.llm import AgentLLM  # noqa: E402
from agentic.policy import AgenticPolicy  # noqa: E402

N_EPISODES = 300
ROLLOUT_TRANCHE = 92  # identical to scripts/policy_rollout.py in the factory
# tranche 98 = the pre-registered v0.4 rerun (preregistration/
# comparison_v4_tranche98.json); run with AMMONIX_FACTORY_ROOT=<v4-build>
# --tranche 98 --report comparison_v4_tranche98.json --tag _v4t98
ALLOCATION = {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}
CONFIDENCE_FLOOR = 0.05
LLM_WORKERS = 6  # 12 concurrent decisions crashed the 24GB laptop vLLM mid-run


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


def outcome_v2(env, case, pending):
    outcome = env.outcome(case)
    mistakes = compute_mistakes(pending, case, outcome.success)
    n = sum(mistakes.values())
    return outcome.success, max(0.0, outcome.score - MISTAKE_PENALTY * n), n


def human_action(env, case, payer):
    """Persona decision with the unknown-CARC close-out fallback (CO-18:
    the clerk scripts predate it and can only meet it on an escalated touch
    after an LLM arm resubmitted a paid claim). Arm-agnostic wrapper."""
    return persona_action(
        choose_action, case.persona, case_view(case, payer), MASTER_SEED,
        case.episode_id, case.patient_share_pending, env.legal_actions(case),
    )


def system_router(world, engine):
    """The shipped v0.2 Ammonix runtime, loaded lazily (needs basis + data)."""
    from cardessa.features import build_round1_features
    from cardessa.harness import BasisRuntime, route_case

    rt = BasisRuntime.load(FACTORY_ROOT)
    working = pl.read_parquet(
        FACTORY_ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )

    def route_step(snap_rows):
        snaps = pl.DataFrame(
            [{c: r.get(c) for c in working.columns} for r in snap_rows],
            schema_overrides=working.schema,
        )
        frame, _ = build_round1_features(
            pl.concat([working, snaps], how="vertical_relaxed")
        )
        names = rt.feature_names
        feats = {
            r["state_id"]: {n: r[n] for n in names}
            for r in frame.filter(
                pl.col("state_id").is_in(snaps["state_id"].to_list())
            ).to_dicts()
        }

        def pick(row):
            scores, known, forced = route_case(rt, row, feats[row["state_id"]])
            if known and forced is None and scores:
                best_action, best = max(scores.items(), key=lambda kv: kv[1])
                if best >= CONFIDENCE_FLOOR:
                    return best_action
            return None  # escalate

        return pick

    return route_step


def play(policy_name, world, engine, indices, agentic_policy=None,
         route_step=None):
    """Step-synchronous rollout; identical structure to the factory script."""
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    live = {}
    for index in indices:
        case = env.reset(index)
        live[case.episode_id] = {"case": case, "pending": [], "escalated_touches": 0}
    results = {}
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
                "clinical_indication_text": "", "payer_correspondence_text": "",
            })
            snap_rows.append(row)

        chosen: dict[str, str | None] = {}
        if agentic_policy is not None:  # LLM-agent arms: "agentic", "claude", "agentic_rag"
            agentic_policy.prepare(snap_rows)  # batch hook (retrieval arm computes features)
            def decide(row):
                case = live[row["episode_id"]]["case"]
                legal = env.legal_actions(case)
                try:
                    return row["episode_id"], agentic_policy.decide(row, legal).action
                except Exception as err:  # a dead LLM call must not kill the run:
                    print(f"decision failed for {row['state_id']}: {err!r}",
                          file=sys.stderr)
                    agentic_policy.decision_errors += 1
                    return row["episode_id"], None  # the human works this touch

            with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
                for episode_id, action in pool.map(decide, snap_rows):
                    chosen[episode_id] = action
        elif policy_name == "system":
            pick = route_step(snap_rows)
            for row in snap_rows:
                chosen[row["episode_id"]] = pick(row)
        else:  # personas
            for row in snap_rows:
                case = live[row["episode_id"]]["case"]
                chosen[row["episode_id"]] = human_action(
                    env, case, engine.payers[case.payer_id]
                )

        finished = []
        for row in snap_rows:
            episode_id = row["episode_id"]
            slot = live[episode_id]
            case = slot["case"]
            payer = engine.payers[case.payer_id]
            action = chosen[episode_id]
            if action is None:  # escalated: the human works this touch
                slot["escalated_touches"] += 1
                action = human_action(env, case, payer)
            snapshot = state_tabular(case, payer, 0.0)
            slot["pending"].append((snapshot, action))
            case = env.apply(case, action)
            slot["case"] = case
            if case.terminal:
                success, score, n_mistakes = outcome_v2(env, case, slot["pending"])
                results[episode_id] = {
                    "success": success, "score": score, "mistakes": n_mistakes,
                    "touches": len(case.action_history),
                    "collected_payer": case.collected_payer,
                    "collected_patient": case.collected_patient,
                    "escalated_touches": slot["escalated_touches"],
                    "resolution": case.resolution,
                    "allowed": case.allowed,
                    "actions": list(case.action_history),
                }
                finished.append(episode_id)
        for episode_id in finished:
            del live[episode_id]
    return results


def summary(results):
    n = len(results)
    return {
        "episodes": n,
        "success_rate_v2": round(sum(r["success"] for r in results.values()) / n, 4),
        "mean_score": round(float(np.mean([r["score"] for r in results.values()])), 4),
        "payer_collected_total": round(
            sum(r["collected_payer"] for r in results.values()), 2
        ),
        "payer_collected_capped": round(
            sum(min(r["collected_payer"], r["allowed"]) for r in results.values()), 2
        ),
        "payer_over_collection": round(
            sum(max(0.0, r["collected_payer"] - r["allowed"])
                for r in results.values()), 2
        ),
        "patient_collected_total": round(
            sum(r["collected_patient"] for r in results.values()), 2
        ),
        "mean_touches": round(
            float(np.mean([r["touches"] for r in results.values()])), 3
        ),
        "episodes_with_mistakes": sum(
            1 for r in results.values() if r["mistakes"] > 0
        ),
        "escalated_touches": sum(r["escalated_touches"] for r in results.values()),
        "resolutions": dict(Counter(r["resolution"] for r in results.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="agentic,claude,system,personas")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--tranche", type=int, default=ROLLOUT_TRANCHE,
                        help="fresh-episode tranche number (seed lineage draw)")
    parser.add_argument("--report", default="comparison.json",
                        help="report filename under runs/reports/")
    parser.add_argument("--tag", default="",
                        help="suffix for episodes_<arm><tag>.json artefacts")
    args = parser.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    world_ext, engine = training_world(FACTORY_ROOT, MASTER_SEED)
    start = world_ext.studies.height
    rollout_studies = expansion_studies(
        world_ext, MASTER_SEED, args.tranche, ALLOCATION, start
    )
    world = extend_world(world_ext, rollout_studies)
    indices = list(range(start, start + args.episodes))
    print(f"comparing {arms} on {args.episodes} fresh episodes of tranche "
          f"{args.tranche} (indices {start}..{start + args.episodes - 1})")

    report = {
        "analysis": "agentic baseline vs ammonix system vs personas",
        "factory_root": str(FACTORY_ROOT),
        "tranche": args.tranche,
        "n_episodes": args.episodes,
        "note": (
            "identical fresh episodes and engine randomness across arms; "
            "the 'agentic' arm runs the same pinned 27B as Ammonix M1 "
            "(architecture comparison); the 'claude' arm runs the same agent "
            "loop on claude-sonnet-5 via API (frontier-model comparison; not "
            "sampling-deterministic - the decision cache freezes the run); "
            "escalated touches fall back to the persona (human) policy in "
            "every arm"
        ),
    }
    for arm in arms:
        started = time.perf_counter()
        agentic_policy, route_step = None, None
        if arm == "agentic":
            llm = AgentLLM(cache_dir=BASE / "cache" / "decisions")
            agentic_policy = AgenticPolicy(llm)
        elif arm == "claude":
            from agentic.llm_claude import ClaudeLLM

            llm = ClaudeLLM(cache_dir=BASE / "cache" / "decisions_claude")
            agentic_policy = AgenticPolicy(llm)
        elif arm == "agentic_rag":
            from agentic.rag_policy import RagPolicy

            llm = AgentLLM(cache_dir=BASE / "cache" / "decisions_rag")
            agentic_policy = RagPolicy(llm, FACTORY_ROOT)
        elif arm in ("sol", "sol_rag"):
            # GPT-5.6 Sol through the factory UI's OpenAI client (key read
            # in-process from <factory>/data/openai_key.txt; own cache dir)
            sys.path.insert(0, str(FACTORY_ROOT))
            from ui.openai_llm import SolAgentLLM

            llm = SolAgentLLM(cache_dir=FACTORY_ROOT / "data" / f"sol_decisions_{arm}")
            if arm == "sol":
                agentic_policy = AgenticPolicy(llm)
            else:
                from agentic.rag_policy import RagPolicy

                agentic_policy = RagPolicy(llm, FACTORY_ROOT)
        elif arm in ("sol_tuned", "sol_rag_tuned"):
            # the pre-registered TUNED Sol configuration (P2 prompt, k=25):
            # preregistration/comparison_tranches101_102.json
            sys.path.insert(0, str(FACTORY_ROOT))
            from agentic.briefing import TUNED_SYSTEM_PROMPT
            from ui.openai_llm import SolAgentLLM

            llm = SolAgentLLM(cache_dir=FACTORY_ROOT / "data" / f"sol_decisions_{arm}")
            if arm == "sol_tuned":
                agentic_policy = AgenticPolicy(llm, system_prompt=TUNED_SYSTEM_PROMPT)
            else:
                from agentic.rag_policy import RagPolicy

                agentic_policy = RagPolicy(llm, FACTORY_ROOT, k=25,
                                           system_prompt=TUNED_SYSTEM_PROMPT)
        elif arm == "system":
            route_step = system_router(world, engine)
        results = play(arm, world, engine, indices,
                       agentic_policy=agentic_policy, route_step=route_step)
        episodes_out = BASE / "runs" / "reports" / f"episodes_{arm}{args.tag}.json"
        episodes_out.write_text(json.dumps(results, indent=1), encoding="utf-8",
                                newline="\n")
        arm_summary = summary(results)
        arm_summary["wall_seconds"] = round(time.perf_counter() - started, 1)
        if agentic_policy is not None:
            arm_summary["policy"] = agentic_policy.stats()
            arm_summary["llm"] = agentic_policy.llm.stats()
        report[arm] = arm_summary
        print(f"[{arm}] {json.dumps(arm_summary, indent=1)}")

    out = BASE / "runs" / "reports" / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():  # partial runs merge into the existing report
        merged = json.loads(out.read_text(encoding="utf-8"))
        merged.update(report)
        report = merged
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
