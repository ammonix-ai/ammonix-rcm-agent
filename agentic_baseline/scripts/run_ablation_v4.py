"""Episodic-memory ablation of the Ammonix v0.4 arm on tranche 98.

Question: what does the Knowledge-Universe layer contribute versus the bare
classifiers trained on the same data? The "ammonix_v4_ablated" arm is
byte-identical to run_v4_arm.py's policy EXCEPT that escalations triggered by
universe/memory statistics are disabled - the system always acts on the
masked calibrated argmax instead of handing the touch to the clerk.

Gate classification (every escalation trigger in harness.py route_case +
the rollout loop):
- KEPT (hard rules, not memory):
  * applicability mask (applicable_actions): published payer rules + case
    facts;
  * forced CO-16 responsive action (the denial names provide_requested_info,
    first attempt only);
  * blown success clock (cumulative_delay_days > 120): the published
    outcome-definition constant SUCCESS_WINDOW_DAYS, not a universe
    statistic - this escalation remains, so ablated escalated_touches counts
    exactly the hard-rule hand-offs;
  * EV close-out on a sub-floor best score: model-derived (published money
    constants x calibrated scores); removing it would change too many
    things at once - KEPT by design, documented in the report.
- DISABLED (memory gates):
  * OOD/unknown-payer escalation (payer not in the working corpus =
    coverage of the episodic memory);
  * confidence-floor escalation (best < 0.05 AND every close-out EV
    worthless -> full arm escalates; ablated arm executes the best
    applicable argmax action anyway);
  * ambiguous-tribe ask_before_deciding routing: a memory gate, but the
    batch rollout loop (factory policy_rollout.py and run_v4_arm.py alike)
    ignores route_case's ambiguous_override - inactive here, so disabling
    it is a no-op; same for the fold-ensemble score_spread ambiguity,
    which only feeds the interactive harness retrieval path.
- Empty mask / empty scores cannot occur outside the blown-clock return:
  write_off is always applicable and always carries a constant prior. If it
  ever fired anyway it is recorded ("escalate_empty_scores") and takes the
  structural fallback = persona hand-off, like the loop always did.

Episode identity is proven twice: the personas replay must match the stored
tranche-98 comparison summary, and the "full" policy is replayed here with
gate logging and must reproduce episodes_ammonix_v4_v4t98.json per episode,
bit-exactly, before the ablated arm is played on the same dice.

Run with system python:
  python scripts\\run_ablation_v4.py
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
WORKTREES = Path(os.environ.get("CARDESSA_WORKTREES_ROOT", BASE.parents[1] / "Cardessa_worktrees"))  # historical builds; not part of the release (see REPRODUCING)
V4_ROOT = Path(os.environ.get("AMMONIX_V4_ROOT", WORKTREES / "v4-build"))
for entry in (str(BASE), str(V4_ROOT), str(V4_ROOT / "ammonix_core"),
              str(V4_ROOT / "scripts")):
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
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.harness import (  # noqa: E402
    BasisRuntime,
    close_out_choice,
    expected_values,
    route_case,
)
from cardessa.livecases import training_world  # noqa: E402
from cardessa.personas import CaseView, choose_action  # noqa: E402

from agentic.fallback import persona_action  # noqa: E402

N_EPISODES = 300
ROLLOUT_TRANCHE = int(os.environ.get("AMMONIX_TRANCHE", "98"))
_TAG = os.environ.get("AMMONIX_TAG", "_v4t98")  # e.g. _v6t99
_CMP = os.environ.get("AMMONIX_REPORT", "comparison_v4_tranche98.json")
_ARM = os.environ.get("AMMONIX_SYSTEM_ARM", "ammonix_v4")  # e.g. ammonix_v6
ALLOCATION = {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}
CONFIDENCE_FLOOR = 0.05
LINEAGE_EPISODES = 5000

GATE_CLASSIFICATION = [
    {"trigger": "applicability mask (applicable_actions)",
     "source": "published payer rules + case facts",
     "class": "hard-rule", "treatment": "kept"},
    {"trigger": "forced CO-16 responsive action (route_case early return)",
     "source": "the denial names its own responsive action; deterministic",
     "class": "hard-rule", "treatment": "kept"},
    {"trigger": "blown success clock: cumulative_delay_days > 120 -> escalate",
     "source": "published outcome constant SUCCESS_WINDOW_DAYS, not memory",
     "class": "hard-rule", "treatment": "kept (still escalates to the clerk)"},
    {"trigger": "EV close-out when best calibrated score < 0.05",
     "source": "published money constants x calibrated scores (model-derived)",
     "class": "model-derived, floor-triggered",
     "treatment": "kept by design: removing it would change too many things "
                  "at once; the floor survives ONLY as this branch's trigger"},
    {"trigger": "OOD/unknown-payer: payer_id not in working_payers -> escalate",
     "source": "working-corpus coverage = episodic-memory footprint",
     "class": "memory-gate", "treatment": "disabled (acts on scores anyway)"},
    {"trigger": "confidence-floor escalation: best < 0.05 AND close-out EV "
                "worthless -> escalate",
     "source": "universe/calibration statistics",
     "class": "memory-gate",
     "treatment": "disabled (executes the best applicable argmax action)"},
    {"trigger": "ambiguous-tribe ask_before_deciding (route_case "
                "ambiguous_override; tribe-scoped skills)",
     "source": "universe/tribe resolution-margin statistics",
     "class": "memory-gate",
     "treatment": "inactive in the batch rollout loop (ablation is a no-op "
                  "here; documented)"},
    {"trigger": "fold-ensemble score_spread ambiguity (RetrievalContext)",
     "source": "universe statistics",
     "class": "memory-gate",
     "treatment": "not part of this loop (interactive harness path only)"},
    {"trigger": "empty scores without a blown clock",
     "source": "cannot occur: write_off is always applicable and carries a "
               "constant prior",
     "class": "n/a",
     "treatment": "recorded as escalate_empty_scores if it ever fires; "
                  "structural fallback = persona hand-off"},
]


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


def route_decision(rt, row, feats, ablated):
    """One system decision + the gate that produced it.

    Mirrors run_v4_arm.py/policy_rollout.py exactly for ablated=False; for
    ablated=True only the memory-gated escalations are replaced (see module
    docstring). Returns (action_or_None, gate, detail)."""
    scores, known, forced, _amb = route_case(rt, row, feats)
    if forced is not None:
        return forced, "forced_co16", {}
    if not scores:
        gate = ("escalate_blown_clock" if row["cumulative_delay_days"] > 120
                else "escalate_empty_scores")
        return None, gate, {}
    best_action, best = max(scores.items(), key=lambda kv: kv[1])
    detail = {
        "best_action": best_action,
        "best_score": round(best, 4),
        "scores": {a: round(v, 4) for a, v in scores.items()},
    }
    if not known:
        if not ablated:
            return None, "escalate_ood", detail
        detail["ood_overridden"] = True  # memory gate disabled
    if best >= CONFIDENCE_FLOOR:
        return best_action, "argmax", detail
    ev = expected_values(row, scores)
    detail["close_out_ev"] = {
        a: ev[a] for a in ("bill_secondary", "bill_patient", "write_off")
        if a in ev
    }
    choice = close_out_choice(ev, scores)
    if choice is not None:
        return choice, "ev_close_out", detail
    if ablated:  # memory gate disabled: act on the argmax anyway
        return best_action, "argmax_below_floor", detail
    return None, "escalate_floor", detail


def play(policy_name, world, engine, rt, working, indices):
    """policy_name in ('personas', 'full', 'ablated'). Returns (results, logs):
    logs[episode_id] = {facts, touches: [per-touch gate records]}."""
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    live = {}
    logs = {}
    for index in indices:
        case = env.reset(index)
        live[case.episode_id] = {"case": case, "pending": [], "escalated_touches": 0}
        logs[case.episode_id] = {
            "facts": {
                "payer_id": case.payer_id,
                "cpt": case.study["cpt"],
                "family": case.study["family"],
                "allowed": round(case.allowed, 2),
                "persona": case.persona,
            },
            "touches": [],
        }
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
        feats = {}
        if policy_name != "personas":
            snaps = pl.DataFrame(
                [{c: r.get(c) for c in working.columns} for r in snap_rows],
                schema_overrides=working.schema,
            )
            frame, _names, _round, _pin = build_kept_features(
                V4_ROOT, pl.concat([working, snaps], how="vertical_relaxed")
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
            action, gate, detail = None, "personas", {}
            if policy_name != "personas":
                action, gate, detail = route_decision(
                    rt, row, feats[row["state_id"]],
                    ablated=(policy_name == "ablated"),
                )
            if action is None and policy_name != "personas":
                slot["escalated_touches"] += 1
            if action is None:  # personas arm, or an escalated (hard-rule) touch
                action = persona_action(
                    choose_action, case.persona, case_view(case, payer),
                    MASTER_SEED, case.episode_id, case.patient_share_pending,
                    env.legal_actions(case),
                )
            logs[episode_id]["touches"].append({
                "touch": case.touch_seq,
                "carc": row["carc"],
                "balance": row["balance"],
                "gate": gate,
                "action": action,
                **detail,
            })
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
    return results, logs


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


def outcome_tuple(r):
    """Lexicographic outcome order: success, then v2 score, then capped
    payer dollars. Used to classify divergences as saved/cost/neutral."""
    return (int(r["success"]), round(r["score"], 4),
            round(min(r["collected_payer"], r["allowed"]), 2))


def brief(r):
    return {
        "success": r["success"], "score": r["score"], "mistakes": r["mistakes"],
        "collected_payer": round(r["collected_payer"], 2),
        "collected_patient": round(r["collected_patient"], 2),
        "resolution": r["resolution"], "actions": r["actions"],
    }


def main() -> int:
    report_path = BASE / "runs" / "reports" / _CMP
    comparison = json.loads(report_path.read_text(encoding="utf-8"))
    stored_full = json.loads(
        (BASE / "runs" / "reports" / f"episodes_{_ARM}{_TAG}.json").read_text(
            encoding="utf-8"
        )
    )
    stored_llm = json.loads(
        (BASE / "runs" / "reports" / f"episodes_agentic{_TAG}.json").read_text(
            encoding="utf-8"
        )
    )

    world_ext, engine = training_world(V4_ROOT, MASTER_SEED)
    start = world_ext.studies.height
    assert start == LINEAGE_EPISODES
    rollout_studies = expansion_studies(
        world_ext, MASTER_SEED, ROLLOUT_TRANCHE, ALLOCATION, start
    )
    world = extend_world(world_ext, rollout_studies)
    indices = list(range(start, start + N_EPISODES))
    print(f"ablation on tranche {ROLLOUT_TRANCHE} episodes "
          f"{start}..{start + N_EPISODES - 1}")

    rt = BasisRuntime.load(V4_ROOT)
    working = pl.read_parquet(
        V4_ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )

    # CANARY 1: personas replay vs the stored comparison summary
    per_results, _ = play("personas", world, engine, rt, working, indices)
    canary = {k: v for k, v in summary(per_results).items()
              if k not in ("wall_seconds", "escalated_touches")}
    stored_per = {k: v for k, v in comparison["personas"].items()
                  if k not in ("wall_seconds", "escalated_touches")}
    if canary != stored_per:
        print("CANARY MISMATCH (personas) - aborting.")
        return 1
    print("canary ok: personas replay matches the stored summary exactly")

    # CANARY 2: the full policy replayed WITH gate logging must reproduce the
    # stored ammonix_v4 per-episode results bit-exactly
    full_results, full_logs = play("full", world, engine, rt, working, indices)
    if full_results != stored_full:
        diffs = [e for e in stored_full if full_results.get(e) != stored_full[e]]
        print(f"CANARY MISMATCH (full replay) on {len(diffs)} episodes, "
              f"e.g. {diffs[:3]} - aborting.")
        return 1
    print("canary ok: full-arm replay reproduces episodes_ammonix_v4_v4t98.json")

    started = time.perf_counter()
    ablated_results, ablated_logs = play("ablated", world, engine, rt, working,
                                         indices)
    wall = round(time.perf_counter() - started, 1)

    (BASE / "runs" / "reports" / f"episodes_{_ARM}_ablated{_TAG}.json").write_text(
        json.dumps(ablated_results, indent=1), encoding="utf-8", newline="\n"
    )

    gate_counts_full = Counter(
        t["gate"] for log in full_logs.values() for t in log["touches"]
    )
    gate_counts_ablated = Counter(
        t["gate"] for log in ablated_logs.values() for t in log["touches"]
    )

    # per-episode diff: first divergent touch, the gate that fired in the
    # full arm there (identical action prefix => identical state => the full
    # replay log entry at that index IS the full arm's decision record)
    divergences = []
    saved = cost = neutral = 0
    for episode_id in stored_full:
        fa = full_results[episode_id]["actions"]
        ab = ablated_results[episode_id]["actions"]
        if fa == ab:
            assert outcome_tuple(full_results[episode_id]) == outcome_tuple(
                ablated_results[episode_id]
            ), f"{episode_id}: same actions, different outcome"
            continue
        idx = next(i for i in range(min(len(fa), len(ab))) if fa[i] != ab[i])
        full_touch = full_logs[episode_id]["touches"][idx]
        ablated_touch = ablated_logs[episode_id]["touches"][idx]
        ft, at = outcome_tuple(full_results[episode_id]), outcome_tuple(
            ablated_results[episode_id]
        )
        if ft > at:
            verdict = "memory_saved"
            saved += 1
        elif ft < at:
            verdict = "memory_cost"
            cost += 1
        else:
            verdict = "neutral"
            neutral += 1
        divergences.append({
            "episode_id": episode_id,
            "facts": full_logs[episode_id]["facts"],
            "diverged_at_touch": idx,
            "carc_on_table": full_touch["carc"] or None,
            "balance_at_divergence": full_touch["balance"],
            "full_gate": full_touch["gate"],
            "full_action": fa[idx],
            "ablated_gate": ablated_touch["gate"],
            "ablated_action": ab[idx],
            "argmax_wanted": ablated_touch.get("best_action"),
            "argmax_score": ablated_touch.get("best_score"),
            "verdict": verdict,
            "full_outcome": brief(full_results[episode_id]),
            "ablated_outcome": brief(ablated_results[episode_id]),
        })

    # top-3 memory-saved narratives, ranked success-flip first, then by how
    # many capped dollars the gate saved, then by score delta
    def rank(d):
        f, a = d["full_outcome"], d["ablated_outcome"]
        return (
            int(f["success"]) - int(a["success"]),
            round(min(f["collected_payer"], d["facts"]["allowed"])
                  - min(a["collected_payer"], d["facts"]["allowed"]), 2),
            round(f["score"] - a["score"], 4),
        )

    top_saved = sorted(
        (d for d in divergences if d["verdict"] == "memory_saved"),
        key=rank, reverse=True,
    )[:3]
    narratives = []
    for d in top_saved:
        episode_id = d["episode_id"]
        signal = {
            "escalate_floor": "confidence: no applicable action cleared the "
                              "0.05 calibrated floor and every close-out EV "
                              "was worthless",
            "escalate_ood": "OOD: payer outside the working corpus",
            "escalate_blown_clock": "hard rule (not memory): success clock "
                                    "blown",
        }.get(d["full_gate"], d["full_gate"])
        narratives.append({
            "episode_id": episode_id,
            "claim": d["facts"],
            "diverged_at_touch": d["diverged_at_touch"],
            "carc_on_table": d["carc_on_table"],
            "balance": d["balance_at_divergence"],
            "argmax_wanted": {
                "action": d["argmax_wanted"], "score": d["argmax_score"],
            },
            "universe_signal": signal,
            "escalation_led_to": {
                "clerk_action": d["full_action"],
                "full_arm_actions": d["full_outcome"]["actions"],
            },
            "ablated_instead": {
                "action": d["ablated_action"], "gate": d["ablated_gate"],
                "ablated_arm_actions": d["ablated_outcome"]["actions"],
            },
            "outcome_full": {k: v for k, v in d["full_outcome"].items()
                             if k != "actions"},
            "outcome_ablated": {k: v for k, v in d["ablated_outcome"].items()
                                if k != "actions"},
            "llm_agent_same_episode": (
                brief(stored_llm[episode_id]) if episode_id in stored_llm
                else None
            ),
        })

    report = {
        "_provenance": {
            "analysis": "episodic-memory ablation (post-hoc analysis; NOT a "
                        "pre-registered headline result)",
            "date": "2026-07-22",
            "arm_definition": (
                "ammonix_v4_ablated = the shipped v0.4 policy with every "
                "universe/memory-gated escalation disabled: acts on the "
                "masked calibrated argmax where the full arm handed the "
                "touch to the clerk. Kept: applicability mask, forced CO-16 "
                "routing, blown-clock escalation (hard rules), and the EV "
                "close-out branch (model-derived; kept by design so only "
                "escalation behaviour changes). Same tranche-98 episodes, "
                "same dice; personas + full-replay canaries enforced."
            ),
            "gate_classification": GATE_CLASSIFICATION,
            "outcome_order": "divergence verdicts compare (success, v2 "
                             "score, capped payer dollars) lexicographically",
        },
        "aggregate": {
            f"{_ARM}_full": comparison[_ARM],
            "ammonix_v4_ablated": {**summary(ablated_results),
                                   "wall_seconds": wall},
            "agentic_llm": comparison["agentic"],
            "personas": comparison["personas"],
        },
        "decisions_by_gate": {
            "full": dict(sorted(gate_counts_full.items())),
            "ablated": dict(sorted(gate_counts_ablated.items())),
        },
        "divergences": {
            "n_divergent_episodes": len(divergences),
            "memory_saved": saved,
            "memory_cost": cost,
            "neutral": neutral,
            "episodes": divergences,
        },
        "top_memory_saved": narratives,
    }
    out = BASE / "runs" / "reports" / f"ablation{_TAG}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({
        "ablated": report["aggregate"]["ammonix_v4_ablated"],
        "decisions_by_gate": report["decisions_by_gate"],
        "divergent_episodes": len(divergences),
        "memory_saved": saved, "memory_cost": cost, "neutral": neutral,
    }, indent=1))
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
