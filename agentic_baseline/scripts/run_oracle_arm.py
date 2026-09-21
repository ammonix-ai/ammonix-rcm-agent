"""Oracle-greedy arm on tranche 98: how much of the world is winnable at all?

EVALUATION-ONLY UPPER-BOUND ESTIMATE. At each touch the policy picks, over
env.legal_actions(case), the action with the highest TRUE per-touch
resolution value, taken from the engine's own parameters
(engine.action_resolution_prob + the deterministic adjudication cascade).
It is GREEDY PER-TOUCH, not a full sequential planner: it ignores the
120-day clock and the touch budget when choosing, so a perfect planner
could only do better - and a lucky myopic line can occasionally do worse
than planning. Treat the number as an estimate of the world's ceiling, not
the exact optimum.

Per-action oracle value (all evaluation-only; no realized dice are peeked -
probabilities and deterministic rules only):
- request_retro_auth / appeal_with_necessity / request_peer_to_peer /
  provide_requested_info: engine.action_resolution_prob(payer, action,
  oracle_context(row)) - exactly the factory's final_exam_v4.py oracle.
  A contest already attempted this episode is valued 0: resolution draws
  are seeded per (purpose, payer, episode), so a failed contest repeats
  identically (engine truth). appeal/p2p are valued only with a denial
  standing; the corpus would mechanically allow contesting a claim that
  was never adjudicated, but that pathway is a world artifact and using
  it would inflate the bound.
- submit_clean / submit_with_records / correct_and_resubmit: 0 when the
  claim is already paid up (the v0.4 pays-once rule answers CO-18); else a
  PURE probe of engine.adjudicate on the exact Claim the env would submit
  (records attached so the seeded CO-16 draw is never consulted: every
  rule before it is deterministic). Probe pays with payer money -> 1.0;
  probe "pays" $0 (all-deductible claim) -> 1.0 only when the first
  adjudication would establish a patient share a secondary payer then
  settles, else 0 (resubmitting an all-deductible claim moves nothing)
  (submit_clean scaled by 1 - missing_info_rate for its CO-16 exposure);
  probe denies with CARC D -> the value of submitting is the best
  action_resolution_prob still available AFTER D (appeal / p2p / retro on
  CO-197) - one lookahead step, because the denial is the gateway to the
  contest. provide_requested_info is likewise scaled by what its
  reprocessing would then do.
- bill_secondary: 1.0 when the pending patient share exists and closing it
  reaches the success fraction (the corpus pays it deterministically);
  else 0.
- bill_patient: the engine's own patient-pay probability (0.7 <= $300,
  0.4 above) when the pending share exists and patient money (capped at
  the contractual share) would reach the success fraction; else 0.
- write_off: 0. No escalation in this arm.
Choice: argmax over legal actions with value > 0; deterministic tie-break
by the fixed ORDER list. If every value is 0 the claim is closed out:
bill_patient when a patient share is pending, else write_off.

Same 300 tranche-98 episodes, same dice, same loop, personas canary.
Appends arm "oracle_greedy" to comparison_v4_tranche98.json and writes the
structurally-lost analysis to runs/reports/oracle_v4_tranche98.json.
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
WORKTREES = BASE.parent / "Ammonix_Generic" / "Cardessa_worktrees"
V4_ROOT = Path(os.environ.get("AMMONIX_V4_ROOT", WORKTREES / "v4-build"))
for entry in (str(BASE), str(V4_ROOT), str(V4_ROOT / "ammonix_core"),
              str(V4_ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import (  # noqa: E402
    MISTAKE_PENALTY,
    SUCCESS_FRACTION,
    CardessaEnvironment,
    compute_mistakes,
    state_tabular,
)
from cardessa.engine import patient_share  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.livecases import training_world  # noqa: E402
from cardessa.personas import CaseView, choose_action  # noqa: E402

from agentic.fallback import persona_action  # noqa: E402

N_EPISODES = 300
ROLLOUT_TRANCHE = int(os.environ.get("AMMONIX_TRANCHE", "98"))
_TAG = os.environ.get("AMMONIX_TAG", "_v4t98")  # e.g. _v6t99
_CMP = os.environ.get("AMMONIX_REPORT", "comparison_v4_tranche98.json")
_ARM = os.environ.get("AMMONIX_SYSTEM_ARM", "ammonix_v4")  # e.g. ammonix_v6
ALLOCATION = {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}
LINEAGE_EPISODES = 5000

# deterministic tie-break: at equal value the earlier action wins
ORDER = [
    "provide_requested_info", "request_retro_auth", "appeal_with_necessity",
    "request_peer_to_peer", "submit_with_records", "correct_and_resubmit",
    "submit_clean", "bill_secondary", "bill_patient", "write_off",
]

CONTESTS = ("request_retro_auth", "appeal_with_necessity",
            "request_peer_to_peer", "provide_requested_info")


def oracle_context(row):
    """Verbatim from v4-build scripts/final_exam_v4.py."""
    return {
        "days_since_service": row["days_since_service"],
        "strong_documentation": row["clinic_doc_quality"] >= 0.6,
    }


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


def _tried(case, action):
    if action == "request_retro_auth":
        return case.retro_requested
    if action == "appeal_with_necessity":
        return case.appealed
    return action in case.action_history


def _contest_prob(engine, case, row, action):
    """action_resolution_prob with the futile-retry and applicability gates."""
    if _tried(case, action):
        return 0.0  # per-episode seeded draw: a failed contest repeats
    carc = case.carc or ""
    if action in ("appeal_with_necessity", "request_peer_to_peer") and not carc:
        return 0.0  # only a standing denial is legitimately contestable
    if action == "provide_requested_info" and carc != "CO-16":
        return 0.0
    if action == "request_retro_auth" and (
        case.auth_ref is not None
        or case.study["cpt"] not in engine.payers[case.payer_id].auth_required_cpts
    ):
        return 0.0
    return engine.action_resolution_prob(
        case.payer_id, action, oracle_context(row)
    )


def _best_contest_after(engine, case, row, denial_carc):
    """Best oracle prob still available once denial_carc is on the table."""
    payer = engine.payers[case.payer_id]
    best = 0.0
    if not case.appealed:
        best = max(best, engine.action_resolution_prob(
            case.payer_id, "appeal_with_necessity", oracle_context(row)))
    if payer.p2p_available and "request_peer_to_peer" not in case.action_history:
        best = max(best, engine.action_resolution_prob(
            case.payer_id, "request_peer_to_peer", oracle_context(row)))
    if (denial_carc == "CO-197" and not case.retro_requested
            and case.auth_ref is None):
        best = max(best, engine.action_resolution_prob(
            case.payer_id, "request_retro_auth", oracle_context(row)))
    if denial_carc == "CO-16" and "provide_requested_info" not in case.action_history:
        best = max(best, engine.action_resolution_prob(
            case.payer_id, "provide_requested_info", oracle_context(row)))
    return best


def oracle_values(env, engine, case, row, legal):
    """True per-touch resolution value for every legal action (see docstring)."""
    payer = engine.payers[case.payer_id]
    share = patient_share(
        case.allowed,
        case.coverage["deductible_remaining"],
        case.coverage["coinsurance_pct"],
    )
    payer_remaining = round(case.allowed - share - case.collected_payer, 2)
    pending = case.patient_share_pending
    has_secondary = case.coverage["secondary_payer_id"] is not None

    submit_base = None  # lazily probed once

    def probe(corrected):
        # PURE call: adjudicate mutates nothing; records attached so the
        # seeded CO-16 draw (the only stochastic rule) is never consulted
        return engine.adjudicate(env._claim(case, True, corrected))

    values = {}
    for action in legal:
        if action in CONTESTS:
            p = _contest_prob(engine, case, row, action)
            if action == "provide_requested_info" and p > 0:
                # its reprocessing runs the same deterministic cascade
                result = probe(False)
                p *= (1.0 if result.carc in (None, "CO-16")
                      else _best_contest_after(engine, case, row, result.carc))
            values[action] = p
        elif action in ("submit_clean", "submit_with_records",
                        "correct_and_resubmit"):
            if case.collected_payer > 0 and payer_remaining <= 0.005:
                values[action] = 0.0  # pays-once rule: a CO-18 awaits
                continue
            corrected = action == "correct_and_resubmit"
            result = probe(corrected) if corrected else (
                submit_base := submit_base or probe(False)
            )
            if result.carc is None:
                if payer_remaining > 0.005:
                    base = 1.0  # payer money moves
                elif pending <= 0.005 and has_secondary:
                    # all-deductible claim, not yet adjudicated: paying $0
                    # still ESTABLISHES the patient share, which the
                    # secondary then settles as payer money (deterministic)
                    base = 1.0
                else:
                    # $0 moves and no share is unlocked: resubmitting an
                    # all-deductible claim forever is the trap this closes
                    base = 0.0
            else:
                base = _best_contest_after(engine, case, row, result.carc)
            if action == "submit_clean" and result.carc is None:
                base *= 1.0 - payer.missing_info_rate
            values[action] = base
        elif action == "bill_secondary":
            closes = case.collected_payer + pending >= SUCCESS_FRACTION * case.allowed
            values[action] = 1.0 if pending > 0 and closes else 0.0
        elif action == "bill_patient":
            counted = min(pending, case.contractual_share)
            closes = (case.collected_payer + counted
                      >= SUCCESS_FRACTION * case.allowed)
            if pending > 0 and closes:
                amount = pending or case.allowed
                values[action] = 0.7 if amount <= 300 else 0.4
            else:
                values[action] = 0.0
        else:  # write_off
            values[action] = 0.0
    return values


def oracle_action(env, engine, case, row):
    legal = env.legal_actions(case)
    values = oracle_values(env, engine, case, row, legal)
    best = max(values.values())
    if best > 0:
        return min((a for a, v in values.items() if v == best),
                   key=ORDER.index)
    # nothing has any true resolution value: close the claim out
    return ("bill_patient" if case.patient_share_pending > 0
            and "bill_patient" in legal else "write_off")


def play(policy_name, world, engine, indices):
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    live = {}
    for index in indices:
        case = env.reset(index)
        live[case.episode_id] = {"case": case, "pending": []}
    results = {}
    step = 0
    while live and step < 8:
        step += 1
        finished = []
        for episode_id, slot in list(live.items()):
            case = slot["case"]
            payer = engine.payers[case.payer_id]
            row = state_tabular(case, payer, 0.0)
            if policy_name == "oracle":
                action = oracle_action(env, engine, case, row)
            else:  # personas canary
                action = persona_action(
                    choose_action, case.persona, case_view(case, payer),
                    MASTER_SEED, case.episode_id, case.patient_share_pending,
                    env.legal_actions(case),
                )
            snapshot = state_tabular(case, payer, 0.0)
            slot["pending"].append((snapshot, action))
            case = env.apply(case, action)
            slot["case"] = case
            if case.terminal:
                success, score, n_mistakes = outcome_v2(env, case, slot["pending"])
                fraction = (
                    (case.collected_payer
                     + min(case.collected_patient, case.contractual_share))
                    / case.allowed if case.allowed > 0 else 0.0
                )
                results[episode_id] = {
                    "success": success, "score": score, "mistakes": n_mistakes,
                    "touches": len(case.action_history),
                    "collected_payer": case.collected_payer,
                    "collected_patient": case.collected_patient,
                    "escalated_touches": 0,
                    "resolution": case.resolution,
                    "allowed": case.allowed,
                    "actions": list(case.action_history),
                    "days_elapsed": case.days_elapsed,
                    "collected_fraction_v2": round(fraction, 4),
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
        "escalated_touches": 0,
        "resolutions": dict(Counter(r["resolution"] for r in results.values())),
    }


def main() -> int:
    report_path = BASE / "runs" / "reports" / _CMP
    comparison = json.loads(report_path.read_text(encoding="utf-8"))

    world_ext, engine = training_world(V4_ROOT, MASTER_SEED)
    start = world_ext.studies.height
    assert start == LINEAGE_EPISODES
    rollout_studies = expansion_studies(
        world_ext, MASTER_SEED, ROLLOUT_TRANCHE, ALLOCATION, start
    )
    world = extend_world(world_ext, rollout_studies)
    indices = list(range(start, start + N_EPISODES))
    print(f"oracle-greedy arm on tranche {ROLLOUT_TRANCHE} episodes "
          f"{start}..{start + N_EPISODES - 1}")

    # CANARY: personas replay must match the stored comparison summary
    canary = {k: v for k, v in summary(play("personas", world, engine,
                                            indices)).items()
              if k not in ("wall_seconds", "escalated_touches")}
    stored = {k: v for k, v in comparison["personas"].items()
              if k not in ("wall_seconds", "escalated_touches")}
    if canary != stored:
        print("CANARY MISMATCH (personas) - aborting.")
        return 1
    print("canary ok: personas replay matches the stored summary exactly")

    started = time.perf_counter()
    results = play("oracle", world, engine, indices)
    wall = round(time.perf_counter() - started, 1)
    (BASE / "runs" / "reports" / f"episodes_oracle_greedy{_TAG}.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8", newline="\n"
    )

    arm = summary(results)
    arm["wall_seconds"] = wall
    arm["note"] = (
        "EVALUATION-ONLY upper-bound ESTIMATE: greedy per-touch on the "
        "engine's true resolution probabilities (action_resolution_prob + "
        "deterministic adjudication cascade), not a full sequential planner; "
        "see scripts/run_oracle_arm.py and runs/reports/oracle_v4_tranche98.json"
    )
    comparison["oracle_greedy"] = arm
    report_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8",
                           newline="\n")

    # structurally-lost breakdown among oracle failures. The greedy line is
    # an estimate, so cross-check against every other arm: an episode any
    # arm won is winnable regardless of what the greedy oracle did.
    other_arms = {}
    for name, fn in ((_ARM, f"episodes_{_ARM}{_TAG}.json"),
                     ("agentic", f"episodes_agentic{_TAG}.json"),
                     ("personas", f"episodes_personas{_TAG}.json")):
        other_arms[name] = json.loads(
            (BASE / "runs" / "reports" / fn).read_text(encoding="utf-8")
        )
    won_by_any = {
        e for e in results
        if results[e]["success"] or any(arm[e]["success"]
                                        for arm in other_arms.values())
    }
    failures = {e: r for e, r in results.items() if not r["success"]}
    lost_on_clock = {
        e for e, r in failures.items()
        if r["collected_fraction_v2"] >= SUCCESS_FRACTION
        and r["days_elapsed"] > 120
    }
    never_resolved = {e for e in failures if e not in lost_on_clock}

    def brief(arm_summary):
        return {
            "success_rate_v2": arm_summary["success_rate_v2"],
            "payer_collected_capped": arm_summary["payer_collected_capped"],
            "patient_collected_total": arm_summary["patient_collected_total"],
            "episodes_with_mistakes": arm_summary["episodes_with_mistakes"],
        }

    gap_table = {
        "oracle_greedy": brief(arm),
        _ARM: brief(comparison[_ARM]),
        "agentic_llm": brief(comparison["agentic"]),
        "personas": brief(comparison["personas"]),
    }
    oracle_report = {
        "_provenance": {
            "analysis": "oracle-greedy world-ceiling estimate (evaluation "
                        "only; NOT a pre-registered headline result)",
            "date": "2026-07-22",
            "definition": (
                "greedy per-touch argmax of the engine's true resolution "
                "value over env.legal_actions; rework actions valued by "
                "engine.action_resolution_prob (final_exam_v4.py "
                "oracle_context), submits by a pure probe of the "
                "deterministic adjudication cascade with a one-step "
                "contest lookahead behind a forecast denial; failed "
                "contests never retried (per-episode seeded draws); no "
                "escalation; deterministic tie-break order. Greedy, not a "
                "planner: it ignores the 120-day clock and the 8-touch cap "
                "when choosing, so the true ceiling can only be higher - "
                "and single realized dice mean it is an estimate either way."
            ),
        },
        "oracle_greedy": arm,
        "structurally_lost": {
            "n_oracle_failures": len(failures),
            "lost_on_time_window": {
                "n": len(lost_on_clock),
                "definition": "collected the success fraction but past the "
                              "120-day clock",
                "episodes": sorted(lost_on_clock),
            },
            "never_resolved": {
                "n": len(never_resolved),
                "definition": "below the 98% collected fraction within 8 "
                              "touches under this world+dice (greedy line)",
                "episodes": sorted(never_resolved),
            },
            "cross_check_union": {
                "won_by_any_arm": len(won_by_any),
                "lost_by_every_arm": N_EPISODES - len(won_by_any),
                "note": "episodes some arm (oracle/ammonix/agentic/personas) "
                        "won on these dice: a sharper lower bound on the "
                        "world ceiling than the greedy oracle line alone",
            },
        },
        "gap_table": gap_table,
    }
    out = BASE / "runs" / "reports" / f"oracle{_TAG}.json"
    out.write_text(json.dumps(oracle_report, indent=2), encoding="utf-8",
                   newline="\n")
    print(json.dumps({"oracle_greedy": brief(arm),
                      "failures": len(failures),
                      "lost_on_time_window": len(lost_on_clock),
                      "never_resolved": len(never_resolved)}, indent=1))
    print(f"comparison updated: {report_path}")
    print(f"oracle analysis written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
