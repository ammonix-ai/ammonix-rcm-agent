"""Play the shipped Ammonix v0.4 runtime (v4-build worktree) on the SAME
fresh episodes as the pre-registered tranche-98 comparison, and merge the
result into the given report as "ammonix_v4".

Unlike run_v3_arm.py (which had to reconstruct the v0.2-lineage world under
v3 code), everything here runs natively under the v0.4 codebase: the world is
the v0.4 training lineage (1000 + tranches 1-8 = 5000 episodes, replayed from
v4-build's recorded expansion.json), and the comparison tranche is appended
after it - the identical generation path scripts/run_comparison.py uses when
pointed at the v4 root. Episode identity is proven, not assumed: the personas
arm is replayed here and must reproduce the personas summary stored in the
report bit-exactly, else the script aborts.

Policy is the shipped v0.4 system exactly as the factory's
scripts/policy_rollout.py plays it: route_case (forced CO-16 responsive
action honoured) + confidence floor + EV close-out (expected_values /
close_out_choice); escalations fall back to the persona policy.

Run with system python, e.g.:
  python scripts\\run_v4_arm.py --report comparison_v4_tranche98.json --tag _v4t98
Smoke-load the basis only (no episode generation): --smoke-load
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
WORKTREES = BASE.parent / "Ammonix_Generic" / "Cardessa_worktrees"
V4_ROOT = Path(os.environ.get("AMMONIX_V4_ROOT", WORKTREES / "v4-build"))
ARM_NAME = os.environ.get("AMMONIX_SYSTEM_ARM", "ammonix_v4")  # e.g. ammonix_v6
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
ROLLOUT_TRANCHE = 98  # pre-registered: preregistration/comparison_v4_tranche98.json
ALLOCATION = {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}
CONFIDENCE_FLOOR = 0.05
LINEAGE_EPISODES = 5000  # v0.4 lineage: 1000 world + 8 x 500 expansion


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


def play(policy_name, world, engine, rt, working, indices):
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
        feats = {}
        if policy_name == "system":
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
            action = None
            if policy_name == "system":
                # the shipped v0.4 policy, verbatim from the factory's
                # policy_rollout.py: forced action honoured, confidence
                # floor, EV close-out instead of abandoning value
                scores, known, forced, _amb = route_case(
                    rt, row, feats[row["state_id"]]
                )
                action = forced
                if action is None and known and scores:
                    best_action, best = max(scores.items(), key=lambda kv: kv[1])
                    if best >= CONFIDENCE_FLOOR:
                        action = best_action
                    else:
                        action = close_out_choice(
                            expected_values(row, scores), scores
                        )
            if action is None:  # personas arm, or escalated system touch
                if policy_name == "system":
                    slot["escalated_touches"] += 1
                # arm-agnostic persona hand-off with the unknown-CARC
                # close-out fallback (see agentic/fallback.py); neither the
                # personas nor the system arm can structurally reach a CO-18
                # state, so this is identity here - kept for symmetry
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
    parser.add_argument("--report", default="comparison_v4_tranche98.json",
                        help="report filename under runs/reports/ (must already "
                             "hold the personas arm to verify against)")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--tranche", type=int, default=ROLLOUT_TRANCHE)
    parser.add_argument("--tag", default="_v4t98",
                        help="suffix for the episodes_ammonix_v4<tag>.json artefact")
    parser.add_argument("--smoke-load", action="store_true",
                        help="load the v0.4 BasisRuntime and exit (no episodes)")
    args = parser.parse_args()

    if args.smoke_load:
        rt = BasisRuntime.load(V4_ROOT)
        print(json.dumps({
            "basis_id": rt.manifest.basis_id,
            "n_features": len(rt.feature_names),
            "trained_actions": rt.trained_actions,
            "n_tribes": len(rt.tribes),
            "fold_models": sorted(rt.fold_models or {}),
            "resolution_heads": sorted(rt.resolution_models or {}),
        }, indent=1))
        return 0

    report_path = BASE / "runs" / "reports" / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "personas" not in report:
        print(f"{args.report} has no personas arm to verify against; run the "
              "main comparison (personas arm) first")
        return 1

    world_ext, engine = training_world(V4_ROOT, MASTER_SEED)
    start = world_ext.studies.height
    assert start == LINEAGE_EPISODES, (
        f"v0.4 lineage should end at {LINEAGE_EPISODES} studies, got {start}"
    )
    rollout_studies = expansion_studies(
        world_ext, MASTER_SEED, args.tranche, ALLOCATION, start
    )
    world = extend_world(world_ext, rollout_studies)
    indices = list(range(start, start + args.episodes))
    print(f"v0.4 arm on tranche {args.tranche} episodes "
          f"{start}..{start + args.episodes - 1}")

    # CANARY: the personas replay must reproduce the stored summary exactly,
    # proving these are byte-identical episodes and engine randomness.
    canary = summary(play("personas", world, engine, None, None, indices))
    stored = {k: v for k, v in report["personas"].items()
              if k not in ("wall_seconds", "escalated_touches")}
    replayed = {k: v for k, v in canary.items()
                if k not in ("wall_seconds", "escalated_touches")}
    if replayed != stored:
        print("CANARY MISMATCH - episodes are not identical; aborting.")
        print("stored:  ", json.dumps(stored, sort_keys=True))
        print("replayed:", json.dumps(replayed, sort_keys=True))
        return 1
    print("canary ok: personas replay matches the stored summary exactly")

    rt = BasisRuntime.load(V4_ROOT)
    working = pl.read_parquet(
        V4_ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    started = time.perf_counter()
    results = play("system", world, engine, rt, working, indices)
    (BASE / "runs" / "reports" / f"episodes_{ARM_NAME}{args.tag}.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8", newline="\n"
    )
    arm_summary = summary(results)
    arm_summary["wall_seconds"] = round(time.perf_counter() - started, 1)
    arm_summary["factory_root"] = str(V4_ROOT)
    arm_summary["decision_field"] = getattr(rt, "decision_field", "model")
    report[ARM_NAME] = arm_summary
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"[{ARM_NAME}] {json.dumps(arm_summary, indent=1)}")
    print(f"report updated: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
