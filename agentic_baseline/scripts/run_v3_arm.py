"""Play the Ammonix v0.3 runtime (v3-build worktree) on the SAME 300 fresh
episodes as the main comparison, and merge the result into
runs/reports/comparison.json as "system_v3".

The v0.3 codebase extends the training lineage to 5000 episodes, so its own
recorded rollout (tranche 92 at start 5000) is a DIFFERENT draw than this
comparison's (tranche 92 at start 4000 after the v0.2 lineage). This script
therefore runs under the v3 code but reconstructs the v0.2-lineage world
(engine/world code is byte-identical between the two worktrees; only the
tranche cap constant differs). Episode identity is proven, not assumed: the
personas arm is replayed here and must reproduce the stored personas summary
exactly, else the script aborts.

Run with the v3 worktree's venv:
  <v3-build>\\.venv\\Scripts\\python.exe scripts\\run_v3_arm.py
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
WORKTREES = Path(os.environ.get("CARDESSA_WORKTREES_ROOT", BASE.parents[1] / "Cardessa_worktrees"))  # historical builds; not part of the release (see REPRODUCING)
V3_ROOT = Path(os.environ.get("AMMONIX_V3_ROOT", WORKTREES / "v3-build"))
V2_ROOT = Path(os.environ.get("AMMONIX_FACTORY_ROOT", BASE.parent))  # registered run used the historical v2-agent build (does not ship); see REPRODUCING
for entry in (str(BASE), str(V3_ROOT), str(V3_ROOT / "ammonix_core"),
              str(V3_ROOT / "scripts")):
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
from cardessa.engine import PayerEngine  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.harness import BasisRuntime, route_case  # noqa: E402
from cardessa.personas import CaseView, choose_action  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

N_EPISODES = 300
ROLLOUT_TRANCHE = 92
ALLOCATION = {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}
CONFIDENCE_FLOOR = 0.05


def v2_lineage_world():
    """The v0.2-lineage world (tranches 1-4 from the v2-agent worktree's
    recorded allocation), built with the v3 code (world code is identical)."""
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    expansion_report = json.loads(
        (V2_ROOT / "runs" / "reports" / "expansion.json").read_text(encoding="utf-8")
    )
    extra = []
    start = 1000
    for tranche in expansion_report["tranches"]:
        allocation = {
            (None if k == "None" else k): v for k, v in tranche["allocation"].items()
        }
        studies = expansion_studies(
            world, MASTER_SEED, tranche["tranche"], allocation, start
        )
        extra.append(studies)
        start += studies.height
    world_ext = extend_world(world, pl.concat(extra)) if extra else world
    return world_ext, engine


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
                V3_ROOT, pl.concat([working, snaps], how="vertical_relaxed")
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
                scores, known, forced, _amb = route_case(
                    rt, row, feats[row["state_id"]]
                )
                if known and forced is None and scores:
                    best_action, best = max(scores.items(), key=lambda kv: kv[1])
                    if best >= CONFIDENCE_FLOOR:
                        action = best_action
            if action is None:  # personas arm, or escalated system touch
                if policy_name == "system":
                    slot["escalated_touches"] += 1
                action = choose_action(case.persona, case_view(case, payer),
                                       MASTER_SEED, case.episode_id)
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
    report_path = BASE / "runs" / "reports" / "comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if "personas" not in report:
        print("comparison.json has no personas arm to verify against; run the "
              "main comparison first")
        return 1

    world_ext, engine = v2_lineage_world()
    start = world_ext.studies.height
    assert start == 4000, f"v0.2 lineage should end at 4000 studies, got {start}"
    rollout_studies = expansion_studies(
        world_ext, MASTER_SEED, ROLLOUT_TRANCHE, ALLOCATION, start
    )
    world = extend_world(world_ext, rollout_studies)
    indices = list(range(start, start + N_EPISODES))
    print(f"v0.3 arm on episodes {start}..{start + N_EPISODES - 1} "
          f"(v0.2-lineage world rebuilt under v3 code)")

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

    rt = BasisRuntime.load(V3_ROOT)
    working = pl.read_parquet(
        V3_ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    started = time.perf_counter()
    results = play("system", world, engine, rt, working, indices)
    (BASE / "runs" / "reports" / "episodes_system_v3.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8", newline="\n"
    )
    arm_summary = summary(results)
    arm_summary["wall_seconds"] = round(time.perf_counter() - started, 1)
    arm_summary["factory_root"] = str(V3_ROOT)
    report["system_v3"] = arm_summary
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"[system_v3] {json.dumps(arm_summary, indent=1)}")
    print(f"report updated: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
