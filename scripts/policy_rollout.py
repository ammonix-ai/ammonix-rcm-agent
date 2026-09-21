"""Policy rollout: the SYSTEM works whole claims; the personas work the same
claims; compare collections under outcome v2. Analysis only - no build
artefact changes, no sealed data (fresh episodes), no LLM (action choice
does not involve M1; text channels are irrelevant to the 75 features).

When the system escalates, the human takes that touch (persona policy):
escalation hands off, it does not freeze the claim.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
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
from cardessa.harness import (  # noqa: E402
    BasisRuntime,
    close_out_choice,
    expected_values,
    route_case,
)  # noqa: E402
from cardessa.livecases import training_world  # noqa: E402
from cardessa.personas import CaseView, choose_action  # noqa: E402

N_EPISODES = 300
ROLLOUT_TRANCHE = 92
CONFIDENCE_FLOOR = 0.05


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


def play(policy_name, world, engine, rt, working, indices, use_ev=True):
    """Play the episodes step-synchronously; returns per-episode results.
    use_ev=False disables the v0.4 close-out EV (the v0.3-policy baseline
    for the spec s.6 deductible_heavy comparison, same dice)."""
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    live = {}
    for index in indices:
        case = env.reset(index)
        live[case.episode_id] = {"case": case, "pending": [], "escalated_touches": 0}
    results = {}
    step = 0
    while live and step < 8:
        step += 1
        # snapshot every live case's current state, batch-tokenise once
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
        snaps = pl.DataFrame(
            [{c: r.get(c) for c in working.columns} for r in snap_rows],
            schema_overrides=working.schema,
        )
        from cardessa.features_kept import build_kept_features
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
            if policy_name == "system":
                scores, known, forced, _amb = route_case(rt, row, feats[row["state_id"]])
                action = forced
                if action is None and known and scores:
                    best_action, best = max(scores.items(), key=lambda kv: kv[1])
                    if best >= CONFIDENCE_FLOOR:
                        action = best_action
                    elif use_ev:
                        # v0.4 s.3: EV close-out instead of abandoning value
                        action = close_out_choice(
                            expected_values(row, scores), scores
                        )
                if action is None:  # escalated: the human works this touch
                    slot["escalated_touches"] += 1
                    action = choose_action(case.persona, case_view(case, payer),
                                           MASTER_SEED, case.episode_id)
            else:
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
                    "contractual_share": case.contractual_share,
                    "resubmitted_after_paid": case.resubmitted_after_paid,
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
        "payer_collected_total": round(sum(r["collected_payer"] for r in results.values()), 2),
        "patient_collected_total": round(
            sum(r["collected_patient"] for r in results.values()), 2
        ),
        "mean_touches": round(float(np.mean([r["touches"] for r in results.values()])), 3),
        "episodes_with_mistakes": sum(1 for r in results.values() if r["mistakes"] > 0),
        "resolutions": dict(Counter(r["resolution"] for r in results.values())),
    }


def main() -> int:
    world_ext, engine = training_world(ROOT, MASTER_SEED)
    start = world_ext.studies.height
    rollout_studies = expansion_studies(
        world_ext, MASTER_SEED, ROLLOUT_TRANCHE,
        {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}, start,
    )
    world = extend_world(world_ext, rollout_studies)
    indices = list(range(start, start + N_EPISODES))
    rt = BasisRuntime.load(ROOT)
    working = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    print(f"rolling out {N_EPISODES} fresh episodes (indices {start}..{start + N_EPISODES - 1})")

    system = play("system", world, engine, rt, working, indices)
    personas = play("personas", world, engine, rt, working, indices)
    sys_summary, per_summary = summary(system), summary(personas)
    escalated = sum(r["escalated_touches"] for r in system.values())

    report = {
        "analysis": "policy_rollout (post-sign-off analysis; no build change)",
        "n_episodes": N_EPISODES,
        "system": sys_summary,
        "personas_baseline": per_summary,
        "system_escalated_touches": escalated,
        "note": (
            "identical fresh episodes, identical engine randomness per case; "
            "escalated touches fall back to the persona (human) policy"
        ),
    }
    (ROOT / "runs" / "reports" / "policy_rollout.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n",
    )
    print(json.dumps({"system": sys_summary, "personas": per_summary,
                      "system_escalated_touches": escalated}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
