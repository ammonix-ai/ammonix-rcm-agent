"""Kernel-Ether rollout (round 2; analysis only - no build artefact changes,
no sealed data, no LLM).

Round 1 (scripts/kernel_ether_probe.py) showed that counting the outcomes of
label-space neighbours estimates P(success | action, state) as well as the
shipped fitted models (AUROC 0.874 vs 0.875). Round 2 asks whether an agent
that DECIDES by counting collects the same dollars.

Two arms play the same 300 fresh tranche-92 episodes under identical engine
randomness (the policy_rollout setup):

  system  - the shipped v0.4/v0.5 decision rule (route_case unchanged)
  kernel  - identical control flow, but the per-action episode-success scores
            are neighbour counts: the live case's u coordinate is computed by
            the swarm (Lambda places, as in [1]), the stored label-space index
            supplies the neighbours, and each applicable trained action is
            scored by the Laplace-smoothed success rate of up to K=200
            neighbours that took it (minimum support 3; below that, the
            corpus-level success rate for the action - the wide-bandwidth
            limit of the kernel). Constant-prior actions, the CO-16 forcing,
            the blown-clock rule, the resolution path-value block, the 0.05
            floor, the EV close-out, and persona escalation are identical in
            both arms.

Personas are replayed too, as the shared baseline. Threads capped for the workstation's
laptop. Writes runs/reports/kernel_ether_rollout.json.
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from policy_rollout import (  # noqa: E402
    CONFIDENCE_FLOOR,
    N_EPISODES,
    ROLLOUT_TRANCHE,
    case_view,
    outcome_v2,
    summary,
)
from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import CardessaEnvironment, state_tabular  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.harness import (  # noqa: E402
    BasisRuntime,
    applicable_actions,
    close_out_choice,
    expected_values,
    route_case,
)
from cardessa.livecases import training_world  # noqa: E402
from cardessa.personas import choose_action  # noqa: E402

K_NEIGHBOURS = 200
MIN_SUPPORT = 3
LAPLACE = 2
QUERY_N = 2000  # neighbours fetched per query before per-action counting


class KernelEther:
    """Neighbour-count scorer over the stored label-space index."""

    def __init__(self, rt: BasisRuntime):
        meta = json.loads(
            (ROOT / "basis" / "feat_index" / "meta.json").read_text(
                encoding="utf-8"
            )
        )
        by_id = {
            s: (a, int(y))
            for s, a, y in zip(
                meta["state_id"], meta["true_action_id"], meta["outcome_success"]
            )
        }
        self.actions = np.asarray(
            [by_id[s][0] for s in rt.index_state_ids]
        )
        self.success = np.asarray(
            [by_id[s][1] for s in rt.index_state_ids], dtype=int
        )
        self.rt = rt
        # corpus-level success rate per action: the wide-bandwidth fallback
        self.global_rate = {
            a: float(self.success[self.actions == a].mean())
            for a in rt.trained_actions
        }

    def live_u(self, x: np.ndarray) -> np.ndarray:
        rt = self.rt
        u = np.empty((1, len(rt.u_order)), dtype="float64")
        for j, action in enumerate(rt.u_order):
            raw = rt.refit_models[action].predict_proba(x)[0, 1]
            u[0, j] = float(rt.calibrators[action].predict([raw])[0])
        return u

    def scores_for(self, x: np.ndarray, wanted: set[str]) -> dict[str, float]:
        u = self.live_u(x)
        n = min(QUERY_N, len(self.actions))
        _, idx = self.rt.index.kneighbors(u, n_neighbors=n)
        nbr_actions = self.actions[idx[0]]
        nbr_success = self.success[idx[0]]
        out: dict[str, float] = {}
        for action in wanted:
            m = nbr_actions == action
            take = np.where(m)[0][:K_NEIGHBOURS]
            support = len(take)
            if support >= MIN_SUPPORT:
                out[action] = float(
                    nbr_success[take].sum() / (support + LAPLACE)
                )
            else:
                out[action] = self.global_rate[action]
        return out


def route_case_kernel(rt, case_row, features, kernel: KernelEther):
    """route_case with one swap: episode-success scores come from neighbour
    counts instead of the fitted models. Control flow mirrors
    cardessa.harness.route_case; the resolution path-value block reuses the
    shipped machinery via route_case itself."""
    shipped_scores, known, forced, amb = route_case(rt, case_row, features)
    if forced is not None or not shipped_scores:
        return shipped_scores, known, forced, amb
    x = np.array([[features[n] for n in rt.feature_names]])
    applicable = applicable_actions(case_row)
    trained_wanted = {
        a for a in rt.trained_actions if a in applicable and a in shipped_scores
    }
    kernel_scores = kernel.scores_for(x, trained_wanted)
    scores = dict(shipped_scores)
    for action, val in kernel_scores.items():
        # keep a path-value substitution if it exceeded the base score
        # (that block is shared machinery, identical in both arms)
        base_was_pathvalue = shipped_scores[action] > val and action in (
            rt.resolution_models or {}
        )
        scores[action] = max(val, shipped_scores[action]) if base_was_pathvalue \
            else val
    return scores, known, None, amb


def play(policy_name, world, engine, rt, working, indices, kernel=None):
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
            if policy_name in ("system", "kernel"):
                if policy_name == "kernel":
                    scores, known, forced, _amb = route_case_kernel(
                        rt, row, feats[row["state_id"]], kernel
                    )
                else:
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
                if action is None:
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
                    "actions": list(case.action_history),
                }
                finished.append(episode_id)
        for episode_id in finished:
            del live[episode_id]
    return results


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
    kernel = KernelEther(rt)
    working = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    print(f"rolling out {N_EPISODES} fresh episodes, three arms...")

    system = play("system", world, engine, rt, working, indices)
    kernel_arm = play("kernel", world, engine, rt, working, indices, kernel)
    personas = play("personas", world, engine, rt, working, indices)

    def dollars(res):
        return round(sum(r["collected_payer"] + r["collected_patient"]
                         for r in res.values()), 2)

    divergent = [
        e for e in system
        if system[e]["actions"] != kernel_arm[e]["actions"]
    ]
    report = {
        "analysis": "kernel-Ether rollout (round 2; analysis only)",
        "n_episodes": N_EPISODES,
        "tranche": ROLLOUT_TRANCHE,
        "kernel_config": {"K": K_NEIGHBOURS, "min_support": MIN_SUPPORT,
                          "laplace": LAPLACE, "query_n": QUERY_N},
        "system": summary(system),
        "kernel": summary(kernel_arm),
        "personas": summary(personas),
        "dollars": {"system": dollars(system), "kernel": dollars(kernel_arm),
                    "personas": dollars(personas)},
        "escalated_touches": {
            "system": sum(r["escalated_touches"] for r in system.values()),
            "kernel": sum(r["escalated_touches"] for r in kernel_arm.values()),
        },
        "divergent_episodes": {"n": len(divergent), "ids": divergent[:40]},
        "note": (
            "identical fresh episodes and engine randomness per arm; escalated "
            "touches fall back to the persona policy in every arm; the kernel "
            "arm swaps only the episode-success field for neighbour counts"
        ),
    }
    out = ROOT / "runs" / "reports" / "kernel_ether_rollout.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({
        "probe": "kernel_ether_rollout", "verdict": True,
        "dollars_system": report["dollars"]["system"],
        "dollars_kernel": report["dollars"]["kernel"],
        "dollars_personas": report["dollars"]["personas"],
        "success_system": report["system"]["success_rate_v2"],
        "success_kernel": report["kernel"]["success_rate_v2"],
        "mistakes_system": report["system"]["episodes_with_mistakes"],
        "mistakes_kernel": report["kernel"]["episodes_with_mistakes"],
        "divergent_episodes": len(divergent),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
