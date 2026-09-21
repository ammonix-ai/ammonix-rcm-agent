"""Learning curve: sample efficiency of the episodic basis (analysis only).

For each memory size N, rebuild the swarm + calibrators + resolution heads
from a patient-grouped subset of the working episodes ALONE (in memory, same
functions and seeds as the P7/v0.3 builds; the shipped basis/ is untouched),
then play the SAME 200 demo claims with the same engine dice as the inbox
rollout and score them with the same capped outcome-v2 money.

The tokenizer (feature pipeline) stays fixed - the curve measures what the
classifier swarm and its universe learn from N remembered claims, mirroring
the architecture paper's data-efficiency design (phi fixed, Lambda trained
on N). Flat references (simulated human, LLM agent) come from
data/ui_rollout.json: they do not change with memory size.

Incremental: finished (n, seed) points are kept in data/ui_curve.json and
skipped on re-run. The full-size point must reproduce the inbox scoreboard
exactly (anchor check).
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.schema import FeatureSpec, FoldPlan  # noqa: E402
from ammonix_core.split import build_grouped_fold_plan  # noqa: E402
from ammonix_core.universe import build_universe  # noqa: E402

from policy_rollout import CONFIDENCE_FLOOR, case_view, outcome_v2  # noqa: E402
from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import CardessaEnvironment, state_tabular  # noqa: E402
from cardessa.features import to_qpsi_records  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.harness import (  # noqa: E402
    BasisRuntime,
    close_out_choice,
    expected_values,
    route_case,
)
from cardessa.livecases import (  # noqa: E402
    DEMO_TRANCHE_NO,
    demo_allocation,
    training_world,
)
from cardessa.personas import choose_action  # noqa: E402
from cardessa.resolution import train_resolution_heads  # noqa: E402

import os as _os  # noqa: E402

# v0.6: the curve runs with the shipped decision field (kernel) unless
# AMMONIX_DECISION_FIELD=model is set; each mode keeps its own data file so
# neither curve overwrites the other.
DECISION_FIELD = _os.environ.get("AMMONIX_DECISION_FIELD", "kernel").lower()
CURVE = ROOT / "data" / (
    "ui_curve.json" if DECISION_FIELD == "model" else "ui_curve_kernel.json"
)
SIZES = [50, 100, 250, 500, 1000, 2500]
SEEDS = [0, 1, 2]
FULL = 4750

_ACTION_IN_ERROR = re.compile(r"action '([a-z_]+)'")


def sample_episodes(episodes: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Patient-grouped subset: whole patients until n episodes, cut to n."""
    rng = np.random.default_rng(MASTER_SEED * 1000 + n * 10 + seed)
    patients = episodes["patient_id"].unique().sort().to_list()
    rng.shuffle(patients)
    chosen: list[str] = []
    for patient in patients:
        chosen.extend(
            episodes.filter(pl.col("patient_id") == patient)["episode_id"].to_list()
        )
        if len(chosen) >= n:
            break
    return episodes.filter(pl.col("episode_id").is_in(chosen[:n]))


def train_subset_runtime(
    subset_eps, subset_states, frame, names, swarm, engine, fold_plan
):
    """The swarm + calibrators + resolution heads from this memory alone.

    Actions whose subset support cannot train a classifier (no states, one
    outcome class, or a single-class fold partition) are demoted to their
    empirical constant prior - exactly what the P6 swarm round did with
    bill_secondary and write_off on the full corpus.
    """
    outcome_by_episode = {
        row["episode_id"]: (bool(row["success"]), float(row["outcome_score"]))
        for row in subset_eps.select(
            "episode_id", "success", "outcome_score"
        ).to_dicts()
    }
    subset_frame = frame.filter(
        pl.col("state_id").is_in(subset_states["state_id"].to_list())
    )
    specs = [
        FeatureSpec(name=n, dtype="float", description=f"kept tokenizer feature {n}")
        for n in names
    ]
    records = to_qpsi_records(subset_frame, specs, outcome_by_episode)

    def empirical_prior(action: str) -> float:
        own = [r.outcome_success for r in records if r.action_id == action]
        return float(np.mean(own)) if own else 0.0

    trained = list(swarm["trained_actions"])
    priors = {a: empirical_prior(a) for a in swarm["constant_prior"]}
    demoted: list[str] = []
    build = None
    while build is None:
        # pre-demote actions build_universe would reject: single outcome class
        for action in list(trained):
            own_y = {r.outcome_success for r in records if r.action_id == action}
            if len(own_y) < 2:
                trained.remove(action)
                demoted.append(action)
                priors[action] = empirical_prior(action)
        if not trained:
            break
        try:
            build = build_universe(
                records, fold_plan, names, trained, priors,
                seed=MASTER_SEED,
                hyperparams=swarm["classifier_spec"]["hyperparams"],
                balanced=swarm["label_policy"]["class_weight"] == "balanced",
            )
        except ValueError as exc:  # single-class fold partition -> demote
            match = _ACTION_IN_ERROR.search(str(exc))
            if not match or match.group(1) not in trained:
                raise
            trained.remove(match.group(1))
            demoted.append(match.group(1))
            priors[match.group(1)] = empirical_prior(match.group(1))

    resolution = train_resolution_heads(
        engine, subset_states, subset_eps, subset_frame, names, fold_plan,
        MASTER_SEED,
    )
    rt = BasisRuntime(
        manifest=None, tribes=[], scaler=None, index=None, index_state_ids=[],
        calibrators=dict(build.calibrator_models) if build else {},
        refit_models=dict(build.refit_models) if build else {},
        trained_actions=list(trained),
        constant_priors=dict(priors),
        working_payers=set(subset_eps["payer_id"].unique().to_list()),
        feature_names=list(names),
        resolution_models=dict(resolution.models),
        resolution_calibrators=dict(resolution.calibrators),
        success_given_granted=dict(resolution.success_given_granted),
    )
    # v0.6: the kernel decision field for this subset. Each universe record
    # carries its out-of-fold calibrated score vector (u), the taken action
    # and the success label: exactly the stored universe the field counts
    # over. Built from the subset ALONE, so the curve measures what N
    # remembered claims give the kernel read of the Ether.
    if build is not None and DECISION_FIELD == "kernel":
        from cardessa.kernel_field import KernelField

        u_order = list(trained)
        rt.kernel_field = KernelField(
            U=np.array([[r.scores_cal[a] for a in u_order] for r in build.records]),
            state_ids=[r.state_id for r in build.records],
            actions=[r.true_action_id for r in build.records],
            success=[int(r.outcome_success) for r in build.records],
            u_order=u_order,
        )
        rt.decision_field = "kernel"
    return rt, demoted, sorted(resolution.models)


_FEATURE_PATH_CHECKED = False
_USE_CONCAT = False


def snap_features(working, snap_rows, names):
    """Features for live snapshots; tokenizer features are row-local, so the
    small-frame build equals the full-concat build (validated on first use)."""
    global _FEATURE_PATH_CHECKED, _USE_CONCAT
    snaps = pl.DataFrame(
        [{c: r.get(c) for c in working.columns} for r in snap_rows],
        schema_overrides=working.schema,
    )

    def from_frame(frame):
        return {
            r["state_id"]: {n: float(r.get(n) or 0.0) for n in names}
            for r in frame.filter(
                pl.col("state_id").is_in(snaps["state_id"].to_list())
            ).to_dicts()
        }

    fast, _, _, _ = build_kept_features(ROOT, snaps)
    fast_feats = from_frame(fast)
    if not _FEATURE_PATH_CHECKED:
        _FEATURE_PATH_CHECKED = True
        slow, _, _, _ = build_kept_features(
            ROOT, pl.concat([working, snaps], how="vertical_relaxed")
        )
        slow_feats = from_frame(slow)
        drift = max(
            abs(fast_feats[s][n] - slow_feats[s][n])
            for s in fast_feats for n in names
        )
        if drift > 1e-9:
            print(f"feature fast path drift {drift}: falling back to concat")
            _USE_CONCAT = True
        else:
            print("feature fast path validated against full-concat build")
    if _USE_CONCAT:
        slow, _, _, _ = build_kept_features(
            ROOT, pl.concat([working, snaps], how="vertical_relaxed")
        )
        return from_frame(slow)
    return fast_feats


def play_system(world, engine, rt, working, indices):
    """The inbox rollout's system lane, byte-for-byte semantics: confidence
    floor, EV close-out, escalation to the persona, capped outcome-v2 money."""
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    live = {}
    for index in indices:
        case = env.reset(index)
        live[case.episode_id] = {"case": case, "pending": [], "escalated": 0}
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
        feats = snap_features(working, snap_rows, rt.feature_names)
        finished = []
        for row in snap_rows:
            episode_id = row["episode_id"]
            slot = live[episode_id]
            case = slot["case"]
            payer = engine.payers[case.payer_id]
            scores, known, forced, _amb = route_case(rt, row, feats[row["state_id"]])
            action = forced
            if action is None and known and scores:
                best_action, best = max(scores.items(), key=lambda kv: kv[1])
                if best >= CONFIDENCE_FLOOR:
                    action = best_action
                else:
                    action = close_out_choice(expected_values(row, scores), scores)
            if action is None:  # escalated: the human works this touch
                slot["escalated"] += 1
                action = choose_action(
                    case.persona, case_view(case, payer), MASTER_SEED, episode_id
                )
            snapshot = state_tabular(case, payer, 0.0)
            slot["pending"].append((snapshot, action))
            after = env.apply(case, action)
            slot["case"] = after
            if after.terminal:
                success, _score, n_mistakes = outcome_v2(env, after, slot["pending"])
                patient_valid = round(
                    min(after.collected_patient, after.contractual_share), 2
                )
                payer_valid = round(
                    min(after.collected_payer, after.allowed - patient_valid), 2
                )
                results[episode_id] = {
                    "collected": round(payer_valid + patient_valid, 2),
                    "payer_counted": payer_valid,
                    "patient_counted": patient_valid,
                    "mistakes": n_mistakes,
                    "success": bool(success),
                    "touches": len(after.action_history),
                    "escalated": slot["escalated"],
                }
                finished.append(episode_id)
        for episode_id in finished:
            del live[episode_id]
    return results


def verdicts(results, human_claims):
    wins = same = losses = 0
    for eid, r in results.items():
        h = human_claims[eid]["human"]
        delta = round(r["collected"] - h["collected"], 2)
        if abs(delta) > 0.005:
            wins, losses = (wins + 1, losses) if delta > 0 else (wins, losses + 1)
        elif r["mistakes"] != h["mistakes"]:
            better = r["mistakes"] < h["mistakes"]
            wins, losses = (wins + 1, losses) if better else (wins, losses + 1)
        else:
            same += 1
    return wins, same, losses


def main() -> int:
    rollout_file = ROOT / "data" / (
        "ui_rollout.json" if DECISION_FIELD == "model"
        else f"ui_rollout_{DECISION_FIELD}.json"
    )
    if not rollout_file.is_file():
        raise SystemExit(
            f"{rollout_file.name} missing: start the UI once with the "
            f"{DECISION_FIELD} decision field so the demo scoreboard is computed"
        )
    rollout = json.loads(rollout_file.read_text("utf-8"))
    human_claims = rollout["claims"]
    doc = (
        json.loads(CURVE.read_text("utf-8")) if CURVE.exists()
        else {"metric": rollout["metric"], "points": []}
    )
    done = {(p["n"], p["seed"]) for p in doc["points"]}

    states = pl.read_parquet(ROOT / "data/working/cardessa_sim/states.parquet")
    episodes = pl.read_parquet(ROOT / "data/working/cardessa_sim/episodes.parquet")
    frame, names, _, _ = build_kept_features(ROOT, states)

    rounds = json.loads((ROOT / "runs/state/swarm_rounds.json").read_text("utf-8"))
    kept = [r["round"] for r in rounds if r["kept"]][-1]
    swarm = json.loads(
        (ROOT / f"runs/manifests/swarm_round{kept}.json").read_text("utf-8")
    )

    from cardessa.expansion import expansion_studies, extend_world
    world_ext, engine = training_world(ROOT, MASTER_SEED)
    start = world_ext.studies.height
    demo_studies = expansion_studies(
        world_ext, MASTER_SEED, DEMO_TRANCHE_NO, demo_allocation(), start
    )
    world = extend_world(world_ext, demo_studies)
    indices = list(range(start, start + sum(demo_allocation().values())))
    expected_ids = {f"ep-{i:05d}" for i in indices}
    if set(human_claims) != expected_ids:
        raise SystemExit("demo episode ids do not match ui_rollout.json claims")

    todo = [(n, s) for n in SIZES for s in SEEDS] + [(FULL, 0)]
    for n, seed in todo:
        if (n, seed) in done:
            continue
        t0 = time.time()
        if n == FULL:
            subset_eps, subset_states = episodes, states
            fold_plan = FoldPlan.model_validate_json(
                (ROOT / "runs/manifests/fold_plan.json").read_text("utf-8")
            )
        else:
            subset_eps = sample_episodes(episodes, n, seed)
            subset_states = states.filter(
                pl.col("episode_id").is_in(subset_eps["episode_id"].to_list())
            )
            fold_plan = build_grouped_fold_plan(
                subset_eps, "episode_id", "success", "patient_id",
                n_folds=5 if n >= 250 else 3, seed=MASTER_SEED + seed,
            )
        rt, demoted, resolution_actions = train_subset_runtime(
            subset_eps, subset_states, frame, names, swarm, engine, fold_plan
        )
        if n == FULL:
            # the anchor point must be the shipped swarm exactly
            rt.constant_priors = {
                a: info["prior"] for a, info in swarm["constant_prior"].items()
            }
        train_s = time.time() - t0
        results = play_system(world, engine, rt, working=states, indices=indices)
        wins, same, losses = verdicts(results, human_claims)
        point = {
            "n": n, "seed": seed,
            "n_patients": subset_eps["patient_id"].n_unique(),
            "claims": len(results),
            "collected": round(sum(r["collected"] for r in results.values()), 2),
            "mistake_claims": sum(
                1 for r in results.values() if r["mistakes"] > 0
            ),
            "wins": wins, "same": same, "losses": losses,
            "escalated_touches": sum(r["escalated"] for r in results.values()),
            "trained_actions": rt.trained_actions,
            "demoted_to_prior": demoted,
            "resolution_actions": resolution_actions,
            "train_seconds": round(train_s, 1),
            "play_seconds": round(time.time() - t0 - train_s, 1),
        }
        if n == FULL:
            anchor = rollout["scoreboard"]["system_collected"]
            point["anchor_match"] = abs(point["collected"] - anchor) < 0.01
            point["anchor_delta"] = round(point["collected"] - anchor, 2)
        doc["points"].append(point)
        doc["human_collected"] = rollout["scoreboard"]["human_collected"]
        doc["human_mistake_claims"] = rollout["scoreboard"]["human_mistake_claims"]
        CURVE.write_text(
            json.dumps(doc, indent=2), encoding="utf-8", newline="\n"
        )
        print(json.dumps(point))

    full_points = [p for p in doc["points"] if p["n"] == FULL]
    ok = bool(full_points) and all(p.get("anchor_match") for p in full_points)
    print(json.dumps({
        "loop": "CURVE", "verdict": ok and len(doc["points"]) >= len(todo),
        "points": len(doc["points"]), "anchor_match": ok,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
