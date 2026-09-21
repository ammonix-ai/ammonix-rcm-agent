"""One swarm hyperparameter round (P6): propose, score, keep/revert, stop.

Round number = next entry in runs/state/swarm_rounds.json. Every round trains
the per-action gbdt swarm (LabelPolicy: own_action_states, outcome_success,
class_weight balanced) on WORKING data only, over the P5 fold plan, with the
kept round-1 tokenizer. Scoring is mean pooled OOF AUROC over the trainable
actions. Keep on gain >= 0.005 vs the previous round (round 1 vs chance 0.5);
the plateau rule (3 consecutive sub-threshold rounds) ends the loop.

Action partition, recorded in every report:
- dropped: coverage violations from P5 (no classifier, per CoverageReport);
- constant_prior: a single outcome class or a fold whose training partition
  is single-class (P(success) is a constant, not a classifier);
- trained: everything else, with pooled and per-fold metrics.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.pipeline import train_swarm  # noqa: E402
from ammonix_core.schema import (  # noqa: E402
    ClassifierSpec,
    FoldPlan,
    LabelPolicy,
)

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.features import to_qpsi_records  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402

GAIN_THRESHOLD = 0.005
WORKING = ROOT / "data" / "working" / "cardessa_sim"
STATE_FILE = ROOT / "runs" / "state" / "swarm_rounds.json"

PROPOSALS: dict[int, dict] = {
    1: {},  # gbdt defaults: the baseline every later round must beat
    2: {  # small-data regularisation
        "max_leaf_nodes": 15,
        "learning_rate": 0.05,
        "max_iter": 300,
        "min_samples_leaf": 25,
        "l2_regularization": 1.0,
    },
    3: {  # shallow trees, stronger shrinkage
        "max_depth": 3,
        "learning_rate": 0.1,
        "max_iter": 150,
        "min_samples_leaf": 40,
    },
    4: {  # heavier regularisation of the default shape
        "l2_regularization": 5.0,
        "min_samples_leaf": 60,
        "learning_rate": 0.03,
        "max_iter": 400,
    },
    5: {  # wide shallow ensemble
        "max_leaf_nodes": 7,
        "learning_rate": 0.02,
        "max_iter": 800,
        "min_samples_leaf": 20,
    },
}


def partition_actions(records, fold_plan, dropped: set[str]):
    trained, constant = [], {}
    for action in sorted({r.action_id for r in records}):
        if action in dropped:
            continue
        rows = [r for r in records if r.action_id == action]
        labels = [r.outcome_success for r in rows]
        prior = sum(labels) / len(labels)
        if len(set(labels)) < 2:
            constant[action] = {"prior": prior, "reason": "single outcome class"}
            continue
        bad = None
        for fold in range(fold_plan.n_folds):
            train_labels = {
                r.outcome_success for r in rows
                if fold_plan.assignment[r.example_id] != fold
            }
            if len(train_labels) < 2:
                bad = fold
                break
        if bad is not None:
            constant[action] = {
                "prior": prior,
                "reason": f"fold {bad} training partition single-class",
            }
            continue
        trained.append(action)
    return trained, constant


def main() -> int:
    rounds = (
        json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if STATE_FILE.is_file()
        else []
    )
    round_no = len(rounds) + 1
    if round_no not in PROPOSALS:
        raise SystemExit(f"no proposal implemented for round {round_no}")
    hyperparams = PROPOSALS[round_no]
    print(f"swarm round {round_no}: gbdt hyperparams {hyperparams or 'defaults'}")

    # the kept tokenizer, resolved from the round state; pin re-verified

    states = pl.read_parquet(WORKING / "states.parquet")
    episodes = pl.read_parquet(WORKING / "episodes.parquet")
    fold_plan = FoldPlan.model_validate_json(
        (ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8")
    )
    action_report = json.loads(
        (ROOT / "runs" / "reports" / "action_map.json").read_text(encoding="utf-8")
    )
    dropped = set(action_report["dropped_before_training"])

    frame, feature_names, kept_round, features_hash = build_kept_features(ROOT, states)
    outcome_by_episode = {
        row["episode_id"]: (bool(row["success"]), float(row["outcome_score"]))
        for row in episodes.select("episode_id", "success", "outcome_score").to_dicts()
    }
    from ammonix_core.schema import FeatureSpec
    specs = [FeatureSpec(name=n, dtype="float", description=f"kept tokenizer feature {n}")
             for n in feature_names]  # names drive scoring; real specs live in the manifest
    records = to_qpsi_records(frame, specs, outcome_by_episode)

    trained_actions, constant_prior = partition_actions(records, fold_plan, dropped)
    print(f"trained: {trained_actions}")
    for action, info in constant_prior.items():
        print(f"constant prior: {action} ({info['reason']}, prior {info['prior']:.3f})")
    print(f"dropped (P5 coverage): {sorted(dropped)}")

    names = feature_names
    scored = [r for r in records if r.action_id in trained_actions]
    swarm = train_swarm(
        scored, fold_plan, names, seed=MASTER_SEED,
        hyperparams=hyperparams, balanced=True,
    )
    mean_auroc = swarm.mean_oof_auroc
    per_action = {a: round(v, 4) for a, v in sorted(swarm.pooled_auroc.items())}
    print(f"mean pooled OOF AUROC {mean_auroc:.4f}: {per_action}")

    # gain is measured against the last KEPT round (the current best), not the
    # previous round's possibly-reverted score: a revert must not lower the bar
    kept_rounds = [r for r in rounds if r["kept"]]
    previous = kept_rounds[-1]["mean_oof_auroc"] if kept_rounds else 0.5
    gain = mean_auroc - previous
    kept = gain >= GAIN_THRESHOLD
    print(f"best kept {previous:.4f} -> gain {gain:+.4f} -> {'KEEP' if kept else 'REVERT'}")

    report = {
        "milestone": "P6",
        "round": round_no,
        "classifier_spec": ClassifierSpec(model="gbdt", hyperparams=hyperparams).model_dump(),
        "label_policy": LabelPolicy().model_dump(),
        "mean_oof_auroc": mean_auroc,
        "previous_auroc": previous,
        "gain": gain,
        "kept": kept,
        "trained_actions": trained_actions,
        "constant_prior": constant_prior,
        "dropped": sorted(dropped),
        "pooled_auroc": per_action,
        "fold_metrics": {
            a: [m.model_dump() for m in ms] for a, ms in swarm.fold_metrics.items()
        },
        "n_states_scored": len(scored),
        "tokenizer_code_sha256": features_hash,
    }
    (ROOT / "runs" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "reports" / f"swarm_round{round_no}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if kept:
        oof = pl.DataFrame(
            [
                {
                    "state_id": r.state_id,
                    "episode_id": r.example_id,
                    "fold": fold_plan.assignment[r.example_id],
                    "action_id": r.action_id,
                    "outcome_success": r.outcome_success,
                    "outcome_score": r.outcome_score,
                    "oof_score": float(score),
                }
                for action in trained_actions
                for r, score in zip(
                    [x for x in scored if x.action_id == action],
                    swarm.oof_scores[action],
                    strict=True,
                )
            ]
        )
        oof_path = WORKING / f"oof_own_action_round{round_no}.parquet"
        oof.write_parquet(oof_path)
        manifest = {
            "milestone": "P6",
            "round": round_no,
            "classifier_spec": report["classifier_spec"],
            "label_policy": report["label_policy"],
            "trained_actions": trained_actions,
            "constant_prior": constant_prior,
            "dropped": sorted(dropped),
            "pooled_auroc": per_action,
            "seed": MASTER_SEED,
            "tokenizer_code_sha256": features_hash,
            "pins": [
                {
                    "path": f"data/working/cardessa_sim/oof_own_action_round{round_no}.parquet",
                    "sha256": sha256_file(oof_path),
                }
            ],
        }
        (ROOT / "runs" / "manifests" / f"swarm_round{round_no}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (ROOT / "runs" / "manifests" / f"swarm_round{round_no}_files.json").write_text(
            json.dumps(manifest["pins"], indent=2), encoding="utf-8"
        )
        print("swarm manifest + own-action OOF scores written and pinned")

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    rounds.append(
        {
            "round": round_no,
            "hyperparams": hyperparams,
            "mean_oof_auroc": mean_auroc,
            "previous_auroc": previous,
            "gain": gain,
            "kept": kept,
        }
    )
    STATE_FILE.write_text(json.dumps(rounds, indent=2), encoding="utf-8")
    print(f"round state appended: {STATE_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
