"""`make toy`: build the toy Basis end to end and write the M1 report.

Generates the toy corpus twice (double-generation hash equality), tokenizes,
builds grouped stratified folds, trains the per-Action swarm, calibrates,
recommends on fresh oracle-labelled test states, and writes
runs/reports/toy_build.json plus determinism pins in runs/manifests/toy.json.
"""

import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ammonix_core"))

from ammonix_core.hashing import dataset_fingerprint, sha256_file  # noqa: E402
from ammonix_core.pipeline import (  # noqa: E402
    build_fold_plan,
    recommend,
    tokenize_tabular,
    train_swarm,
)
from ammonix_core.schema import FeatureSpec  # noqa: E402

from ammonix_core import toy  # noqa: E402

SEED = 42
N_EPISODES = 4_000
N_TEST_STATES = 1_000
N_FOLDS = 5

FEATURE_SPECS = [
    FeatureSpec(
        name="x1",
        dtype="float",
        description="Planted signal: act_a is right above 0.5, act_b/act_c below",
        valid_range=(0.0, 1.0),
    ),
    FeatureSpec(
        name="x2",
        dtype="float",
        description="Decoy: uniform noise with no bearing on any outcome",
        valid_range=(0.0, 1.0),
    ),
    FeatureSpec(
        name="x3",
        dtype="bool",
        description="Planted signal: selects between act_b (true) and act_c (false)",
    ),
    FeatureSpec(
        name="x4",
        dtype="float",
        description="Decoy: uniform noise with no bearing on any outcome",
        valid_range=(0.0, 1.0),
    ),
]

ACTION_MAP = {a: a for a in toy.ACTIONS}  # toy labels are already canonical


def main() -> int:
    print(f"generating toy corpus: {N_EPISODES} episodes, seed {SEED}")
    corpus = toy.generate_corpus(N_EPISODES, SEED)
    corpus_hash = corpus.content_sha256()
    double_ok = toy.generate_corpus(N_EPISODES, SEED).content_sha256() == corpus_hash
    print(f"double-generation hash equality: {double_ok} ({corpus_hash[:12]}...)")

    toy_dir = ROOT / "data" / "toy"
    toy_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "state_id": s.state_id,
                "example_id": s.example_id,
                "seq": s.seq,
                "action_raw": s.action_raw,
                **{k: s.data.inline[k] for k in toy.FEATURE_NAMES},
            }
            for s in corpus.states
        ]
    ).write_parquet(toy_dir / "states.parquet")
    pl.DataFrame(
        [
            {"example_id": e.example_id, "success": e.outcome.success}
            for e in corpus.examples
        ]
    ).write_parquet(toy_dir / "episodes.parquet")

    records = tokenize_tabular(corpus.states, corpus.examples, FEATURE_SPECS, ACTION_MAP)
    fold_plan = build_fold_plan(corpus.examples, N_FOLDS, SEED)
    action_counts = {
        a: sum(1 for r in records if r.action_id == a) for a in sorted(ACTION_MAP.values())
    }
    print(f"states per action: {action_counts}")

    swarm = train_swarm(records, fold_plan, [s.name for s in FEATURE_SPECS], seed=SEED)
    rounded_auroc = {a: round(v, 4) for a, v in swarm.pooled_auroc.items()}
    print(f"pooled OOF AUROC per action: {rounded_auroc}")

    test_rows = toy.generate_test_states(N_TEST_STATES, SEED)
    recommended = recommend(swarm, [row["features"] for row in test_rows])
    matches = sum(
        rec == row["optimal_action"] for rec, row in zip(recommended, test_rows, strict=True)
    )
    match_rate = matches / len(test_rows)

    report = {
        "milestone": "M1",
        "seed": SEED,
        "n_episodes": N_EPISODES,
        "n_test_states": N_TEST_STATES,
        "states_per_action": action_counts,
        "double_generation_equal": double_ok,
        "corpus_sha256": corpus_hash,
        "dataset_fingerprint": dataset_fingerprint(
            [(e.example_id, e.state_ids, e.outcome.success) for e in corpus.examples]
        ),
        "pooled_oof_auroc": swarm.pooled_auroc,
        "mean_oof_auroc": swarm.mean_oof_auroc,
        "recommended_matches_optimal": match_rate,
    }
    reports_dir = ROOT / "runs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "toy_build.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    manifests_dir = ROOT / "runs" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    pins = [
        {"path": "data/toy/states.parquet", "sha256": sha256_file(toy_dir / "states.parquet")},
        {"path": "data/toy/episodes.parquet", "sha256": sha256_file(toy_dir / "episodes.parquet")},
    ]
    (manifests_dir / "toy.json").write_text(json.dumps(pins, indent=2), encoding="utf-8")

    print(
        f"mean_oof_auroc={swarm.mean_oof_auroc:.4f} "
        f"recommended_matches_optimal={match_rate:.4f}"
    )
    print("report written: runs/reports/toy_build.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
