"""`make split`: materialise the quarantine manifest and physical split (P3).

The side of every patient was fixed once at world creation (W1); this script
re-derives it from the world regeneration (source of truth), asserts the
corpus agrees, writes the QuarantineManifest + physical side directories +
grouped FoldPlan + dev-slice manifest, and pins everything.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.platform import SplitPolicy  # noqa: E402
from ammonix_core.split import (  # noqa: E402
    SIDE_DIRS,
    build_grouped_fold_plan,
    build_quarantine_manifest,
    pick_dev_slice,
    split_dataset_files,
)

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.descriptor import cardessa_descriptor  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

DEV_SLICE_TARGET_STATES = 500


def main() -> int:
    descriptor = cardessa_descriptor()
    world = generate_world(MASTER_SEED)
    side_of_patient = dict(world.patients.select("patient_id", "split").rows())

    episodes = pl.read_parquet(
        ROOT / descriptor.root_uri / descriptor.example_table
    )
    corpus_sides = dict(episodes.select("patient_id", "split").rows())
    mismatched = {
        p: (s, side_of_patient.get(p))
        for p, s in corpus_sides.items()
        if side_of_patient.get(p) != s
    }
    if mismatched:
        raise SystemExit(f"corpus split column disagrees with world: {mismatched}")
    print("corpus split assignments agree with the world snapshot")

    policy = SplitPolicy(
        test_fraction=0.15, reserve_fraction=0.10, n_folds=5,
        group_by="patient_id", stratify_by="outcome", seed=MASTER_SEED,
    )
    manifest = build_quarantine_manifest(
        manifest_id="cardessa-split-v1",
        policy=policy,
        group_field="patient_id",
        side_of_group=side_of_patient,
        forced_rules=["primary payer in (granite, pelican) -> quarantine_a"],
    )
    print(f"quarantine manifest hash: {manifest.content_sha256[:16]}...")

    counts = split_dataset_files(descriptor, ROOT, manifest, "patient_id")
    for side, c in counts.items():
        print(f"  {side}: {c['examples']} episodes, {c['states']} states")

    working_dir = ROOT / SIDE_DIRS["working"] / Path(descriptor.root_uri).name
    working_examples = pl.read_parquet(working_dir / descriptor.example_table)
    working_states = pl.read_parquet(working_dir / descriptor.state_table)
    fold_plan = build_grouped_fold_plan(
        working_examples,
        example_col=descriptor.id_fields["example_id"],
        outcome_col=descriptor.outcome_field,
        group_column="patient_id",
        n_folds=policy.n_folds,
        seed=MASTER_SEED,
    )
    dev_slice = pick_dev_slice(
        working_states,
        example_col=descriptor.id_fields["example_id"],
        fold_plan=fold_plan,
        target_states=DEV_SLICE_TARGET_STATES,
        seed=MASTER_SEED + 8,
    )
    print(f"fold plan: {policy.n_folds} folds over {len(fold_plan.assignment)} episodes")
    print(f"dev slice: {len(dev_slice['example_ids'])} episodes, {dev_slice['n_states']} states")

    manifests = ROOT / "runs" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "quarantine_manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    (manifests / "fold_plan.json").write_text(
        fold_plan.model_dump_json(indent=2), encoding="utf-8"
    )
    (manifests / "dev_slice.json").write_text(
        json.dumps(dev_slice, indent=2), encoding="utf-8"
    )
    # pin only the working/reserve side files; the quarantine directory is
    # sealed and never re-read outside the whitelisted final eval
    pins = []
    for side in ("working", "reserve_b"):
        out_dir = ROOT / SIDE_DIRS[side] / Path(descriptor.root_uri).name
        for name in (descriptor.example_table, descriptor.state_table):
            rel = f"{SIDE_DIRS[side]}/{Path(descriptor.root_uri).name}/{name}"
            pins.append({"path": rel, "sha256": sha256_file(out_dir / name)})
    (manifests / "split_files.json").write_text(json.dumps(pins, indent=2), encoding="utf-8")

    (ROOT / "runs" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "reports" / "split_build.json").write_text(
        json.dumps(
            {
                "milestone": "P3",
                "manifest_sha256": manifest.content_sha256,
                "counts": counts,
                "n_working_folds": policy.n_folds,
                "dev_slice_states": dev_slice["n_states"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("split artefacts written and pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
