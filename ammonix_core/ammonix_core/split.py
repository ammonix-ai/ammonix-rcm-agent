"""Stage 1: splits and quarantine (platform plan Section 3, row 1).

Generic over any grouping key: given each group's fixed side assignment
(decided once, at world/dataset creation), this module materialises the
QuarantineManifest, physically separates the dataset into working / reserve /
quarantine directories, builds the grouped stratified FoldPlan on the working
side, and reserves a dev slice for the harness milestone.

The vault property is structural: after the split, build loops only ever read
the working directory, the PreToolUse hook denies data/quarantine/** to every
agent, and the manifest's content hash pins the assignment so any later
change is detectable.
"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from ammonix_core.hashing import sha256_json
from ammonix_core.platform import DatasetDescriptor, QuarantineManifest, SplitPolicy
from ammonix_core.schema import FoldPlan

SIDE_DIRS = {
    "working": "data/working",
    "reserve_b": "data/reserve",
    "quarantine_a":"data/quarantine"}


def manifest_content_hash(manifest: QuarantineManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("content_sha256")
    return sha256_json(payload)


def build_quarantine_manifest(
    manifest_id: str,
    policy: SplitPolicy,
    group_field: str,
    side_of_group: dict[str, str],
    forced_rules: list[str],
    created_at: datetime | None = None,
) -> QuarantineManifest:
    groups: dict[str, list[str]] = {"quarantine_a": [], "reserve_b": [], "working": []}
    for group, side in sorted(side_of_group.items()):
        if side not in groups:
            raise ValueError(f"group {group!r}: unknown split side {side!r}")
        groups[side].append(group)
    manifest = QuarantineManifest(
        manifest_id=manifest_id,
        created_at=created_at or datetime(2026, 7, 11, tzinfo=UTC),
        policy=policy,
        group_field=group_field,
        quarantine_groups=groups["quarantine_a"],
        reserve_groups=groups["reserve_b"],
        working_groups=groups["working"],
        forced_quarantine_rules=forced_rules,
        content_sha256="0" * 64,
    )
    return manifest.model_copy(update={"content_sha256": manifest_content_hash(manifest)})


def split_dataset_files(
    descriptor: DatasetDescriptor,
    base_dir: str | Path,
    manifest: QuarantineManifest,
    group_column: str,
) -> dict[str, dict[str, int]]:
    """Physically separate the example and state tables by split side.

    Reads the raw dataset once, writes one directory per side, and returns
    per-side row counts. Every episode inherits its group's side; an example
    whose group is missing from the manifest fails the split loudly.
    """
    base = Path(base_dir)
    root = base / descriptor.root_uri
    example_df = pl.read_parquet(root / descriptor.example_table)
    state_df = pl.read_parquet(root / descriptor.state_table)
    example_col = descriptor.id_fields["example_id"]

    side_of_group: dict[str, str] = {}
    for side, groups in (
        ("quarantine_a", manifest.quarantine_groups),
        ("reserve_b", manifest.reserve_groups),
        ("working", manifest.working_groups),
    ):
        for group in groups:
            side_of_group[group] = side

    missing = [
        g for g in example_df[group_column].unique().to_list() if g not in side_of_group
    ]
    if missing:
        raise ValueError(f"groups without split assignment: {missing[:5]}...")

    sides = example_df[group_column].replace_strict(side_of_group).alias("_side")
    example_df = example_df.with_columns(sides)
    example_side = dict(example_df.select(example_col, "_side").rows())
    state_df = state_df.with_columns(
        pl.col(example_col).replace_strict(example_side).alias("_side")
    )

    counts: dict[str, dict[str, int]] = {}
    for side, out_rel in SIDE_DIRS.items():
        out_dir = base / out_rel / Path(descriptor.root_uri).name
        out_dir.mkdir(parents=True, exist_ok=True)
        examples_part = example_df.filter(pl.col("_side") == side).drop("_side")
        states_part = state_df.filter(pl.col("_side") == side).drop("_side")
        examples_part.write_parquet(out_dir / descriptor.example_table)
        states_part.write_parquet(out_dir / descriptor.state_table)
        counts[side] = {
            "examples": examples_part.height,
            "states": states_part.height,
        }
    return counts


def build_grouped_fold_plan(
    working_examples: pl.DataFrame,
    example_col: str,
    outcome_col: str,
    group_column: str,
    n_folds: int,
    seed: int,
) -> FoldPlan:
    """Grouped by the domain key (e.g. patient), stratified by outcome."""
    example_ids = working_examples[example_col].to_list()
    outcomes = working_examples[outcome_col].cast(pl.Int8).to_list()
    groups = working_examples[group_column].to_list()
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assignment: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(
        splitter.split(np.zeros(len(example_ids)), outcomes, groups=groups)
    ):
        for i in test_idx:
            assignment[example_ids[i]] = fold
    return FoldPlan(n_folds=n_folds, seed=seed, assignment=assignment)


def pick_dev_slice(
    working_states: pl.DataFrame,
    example_col: str,
    fold_plan: FoldPlan,
    target_states: int,
    seed: int,
) -> dict:
    """Reserve a dev slice for harness optimisation, filling folds in order.

    Starts with fold 0's episodes (shuffled); if one fold holds fewer than
    target_states states, spills into the next fold until the target is met.
    """
    states_per_example = dict(
        working_states.group_by(example_col).len().rows()
    )
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    folds_used: list[int] = []
    n_states = 0
    for fold in range(fold_plan.n_folds):
        fold_examples = sorted(e for e, f in fold_plan.assignment.items() if f == fold)
        folds_used.append(fold)
        for i in rng.permutation(len(fold_examples)):
            example_id = fold_examples[i]
            picked.append(example_id)
            n_states += states_per_example.get(example_id, 0)
            if n_states >= target_states:
                return {
                    "purpose": "harness dev slice (P8)",
                    "from_folds": folds_used,
                    "example_ids": sorted(picked),
                    "n_states": int(n_states),
                }
    raise ValueError(
        f"working side holds only {n_states} states; cannot reserve {target_states}"
    )
