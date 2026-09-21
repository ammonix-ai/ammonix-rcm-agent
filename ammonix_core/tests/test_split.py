"""Splits and quarantine: manifest hashing, physical separation, grouped folds.

Hermetic: builds a tiny synthetic dataset in tmp_path; no domain code involved.
"""

import polars as pl
import pytest

from ammonix_core.platform import DatasetDescriptor, SplitPolicy
from ammonix_core.split import (
    build_grouped_fold_plan,
    build_quarantine_manifest,
    manifest_content_hash,
    pick_dev_slice,
    split_dataset_files,
)


def make_policy():
    return SplitPolicy(
        test_fraction=0.15, reserve_fraction=0.10, n_folds=3,
        group_by="patient", stratify_by="outcome", seed=7,
    )


def side_for(i: int) -> str:
    if i % 10 < 2:
        return "quarantine_a"
    if i % 10 == 2:
        return "reserve_b"
    return "working"


def make_dataset(tmp_path, n_groups=60):
    root = tmp_path / "data" / "raw" / "tiny"
    root.mkdir(parents=True)
    example_rows, state_rows = [], []
    for i in range(n_groups):
        group = f"g{i:03d}"
        for j in range(1 + i % 2):  # one or two episodes per group
            example_id = f"e{i:03d}-{j}"
            example_rows.append(
                {"ex_id": example_id, "patient": group, "won": i % 3 != 0}
            )
            for seq in range(1 + (i + j) % 3):
                state_rows.append(
                    {"ex_id": example_id, "st_id": f"{example_id}-{seq}",
                     "step": seq, "move": "hold", "x": float(i)}
                )
    pl.DataFrame(example_rows).write_parquet(root / "examples.parquet")
    pl.DataFrame(state_rows).write_parquet(root / "states.parquet")
    descriptor = DatasetDescriptor(
        dataset_id="tiny_v1",
        root_uri="data/raw/tiny/",
        example_table="examples.parquet",
        state_table="states.parquet",
        id_fields={"example_id": "ex_id", "state_id": "st_id", "seq": "step"},
        action_field="move",
        outcome_field="won",
        modality_map={"*": "tabular"},
    )
    side_of_group = {f"g{i:03d}": side_for(i) for i in range(n_groups)}
    return descriptor, side_of_group


def make_manifest(side_of_group):
    return build_quarantine_manifest(
        manifest_id="tiny-split",
        policy=make_policy(),
        group_field="patient",
        side_of_group=side_of_group,
        forced_rules=[],
    )


def test_manifest_hash_is_stable_and_tamper_evident():
    manifest = make_manifest({"g0": "working", "g1": "quarantine_a"})
    assert manifest.content_sha256 == manifest_content_hash(manifest)
    tampered = manifest.model_copy(
        update={"working_groups": [*manifest.working_groups, "g999"]}
    )
    assert manifest_content_hash(tampered) != manifest.content_sha256


def test_manifest_partitions_groups_by_side():
    side_of_group = {f"g{i:03d}": side_for(i) for i in range(30)}
    manifest = make_manifest(side_of_group)
    assert len(manifest.quarantine_groups) == 6
    assert len(manifest.reserve_groups) == 3
    assert len(manifest.working_groups) == 21
    assert not set(manifest.quarantine_groups) & set(manifest.working_groups)


def test_manifest_rejects_unknown_side():
    with pytest.raises(ValueError, match="unknown split side"):
        make_manifest({"g0": "working", "g1": "sideways"})


def test_split_files_partition_completely(tmp_path):
    descriptor, side_of_group = make_dataset(tmp_path)
    manifest = make_manifest(side_of_group)
    counts = split_dataset_files(descriptor, tmp_path, manifest, "patient")
    total_examples = pl.read_parquet(
        tmp_path / "data/raw/tiny/examples.parquet"
    ).height
    total_states = pl.read_parquet(tmp_path / "data/raw/tiny/states.parquet").height
    assert sum(c["examples"] for c in counts.values()) == total_examples
    assert sum(c["states"] for c in counts.values()) == total_states
    working = pl.read_parquet(tmp_path / "data/working/tiny/examples.parquet")
    assert set(working["patient"].to_list()) <= set(manifest.working_groups)
    assert "_side" not in working.columns


def test_split_files_fail_on_unassigned_group(tmp_path):
    descriptor, side_of_group = make_dataset(tmp_path)
    del side_of_group["g000"]
    manifest = make_manifest(side_of_group)
    with pytest.raises(ValueError, match="without split assignment"):
        split_dataset_files(descriptor, tmp_path, manifest, "patient")


def test_fold_plan_groups_never_straddle(tmp_path):
    descriptor, side_of_group = make_dataset(tmp_path)
    manifest = make_manifest(side_of_group)
    split_dataset_files(descriptor, tmp_path, manifest, "patient")
    working = pl.read_parquet(tmp_path / "data/working/tiny/examples.parquet")
    plan = build_grouped_fold_plan(
        working, example_col="ex_id", outcome_col="won",
        group_column="patient", n_folds=3, seed=7,
    )
    assert set(plan.assignment.values()) == {0, 1, 2}
    assert set(plan.assignment) == set(working["ex_id"].to_list())
    group_of = dict(working.select("ex_id", "patient").rows())
    folds_per_group: dict[str, set[int]] = {}
    for example_id, fold in plan.assignment.items():
        folds_per_group.setdefault(group_of[example_id], set()).add(fold)
    assert all(len(folds) == 1 for folds in folds_per_group.values())


def test_dev_slice_spills_into_next_fold(tmp_path):
    descriptor, side_of_group = make_dataset(tmp_path)
    manifest = make_manifest(side_of_group)
    split_dataset_files(descriptor, tmp_path, manifest, "patient")
    working = pl.read_parquet(tmp_path / "data/working/tiny/examples.parquet")
    states = pl.read_parquet(tmp_path / "data/working/tiny/states.parquet")
    plan = build_grouped_fold_plan(
        working, example_col="ex_id", outcome_col="won",
        group_column="patient", n_folds=3, seed=7,
    )
    per_fold = {f: 0 for f in range(3)}
    states_per_example = dict(states.group_by("ex_id").len().rows())
    for example_id, fold in plan.assignment.items():
        per_fold[fold] += states_per_example.get(example_id, 0)
    target = per_fold[0] + 1  # force a spill past fold 0
    dev = pick_dev_slice(states, "ex_id", plan, target_states=target, seed=3)
    assert dev["n_states"] >= target
    assert dev["from_folds"][0] == 0 and len(dev["from_folds"]) >= 2
    assert len(set(dev["example_ids"])) == len(dev["example_ids"])


def test_dev_slice_fails_when_target_unreachable(tmp_path):
    descriptor, side_of_group = make_dataset(tmp_path)
    manifest = make_manifest(side_of_group)
    split_dataset_files(descriptor, tmp_path, manifest, "patient")
    working = pl.read_parquet(tmp_path / "data/working/tiny/examples.parquet")
    states = pl.read_parquet(tmp_path / "data/working/tiny/states.parquet")
    plan = build_grouped_fold_plan(
        working, example_col="ex_id", outcome_col="won",
        group_column="patient", n_folds=3, seed=7,
    )
    with pytest.raises(ValueError, match="cannot reserve"):
        pick_dev_slice(states, "ex_id", plan, target_states=10_000, seed=3)


def test_dev_slice_is_deterministic(tmp_path):
    descriptor, side_of_group = make_dataset(tmp_path)
    manifest = make_manifest(side_of_group)
    split_dataset_files(descriptor, tmp_path, manifest, "patient")
    working = pl.read_parquet(tmp_path / "data/working/tiny/examples.parquet")
    states = pl.read_parquet(tmp_path / "data/working/tiny/states.parquet")
    plan = build_grouped_fold_plan(
        working, example_col="ex_id", outcome_col="won",
        group_column="patient", n_folds=3, seed=7,
    )
    a = pick_dev_slice(states, "ex_id", plan, target_states=20, seed=3)
    b = pick_dev_slice(states, "ex_id", plan, target_states=20, seed=3)
    assert a == b
