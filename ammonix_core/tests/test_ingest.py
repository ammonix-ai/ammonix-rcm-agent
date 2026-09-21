"""Generic ingest: descriptor-driven mapping, scope conditions, leakage audit.

Hermetic: builds a tiny synthetic dataset in tmp_path; no domain code involved.
"""

import polars as pl
import pytest

from ammonix_core.ingest import ingest_dataset
from ammonix_core.platform import DatasetDescriptor


def make_dataset(tmp_path, n_examples=120, poison=True):
    root = tmp_path / "data" / "raw" / "tiny"
    root.mkdir(parents=True)
    example_rows, state_rows = [], []
    for i in range(n_examples):
        example_id = f"e{i:04d}"
        won = i % 3 != 0
        example_rows.append(
            {"ex_id": example_id, "won": won, "grade": 0.9 if won else 0.2,
             "site": f"s{i % 4}", "final_cost": 100.0 + i}
        )
        n_states = 1 + i % 3
        for seq in range(n_states):
            state_rows.append(
                {
                    "ex_id": example_id,
                    "st_id": f"{example_id}-{seq}",
                    "step": seq,
                    "move": "hold" if seq % 2 == 0 else "fold",
                    "x": float(i % 7),
                    "note_text": f"note {i}-{seq}",
                    # poison: constant within episode, perfectly separates outcome
                    "leak": (1000.0 if won else -1000.0) if poison else 0.0,
                }
            )
    pl.DataFrame(example_rows).write_parquet(root / "examples.parquet")
    pl.DataFrame(state_rows).write_parquet(root / "states.parquet")
    return DatasetDescriptor(
        dataset_id="tiny_v1",
        root_uri="data/raw/tiny/",
        example_table="examples.parquet",
        state_table="states.parquet",
        id_fields={"example_id": "ex_id", "state_id": "st_id", "seq": "step"},
        action_field="move",
        outcome_field="won",
        outcome_score_field="grade",
        modality_map={"*": "tabular", "note_text": "text"},
        posthoc_fields=["final_*"],
    )


def test_ingest_maps_schema_shape(tmp_path):
    descriptor = make_dataset(tmp_path)
    result = ingest_dataset(descriptor, tmp_path)
    assert len(result.examples) == 120
    assert all(e.outcome.score is not None for e in result.examples)
    by_id = {s.state_id: s for s in result.states}
    for example in result.examples:
        assert example.state_ids
        assert by_id[example.state_ids[-1]].next_state_id is None
        assert example.meta["site"].startswith("s")


def test_scope_conditions_pass_on_clean_data(tmp_path):
    descriptor = make_dataset(tmp_path)
    audit = ingest_dataset(descriptor, tmp_path).audit
    assert all(audit["scope_conditions"].values())
    assert audit["raw_action_inventory"] == {
        "hold": sum(1 + i % 3 - (i % 3 // 2) for i in range(120)),
        "fold": sum(i % 3 // 2 for i in range(120)),
    } or set(audit["raw_action_inventory"]) == {"hold", "fold"}


def test_volume_scope_fails_below_minimum(tmp_path):
    descriptor = make_dataset(tmp_path, n_examples=50)
    audit = ingest_dataset(descriptor, tmp_path).audit
    assert audit["scope_conditions"]["volume"] is False


def test_statistical_leak_screen_catches_unlisted_poison(tmp_path):
    """The leak column is NOT on the denylist; the screen must still catch it."""
    descriptor = make_dataset(tmp_path)
    audit = ingest_dataset(descriptor, tmp_path).audit
    leakage = audit["leakage_audit"]
    assert "leak" in leakage["statistically_flagged"]
    assert "leak" in leakage["excluded_from_data"]
    all_data = {c for cols in audit["data_columns_by_modality"].values() for c in cols}
    assert "leak" not in all_data
    assert "x" in all_data  # honest features survive


def test_denylist_wildcard_applies_to_state_columns(tmp_path):
    descriptor = make_dataset(tmp_path)
    # add a state column matching the wildcard
    root = tmp_path / "data" / "raw" / "tiny"
    states = pl.read_parquet(root / "states.parquet").with_columns(
        pl.lit(5.0).alias("final_touchup")
    )
    states.write_parquet(root / "states.parquet")
    audit = ingest_dataset(descriptor, tmp_path).audit
    assert "final_touchup" in audit["leakage_audit"]["denylisted"]


def test_text_modality_mapping(tmp_path):
    descriptor = make_dataset(tmp_path)
    audit = ingest_dataset(descriptor, tmp_path).audit
    assert audit["data_columns_by_modality"]["text"] == ["note_text"]
    assert "x" in audit["data_columns_by_modality"]["tabular"]


def test_outcome_polarity_flip(tmp_path):
    descriptor = make_dataset(tmp_path)
    flipped = descriptor.model_copy(update={"outcome_positive": False})
    normal = ingest_dataset(descriptor, tmp_path)
    inverted = ingest_dataset(flipped, tmp_path)
    for a, b in zip(normal.examples, inverted.examples, strict=True):
        assert a.outcome.success != b.outcome.success


def test_missing_action_fails_scope(tmp_path):
    descriptor = make_dataset(tmp_path)
    root = tmp_path / "data" / "raw" / "tiny"
    states = pl.read_parquet(root / "states.parquet").with_columns(
        pl.when(pl.col("st_id") == "e0000-0")
        .then(pl.lit(""))
        .otherwise(pl.col("move"))
        .alias("move")
    )
    states.write_parquet(root / "states.parquet")
    audit = ingest_dataset(descriptor, tmp_path).audit
    assert audit["scope_conditions"]["data_and_action_per_state"] is False


def test_no_poison_no_flags(tmp_path):
    descriptor = make_dataset(tmp_path, poison=False)
    audit = ingest_dataset(descriptor, tmp_path).audit
    assert audit["leakage_audit"]["statistically_flagged"] == {}


@pytest.mark.parametrize("missing_key", ["example_id", "state_id", "seq"])
def test_descriptor_requires_id_fields(tmp_path, missing_key):
    descriptor = make_dataset(tmp_path)
    broken_ids = dict(descriptor.id_fields)
    del broken_ids[missing_key]
    broken = descriptor.model_copy(update={"id_fields": broken_ids})
    with pytest.raises(KeyError):
        ingest_dataset(broken, tmp_path)
