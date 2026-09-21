"""Platform entities (schema v0.2): descriptor, split, quarantine, loops, test report."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ammonix_core.platform import (
    DatasetDescriptor,
    LoopIteration,
    LoopRun,
    LoopSpec,
    QuarantineManifest,
    SplitPolicy,
    TestReport,
)
from ammonix_core.schema import FoldMetrics


def cardessa_descriptor() -> DatasetDescriptor:
    """The descriptor of the Cardessa build package Section 2, verbatim fields."""
    return DatasetDescriptor(
        dataset_id="cardessa_sim_v1",
        root_uri="data/raw/cardessa_sim/",
        example_table="episodes.parquet",
        state_table="states.parquet",
        id_fields={"example_id": "episode_id", "state_id": "state_id", "seq": "touch_seq"},
        action_field="action_raw",
        outcome_field="success",
        outcome_positive=True,
        modality_map={
            "*": "tabular",
            "payer_correspondence_text": "text",
            "clinical_indication_text": "text",
        },
        notes="Grouping key meta.patient_id.",
    )


def test_cardessa_descriptor_constructs():
    descriptor = cardessa_descriptor()
    assert descriptor.modality_map["*"] == "tabular"
    assert descriptor.id_fields["example_id"] == "episode_id"


def test_modality_map_rejects_unknown_modality():
    with pytest.raises(ValidationError):
        DatasetDescriptor(
            dataset_id="x",
            root_uri="data/raw/x/",
            example_table="e.parquet",
            state_table="s.parquet",
            id_fields={},
            action_field="action_raw",
            outcome_field="success",
            modality_map={"*": "hologram"},
        )


def test_split_policy_platform_defaults():
    policy = SplitPolicy(seed=7)
    assert policy.test_fraction == 0.15
    assert policy.reserve_fraction == 0.10
    assert policy.n_folds == 5
    assert policy.stratify_by == "outcome"


def test_quarantine_manifest_round_trip():
    manifest = QuarantineManifest(
        manifest_id="01QM",
        created_at=datetime(2026, 7, 9, tzinfo=UTC),
        policy=SplitPolicy(seed=7, group_by="patient_id"),
        group_field="patient_id",
        quarantine_groups=["p1", "p2"],
        reserve_groups=["p3"],
        working_groups=["p4", "p5", "p6"],
        forced_quarantine_rules=["payer in (Granite Shield, Pelican Care)"],
        content_sha256="c" * 64,
    )
    restored = QuarantineManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    groups = (
        set(restored.quarantine_groups) | set(restored.reserve_groups)
        | set(restored.working_groups)
    )
    assert len(groups) == 6  # no group on two sides of a boundary


def test_loop_run_bookkeeping():
    spec = LoopSpec(
        loop_id="L0",
        purpose="Scaffold, schema package, CI, hooks",
        goal_condition=(
            "pytest -q and ruff check . exit 0 in ammonix_core, "
            "as printed by scripts/check_m0.py"
        ),
        checker_script="scripts/check_m0.py",
        max_turns=30,
    )
    run = LoopRun(
        run_id="01LR",
        spec=spec,
        worktree="m0-scaffold",
        iterations=[
            LoopIteration(
                n=1,
                started_at=datetime(2026, 7, 9, tzinfo=UTC),
                summary="scaffold written, checker green",
                verdict=True,
            )
        ],
        verdict=True,
    )
    assert run.spec.max_turns == 30
    assert run.iterations[-1].verdict is True


def test_test_report_taints_explicitly():
    report = TestReport(
        report_id="01TR",
        evaluated_at=datetime(2026, 7, 9, tzinfo=UTC),
        basis_manifest_sha256="d" * 64,
        quarantine_manifest_sha256="e" * 64,
        per_action_metrics={"submit_clean": FoldMetrics(auroc=0.9, auprc=0.8, n_pos=50, n_neg=50)},
        calibration_ece=0.03,
        recommendation_accuracy=0.91,
        trap_results={"P1": True, "P5": True},
    )
    assert report.tainted is False
    assert report.trap_results["P1"] is True
