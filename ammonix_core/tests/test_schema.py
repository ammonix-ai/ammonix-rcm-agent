"""Schema v0.1 entities: construction, defaults, round-trips, rejection of bad data."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ammonix_core.schema import (
    ActionMap,
    ActionMapRule,
    AmmonixConfig,
    CanonicalAction,
    CoverageReport,
    Example,
    FeatureSpec,
    FoldPlan,
    LabelPolicy,
    LLMPin,
    Outcome,
    QPsiRecord,
    RawData,
    RawState,
    RecommendationPolicy,
    UniverseRecord,
)


def make_state(state_id: str, example_id: str, seq: int, nxt: str | None) -> RawState:
    return RawState(
        state_id=state_id,
        example_id=example_id,
        seq=seq,
        data=RawData(modality="tabular", inline={"x": 1}, sha256="a" * 64),
        action_raw="submit_clean",
        next_state_id=nxt,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_example_round_trip():
    example = Example(
        example_id="01EX",
        state_ids=["01S1", "01S2"],
        outcome=Outcome(success=True, score=0.95),
        meta={"patient_id": "01PT"},
    )
    restored = Example.model_validate_json(example.model_dump_json())
    assert restored == example
    assert restored.outcome.success is True


def test_example_meta_defaults_are_independent():
    a, b = Example(example_id="a", state_ids=[], outcome=Outcome(success=False)), None
    b = Example(example_id="b", state_ids=[], outcome=Outcome(success=False))
    a.meta["k"] = "v"
    assert b.meta == {}


def test_raw_state_terminal_marker():
    chain = [make_state("s1", "e1", 0, "s2"), make_state("s2", "e1", 1, None)]
    assert chain[-1].next_state_id is None
    assert [s.seq for s in chain] == [0, 1]


def test_raw_data_requires_sha256():
    with pytest.raises(ValidationError):
        RawData(modality="text")  # type: ignore[call-arg]


def test_qpsi_record_has_no_raw_data_field():
    record = QPsiRecord(
        state_id="s1",
        example_id="e1",
        seq=0,
        features={"days_since_service": 12, "auth_missing": True},
        action_id="submit_clean",
        outcome_success=True,
        next_state_id=None,
    )
    assert "data" not in QPsiRecord.model_fields
    assert record.outcome_score == 1.0


def test_feature_spec_rejects_unknown_dtype():
    with pytest.raises(ValidationError):
        FeatureSpec(name="f", dtype="complex", description="not a legal dtype")


def test_action_map_shape():
    action_map = ActionMap(
        version="0.1",
        actions=[
            CanonicalAction(
                action_id="submit_clean",
                name="Standard claim submission",
                description="CMS-1500 submission without attachments",
                payload_schema={"type": "object"},
            )
        ],
        rules=[ActionMapRule(pattern="submit_clean", action_id="submit_clean")],
    )
    assert action_map.rules[0].action_id == "submit_clean"


def test_coverage_report_violations():
    report = CoverageReport(
        states_total=150,
        counts={"submit_clean": 120, "write_off": 30},
        min_states_per_action=100,
        violations=["write_off"],
    )
    assert report.violations == ["write_off"]


def test_fold_plan_and_label_policy_defaults():
    plan = FoldPlan(seed=42, assignment={"e1": 0})
    assert plan.n_folds == 5
    assert plan.group_by == "example"
    assert plan.stratify_by == "outcome"
    policy = LabelPolicy()
    assert policy.scope == "own_action_states"
    assert policy.label == "outcome_success"
    assert policy.class_weight == "balanced"


def test_universe_record_calibrated_and_raw_scores_are_separate():
    record = UniverseRecord(
        state_id="s1",
        example_id="e1",
        fold=3,
        scores_raw={"submit_clean": 0.9, "write_off": 0.4},
        scores_cal={"submit_clean": 0.82, "write_off": 0.35},
        true_action_id="submit_clean",
        outcome_success=True,
        outcome_score=0.95,
        top_features=[],
    )
    assert record.tribe_id is None
    assert record.scores_raw != record.scores_cal


def test_llm_pin_defaults_deterministic():
    pin = LLMPin(name="qwen-27b", weights_sha256="b" * 64)
    assert pin.temperature == 0.0
    assert pin.constrained_decoding is True


def test_ammonix_config_platform_defaults():
    config = AmmonixConfig()
    assert config.n_folds == 5
    assert config.min_states_per_action == 100
    assert config.ambiguity_margin == 0.05
    assert config.calibration == "isotonic"
    assert config.max_iterations == 3
    assert config.recommendation == RecommendationPolicy(
        action_source="swarm_scores", inference_model="refit_full", k_neighbours=25
    )
    assert config.region_model.algorithm == "hdbscan"
