"""v0.4 platform hardening: uncertainty, validator, genericity, crash guards.

Covers the three fixes of the v4-fixes worktree:
1. fold-ensemble uncertainty (scores_std, fold_ensemble inference,
   uncertainty-aware ambiguity in retrieve);
2. hygiene (strict payload validator, loud errors instead of asserts,
   NaN-tolerant leak screen, one balanced-weights implementation);
3. genericity (ClassifierSpec model choice, NaN/bool feature handling,
   domain-free prompt fields from context_sources).
"""

import numpy as np
import polars as pl
import pytest

from ammonix_core import pipeline as pl_mod
from ammonix_core import runtime as rt_mod
from ammonix_core import universe as uni_mod
from ammonix_core.ingest import _leak_screen
from ammonix_core.schema import (
    ClassifierSpec,
    ContextSource,
    ExpectedResultSpec,
    FoldPlan,
    LLMPin,
    PromptTemplate,
    QPsiRecord,
    Scope,
    Skill,
)

SEED = 7


def synthetic_records(n=300, actions=("act_a", "act_b")) -> list[QPsiRecord]:
    rng = np.random.default_rng(SEED)
    records = []
    for i in range(n):
        action = actions[i % len(actions)]
        x1 = float(rng.normal())
        p = 1.0 / (1.0 + np.exp(-3.0 * x1))
        success = bool(rng.random() < p)
        records.append(
            QPsiRecord(
                state_id=f"s{i}",
                example_id=f"e{i}",
                seq=0,
                features={"x1": x1, "x2": float(rng.normal())},
                action_id=action,
                outcome_success=success,
                outcome_score=1.0,
                next_state_id=None,
            )
        )
    return records


def fold_plan_for(records: list[QPsiRecord], n_folds=5) -> FoldPlan:
    assignment = {r.example_id: i % n_folds for i, r in enumerate(records)}
    return FoldPlan(n_folds=n_folds, seed=SEED, assignment=assignment)


# -- fix 1: uncertainty ------------------------------------------------------


def test_universe_records_carry_fold_spread():
    records = synthetic_records()
    plan = fold_plan_for(records)
    build = uni_mod.build_universe(
        records, plan, ["x1", "x2"], ["act_a", "act_b"],
        constant_priors={"act_c": 0.9}, seed=SEED,
    )
    rec = build.records[0]
    assert rec.scores_std is not None
    assert set(rec.scores_std) == {"act_a", "act_b", "act_c"}
    assert all(v >= 0.0 for v in rec.scores_std.values())
    assert rec.scores_std["act_c"] == 0.0  # constant prior: no model uncertainty
    assert any(
        r.scores_std["act_a"] > 0.0 or r.scores_std["act_b"] > 0.0
        for r in build.records
    )
    assert build.fold_models  # persisted for live uncertainty


def test_swarm_fold_ensemble_inference():
    records = synthetic_records()
    plan = fold_plan_for(records)
    swarm = pl_mod.train_swarm(records, plan, ["x1", "x2"], seed=SEED)
    assert swarm.fold_models["act_a"]
    rows = [{"x1": 2.0, "x2": 0.0}, {"x1": -2.0, "x2": 0.0}]
    cal, spread = pl_mod.score_live(swarm, rows, inference_model="fold_ensemble")
    assert set(cal) == {"act_a", "act_b"}
    assert all(0.0 <= v <= 1.0 for arr in cal.values() for v in arr)
    assert all((arr >= 0.0).all() for arr in spread.values())
    assert pl_mod.recommend(swarm, rows, inference_model="fold_ensemble")
    with pytest.raises(ValueError, match="inference_model"):
        pl_mod.score_live(swarm, rows, inference_model="nope")


def make_skills() -> list[Skill]:
    def skill(skill_id, level, ref, kind="execute"):
        return Skill(
            skill_id=skill_id, version="v1", scope=Scope(level=level, ref=ref),
            kind=kind, context_schema={}, context_sources=[],
            m1_prompt_ref="", expected_result=ExpectedResultSpec(kind="schema"),
        )

    return [
        skill("default", "default", None, kind="escalate"),
        skill("skill-a", "cluster", "act_a"),
        skill("skill-b", "cluster", "act_b"),
    ]


def retrieve_with(scores_std, scores_cal):
    return rt_mod.retrieve(
        rt_mod.RetrievalContext(
            case_id="c1", features={"x1": 0.0}, scores_cal=scores_cal,
            known=True, scores_std=scores_std,
        ),
        [], make_skills(), np.zeros(1), [], [], ["act_a", "act_b"],
    )


def test_retrieve_flags_margin_inside_model_noise():
    scores = {"act_a": 0.60, "act_b": 0.52}  # margin 0.08 > 0.05 threshold
    assert retrieve_with(None, scores).ambiguous is False
    # fold models disagree by more than the margin: inside the noise floor
    # (v0.4: noise of the gap = hypot of the two spreads, not their sum)
    assert retrieve_with({"act_a": 0.08, "act_b": 0.05}, scores).ambiguous is True
    assert retrieve_with({"act_a": 0.01, "act_b": 0.01}, scores).ambiguous is False


# -- fix 2: hygiene ----------------------------------------------------------


def test_validator_new_keywords():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 0, "maximum": 10},
            "s": {"type": "string", "maxLength": 5, "pattern": "^ab"},
            "arr": {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
        },
        "required": ["n"],
        "additionalProperties": False,
    }
    assert rt_mod.validate_payload(schema, {"n": 3, "s": "abc", "arr": [1]}) == []
    failures = rt_mod.validate_payload(
        schema, {"n": 11.5, "s": "zzzzzzz", "arr": [1, 2, 3]}
    )
    assert any("expected integer" in f for f in failures)
    assert any("maxLength" in f for f in failures)
    assert any("pattern" in f for f in failures)
    assert any("maxItems" in f for f in failures)


def test_validator_refuses_unknown_keywords():
    with pytest.raises(NotImplementedError, match="oneOf"):
        rt_mod.validate_payload({"type": "object", "oneOf": []}, {})


def test_no_default_skill_is_a_clear_error():
    skills = [s for s in make_skills() if s.scope.level != "default"]
    with pytest.raises(ValueError, match="default skill"):
        rt_mod.retrieve(
            rt_mod.RetrievalContext(
                case_id="c1", features={}, scores_cal={"act_a": 0.5}, known=True
            ),
            [], skills, np.zeros(1), [], [], ["act_a"],
        )


def test_single_class_action_is_a_clear_error():
    records = synthetic_records()
    for r in records:
        if r.action_id == "act_b":
            r.outcome_success = True  # single class by construction
    with pytest.raises(ValueError, match="single outcome class"):
        pl_mod.train_swarm(records, fold_plan_for(records), ["x1", "x2"], seed=SEED)


def test_leak_screen_catches_nan_bearing_posthoc_column():
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(120):
        outcome = i % 2 == 0
        leak = 5.0 if outcome else 1.0
        rows.append(
            {
                "episode_id": f"e{i}",
                "honest": float(rng.normal()),
                "posthoc": None if i == 0 else leak,  # one NaN must not exempt it
            }
        )
    frame = pl.DataFrame(rows)
    outcomes = {f"e{i}": i % 2 == 0 for i in range(120)}
    flagged = _leak_screen(frame, outcomes, "episode_id")
    assert "posthoc" in flagged
    assert "honest" not in flagged


def test_balanced_weights_single_source():
    assert uni_mod.balanced_weights is pl_mod.balanced_weights


# -- fix 3: genericity -------------------------------------------------------


def test_classifier_spec_logreg():
    records = synthetic_records()
    plan = fold_plan_for(records)
    swarm = pl_mod.train_swarm(
        records, plan, ["x1", "x2"], seed=SEED, balanced=True,
        classifier=ClassifierSpec(model="logreg"),
    )
    assert swarm.pooled_auroc["act_a"] > 0.8  # planted signal is linear
    with pytest.raises(ValueError, match="not supported"):
        pl_mod.build_classifier(SEED, ClassifierSpec(model="mystery"))


def test_feature_matrix_handles_missing_and_bool_refuses_strings():
    x = pl_mod.matrix_from_rows([{"a": None, "b": True}], ["a", "b"])
    assert np.isnan(x[0, 0]) and x[0, 1] == 1.0
    with pytest.raises(ValueError, match="one-hot"):
        pl_mod.matrix_from_rows([{"a": "red"}], ["a"])


def test_gbdt_trains_through_missing_values():
    records = synthetic_records()
    for r in records[::10]:
        r.features["x2"] = None  # gbdt handles NaN natively
    swarm = pl_mod.train_swarm(records, fold_plan_for(records), ["x1", "x2"], seed=SEED)
    assert swarm.pooled_auroc["act_a"] > 0.7


def test_runtime_has_no_domain_fields_and_default_prompt_uses_context_sources():
    import inspect

    source = inspect.getsource(rt_mod)
    for token in ("cpt", "carc", "payer", "clinical_indication"):
        assert token not in source, f"domain token {token!r} left in generic runtime"

    skills = make_skills()
    skills[1].context_sources = [
        ContextSource(field="claim_ref", source="case_record", locator="claim_id")
    ]
    retrieval = retrieve_with(None, {"act_a": 0.9, "act_b": 0.1})
    prompts = []
    trace = rt_mod.run_case(
        retrieval, skills[1], {"claim_id": "C-77"},
        PromptTemplate(template_id="m1", version="t", text="{action_id}|{claim_ref}|{feedback}"),
        PromptTemplate(template_id="m2", version="t", text="failed: {failures}"),
        LLMPin(name="test", weights_sha256="0" * 64),
        lambda p, s: (prompts.append(p), "{}")[1],
        lambda rule, payload, row: (True, ""),
        "b", "h",
    )
    assert trace.status == "executed"
    assert prompts[0] == "act_a|C-77|"
