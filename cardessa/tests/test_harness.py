"""P8 harness pieces: payload validation, M2 rules, routing masks, skills,
and the M1/M2 loop with a fake provider. Hermetic - no vLLM involved."""

import json

import pytest
from ammonix_core.runtime import (
    RetrievalContext,
    retrieve,
    run_case,
    validate_payload,
)
from ammonix_core.schema import LLMPin, PromptTemplate

from cardessa.harness import applicable_actions, build_skills
from cardessa.payloads import PAYLOAD_SCHEMAS, RULES, evaluate_rule

PIN = LLMPin(name="Qwen3.6-27B-AWQ-INT4", weights_sha256="0" * 64)
M2 = PromptTemplate(template_id="m2", version="t", text="failed: {failures}")


def base_case(**overrides) -> dict:
    row = {
        "episode_id": "ep-1", "state_id": "ep-1-s0", "cpt": "93229",
        "dx_codes": "I48.0,R55", "pairing_valid": True, "payer_id": "meridian",
        "carc": "", "balance": 700.0, "days_since_service": 3,
        "days_to_filing_deadline": 87, "retro_window_days": 10,
        "auth_required": True, "auth_status": "missing", "p2p_available": True,
        "has_secondary": False, "touches_so_far": 0, "cob_position_ok": True,
        "payer_correspondence_text": "", "clinical_indication_text": "",
    }
    row.update(overrides)
    return row


def test_validate_payload_subset():
    schema = PAYLOAD_SCHEMAS["request_retro_auth"]
    good = {"episode_id": "e", "cpt": "93229",
            "justification_code": "auth_not_obtained_before_service"}
    assert validate_payload(schema, good) == []
    assert validate_payload(schema, {"episode_id": "e", "cpt": "93229"})
    assert validate_payload(schema, {**good, "justification_code": "made_up"})
    assert validate_payload(schema, {**good, "extra": 1})


def test_rules_retro_window():
    payload = {"episode_id": "ep-1", "cpt": "93229",
               "justification_code": "auth_not_obtained_before_service"}
    ok, _ = evaluate_rule("retro_window_open", payload, base_case())
    assert ok
    ok, _ = evaluate_rule("retro_window_open", payload, base_case(days_since_service=40))
    assert not ok


def test_rules_appeal_grounding():
    case = base_case(carc="CO-50")
    payload = {"episode_id": "ep-1", "cpt": "93229", "denial_carc": "CO-50",
               "necessity_paragraph": "Monitoring with CPT 93229 is necessary "
               "given documented paroxysmal atrial fibrillation and syncope."}
    for rule in RULES["appeal_with_necessity"]:
        ok, detail = evaluate_rule(rule, payload, case)
        assert ok, (rule, detail)
    bad = {**payload, "necessity_paragraph": "The patient clearly needs this test."}
    ok, _ = evaluate_rule("paragraph_grounded_in_case", bad, case)
    assert not ok


def test_applicability_masks():
    fresh = applicable_actions(base_case())
    assert "submit_clean" in fresh and "request_retro_auth" in fresh
    assert "appeal_with_necessity" not in fresh  # no denial yet
    assert "bill_secondary" not in fresh  # no secondary
    denied = applicable_actions(base_case(carc="CO-197", touches_so_far=1))
    assert {"appeal_with_necessity", "request_peer_to_peer"} <= denied
    dead = applicable_actions(
        base_case(days_to_filing_deadline=-10, days_since_service=400, carc="CO-29")
    )
    assert "submit_clean" not in dead and "request_retro_auth" not in dead
    assert "write_off" in dead
    underpaid = applicable_actions(
        base_case(has_secondary=True, touches_so_far=1, balance=120.0)
    )
    assert "bill_secondary" in underpaid
    # a submission the engine would reject on published rules is not a plan
    bad_pairing = applicable_actions(base_case(pairing_valid=False))
    assert "submit_clean" not in bad_pairing
    assert "correct_and_resubmit" not in bad_pairing
    wrong_cob = applicable_actions(base_case(cob_position_ok=False))
    assert "submit_clean" not in wrong_cob
    assert "correct_and_resubmit" in wrong_cob  # even pre-denial


def test_skills_cover_all_actions_and_default():
    skills = build_skills([])
    refs = {s.scope.ref for s in skills if s.scope.level == "cluster"}
    assert len(refs) == 10
    assert sum(1 for s in skills if s.scope.level == "default") == 1
    info = next(s for s in skills if s.scope.ref == "provide_requested_info")
    assert info.kind == "execute"  # v0.2: trained (coverage 108), gate-pending
    assert "requested_item_in_record" in info.expected_result.rules
    execute = next(s for s in skills if s.scope.ref == "request_retro_auth")
    assert execute.kind == "execute"
    assert execute.expected_result.rules == RULES["request_retro_auth"]


def make_retrieval(skills, case_id="ep-1-s0", action="request_retro_auth"):
    import numpy as np

    return retrieve(
        RetrievalContext(
            case_id=case_id, features={"x": 1.0},
            scores_cal={action: 0.8, "submit_clean": 0.4}, known=True,
        ),
        [], skills, np.zeros(1), [], [], ["request_retro_auth", "submit_clean"],
    )


def test_loop_executes_on_valid_first_attempt():
    skills = build_skills([])
    retrieval = make_retrieval(skills)
    skill = next(s for s in skills if s.scope.ref == "request_retro_auth")
    payload = {"episode_id": "ep-1", "cpt": "93229",
               "justification_code": "auth_not_obtained_before_service"}
    trace = run_case(
        retrieval, skill, base_case(),
        PromptTemplate(template_id="m1", version="t", text="{action_id}|{feedback}"),
        M2, PIN, lambda p, s: json.dumps(payload), evaluate_rule, "b", "h",
    )
    assert trace.status == "executed"
    assert len(trace.iterations) == 1
    assert trace.final.payload == payload


def test_loop_feeds_back_then_escalates_at_cap():
    skills = build_skills([])
    retrieval = make_retrieval(skills)
    skill = next(s for s in skills if s.scope.ref == "request_retro_auth")
    bad = {"episode_id": "ep-1", "cpt": "WRONG",
           "justification_code": "auth_not_obtained_before_service"}
    prompts_seen = []

    def generate(prompt, schema):
        prompts_seen.append(prompt)
        return json.dumps(bad)

    trace = run_case(
        retrieval, skill, base_case(),
        PromptTemplate(template_id="m1", version="t", text="{action_id}|{feedback}"),
        M2, PIN, generate, evaluate_rule, "b", "h",
    )
    assert trace.status == "escalated"
    assert trace.escalation.reason == "iteration_cap"
    assert len(trace.iterations) == 3
    assert "failed:" in prompts_seen[1]  # M2 feedback reached the retry prompt
    assert all(not it.check.passed for it in trace.iterations)


def test_unknown_payer_escalates_before_m1():
    skills = build_skills([])
    retrieval = make_retrieval(skills)
    skill = next(s for s in skills if s.scope.ref == "request_retro_auth")

    def generate(prompt, schema):  # must never be called
        raise AssertionError("M1 called for an out-of-distribution case")

    trace = run_case(
        retrieval, skill, base_case(payer_id="granite"),
        PromptTemplate(template_id="m1", version="t", text="{action_id}|{feedback}"),
        M2, PIN, generate, evaluate_rule, "b", "h", known=False,
    )
    assert trace.status == "escalated"
    assert trace.escalation.reason == "needs_outside_information"
    assert trace.iterations == []


def test_escalate_skill_never_calls_m1():
    skills = build_skills([])
    retrieval = make_retrieval(skills, action="provide_requested_info")
    # v0.2: PRI executes; the default skill is the remaining escalate example
    skill = next(s for s in skills if s.scope.level == "default")
    trace = run_case(
        retrieval, skill, base_case(carc="CO-16"),
        M2, M2, PIN,
        lambda p, s: pytest.fail("M1 called for an escalate skill"),
        evaluate_rule, "b", "h",
    )
    assert trace.status == "escalated"
    assert trace.escalation.reason == "unresolvable"
