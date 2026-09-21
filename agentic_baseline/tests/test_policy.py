"""Offline tests: policy loop, guardrails, briefing hygiene. No server needed."""

import pytest
import json

from agentic.briefing import CARC_LEGEND, build_briefing, decision_schema
from agentic.policy import AgenticPolicy, applicability_violation

LEGAL = [
    "submit_clean", "submit_with_records", "request_retro_auth",
    "correct_and_resubmit", "appeal_with_necessity", "request_peer_to_peer",
    "provide_requested_info", "bill_patient", "write_off",
]


def snap(**overrides):
    base = {
        "payer_id": "meridian", "payer_archetype": "commercial",
        "auth_required": True, "retro_window_days": 30,
        "timely_filing_days": 180, "p2p_available": False,
        "cpt": "93229", "dx_codes": "I49.9", "pairing_valid": True,
        "days_since_service": 12, "days_to_filing_deadline": 168,
        "auth_status": "missing", "eligibility_status": "active",
        "aob_on_file": True, "subscriber_relationship": "self",
        "cob_position_ok": True, "has_secondary": False, "carc": "",
        "balance": 842.10, "allowed_amount": 842.10,
        "deductible_remaining": 250.0, "coinsurance_pct": 0.2,
        "touches_so_far": 0, "clinic_doc_quality": 0.55,
        "persona_id": "hasty", "prior_actions": "", "prior_carcs": "",
        "cumulative_delay_days": 0, "days_to_payment": 33.0,
    }
    base.update(overrides)
    return base


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, messages, schema):
        self.requests.append((json.loads(json.dumps(messages)), schema))
        return self.replies.pop(0)


def reply(action, reasoning="because"):
    return json.dumps({"reasoning": reasoning, "action": action})


def test_happy_path_returns_action():
    policy = AgenticPolicy(FakeLLM([reply("request_retro_auth")]))
    decision = policy.decide(snap(), LEGAL)
    assert decision.action == "request_retro_auth"
    assert not decision.retried and decision.violations == []


def test_applicability_violation_feedback_then_recovery():
    llm = FakeLLM([reply("request_peer_to_peer"), reply("submit_with_records")])
    policy = AgenticPolicy(llm)
    decision = policy.decide(snap(), LEGAL)  # p2p not offered, no denial
    assert decision.action == "submit_with_records"
    assert decision.retried and len(decision.violations) == 1
    # the retry conversation carries the model's reply and the feedback
    followup = llm.requests[1][0]
    assert followup[-2]["role"] == "assistant"
    assert "not applicable" in followup[-1]["content"]


def test_two_violations_escalate_to_human():
    llm = FakeLLM([reply("appeal_with_necessity"), reply("appeal_with_necessity")])
    policy = AgenticPolicy(llm)
    decision = policy.decide(snap(), LEGAL)  # no denial on the table
    assert decision.action is None
    assert policy.escalations_forced == 1 and len(decision.violations) == 2


def test_explicit_escalate_honoured():
    policy = AgenticPolicy(FakeLLM([reply("escalate")]))
    decision = policy.decide(snap(), LEGAL)
    assert decision.action is None
    assert policy.escalations_explicit == 1 and policy.escalations_forced == 0


def test_invalid_json_then_recovery():
    llm = FakeLLM(["{not json", reply("submit_clean", "fine")])
    policy = AgenticPolicy(llm)
    decision = policy.decide(snap(auth_required=False), LEGAL)
    assert decision.action == "submit_clean" and decision.retried


def test_guardrails_are_applicability_only():
    # strategy is NOT policed: submitting into an open retro door is allowed
    assert applicability_violation("submit_clean", snap()) is None
    # applicability IS policed
    closed = snap(days_since_service=45)  # window 30
    assert "closed" in applicability_violation("request_retro_auth", closed)
    assert applicability_violation("provide_requested_info", snap()) is not None
    assert applicability_violation("provide_requested_info", snap(carc="CO-16")) is None
    assert applicability_violation("bill_secondary", snap()) is not None
    ok = snap(carc="CO-197")
    assert applicability_violation("appeal_with_necessity", ok) is None


def test_briefing_hygiene_and_content():
    briefing = build_briefing(snap(carc="CO-197", prior_actions="submit_clean",
                                   prior_carcs="CO-197"), LEGAL)
    assert "days_to_payment" not in briefing and "33.0" not in briefing  # poison
    assert CARC_LEGEND["CO-197"] in briefing
    assert "retroactive authorization window" in briefing.lower()
    assert "- escalate:" in briefing
    schema = decision_schema([*LEGAL, "escalate"])
    assert schema["properties"]["action"]["enum"][-1] == "escalate"
    assert schema["additionalProperties"] is False


def test_briefing_v4_field_parity():
    """Every decision-time fact the v0.4 basis' features read is rendered:
    persona_id and subscriber_relationship were wrongly withheld pre-v0.4."""
    briefing = build_briefing(snap(allowed_amount=900.0), LEGAL)
    assert "billing specialist assigned to this account" in briefing
    assert "hasty" in briefing  # persona_id (a basis feature) now rendered
    assert "relationship to plan subscriber: self" in briefing
    assert "allowed amount $900.00" in briefing  # allowed_amount, distinct from balance
    # the CO-18 duplicate-claim rule (v0.4 world mechanic) is stated
    assert "must not be resubmitted" in briefing
    assert "CO-18" in briefing


def test_briefing_co18_legend():
    briefing = build_briefing(snap(carc="CO-18", prior_actions="submit_clean",
                                   prior_carcs=""), LEGAL)
    assert CARC_LEGEND["CO-18"] in briefing
    assert "unlisted reason code" not in briefing


def test_claude_split_system():
    pytest.importorskip("anthropic")  # optional extra: .[llm-arms]
    from agentic.llm_claude import split_system

    system, rest = split_system([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "briefing"},
        {"role": "assistant", "content": "{...}"},
        {"role": "user", "content": "feedback"},
    ])
    assert system == "sys"
    assert [m["role"] for m in rest] == ["user", "assistant", "user"]
