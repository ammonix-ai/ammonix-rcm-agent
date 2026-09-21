"""The escalation-fallback wrapper: CO-18 close-out, tight scope guard."""

import pytest

from agentic.fallback import close_out_action, persona_action

LEGAL = [
    "submit_clean", "submit_with_records", "appeal_with_necessity",
    "bill_patient", "write_off",
]


def raising_choose_action(persona, view, master_seed, episode_id):
    # the exact raise from the factory's personas.py _denial_response when an
    # escalated touch hands the clerk a CO-18 (duplicate-denied) state
    raise ValueError("no persona response for CARC 'CO-18'")


def test_co18_with_member_responsibility_bills_patient():
    action = persona_action(raising_choose_action, "hasty", object(),
                            20260709, "ep-99999",
                            patient_share_pending=75.86, legal_actions=LEGAL)
    assert action == "bill_patient"


def test_co18_without_member_responsibility_writes_off():
    action = persona_action(raising_choose_action, "diligent", object(),
                            20260709, "ep-99999",
                            patient_share_pending=0.0, legal_actions=LEGAL)
    assert action == "write_off"


def test_co18_share_pending_but_bill_patient_not_legal():
    assert close_out_action(75.86, ["write_off"]) == "write_off"


def test_other_value_errors_propagate():
    def broken(persona, view, master_seed, episode_id):
        raise ValueError("something else entirely")

    with pytest.raises(ValueError, match="something else"):
        persona_action(broken, "hasty", object(), 20260709, "ep-1",
                       patient_share_pending=10.0, legal_actions=LEGAL)


def test_normal_decisions_pass_through_unchanged():
    def scripted(persona, view, master_seed, episode_id):
        return "appeal_with_necessity"

    action = persona_action(scripted, "hasty", object(), 20260709, "ep-1",
                            patient_share_pending=10.0, legal_actions=LEGAL)
    assert action == "appeal_with_necessity"
