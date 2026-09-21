"""Escalation-fallback wrapper shared by every arm's persona hand-off.

v0.4 harness gap (found on the first full tranche-98 run): the scripted
personas predate CO-18. A clerk never resubmits an already-paid claim, so
`_denial_response` in the factory's personas.py has no CO-18 branch and
raises ValueError when an escalated touch hands over a CO-18 state - a state
only an LLM arm can create (by resubmitting a paid claim despite the
briefing's duplicate rule; that mistake is the agent's and stays on its
ledger). The wrapper is arm-agnostic: the personas and Ammonix arms
structurally never enter this path (clerks don't resubmit paid claims and
the v0.4 mask blocks it), so wrapping them changes nothing.

The clerical close-out for a duplicate-denied claim is objectively
determined: the payer has already adjudicated and paid, so contesting is
over - bill the remaining member responsibility if there is one and billing
the patient is legal in this state, else write the balance off. This is the
same close-out split the persona scripts use on their own exhausted paths
(personas.py choose_action tail / appeal-exhausted branch).
"""

# the exact raise in the factory's personas.py _denial_response
_NO_PERSONA_RESPONSE = "no persona response for CARC"


def close_out_action(patient_share_pending: float, legal_actions) -> str:
    """The scripted clerical close-out for an already-adjudicated claim."""
    if patient_share_pending > 0 and "bill_patient" in legal_actions:
        return "bill_patient"
    return "write_off"


def persona_action(choose_action, persona, view, master_seed, episode_id,
                   patient_share_pending: float, legal_actions) -> str:
    """choose_action, with the unknown-CARC close-out fallback.

    Catches ONLY the personas' no-response ValueError; anything else is a
    real bug and propagates.
    """
    try:
        return choose_action(persona, view, master_seed, episode_id)
    except ValueError as err:
        if _NO_PERSONA_RESPONSE not in str(err):
            raise
        return close_out_action(patient_share_pending, legal_actions)
