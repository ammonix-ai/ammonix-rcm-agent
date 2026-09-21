"""Admin personas (corpus spec Section 3): scripted decision policies.

Three temperaments, rotated per episode, identity recorded as a legitimate
state feature. The planted suboptimal habits live here: in the retro-auth
family (MCT at Meridian/SilverBridge with prior auth missing), Hasty appeals
and Conservative writes off, so the majority of family states carry the wrong
action while the correct one is payer-conditional (retro-auth inside the
window at Meridian, strong appeal at SilverBridge).
"""

from dataclasses import dataclass

import numpy as np

from cardessa.engine import stable_seed
from cardessa.payers import PayerConfig

PERSONA_MIX = {"diligent": 0.40, "hasty": 0.35, "conservative": 0.25}

CONSERVATIVE_EFFORT_THRESHOLD = 300.0  # balances above this are not worth fighting


@dataclass
class CaseView:
    """What the persona can see when deciding (no post-hoc information)."""

    payer: PayerConfig
    touch_seq: int
    submitted_once: bool  # a claim has reached the payer at least once
    carc: str | None  # denial currently on the table, None before/after payment
    underpaid: bool  # payer paid its part, patient share remains
    auth_required: bool
    auth_on_file: bool
    days_since_service: int
    balance: float
    has_secondary: bool
    doc_quality: float
    resubmitted_unchanged_already: bool
    appealed_already: bool
    retro_requested_already: bool


def persona_for_episode(master_seed: int, episode_index: int) -> str:
    rng = np.random.default_rng(stable_seed(master_seed, "persona", episode_index))
    names = list(PERSONA_MIX)
    return names[int(rng.choice(len(names), p=np.array(list(PERSONA_MIX.values()))))]


def _retro_window_open(view: CaseView) -> bool:
    return (
        view.payer.retro_auth_window_days > 0
        and view.days_since_service <= view.payer.retro_auth_window_days
    )


def _initial_action(persona: str, view: CaseView, rng: np.random.Generator) -> str:
    if persona == "diligent" and rng.random() >= 0.05:  # small error rate 0.05
        if view.auth_required and not view.auth_on_file and _retro_window_open(view):
            return "request_retro_auth"  # fix the gap before submitting
        if view.doc_quality < 0.6:
            return "submit_with_records"
        return "submit_clean"
    if persona == "conservative":
        return "submit_with_records"  # requests records excessively
    return "submit_clean"  # hasty (and the diligent error case): never checks auth


def _denial_response(persona: str, view: CaseView, rng: np.random.Generator) -> str:
    carc = view.carc
    if persona == "hasty" and not view.resubmitted_unchanged_already:
        return "submit_clean"  # resubmits unchanged once before escalating
    if carc == "CO-197":
        if persona == "diligent" and rng.random() >= 0.05:
            if _retro_window_open(view) and not view.retro_requested_already:
                return "request_retro_auth"  # correct at Meridian-like payers
            return "appeal_with_necessity"  # correct at SilverBridge-like payers
        if persona == "conservative":
            # the planted habit: give up on big auth problems
            return "write_off" if view.balance > CONSERVATIVE_EFFORT_THRESHOLD else "bill_patient"
        return "appeal_with_necessity"  # hasty: appeal regardless of the window
    if carc == "CO-16":
        return "provide_requested_info"
    if carc == "CO-97":
        if persona == "diligent" and view.payer.p2p_available:
            return "request_peer_to_peer"
        if persona == "conservative":
            return "write_off" if view.balance > CONSERVATIVE_EFFORT_THRESHOLD else "bill_patient"
        return "appeal_with_necessity"
    if carc == "CO-50":
        if persona == "conservative":
            return "write_off" if view.balance > CONSERVATIVE_EFFORT_THRESHOLD else "bill_patient"
        return "appeal_with_necessity"
    if carc == "CO-22":
        return "correct_and_resubmit"
    if carc == "CO-29":
        if persona == "hasty" and not view.appealed_already:
            return "appeal_with_necessity"  # futile, but that is the temperament
        return "write_off"
    if carc == "PR-204":
        if persona == "hasty" and not view.appealed_already:
            return "appeal_with_necessity"
        return "bill_patient"  # no coverage: the balance is the patient's
    if carc == "CO-18":
        # duplicate of an already-adjudicated claim (v0.4): the payer has
        # paid what it owes; the remaining balance is the patient's share
        return "bill_patient"
    raise ValueError(f"no persona response for CARC {carc!r}")


def choose_action(
    persona: str, view: CaseView, master_seed: int, episode_id: str
) -> str:
    """The persona's decision at this state. Deterministic given the seed."""
    rng = np.random.default_rng(
        stable_seed(master_seed, "policy", persona, episode_id, view.touch_seq)
    )
    if view.touch_seq == 0:
        return _initial_action(persona, view, rng)
    if not view.submitted_once:
        # a pre-submission rework (retro-auth request) resolved either way:
        # the claim still has to go to the payer
        return "submit_with_records" if persona != "hasty" else "submit_clean"
    if view.underpaid:
        if persona == "diligent" and view.has_secondary:
            return "bill_secondary"
        return "bill_patient"
    if view.carc is not None:
        if view.appealed_already:
            # the appeal path is exhausted; stop fighting and close the balance
            if view.balance > CONSERVATIVE_EFFORT_THRESHOLD and persona != "diligent":
                return "write_off"
            return "bill_patient"
        return _denial_response(persona, view, rng)
    # exhausted reworks (e.g. appeal denied): close the balance out
    if persona == "diligent" and view.balance <= CONSERVATIVE_EFFORT_THRESHOLD:
        return "bill_patient"
    return "write_off"
