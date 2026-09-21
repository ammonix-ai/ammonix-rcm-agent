"""Rework resolutions and the evaluation-only analytic probabilities."""

import numpy as np
import pytest

from cardessa import MASTER_SEED
from cardessa.engine import Claim, PayerEngine, stable_seed
from cardessa.payers import build_payer_configs

PAYERS = {p.payer_id: p for p in build_payer_configs()}
ENGINE = PayerEngine(PAYERS, MASTER_SEED)


def test_stable_seed_is_deterministic_and_distinct():
    assert stable_seed(1, "a", "b") == stable_seed(1, "a", "b")
    assert stable_seed(1, "a", "b") != stable_seed(1, "a", "c")


def test_adjudication_is_regenerate_identical():
    claim = Claim(
        claim_id="clm-det",
        payer_id="lakeshore",
        plan_variant="standard",
        cpt="93226",
        icd_codes=["I48.0"],
        member_id="M11111111",
        attachments=[],
    )
    first = ENGINE.adjudicate(claim)
    again = PayerEngine(PAYERS, MASTER_SEED).adjudicate(claim)
    assert first == again
    other_world = PayerEngine(PAYERS, MASTER_SEED + 1)
    results_differ = any(
        other_world.adjudicate(
            Claim(
                claim_id=f"clm-{i}",
                payer_id="lakeshore",
                plan_variant="standard",
                cpt="93226",
                icd_codes=["I48.0"],
                member_id="M11111111",
                attachments=[],
            )
        )
        != ENGINE.adjudicate(
            Claim(
                claim_id=f"clm-{i}",
                payer_id="lakeshore",
                plan_variant="standard",
                cpt="93226",
                icd_codes=["I48.0"],
                member_id="M11111111",
                attachments=[],
            )
        )
        for i in range(50)
    )
    assert results_differ  # a different master seed is a different world


def test_allowed_amount_stable_per_member():
    a = ENGINE.allowed_amount(PAYERS["meridian"], "93229", "M123")
    b = ENGINE.allowed_amount(PAYERS["meridian"], "93229", "M123")
    assert a == b
    assert ENGINE.allowed_amount(PAYERS["meridian"], "93229", "M124") != a


def test_medicaid_allowed_amounts_are_low():
    commercial = ENGINE.allowed_amount(PAYERS["meridian"], "93229", "M1")
    medicaid = ENGINE.allowed_amount(PAYERS["prairie"], "93229", "M1")
    assert medicaid < commercial


def test_retro_auth_meridian_window():
    assert ENGINE.action_resolution_prob(
        "meridian", "request_retro_auth", {"days_since_service": 12}
    ) == pytest.approx(0.85)
    assert (
        ENGINE.action_resolution_prob(
            "meridian", "request_retro_auth", {"days_since_service": 45}
        )
        == 0.0
    )
    granted = [
        ENGINE.retro_auth_granted("meridian", 12, f"case-{i}") for i in range(300)
    ]
    rate = float(np.mean(granted))
    assert 0.75 < rate < 0.95  # seeded draws realise the configured 0.85


def test_retro_auth_never_at_silverbridge():
    assert (
        ENGINE.action_resolution_prob(
            "silverbridge", "request_retro_auth", {"days_since_service": 1}
        )
        == 0.0
    )
    assert not any(
        ENGINE.retro_auth_granted("silverbridge", 1, f"case-{i}") for i in range(100)
    )


def test_p1_trap_is_payer_conditional():
    """The planted P1 structure: same situation, opposite correct action."""
    context = {"days_since_service": 12, "strong_documentation": True}
    meridian_retro = ENGINE.action_resolution_prob("meridian", "request_retro_auth", context)
    meridian_appeal = ENGINE.action_resolution_prob("meridian", "appeal_with_necessity", context)
    silver_retro = ENGINE.action_resolution_prob("silverbridge", "request_retro_auth", context)
    silver_appeal = ENGINE.action_resolution_prob("silverbridge", "appeal_with_necessity", context)
    assert meridian_retro > meridian_appeal  # at Meridian: retro-auth wins
    assert silver_appeal > silver_retro  # at SilverBridge: appeal wins


def test_p2p_only_where_available():
    assert ENGINE.action_resolution_prob(
        "cornerstone", "request_peer_to_peer", {}
    ) == pytest.approx(0.70)
    assert ENGINE.action_resolution_prob("meridian", "request_peer_to_peer", {}) == 0.0
    assert not any(ENGINE.p2p_reversed("meridian", f"case-{i}") for i in range(50))


def test_appeal_documentation_matters():
    strong = ENGINE.action_resolution_prob(
        "meridian", "appeal_with_necessity", {"strong_documentation": True}
    )
    weak = ENGINE.action_resolution_prob(
        "meridian", "appeal_with_necessity", {"strong_documentation": False}
    )
    assert strong > weak


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        ENGINE.action_resolution_prob("meridian", "sing_to_the_payer", {})
