"""Every CARC pathway of every payer is reachable and correct (W1 gate).

For each payer, each CARC declared in carc_pathways is triggered by a
constructed claim; the baseline claim pays. Stochastic pathways (reviewer
CO-16) are exercised through the seeded deterministic draws.
"""

import pytest

from cardessa import MASTER_SEED
from cardessa.engine import Claim, PayerEngine
from cardessa.payers import build_payer_configs

PAYERS = {p.payer_id: p for p in build_payer_configs()}
ENGINE = PayerEngine(PAYERS, MASTER_SEED)

ALL_PATHWAYS = sorted(
    {(payer_id, carc) for payer_id, p in PAYERS.items() for carc in p.carc_pathways}
)


def make_claim(payer_id: str, **overrides) -> Claim:
    """A baseline claim that adjudicates to payment at every payer."""
    payer = PAYERS[payer_id]
    base = dict(
        claim_id="clm-test-0",
        payer_id=payer_id,
        plan_variant="plus",  # avoids the Keystone standard-plan exclusion
        cpt="93226",  # Holter: no payer requires auth for it
        icd_codes=["I48.0"],  # justifies every study under strict pairing
        member_id="M00000001",
        auth_ref=None,
        attachments=["clinical_notes"],  # preempts reviewer CO-16
        days_since_service=10,
    )
    if base["cpt"] in payer.auth_required_cpts:
        base["auth_ref"] = "AUTH-1"
    base.update(overrides)
    return Claim(**base)


def trigger_claim(payer_id: str, carc: str) -> Claim:
    """Construct a claim that must hit exactly this CARC at this payer."""
    payer = PAYERS[payer_id]
    if carc == "CO-29":
        return make_claim(payer_id, days_since_service=payer.timely_filing_days + 10)
    if carc == "CO-22":
        return make_claim(payer_id, cob_position_correct=False)
    if carc == "PR-204":
        return make_claim(payer_id, coverage_active=False)
    if carc == "CO-197":
        cpt = payer.auth_required_cpts[0]
        return make_claim(payer_id, cpt=cpt, auth_ref=None)
    if carc == "CO-97":
        return make_claim(payer_id, cpt="93271", prior_holter_within_days=15)
    if carc == "CO-50":
        if payer.mct_palpitations_only_denial:
            claim = make_claim(payer_id, cpt="93229", icd_codes=["R00.2"])
        else:  # strict code pairing (Medicare): unsupported dx for MCT
            claim = make_claim(payer_id, cpt="93229", icd_codes=["I25.10"])
        if claim.cpt in payer.auth_required_cpts:
            claim.auth_ref = "AUTH-1"
        return claim
    if carc == "CO-16":
        if payer.strict_code_pairing and payer.pairing_denial_carc == "CO-16":
            return make_claim(payer_id, cpt="93229", icd_codes=["I25.10"])
        # reviewer pathway: find a seeded claim id that draws a CO-16
        for i in range(500):
            claim = make_claim(payer_id, claim_id=f"clm-co16-{i}", attachments=[])
            if ENGINE.adjudicate(claim).carc == "CO-16":
                return claim
        raise AssertionError(f"{payer_id}: no CO-16 draw in 500 seeded claims")
    raise AssertionError(f"no trigger construction for {carc}")


@pytest.mark.parametrize("payer_id", sorted(PAYERS))
def test_baseline_claim_pays(payer_id):
    result = ENGINE.adjudicate(make_claim(payer_id))
    assert result.carc is None
    assert result.status == "paid"
    assert result.paid_amount == result.allowed_amount > 0


@pytest.mark.parametrize(("payer_id", "carc"), ALL_PATHWAYS)
def test_carc_pathway_reachable(payer_id, carc):
    claim = trigger_claim(payer_id, carc)
    result = ENGINE.adjudicate(claim)
    assert result.carc == carc, (
        f"{payer_id}: expected {carc}, got {result.carc} ({result.remark})"
    )
    assert result.status == "denied"
    assert result.paid_amount == 0.0


def test_prairie_medicare_primary_is_a_second_co22_route():
    result = ENGINE.adjudicate(make_claim("prairie", has_medicare_primary=True))
    assert result.carc == "CO-22"


def test_keystone_plan_exclusion_is_a_second_pr204_route():
    result = ENGINE.adjudicate(make_claim("keystone", plan_variant="standard", cpt="93247"))
    assert result.carc == "PR-204"


def test_enrollment_pr204_routes():
    for payer_id in ("harborview", "pelican"):
        result = ENGINE.adjudicate(
            make_claim(payer_id, ordering_npi_medicaid_enrolled=False)
        )
        assert result.carc == "PR-204", payer_id


def test_records_preempt_reviewer_co16():
    """Claims with clinical notes attached never draw the reviewer CO-16."""
    for i in range(200):
        claim = make_claim("lakeshore", claim_id=f"clm-notes-{i}", cpt="93226")
        assert ENGINE.adjudicate(claim).carc is None


def test_patient_responsibility_creates_underpayment():
    claim = make_claim("meridian", deductible_remaining=400.0, coinsurance_pct=0.2)
    result = ENGINE.adjudicate(claim)
    assert result.status == "underpaid"
    assert result.paid_amount + result.patient_responsibility == pytest.approx(
        result.allowed_amount
    )


def test_every_payer_declares_the_universal_pathways():
    for payer in PAYERS.values():
        assert {"CO-29", "CO-22", "PR-204", "CO-16"} <= set(payer.carc_pathways)


def test_twelve_payers_two_holdout():
    assert len(PAYERS) == 12
    assert sorted(p.payer_id for p in PAYERS.values() if p.holdout) == ["granite", "pelican"]
