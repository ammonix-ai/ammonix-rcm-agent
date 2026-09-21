"""The payer policy engine (corpus spec Section 1.3): one deterministic rules
engine, parameterised per payer, adjudicates every submission.

Where real payers are genuinely stochastic (appeal outcomes, reviewer
strictness, missing-info requests) the engine draws from seeded per-payer
distributions keyed by (master seed, payer, claim, purpose), so chance is real
but regenerate-identical from the master seed.

The engine also exposes, for evaluation only, per-action resolution
probabilities: the analytic answer key behind the P4/P7 calibration gates.
The full P(episode success | state, action) oracle grid is materialised at P7
(seeded rollouts under the corpus continuation policy); the primitives here
are what those rollouts consume.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from ammonix_core.hashing import sha256_text

from cardessa.codes import COVERED_DX, STUDIES
from cardessa.payers import PayerConfig


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed from any key parts (regenerate-identical)."""
    return int(sha256_text("|".join(str(p) for p in parts))[:16], 16) % (2**63)


def _rng(*parts: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


@dataclass
class Claim:
    """A submission touch as the payer sees it."""

    claim_id: str
    payer_id: str
    plan_variant: str
    cpt: str
    icd_codes: list[str]
    member_id: str
    auth_ref: str | None = None
    attachments: list[str] = field(default_factory=list)
    days_since_service: int = 10
    ordering_npi_medicaid_enrolled: bool = True
    cob_position_correct: bool = True
    has_medicare_primary: bool = False  # for secondary_to_medicare payers
    coverage_active: bool = True
    prior_holter_within_days: int | None = None
    deductible_remaining: float = 0.0
    coinsurance_pct: float = 0.0
    billed_amount: float = 0.0


@dataclass
class Adjudication:
    status: Literal["paid", "underpaid", "denied"]
    carc: str | None
    allowed_amount: float
    paid_amount: float
    patient_responsibility: float
    delay_days: int
    remark: str


def patient_share(
    allowed: float, deductible_remaining: float, coinsurance_pct: float
) -> float:
    """Patient responsibility on an allowed amount: deductible first, then
    coinsurance on the remainder. The ONE implementation of this formula -
    adjudication and every corpus payment path must call it, never re-derive
    it (the copies drifted apart is exactly the bug class this prevents)."""
    deductible = min(deductible_remaining, allowed)
    return round(deductible + coinsurance_pct * (allowed - deductible), 2)


class PayerEngine:
    """Adjudication plus rework resolution for one world (one master seed)."""

    def __init__(self, payers: dict[str, PayerConfig], master_seed: int):
        self.payers = payers
        self.master_seed = master_seed

    # -- core adjudication ------------------------------------------------

    def allowed_amount(self, payer: PayerConfig, cpt: str, member_id: str) -> float:
        """Stable per (payer, cpt, member): resubmissions see the same allowed."""
        lo, hi = STUDIES[cpt][1]
        draw = _rng(self.master_seed, "allowed", payer.payer_id, cpt, member_id).uniform(lo, hi)
        return round(draw * payer.allowed_multiplier, 2)

    def response_delay(self, payer: PayerConfig, claim_id: str) -> int:
        lo, hi = payer.response_delay_days
        return int(_rng(self.master_seed, "delay", payer.payer_id, claim_id).integers(lo, hi + 1))

    def adjudicate(self, claim: Claim) -> Adjudication:
        payer = self.payers[claim.payer_id]
        delay = self.response_delay(payer, claim.claim_id)

        def deny(carc: str, remark: str) -> Adjudication:
            return Adjudication(
                status="denied",
                carc=carc,
                allowed_amount=0.0,
                paid_amount=0.0,
                patient_responsibility=0.0,
                delay_days=delay,
                remark=remark,
            )

        # 1. Coordination of benefits
        if not claim.cob_position_correct:
            return deny("CO-22", "Claim submitted out of coordination-of-benefits order.")
        if payer.secondary_to_medicare and claim.has_medicare_primary:
            return deny("CO-22", "Medicare is primary for this member; bill Medicare first.")
        # 2. Coverage in force
        if not claim.coverage_active:
            return deny("PR-204", "No active coverage on the date of service.")
        # 3. Timely filing
        if claim.days_since_service > payer.timely_filing_days:
            return deny("CO-29", "Received after the timely filing limit.")
        # 4. Plan-document exclusions
        excluded = payer.plan_exclusions.get(claim.plan_variant, [])
        if claim.cpt in excluded:
            return deny("PR-204", "Service excluded by the member's plan document.")
        # 5. Ordering-provider enrolment (Medicaid rule)
        if payer.requires_ordering_medicaid_enrollment and not claim.ordering_npi_medicaid_enrolled:
            return deny("PR-204", "Ordering physician is not enrolled with this Medicaid plan.")
        # 6. Prior authorisation
        if claim.cpt in payer.auth_required_cpts and not claim.auth_ref:
            return deny("CO-197", "Prior authorization required and no authorization on file.")
        # 7. Bundling
        if (
            payer.bundling_co97
            and claim.cpt == "93271"
            and claim.prior_holter_within_days is not None
            and claim.prior_holter_within_days <= payer.bundling_window_days
        ):
            return deny("CO-97", "Event monitoring bundled with recent Holter service.")
        # 8. Medical necessity / code pairing
        if payer.mct_palpitations_only_denial and claim.cpt == "93229":
            justified = set(claim.icd_codes) & COVERED_DX["93229"]
            if not justified:
                return deny("CO-50", "MCT not medically necessary for palpitations alone.")
        if payer.strict_code_pairing and not (set(claim.icd_codes) & COVERED_DX[claim.cpt]):
            carc = payer.pairing_denial_carc or "CO-50"
            return deny(carc, "Diagnosis does not support the billed procedure code.")
        # 9. Missing information (seeded reviewer behaviour; records preempt it)
        if "clinical_notes" not in claim.attachments:
            draw = _rng(self.master_seed, "co16", payer.payer_id, claim.claim_id).random()
            if draw < payer.missing_info_rate:
                return deny("CO-16", "Additional documentation required to process this claim.")
        # 10. Pay
        allowed = self.allowed_amount(payer, claim.cpt, claim.member_id)
        patient_resp = patient_share(
            allowed, claim.deductible_remaining, claim.coinsurance_pct
        )
        paid = round(allowed - patient_resp, 2)
        status = "paid" if paid >= 0.9 * allowed else "underpaid"
        return Adjudication(
            status=status,
            carc=None,
            allowed_amount=allowed,
            paid_amount=paid,
            patient_responsibility=patient_resp,
            delay_days=delay,
            remark="Processed per plan benefits.",
        )

    # -- rework resolutions -------------------------------------------------

    def retro_auth_granted(self, payer_id: str, days_since_service: int, case_id: str) -> bool:
        payer = self.payers[payer_id]
        if days_since_service > payer.retro_auth_window_days:
            return False
        draw = _rng(self.master_seed, "retro", payer_id, case_id).random()
        return draw < payer.retro_auth_grant_prob

    def appeal_granted(self, payer_id: str, strong_documentation: bool, case_id: str) -> bool:
        payer = self.payers[payer_id]
        prob = (
            payer.appeal_grant_prob_strong_doc
            if strong_documentation
            else payer.appeal_grant_prob_weak_doc
        )
        return _rng(self.master_seed, "appeal", payer_id, case_id).random() < prob

    def p2p_reversed(self, payer_id: str, case_id: str) -> bool:
        payer = self.payers[payer_id]
        if not payer.p2p_available:
            return False
        return _rng(self.master_seed, "p2p", payer_id, case_id).random() < payer.p2p_reversal_prob

    def info_request_resolved(self, payer_id: str, case_id: str) -> bool:
        payer = self.payers[payer_id]
        return _rng(self.master_seed, "info", payer_id, case_id).random() < (
            payer.info_resolution_prob
        )

    # -- evaluation-only analytic probabilities -----------------------------

    def action_resolution_prob(self, payer_id: str, action: str, context: dict) -> float:
        """The true probability that a rework action resolves in the claimant's
        favour, straight from the engine's own parameters. Evaluation only:
        never available to the build loops as a feature.
        """
        payer = self.payers[payer_id]
        if action == "request_retro_auth":
            if context.get("days_since_service", 0) > payer.retro_auth_window_days:
                return 0.0
            return payer.retro_auth_grant_prob
        if action == "appeal_with_necessity":
            return (
                payer.appeal_grant_prob_strong_doc
                if context.get("strong_documentation", False)
                else payer.appeal_grant_prob_weak_doc
            )
        if action == "request_peer_to_peer":
            return payer.p2p_reversal_prob if payer.p2p_available else 0.0
        if action == "provide_requested_info":
            return payer.info_resolution_prob
        raise ValueError(f"no analytic resolution probability for action {action!r}")
