"""The 12 synthetic payers (corpus spec Section 1.2): invented names, realistic
behaviour, built on real archetypes. Two are OOD holdouts that never appear in
training. Every behavioural parameter the engine reads lives here and is
serialised to payers.yaml as part of the world snapshot.
"""

from pydantic import BaseModel, Field

from cardessa.codes import CARC


class PayerConfig(BaseModel):
    payer_id: str
    name: str
    archetype: str
    holdout: bool = False
    plan_variants: list[str] = Field(default_factory=lambda: ["standard", "plus"])
    # authorisation
    auth_required_cpts: list[str] = Field(default_factory=list)
    retro_auth_window_days: int = 0  # 0 = retro-auth never granted
    retro_auth_grant_prob: float = 0.0  # within the window
    # filing and adjudication
    timely_filing_days: int = 180
    allowed_multiplier: float = 1.0  # scales the CPT base allowed range
    strict_code_pairing: bool = False
    pairing_denial_carc: str | None = None  # CO-50 (necessity) or CO-16 (correctable)
    mct_palpitations_only_denial: bool = False  # the Atlas rule
    bundling_co97: bool = False  # the Cornerstone rule
    bundling_window_days: int = 30
    missing_info_rate: float = 0.05
    requires_ordering_medicaid_enrollment: bool = False
    plan_exclusions: dict[str, list[str]] = Field(default_factory=dict)  # variant -> CPTs
    secondary_to_medicare: bool = False  # Medicaid MCO ordering rule
    # reworks
    appeal_grant_prob_strong_doc: float = 0.5
    appeal_grant_prob_weak_doc: float = 0.15
    appeal_delay_days: tuple[int, int] = (20, 45)
    p2p_available: bool = False
    p2p_reversal_prob: float = 0.0
    info_resolution_prob: float = 0.9
    response_delay_days: tuple[int, int] = (10, 25)
    # every CARC this payer can emit; the engine tests must reach each one
    carc_pathways: list[str] = Field(default_factory=list)


# Pathways every payer shares: timely filing, COB order, lapsed/not-covered,
# and missing-information (rate varies).
_UNIVERSAL = ["CO-29", "CO-22", "PR-204", "CO-16"]


def build_payer_configs() -> list[PayerConfig]:
    payers = [
        PayerConfig(
            payer_id="meridian",
            name="Meridian Health",
            archetype="national_commercial",
            auth_required_cpts=["93229"],
            retro_auth_window_days=30,
            retro_auth_grant_prob=0.85,
            appeal_grant_prob_strong_doc=0.75,
            appeal_grant_prob_weak_doc=0.20,
            carc_pathways=["CO-197", *_UNIVERSAL],
        ),
        PayerConfig(
            payer_id="atlas",
            name="Atlas Mutual",
            archetype="national_commercial",
            mct_palpitations_only_denial=True,
            appeal_grant_prob_strong_doc=0.45,
            appeal_grant_prob_weak_doc=0.10,
            carc_pathways=["CO-50", *_UNIVERSAL],
        ),
        PayerConfig(
            payer_id="cornerstone",
            name="Cornerstone United",
            archetype="national_commercial",
            bundling_co97=True,
            p2p_available=True,
            p2p_reversal_prob=0.70,
            # v0.2 spec section 3: planted near-tie vs p2p (gap 0.03)
            appeal_grant_prob_strong_doc=0.67,
            carc_pathways=["CO-97", *_UNIVERSAL],
        ),
        PayerConfig(
            payer_id="medicare",
            name="Federal Medicare (sim)",
            archetype="medicare",
            timely_filing_days=365,
            strict_code_pairing=True,
            pairing_denial_carc="CO-50",
            allowed_multiplier=0.85,
            appeal_grant_prob_strong_doc=0.60,
            appeal_grant_prob_weak_doc=0.25,
            carc_pathways=["CO-50", *_UNIVERSAL],
        ),
        PayerConfig(
            payer_id="silverbridge",
            name="SilverBridge Advantage",
            archetype="medicare_advantage",
            auth_required_cpts=["93229", "93247"],
            retro_auth_window_days=0,  # retro never granted
            retro_auth_grant_prob=0.0,
            appeal_grant_prob_strong_doc=0.65,  # slow but fair
            appeal_grant_prob_weak_doc=0.30,
            appeal_delay_days=(45, 90),
            allowed_multiplier=0.9,
            carc_pathways=["CO-197", *_UNIVERSAL],
        ),
        PayerConfig(
            payer_id="lakeshore",
            name="Lakeshore Advantage",
            archetype="medicare_advantage",
            auth_required_cpts=["93229", "93247"],  # via radiology-benefit manager
            retro_auth_window_days=14,
            retro_auth_grant_prob=0.40,
            missing_info_rate=0.30,  # frequent CO-16
            allowed_multiplier=0.9,
            carc_pathways=["CO-197", *_UNIVERSAL],
        ),
        PayerConfig(
            payer_id="prairie",
            name="Prairie State Medicaid",
            archetype="medicaid_mco",
            timely_filing_days=90,
            allowed_multiplier=0.55,
            secondary_to_medicare=True,
            carc_pathways=_UNIVERSAL,
        ),
        PayerConfig(
            payer_id="harborview",
            name="Harborview Medicaid",
            archetype="medicaid_mco",
            timely_filing_days=90,
            allowed_multiplier=0.55,
            requires_ordering_medicaid_enrollment=True,
            carc_pathways=_UNIVERSAL,
        ),
        PayerConfig(
            payer_id="bluesummit",
            name="Blue Summit Plan",
            archetype="regional_blues",
            strict_code_pairing=True,
            pairing_denial_carc="CO-16",  # correctable: corrected claims succeed
            allowed_multiplier=1.15,  # generous
            appeal_grant_prob_strong_doc=0.60,
            carc_pathways=_UNIVERSAL,
        ),
        PayerConfig(
            payer_id="keystone",
            name="Keystone TPA",
            archetype="self_funded_tpa",
            plan_exclusions={"standard": ["93247"]},  # plan-document exclusion
            appeal_grant_prob_strong_doc=0.55,  # external review possible
            carc_pathways=_UNIVERSAL,
        ),
        PayerConfig(
            payer_id="granite",
            name="Granite Shield",
            archetype="regional_commercial",
            holdout=True,
            auth_required_cpts=["93229", "93271"],  # unusual: auth even for CEM
            retro_auth_window_days=10,
            retro_auth_grant_prob=0.50,
            mct_palpitations_only_denial=True,
            carc_pathways=["CO-197", "CO-50", *_UNIVERSAL],
        ),
        PayerConfig(
            payer_id="pelican",
            name="Pelican Care",
            archetype="medicaid_mco",
            holdout=True,
            timely_filing_days=90,
            allowed_multiplier=0.50,
            requires_ordering_medicaid_enrollment=True,
            carc_pathways=_UNIVERSAL,
        ),
    ]
    for payer in payers:
        unknown = [c for c in payer.carc_pathways if c not in CARC]
        if unknown:
            raise ValueError(f"{payer.payer_id}: unknown CARC codes {unknown}")
    return payers


def payer_index() -> dict[str, PayerConfig]:
    return {p.payer_id: p for p in build_payer_configs()}
