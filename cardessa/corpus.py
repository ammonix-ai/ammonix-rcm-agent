"""Episode simulation and corpus generation (corpus spec Sections 2-3).

Pure-Python simulation loop: world snapshot + persona policies + payer engine,
one master seed, regenerate-identical. The only LLM involvement is text
generation, behind a prompt-hash cache. Episodes are emitted in factory schema
shape (Example, RawState, RawData, Outcome) via the EnvironmentProtocol
adapter.

Post-hoc quarantine: paid_amount_final, days_to_payment, total_touches and
engine_truth_* live in the episode meta, never in state Data — EXCEPT the one
deliberately poisoned column (days_to_payment) planted in the raw tabular
state data for the P2 leakage audit to catch.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from ammonix_core.hashing import sha256_json
from ammonix_core.schema import Example, Outcome, RawData, RawState

from cardessa.codes import COVERED_DX
from cardessa.engine import Claim, PayerEngine, patient_share, stable_seed
from cardessa.payer_reader import PayerReader, payer_receives
from cardessa.personas import CaseView, choose_action, persona_for_episode
from cardessa.textgen import (
    CachedTextGenerator,
    correspondence_prompt,
    indication_prompt,
    salt_token,
)
from cardessa.world import World

MAX_DECISIONS = 7  # hard cap per episode; write_off forced at the cap
SUCCESS_FRACTION = 0.9
SUCCESS_WINDOW_DAYS = 120
TOUCH_PENALTY = 0.05

ADMIN_ACTION_DELAY_DAYS = (1, 5)  # days the admin takes to execute a decision
PATIENT_PAY_DELAY_DAYS = (20, 40)
RETRO_DECISION_DELAY_DAYS = (5, 12)

TRANCHE_SIZE_INITIAL = 1000
TRANCHE_SIZE_EXPANSION = 500
TRANCHE_CAP_EPISODES = 8000  # v0.3 spec section 5.1


@dataclass
class Case:
    """StateHandle for the Cardessa environment: one claim episode in flight."""

    episode_id: str
    episode_index: int
    persona: str
    patient: dict
    coverage: dict
    study: dict
    clinic: dict
    payer_id: str
    touch_seq: int = 0
    submitted_once: bool = False
    resubmitted_unchanged: bool = False
    resubmitted_after_paid: bool = False
    appealed: bool = False
    retro_requested: bool = False
    auth_ref: str | None = None
    carc: str | None = None
    carc_history: list[str] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)
    correspondence_kind: str | None = None  # drives the letter for this state
    correspondence_extra: bool = False  # CO-16 letter names the missing document
    # paperwork the payer rejected on receipt (wrong codes, wrong denial
    # addressed, item it never asked for, a letter its reviewer could not
    # support); each one costs the touch and a response delay
    paperwork_rejections: int = 0
    paperwork_rejection_log: list[str] = field(default_factory=list)
    days_since_service: int = 0
    days_elapsed: int = 0  # since first submission (the 120-day clock)
    collected_payer: float = 0.0
    collected_patient: float = 0.0
    contractual_share: float = 0.0  # patient share fixed at the paying adjudication (outcome v2)
    allowed: float = 0.0  # engine fee schedule, known to the admin
    patient_share_pending: float = 0.0
    wrong_cob_start: bool = False
    lapsed: bool = False
    terminal: bool = False
    resolution: str | None = None


class CardessaEnvironment:
    """EnvironmentProtocol implementation over the world + payer engine."""

    def __init__(self, world: World, engine: PayerEngine, master_seed: int,
                 reader: PayerReader | None = None):
        self.world = world
        self.engine = engine
        self.master_seed = master_seed
        self.reader = reader  # the payer's reviewer; needed to receive appeal letters
        self._patients = {r["patient_id"]: r for r in world.patients.rows(named=True)}
        self._coverage = {r["patient_id"]: r for r in world.coverage.rows(named=True)}
        self._clinics = {r["clinic_id"]: r for r in world.clinics.rows(named=True)}
        self._studies = world.studies.rows(named=True)

    def reset(self, seed: int) -> Case:
        """seed is the episode index; episodes are generated in indexed order."""
        study = self._studies[seed]
        patient = self._patients[study["patient_id"]]
        coverage = self._coverage[study["patient_id"]]
        clinic = self._clinics[study["clinic_id"]]
        rng = np.random.default_rng(stable_seed(self.master_seed, "episode", seed))
        days_since_service = (study["report_delivered"] - study["service_start"]).days
        # hasty admins sometimes sit on the file long enough to risk timely filing
        persona = persona_for_episode(self.master_seed, seed)
        if persona == "hasty" and rng.random() < 0.1:
            days_since_service += int(rng.integers(70, 120))
        case = Case(
            episode_id=f"ep-{seed:05d}",
            episode_index=seed,
            persona=persona,
            patient=patient,
            coverage=coverage,
            study=study,
            clinic=clinic,
            payer_id=patient["payer_id"],
            auth_ref="AUTH-PRE" if study["auth_obtained"] else None,
            days_since_service=days_since_service,
            allowed=self.engine.allowed_amount(
                self.engine.payers[patient["payer_id"]],
                study["cpt"],
                coverage["member_id"],
            ),
            wrong_cob_start=(study["family"] == "cob")
            and rng.random() < 0.25,  # some COB families start billed out of order
            lapsed=study["coverage_lapsed"],
        )
        case.correspondence_kind = None  # state 0 carries the indication note only
        return case

    def legal_actions(self, case: Case) -> list[str]:
        if case.terminal:
            return []
        actions = [
            "submit_clean",
            "submit_with_records",
            "request_retro_auth",
            "correct_and_resubmit",
            "appeal_with_necessity",
            "request_peer_to_peer",
            "provide_requested_info",
            "bill_patient",
            "write_off",
        ]
        if case.coverage["secondary_payer_id"] is not None:
            actions.append("bill_secondary")
        return actions

    def outcome(self, case: Case) -> Outcome | None:
        """Outcome v2 (change spec section 1): the payer pays what the payer
        owes. Patient money counts only up to the contractual share fixed at
        the paying adjudication; balance-billing a failed claim is failure."""
        if not case.terminal:
            return None
        patient_valid = min(case.collected_patient, case.contractual_share)
        collected_v2 = case.collected_payer + patient_valid
        fraction = min(1.0, collected_v2 / case.allowed) if case.allowed > 0 else 0.0
        success = (
            fraction >= SUCCESS_FRACTION and case.days_elapsed <= SUCCESS_WINDOW_DAYS
        )
        touches = len(case.action_history)
        score = max(0.0, fraction - TOUCH_PENALTY * max(0, touches - 1))
        return Outcome(success=success, score=round(score, 4))

    # -- dynamics ---------------------------------------------------------

    def _rng(self, case: Case, purpose: str) -> np.random.Generator:
        return np.random.default_rng(
            stable_seed(self.master_seed, purpose, case.episode_id, case.touch_seq)
        )

    def _claim(self, case: Case, with_records: bool, corrected: bool) -> Claim:
        pairing_broken_dx = case.patient["diagnoses"]
        return Claim(
            claim_id=f"{case.episode_id}-t{case.touch_seq}",
            payer_id=case.payer_id,
            plan_variant=case.coverage["plan_variant"],
            cpt=case.study["cpt"],
            icd_codes=list(pairing_broken_dx),
            member_id=case.coverage["member_id"],
            auth_ref=case.auth_ref,
            attachments=["clinical_notes"] if with_records else [],
            days_since_service=case.days_since_service,
            ordering_npi_medicaid_enrolled=case.clinic["medicaid_enrolled"],
            cob_position_correct=not case.wrong_cob_start or corrected,
            has_medicare_primary=case.coverage["has_medicare_primary"] and not corrected,
            coverage_active=not case.lapsed,
            prior_holter_within_days=case.study["prior_holter_within_days"],
            deductible_remaining=case.coverage["deductible_remaining"],
            coinsurance_pct=case.coverage["coinsurance_pct"],
            billed_amount=case.allowed,
        )

    def _advance_clock(self, case: Case, days: int) -> None:
        if case.submitted_once:
            case.days_elapsed += days
        case.days_since_service += days

    # -- what the payer does with paperwork it receives ------------------

    def _reject_paperwork(self, case: Case, reason: str) -> None:
        """The payer returns the paperwork unprocessed: the touch is spent,
        a response delay passes, the denial on file stands."""
        case.paperwork_rejections += 1
        case.paperwork_rejection_log.append(f"t{case.touch_seq}:{reason}")
        case.correspondence_kind = "paperwork_rejected"
        self._advance_clock(
            case, int(self._rng(case, "reject_delay").integers(5, 15))
        )
        if not case.terminal and len(case.action_history) >= MAX_DECISIONS:
            case.action_history.append("write_off")
            self._finish(case, "written_off_at_cap")

    def _payer_facts(self, case: Case) -> dict:
        """What the payer holds about this claim, for payer_receives."""
        return {
            "cpt": case.study["cpt"],
            "dx_codes": list(case.patient["diagnoses"]),
            "carc": case.carc,
            "patient_share": case.patient_share_pending or case.allowed,
            "balance": round(case.allowed - case.collected_payer - case.collected_patient, 2),
        }

    def apply(self, case: Case, action_raw: str, payload: dict | None = None,
              records: dict | None = None) -> Case:
        """One touch. `payload` is the paperwork the payer receives (None for
        the scripted clerks, whose forms are correct by construction);
        `records` carries the state's texts (the clinical note and the
        payer's last letter) for the payer's reviewer."""
        if case.terminal:
            raise ValueError(f"{case.episode_id}: episode already terminal")
        case.action_history.append(action_raw)
        case.touch_seq += 1
        admin_delay = int(self._rng(case, "admin_delay").integers(*ADMIN_ACTION_DELAY_DAYS))
        self._advance_clock(case, admin_delay)
        case.correspondence_extra = False
        payload = payload or {}
        if payload:
            reason = payer_receives(action_raw, payload, self._payer_facts(case),
                                    self.reader, records)
            if reason is not None:
                self._reject_paperwork(case, reason)
                return case

        if action_raw in ("submit_clean", "submit_with_records", "correct_and_resubmit"):
            if case.collected_payer > 0 and self._payer_remaining(case) <= 0.005:
                # v0.4: a real payer never pays twice - duplicate-claim denial
                case.resubmitted_after_paid = True
                case.carc = "CO-18"
                case.carc_history.append("CO-18")
                case.correspondence_kind = "denial"
                self._advance_clock(
                    case, int(self._rng(case, "dup_delay").integers(5, 15))
                )
                if not case.terminal and len(case.action_history) >= MAX_DECISIONS:
                    case.action_history.append("write_off")
                    self._finish(case, "written_off_at_cap")
                return case
            corrected = action_raw == "correct_and_resubmit"
            if action_raw == "submit_clean" and case.submitted_once and case.carc:
                case.resubmitted_unchanged = True
            with_records = (
                "clinical_notes" in payload.get("attachments", [])
                if payload and "attachments" in payload
                else action_raw != "submit_clean"
            )
            result = self.engine.adjudicate(self._claim(case, with_records, corrected))
            if corrected:
                case.wrong_cob_start = False
            first_submission = not case.submitted_once
            case.submitted_once = True
            if first_submission:
                case.days_elapsed = 0  # the 120-day clock starts now
            self._advance_clock(case, result.delay_days)
            self._absorb_adjudication(case, result)
        elif action_raw == "request_retro_auth":
            case.retro_requested = True
            granted = self.engine.retro_auth_granted(
                case.payer_id, case.days_since_service, case.episode_id
            )
            self._advance_clock(
                case, int(self._rng(case, "retro_delay").integers(*RETRO_DECISION_DELAY_DAYS))
            )
            if granted:
                case.auth_ref = "AUTH-RETRO"
                case.correspondence_kind = "auth_decision_granted"
                if case.carc == "CO-197":
                    case.carc = None  # cleared: resubmission will now pass the auth check
            else:
                case.correspondence_kind = "auth_decision_denied"
        elif action_raw == "appeal_with_necessity":
            case.appealed = True
            strong = case.clinic["doc_quality"] >= 0.6
            granted = self.engine.appeal_granted(case.payer_id, strong, case.episode_id)
            payer = self.engine.payers[case.payer_id]
            self._advance_clock(
                case,
                int(self._rng(case, "appeal_delay").integers(*payer.appeal_delay_days)),
            )
            self._resolve_overturn(case, granted, "appeal_granted", "appeal_denied")
        elif action_raw == "request_peer_to_peer":
            granted = self.engine.p2p_reversed(case.payer_id, case.episode_id)
            self._advance_clock(case, int(self._rng(case, "p2p_delay").integers(5, 15)))
            self._resolve_overturn(case, granted, "appeal_granted", "appeal_denied")
        elif action_raw == "provide_requested_info":
            resolved = self.engine.info_request_resolved(case.payer_id, case.episode_id)
            self._advance_clock(case, int(self._rng(case, "info_delay").integers(7, 20)))
            if resolved:
                result = self.engine.adjudicate(self._claim(case, True, False))
                if result.carc == "CO-16":  # the reviewer draw would repeat; info stands
                    result.carc = None
                    self._pay_case(case)
                else:
                    self._absorb_adjudication(case, result)
            else:
                case.correspondence_kind = "denial"
                case.carc = "CO-16"
                case.carc_history.append("CO-16")
        elif action_raw == "bill_secondary":
            # simplified: the secondary payer covers the pending patient share
            self._advance_clock(case, int(self._rng(case, "sec_delay").integers(15, 35)))
            case.collected_payer += case.patient_share_pending
            case.patient_share_pending = 0.0
            case.correspondence_kind = "eob_paid"
            self._finish(case, "paid_with_secondary")
        elif action_raw == "bill_patient":
            rng = self._rng(case, "patient_pay")
            self._advance_clock(case, int(rng.integers(*PATIENT_PAY_DELAY_DAYS)))
            amount = case.patient_share_pending or case.allowed
            pay_prob = 0.7 if amount <= 300 else 0.4
            if rng.random() < pay_prob:
                case.collected_patient += amount
            case.patient_share_pending = 0.0
            case.correspondence_kind = None
            self._finish(case, "patient_billed")
        elif action_raw == "write_off":
            case.correspondence_kind = None
            self._finish(case, "written_off")
        else:
            raise ValueError(f"unknown action {action_raw!r}")

        if not case.terminal and len(case.action_history) >= MAX_DECISIONS:
            case.action_history.append("write_off")
            self._finish(case, "written_off_at_cap")
        return case

    def _payer_remaining(self, case: Case) -> float:
        """v0.4: what the payer can still legitimately pay on this claim."""
        share = patient_share(
            case.allowed,
            case.coverage["deductible_remaining"],
            case.coverage["coinsurance_pct"],
        )
        return round(case.allowed - share - case.collected_payer, 2)

    def _absorb_adjudication(self, case: Case, result) -> None:
        if result.carc is not None:
            case.carc = result.carc
            case.carc_history.append(result.carc)
            case.correspondence_kind = "denial"
            if result.carc == "CO-16":
                extra_rng = self._rng(case, "co16_hint")
                case.correspondence_extra = extra_rng.random() < 0.2
            return
        case.carc = None
        # v0.4: payments top up to the payer's obligation, never beyond it
        obligation = round(case.allowed - result.patient_responsibility, 2)
        case.collected_payer += max(
            0.0, min(result.paid_amount, round(obligation - case.collected_payer, 2))
        )
        case.patient_share_pending = result.patient_responsibility
        if result.paid_amount > 0:
            case.contractual_share = result.patient_responsibility
        if result.status == "paid":
            case.correspondence_kind = "eob_paid"
            self._finish(case, "paid")
        else:
            case.correspondence_kind = "eob_underpaid"

    def _resolve_overturn(self, case: Case, granted: bool, kind_yes: str, kind_no: str) -> None:
        if granted:
            share = patient_share(
                case.allowed,
                case.coverage["deductible_remaining"],
                case.coverage["coinsurance_pct"],
            )
            pay = max(0.0, self._payer_remaining(case))  # v0.4: top up only
            case.collected_payer += pay
            case.patient_share_pending = share
            case.contractual_share = share
            case.carc = None
            # v0.4: when reprocessing moves no plan money (all deductible),
            # the letter must say so instead of promising payment
            case.correspondence_kind = (
                "appeal_granted_deductible" if pay <= 0.005 else kind_yes
            )
            if case.collected_payer >= SUCCESS_FRACTION * case.allowed:
                self._finish(case, "paid_on_appeal")
        else:
            case.correspondence_kind = kind_no

    def _pay_case(self, case: Case) -> None:
        share = patient_share(
            case.allowed,
            case.coverage["deductible_remaining"],
            case.coverage["coinsurance_pct"],
        )
        case.collected_payer += max(0.0, self._payer_remaining(case))  # v0.4
        case.patient_share_pending = share
        case.contractual_share = share
        case.carc = None
        if case.collected_payer >= SUCCESS_FRACTION * case.allowed:
            case.correspondence_kind = "eob_paid"
            self._finish(case, "paid")
        else:
            case.correspondence_kind = "eob_underpaid"

    def _finish(self, case: Case, resolution: str) -> None:
        # an underpaid balance still pending is abandoned value, not collected
        case.terminal = True
        case.resolution = resolution


# -- recording ------------------------------------------------------------


def state_tabular(case: Case, payer, days_to_payment_poison: float) -> dict[str, Any]:
    """The admin-visible tabular channel for the CURRENT state, plus the one
    deliberately poisoned post-hoc column (days_to_payment, P2 trap)."""
    dx = list(case.patient["diagnoses"])
    return {
        "payer_id": case.payer_id,
        "payer_archetype": payer.archetype,
        "auth_required": case.study["cpt"] in payer.auth_required_cpts,
        "retro_window_days": payer.retro_auth_window_days,
        "timely_filing_days": payer.timely_filing_days,
        "p2p_available": payer.p2p_available,
        "cpt": case.study["cpt"],
        "dx_codes": ",".join(dx),
        "pairing_valid": bool(set(dx) & COVERED_DX[case.study["cpt"]]),
        "days_since_service": case.days_since_service,
        "days_to_filing_deadline": payer.timely_filing_days - case.days_since_service,
        "auth_status": "on_file" if case.auth_ref else "missing",
        "eligibility_status": "lapsed" if case.lapsed else "active",
        "aob_on_file": case.coverage["aob_on_file"],
        "subscriber_relationship": case.coverage["subscriber_relationship"],
        "cob_position_ok": not case.wrong_cob_start,
        "has_secondary": case.coverage["secondary_payer_id"] is not None,
        "carc": case.carc or "",
        "balance": round(case.allowed - case.collected_payer - case.collected_patient, 2),
        "allowed_amount": case.allowed,
        "deductible_remaining": case.coverage["deductible_remaining"],
        "coinsurance_pct": case.coverage["coinsurance_pct"],
        "touches_so_far": len(case.action_history),
        "clinic_doc_quality": case.clinic["doc_quality"],
        "persona_id": case.persona,
        "prior_actions": ",".join(case.action_history),
        "prior_carcs": ",".join(case.carc_history),
        "cumulative_delay_days": case.days_elapsed,
        "days_to_payment": days_to_payment_poison,  # POISONED: post-hoc, planted for P2
    }


def state_text(
    case: Case, payer, text: CachedTextGenerator, master_seed: int
) -> dict[str, str]:
    indication = text.generate(
        indication_prompt(
            list(case.patient["diagnoses"]),
            case.study["cpt"],
            salt_token(f"{master_seed}|ind|{case.episode_id}"),
        )
    )
    correspondence = ""
    if case.correspondence_kind is not None:
        # v0.4: outcome letters carry the adjudication money so the model
        # cannot promise payment that is not coming
        money = None
        if case.correspondence_kind in (
            "eob_paid", "eob_underpaid", "appeal_granted",
            "appeal_granted_deductible",
        ):
            money = {
                "allowed": round(case.allowed, 2),
                "payer_paid_total": round(case.collected_payer, 2),
                "member_responsibility": round(case.patient_share_pending, 2),
            }
        correspondence = text.generate(
            correspondence_prompt(
                case.correspondence_kind
                if case.correspondence_kind != "denial"
                else "denial",
                payer.name,
                case.study["cpt"],
                case.carc if case.carc else (case.carc_history[-1] if case.carc_history else None),
                salt_token(f"{master_seed}|cor|{case.episode_id}|{case.touch_seq}"),
                names_missing_document=case.correspondence_extra,
                money=money,
            )
        )
    return {
        "clinical_indication_text": indication,
        "payer_correspondence_text": correspondence,
    }


MISTAKE_PENALTY = 0.10  # per process mistake, off the episode score (spec v0.2 s.1)
_SUBMITS = ("submit_clean", "submit_with_records")


def compute_mistakes(pending: list, case: Case, success: bool) -> dict[str, bool]:
    """The process-mistake ledger (change spec section 2): derived purely
    from decision-time snapshots and published payer rules. POST-HOC ONLY -
    these fields must never become features (descriptor posthoc_fields)."""

    def retro_door_open(snap: dict) -> bool:
        return (
            snap["auth_required"]
            and snap["auth_status"] != "on_file"
            and snap["days_since_service"] <= snap["retro_window_days"]
        )

    return {
        # v0.4: re-submitting a claim whose payer share is fully collected
        "mistake_resubmitted_paid_claim": case.resubmitted_after_paid,
        "mistake_submitted_into_missing_auth": any(
            action in _SUBMITS and retro_door_open(snap) for snap, action in pending
        ),
        # appeal carries no expiry in this world; the expiring path is retro
        "mistake_window_expired_unused": (
            not success
            and not case.retro_requested
            and any(retro_door_open(snap) for snap, _ in pending)
        ),
        "mistake_wrong_cob_order_submitted": any(
            action in _SUBMITS and not snap["cob_position_ok"]
            for snap, action in pending
        ),
        "mistake_resubmitted_unchanged": case.resubmitted_unchanged,
        # artifact-level fabrication is checked (and zero-tolerated) by the
        # harness; the simulator itself produces no artifacts
        "mistake_fabrication": False,
    }


@dataclass
class TrancheResult:
    examples: list[Example]
    states: list[RawState]
    episodes_rows: list[dict]
    states_rows: list[dict]

    def content_sha256(self) -> str:
        return sha256_json(
            {
                "episodes": self.episodes_rows,
                "states": [
                    {k: v for k, v in row.items()} for row in self.states_rows
                ],
            }
        )


def generate_tranche(
    world: World,
    engine: PayerEngine,
    master_seed: int,
    text: CachedTextGenerator,
    start_index: int,
    n_episodes: int,
) -> TrancheResult:
    """Run the simulation for episodes [start_index, start_index + n)."""
    env = CardessaEnvironment(world, engine, master_seed)
    examples: list[Example] = []
    raw_states: list[RawState] = []
    episodes_rows: list[dict] = []
    states_rows: list[dict] = []

    for index in range(start_index, start_index + n_episodes):
        case = env.reset(index)
        payer = engine.payers[case.payer_id]
        pending: list[tuple[dict, str]] = []  # (tabular+text snapshot, action)

        while not case.terminal:
            view = CaseView(
                payer=payer,
                touch_seq=case.touch_seq,
                submitted_once=case.submitted_once,
                carc=case.carc,
                underpaid=(
                    case.patient_share_pending > 0
                    and case.carc is None
                    and case.collected_payer > 0
                ),
                auth_required=case.study["cpt"] in payer.auth_required_cpts,
                auth_on_file=case.auth_ref is not None,
                days_since_service=case.days_since_service,
                balance=case.allowed - case.collected_payer - case.collected_patient,
                has_secondary=case.coverage["secondary_payer_id"] is not None,
                doc_quality=case.clinic["doc_quality"],
                resubmitted_unchanged_already=case.resubmitted_unchanged,
                appealed_already=case.appealed,
                retro_requested_already=case.retro_requested,
            )
            action = choose_action(case.persona, view, master_seed, case.episode_id)
            snapshot = state_tabular(case, payer, 0.0)  # poison filled in after terminal
            snapshot.update(state_text(case, payer, text, master_seed))
            pending.append((snapshot, action))
            case = env.apply(case, action)

        outcome = env.outcome(case)
        assert outcome is not None
        mistakes = compute_mistakes(pending, case, outcome.success)
        n_mistakes = sum(mistakes.values())
        outcome = Outcome(
            success=outcome.success,
            score=round(max(0.0, outcome.score - MISTAKE_PENALTY * n_mistakes), 4),
        )
        days_to_payment = float(case.days_elapsed if outcome.success else 999.0)

        state_ids = []
        for seq, (snapshot, action) in enumerate(pending):
            state_id = f"{case.episode_id}-s{seq}"
            state_ids.append(state_id)
            snapshot["days_to_payment"] = days_to_payment  # the planted poison value
            row = {
                "episode_id": case.episode_id,
                "state_id": state_id,
                "touch_seq": seq,
                "action_raw": action,
                **snapshot,
            }
            states_rows.append(row)
            raw_states.append(
                RawState(
                    state_id=state_id,
                    example_id=case.episode_id,
                    seq=seq,
                    data=RawData(
                        modality="composite",
                        inline=dict(snapshot),
                        sha256=sha256_json(
                            {k: snapshot[k] for k in sorted(snapshot)}
                        ),
                    ),
                    action_raw=action,
                    next_state_id=f"{case.episode_id}-s{seq + 1}"
                    if seq + 1 < len(pending)
                    else None,
                )
            )
        examples.append(
            Example(
                example_id=case.episode_id,
                state_ids=state_ids,
                outcome=outcome,
                meta={
                    "patient_id": case.patient["patient_id"],
                    "payer_id": case.payer_id,
                    "payer_archetype": payer.archetype,
                    "persona_id": case.persona,
                    "family_id": case.study["family"],
                    "clinic_id": case.clinic["clinic_id"],
                    "split": case.patient["split"],
                    # post-hoc quarantine: never features
                    "paid_amount_final": round(
                        case.collected_payer + case.collected_patient, 2
                    ),
                    "days_to_payment": days_to_payment,
                    "total_touches": len(case.action_history),
                    "engine_truth_allowed": case.allowed,
                    "resolution": case.resolution,
                    "n_process_mistakes": n_mistakes,
                    "paperwork_rejections": case.paperwork_rejections,
                    **mistakes,
                },
            )
        )
        episodes_rows.append(
            {
                "episode_id": case.episode_id,
                "patient_id": case.patient["patient_id"],
                "payer_id": case.payer_id,
                "persona_id": case.persona,
                "family_id": case.study["family"],
                "clinic_id": case.clinic["clinic_id"],
                "split": case.patient["split"],
                "success": outcome.success,
                "outcome_score": outcome.score,
                "n_states": len(state_ids),
                "resolution": case.resolution,
                "paid_amount_final": round(
                    case.collected_payer + case.collected_patient, 2
                ),
                "days_to_payment": days_to_payment,
                "total_touches": len(case.action_history),
                "engine_truth_allowed": case.allowed,
                "n_process_mistakes": n_mistakes,
                "paperwork_rejections": case.paperwork_rejections,
                **mistakes,
            }
        )
    return TrancheResult(examples, raw_states, episodes_rows, states_rows)


def write_tranche(result: TrancheResult, out_dir) -> dict[str, str]:
    from pathlib import Path

    from ammonix_core.hashing import sha256_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(result.episodes_rows).write_parquet(out / "episodes.parquet")
    pl.DataFrame(result.states_rows).write_parquet(out / "states.parquet")
    return {
        "episodes.parquet": sha256_file(out / "episodes.parquet"),
        "states.parquet": sha256_file(out / "states.parquet"),
    }
