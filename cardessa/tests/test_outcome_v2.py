"""Outcome v2 + the process-mistake ledger (change spec sections 1-2)."""

from cardessa.corpus import MISTAKE_PENALTY, Case, compute_mistakes


def make_case(**overrides) -> Case:
    case = Case(
        episode_id="ep-t", episode_index=0, persona="diligent",
        patient={}, coverage={}, study={}, clinic={}, payer_id="meridian",
    )
    case.allowed = 800.0
    case.terminal = True
    for key, value in overrides.items():
        setattr(case, key, value)
    return case


def snap(**overrides) -> dict:
    base = {
        "auth_required": False, "auth_status": "on_file",
        "days_since_service": 5, "retro_window_days": 30,
        "cob_position_ok": True,
    }
    base.update(overrides)
    return base


class FakeEnv:
    from cardessa.corpus import CardessaEnvironment
    outcome = CardessaEnvironment.outcome


def outcome_of(case):
    return FakeEnv.outcome(FakeEnv, case)


def test_payer_paid_with_contractual_share_is_success():
    case = make_case(
        collected_payer=640.0, collected_patient=160.0,
        contractual_share=160.0, days_elapsed=60,
        action_history=["submit_clean", "bill_patient"],
    )
    outcome = outcome_of(case)
    assert outcome.success  # 800/800, patient paid only their share


def test_balance_billing_a_failed_claim_is_failure():
    # payer never paid; patient paid the FULL balance: v0.1 called this a
    # success, v0.2 must not
    case = make_case(
        collected_payer=0.0, collected_patient=800.0,
        contractual_share=0.0, days_elapsed=60,
        action_history=["submit_clean", "appeal_with_necessity", "bill_patient"],
    )
    outcome = outcome_of(case)
    assert not outcome.success
    assert outcome.score == 0.0  # nothing validly collected


def test_excess_patient_billing_counts_only_the_share():
    case = make_case(
        collected_payer=500.0, collected_patient=300.0,
        contractual_share=100.0, days_elapsed=60,
        action_history=["submit_clean"],
    )
    outcome = outcome_of(case)
    assert not outcome.success  # 600/800 = 0.75 < 0.9


def test_mistake_submitted_into_missing_auth():
    pending = [(snap(auth_required=True, auth_status="missing"), "submit_clean")]
    mistakes = compute_mistakes(pending, make_case(), success=False)
    assert mistakes["mistake_submitted_into_missing_auth"]
    # retro window closed: not this mistake (the door was already shut)
    pending = [(snap(auth_required=True, auth_status="missing",
                     days_since_service=40), "submit_clean")]
    mistakes = compute_mistakes(pending, make_case(), success=False)
    assert not mistakes["mistake_submitted_into_missing_auth"]


def test_mistake_window_expired_unused_only_on_failures():
    pending = [(snap(auth_required=True, auth_status="missing"), "appeal_with_necessity")]
    failed = compute_mistakes(pending, make_case(retro_requested=False), success=False)
    assert failed["mistake_window_expired_unused"]
    succeeded = compute_mistakes(pending, make_case(retro_requested=False), success=True)
    assert not succeeded["mistake_window_expired_unused"]
    tried = compute_mistakes(pending, make_case(retro_requested=True), success=False)
    assert not tried["mistake_window_expired_unused"]


def test_mistake_wrong_cob_and_resubmit_flags():
    pending = [(snap(cob_position_ok=False), "submit_clean")]
    mistakes = compute_mistakes(pending, make_case(resubmitted_unchanged=True), success=False)
    assert mistakes["mistake_wrong_cob_order_submitted"]
    assert mistakes["mistake_resubmitted_unchanged"]
    assert not mistakes["mistake_fabrication"]


def test_mistake_penalty_constant():
    assert MISTAKE_PENALTY == 0.10
