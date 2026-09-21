"""v0.4 spec s.1/s.2 engine properties: the payer never pays twice, letters
know the money. Runs on the generated world (no LLM)."""

import pytest

from cardessa import MASTER_SEED
from cardessa.corpus import CardessaEnvironment, compute_mistakes
from cardessa.engine import PayerEngine, patient_share
from cardessa.textgen import correspondence_prompt
from cardessa.world import generate_world


@pytest.fixture(scope="module")
def env():
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    return CardessaEnvironment(world, engine, MASTER_SEED)


def _payer_obligation(case) -> float:
    share = patient_share(
        case.allowed,
        case.coverage["deductible_remaining"],
        case.coverage["coinsurance_pct"],
    )
    return round(case.allowed - share, 2)


def _drive_to_paid(env, start=0, tries=400):
    """Find an episode where a submit path yields a paying adjudication."""
    for index in range(start, start + tries):
        case = env.reset(index)
        case = env.apply(case, "submit_with_records")
        if case.collected_payer > 0 and not case.terminal:
            return case
    pytest.skip("no non-terminal paying adjudication found in scan range")


def test_no_duplicate_payer_payment(env):
    case = _drive_to_paid(env)
    paid_before = case.collected_payer
    case = env.apply(case, "submit_clean")
    assert case.collected_payer == paid_before, "payer paid twice"
    assert case.carc == "CO-18"
    assert case.resubmitted_after_paid is True
    assert case.correspondence_kind == "denial"


def test_duplicate_flag_reaches_the_ledger(env):
    case = _drive_to_paid(env)
    case = env.apply(case, "submit_clean")
    if not case.terminal:
        case = env.apply(case, "write_off")
    mistakes = compute_mistakes([], case, False)
    assert mistakes["mistake_resubmitted_paid_claim"] is True


def test_payer_never_exceeds_obligation_anywhere(env):
    """Property over many episodes and greedy submit/appeal sequences."""
    for index in range(0, 120):
        case = env.reset(index)
        for action in ("submit_with_records", "appeal_with_necessity",
                       "submit_clean", "request_peer_to_peer", "submit_clean"):
            if case.terminal:
                break
            case = env.apply(case, action)
        assert case.collected_payer <= _payer_obligation(case) + 0.01, (
            f"episode {index}: payer paid {case.collected_payer} "
            f"vs obligation {_payer_obligation(case)}"
        )


def test_overturn_with_zero_plan_money_gets_deductible_kind(env):
    """When the whole allowed amount is deductible, an appeal win must carry
    the no-payment letter kind, never the payment promise."""
    found = False
    for index in range(0, 400):
        case = env.reset(index)
        share = patient_share(
            case.allowed,
            case.coverage["deductible_remaining"],
            case.coverage["coinsurance_pct"],
        )
        if round(case.allowed - share, 2) > 0.005:
            continue  # payer owes something; not the population under test
        case = env.apply(case, "submit_clean")
        if case.terminal or not case.carc:
            continue
        case = env.apply(case, "appeal_with_necessity")
        if case.carc is None and not case.terminal:
            assert case.correspondence_kind == "appeal_granted_deductible"
            assert case.collected_payer == 0.0
            found = True
            break
    if not found:
        pytest.skip("no zero-obligation appeal-win found in scan range")


def test_deductible_letter_prompt_forbids_payment_promise():
    prompt = correspondence_prompt(
        "appeal_granted_deductible", "Meridian Health", "93229", None, "salt",
        money={"allowed": 738.30, "payer_paid_total": 0.0,
               "member_responsibility": 738.30},
    )
    assert "NO payment is due" in prompt
    assert "738.30" in prompt
    assert "will be reprocessed for payment" not in prompt


def test_money_facts_reach_outcome_prompts():
    prompt = correspondence_prompt(
        "eob_underpaid", "Meridian Health", "93229", None, "salt",
        money={"allowed": 800.0, "payer_paid_total": 361.6,
               "member_responsibility": 438.4},
    )
    assert "361.60" in prompt and "438.40" in prompt
    # without money the v0.3 wording is preserved (old caches stay valid)
    legacy = correspondence_prompt(
        "eob_underpaid", "Meridian Health", "93229", None, "salt"
    )
    assert "exact amounts" not in legacy
