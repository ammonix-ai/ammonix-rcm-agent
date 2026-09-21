"""Persona policies: the planted habits are real and payer-conditional."""

from cardessa import MASTER_SEED
from cardessa.payers import payer_index
from cardessa.personas import CaseView, choose_action

PAYERS = payer_index()


def view(payer_id: str, **overrides) -> CaseView:
    base = dict(
        payer=PAYERS[payer_id],
        touch_seq=1,
        submitted_once=True,
        carc="CO-197",
        underpaid=False,
        auth_required=True,
        auth_on_file=False,
        days_since_service=15,
        balance=700.0,
        has_secondary=False,
        doc_quality=0.8,
        resubmitted_unchanged_already=True,
        appealed_already=False,
        retro_requested_already=False,
    )
    base.update(overrides)
    return CaseView(**base)


def test_diligent_is_payer_conditional_on_co197():
    """Aggregated over episodes: diligent has a designed 5% error rate, so a
    single draw can differ; the policy itself is payer-conditional."""
    n = 100
    meridian = [
        choose_action("diligent", view("meridian"), MASTER_SEED, f"ep-{i}")
        for i in range(n)
    ]
    silverbridge = [
        choose_action("diligent", view("silverbridge"), MASTER_SEED, f"ep-{i}")
        for i in range(n)
    ]
    assert meridian.count("request_retro_auth") >= 0.85 * n  # window open at Meridian
    assert silverbridge.count("appeal_with_necessity") >= 0.85 * n  # retro never there
    assert "request_retro_auth" not in silverbridge


def test_hasty_resubmits_unchanged_once_then_appeals():
    first = choose_action(
        "hasty", view("meridian", resubmitted_unchanged_already=False), MASTER_SEED, "ep-x"
    )
    second = choose_action("hasty", view("meridian"), MASTER_SEED, "ep-x")
    assert first == "submit_clean"
    assert second == "appeal_with_necessity"  # the planted wrong habit at Meridian


def test_conservative_writes_off_big_auth_problems():
    action = choose_action("conservative", view("meridian"), MASTER_SEED, "ep-x")
    assert action == "write_off"  # the planted wrong habit
    small = choose_action(
        "conservative", view("meridian", balance=120.0), MASTER_SEED, "ep-x"
    )
    assert small == "bill_patient"


def test_majority_of_family_actions_is_wrong_at_meridian():
    """Hasty (0.35) + Conservative (0.25) beat Diligent (0.40): imitation loses.
    Checked in aggregate because diligent carries a 5% error rate."""
    n = 100
    diligent = [
        choose_action("diligent", view("meridian"), MASTER_SEED, f"ep-{i}") for i in range(n)
    ]
    assert diligent.count("request_retro_auth") >= 0.85 * n
    for i in range(20):
        assert (
            choose_action("hasty", view("meridian"), MASTER_SEED, f"ep-{i}")
            == "appeal_with_necessity"
        )
        assert (
            choose_action("conservative", view("meridian"), MASTER_SEED, f"ep-{i}")
            == "write_off"
        )


def test_co16_gets_information_response():
    for persona in ("diligent", "conservative"):
        action = choose_action(persona, view("meridian", carc="CO-16"), MASTER_SEED, "ep-x")
        assert action == "provide_requested_info", persona


def test_underpaid_routes_by_secondary_coverage():
    underpaid = view("meridian", carc=None, underpaid=True)
    assert choose_action("diligent", underpaid, MASTER_SEED, "ep-x") == "bill_patient"
    with_secondary = view("meridian", carc=None, underpaid=True, has_secondary=True)
    assert choose_action("diligent", with_secondary, MASTER_SEED, "ep-x") == "bill_secondary"


def test_exhausted_appeal_closes_out():
    done = view("meridian", appealed_already=True, carc="CO-50")
    assert choose_action("hasty", done, MASTER_SEED, "ep-x") == "write_off"
    assert choose_action("diligent", done, MASTER_SEED, "ep-x") == "bill_patient"
