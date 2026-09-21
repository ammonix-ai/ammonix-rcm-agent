"""v0.4 spec s.3: expected value beside p(success). The formulas must encode
the demo lessons: ep-05001 (40% of $738 beats certain zero), ep-05004
(a standing denial makes secondary value zero)."""

from cardessa.harness import close_out_choice, expected_values


def _row(balance, allowed=None, deductible=1000.0, coins=0.0, carc=None):
    return {
        "balance": balance,
        "allowed_amount": allowed if allowed is not None else balance,
        "deductible_remaining": deductible,
        "coinsurance_pct": coins,
        "carc": carc,
    }


def test_large_patient_bill_beats_certain_zero():
    # ep-05001: $738.30 all deductible; billing at 40% is worth ~$295
    scores = {"bill_patient": 0.01, "write_off": 0.0}
    ev = expected_values(_row(738.30), scores)
    assert ev["write_off"] == 0.0
    assert abs(ev["bill_patient"] - 0.4 * 738.30) < 0.01
    assert close_out_choice(ev, scores) == "bill_patient"


def test_small_bill_uses_the_high_pay_rate():
    scores = {"bill_patient": 0.5}
    ev = expected_values(_row(200.0), scores)
    assert abs(ev["bill_patient"] - 0.7 * 200.0) < 0.01


def test_standing_denial_makes_secondary_worthless():
    # ep-05004: billing the secondary into a denial collects nothing
    scores = {"bill_secondary": 0.99, "bill_patient": 0.01, "write_off": 0.0}
    denied = expected_values(_row(701.94, carc="CO-197"), scores)
    assert denied["bill_secondary"] == 0.0
    cleared = expected_values(_row(701.94, carc=None), scores)
    assert cleared["bill_secondary"] > 690.0
    assert close_out_choice(cleared, scores) == "bill_secondary"


def test_all_worthless_still_escalates():
    scores = {"write_off": 0.0}
    ev = expected_values(_row(500.0), scores)
    assert close_out_choice(ev, scores) is None


def test_patient_dollars_capped_at_contractual_share():
    # deductible only $100 of a $500 balance: only $100 is the patient's
    scores = {"bill_patient": 0.2}
    ev = expected_values(_row(500.0, deductible=100.0), scores)
    assert abs(ev["bill_patient"] - 0.7 * 100.0) < 0.01


def test_payer_actions_scale_with_outstanding_balance():
    scores = {"submit_clean": 0.5}
    ev = expected_values(_row(400.0), scores)
    assert abs(ev["submit_clean"] - 200.0) < 0.01
