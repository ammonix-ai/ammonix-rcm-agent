"""Round-1 tokenizer: determinism, no leakage, correct encodings, episodic chaining."""

import fnmatch

import polars as pl
import pytest

from cardessa.features import (
    POSTHOC_FORBIDDEN,
    build_round1_features,
    to_qpsi_records,
)

POSTHOC_PATTERNS = (*POSTHOC_FORBIDDEN, "engine_truth_*")
TEXT_COLUMNS = ("payer_correspondence_text", "clinical_indication_text")


def tiny_states() -> pl.DataFrame:
    rows = [
        {
            "state_id": "e1-0", "episode_id": "e1", "touch_seq": 0,
            "action_raw": "submit_clean", "payer_id": "meridian",
            "payer_archetype": "strict_auth", "auth_required": True,
            "retro_window_days": 10, "timely_filing_days": 90,
            "p2p_available": True, "cpt": "93229", "dx_codes": "I48.0,R55",
            "pairing_valid": True, "days_since_service": 3,
            "days_to_filing_deadline": 87, "auth_status": "none",
            "eligibility_status": "active", "aob_on_file": True,
            "subscriber_relationship": "self", "cob_position_ok": True,
            "has_secondary": False, "carc": None, "balance": 700.0,
            "allowed_amount": None, "deductible_remaining": 250.0,
            "coinsurance_pct": 0.2, "touches_so_far": 0,
            "clinic_doc_quality": 0.8, "persona_id": "diligent",
            "prior_actions": "", "prior_carcs": "",
            "cumulative_delay_days": 0, "days_to_payment": 21.0,
            "clinical_indication_text": "x", "payer_correspondence_text": "y",
        },
        {
            "state_id": "e1-1", "episode_id": "e1", "touch_seq": 1,
            "action_raw": "appeal_with_necessity", "payer_id": "meridian",
            "payer_archetype": "strict_auth", "auth_required": True,
            "retro_window_days": 10, "timely_filing_days": 90,
            "p2p_available": True, "cpt": "93229", "dx_codes": "I48.0,R55",
            "pairing_valid": True, "days_since_service": 24,
            "days_to_filing_deadline": 66, "auth_status": "denied",
            "eligibility_status": "active", "aob_on_file": True,
            "subscriber_relationship": "self", "cob_position_ok": True,
            "has_secondary": False, "carc": "CO-197", "balance": 700.0,
            "allowed_amount": None, "deductible_remaining": 250.0,
            "coinsurance_pct": 0.2, "touches_so_far": 1,
            "clinic_doc_quality": 0.8, "persona_id": "diligent",
            "prior_actions": "submit_clean", "prior_carcs": "CO-197",
            "cumulative_delay_days": 21, "days_to_payment": 42.0,
            "clinical_indication_text": "x", "payer_correspondence_text": "y",
        },
        {
            "state_id": "e2-0", "episode_id": "e2", "touch_seq": 0,
            "action_raw": "submit_clean", "payer_id": "silverbridge",
            "payer_archetype": "lenient", "auth_required": False,
            "retro_window_days": 0, "timely_filing_days": 180,
            "p2p_available": False, "cpt": "93226", "dx_codes": "R42",
            "pairing_valid": True, "days_since_service": 5,
            "days_to_filing_deadline": 175, "auth_status": "not_required",
            "eligibility_status": "active", "aob_on_file": False,
            "subscriber_relationship": "spouse", "cob_position_ok": True,
            "has_secondary": True, "carc": None, "balance": 120.0,
            "allowed_amount": 95.0, "deductible_remaining": 0.0,
            "coinsurance_pct": 0.1, "touches_so_far": 0,
            "clinic_doc_quality": 0.55, "persona_id": "hasty",
            "prior_actions": "", "prior_carcs": "",
            "cumulative_delay_days": 0, "days_to_payment": 14.0,
            "clinical_indication_text": "x", "payer_correspondence_text": "y",
        },
    ]
    return pl.DataFrame(rows)


def test_double_build_is_identical():
    states = tiny_states()
    a, specs_a = build_round1_features(states)
    b, specs_b = build_round1_features(states)
    assert a.equals(b)
    assert [s.name for s in specs_a] == [s.name for s in specs_b]


def test_no_posthoc_or_text_feature():
    _, specs = build_round1_features(tiny_states())
    names = [s.name for s in specs]
    leaked = [
        n for n in names
        for pattern in POSTHOC_PATTERNS + TEXT_COLUMNS
        if fnmatch.fnmatch(n, pattern)
    ]
    assert leaked == []


def test_specs_and_columns_agree():
    frame, specs = build_round1_features(tiny_states())
    names = [s.name for s in specs]
    assert len(set(names)) == len(names)
    assert set(names) <= set(frame.columns)
    assert all(s.description.strip() for s in specs)
    feature_frame = frame.select(names)
    assert all(dtype == pl.Float64 for dtype in feature_frame.dtypes)
    assert feature_frame.null_count().sum_horizontal().item() == 0


def test_onehots_and_flags_encode_correctly():
    frame, _ = build_round1_features(tiny_states())
    by_id = {r["state_id"]: r for r in frame.to_dicts()}
    first, denied, other = by_id["e1-0"], by_id["e1-1"], by_id["e2-0"]
    assert first["carc=none"] == 1.0 and first["carc=CO-197"] == 0.0
    assert denied["carc=CO-197"] == 1.0 and denied["carc=none"] == 0.0
    assert first["cpt=93229"] == 1.0 and other["cpt=93226"] == 1.0
    assert first["dx_has_I48_0"] == 1.0 and first["dx_has_R42"] == 0.0
    assert other["dx_has_R42"] == 1.0
    assert first["n_dx_codes"] == 2.0 and other["n_dx_codes"] == 1.0
    assert other["persona_id=hasty"] == 1.0 and first["persona_id=diligent"] == 1.0
    # missing allowed_amount filled with the -1 sentinel
    assert first["allowed_amount"] == -1.0 and other["allowed_amount"] == 95.0


def test_history_features_count_prior_events():
    frame, specs = build_round1_features(tiny_states())
    by_id = {r["state_id"]: r for r in frame.to_dicts()}
    assert by_id["e1-0"]["prior_action_count_submit_clean"] == 0.0
    assert by_id["e1-1"]["prior_action_count_submit_clean"] == 1.0
    assert by_id["e1-1"]["prior_carc_seen_CO_197"] == 1.0
    assert by_id["e2-0"]["prior_carc_seen_CO_197"] == 0.0
    history_specs = [s for s in specs if s.history is not None]
    assert any(s.name.startswith("prior_action_count_") for s in history_specs)
    assert any(s.name == "cumulative_delay_days" for s in history_specs)


def test_qpsi_records_chain_within_episode():
    frame, specs = build_round1_features(tiny_states())
    records = to_qpsi_records(
        frame, specs, {"e1": (True, 700.0), "e2": (False, 0.0)}
    )
    by_id = {r.state_id: r for r in records}
    assert by_id["e1-0"].next_state_id == "e1-1"
    assert by_id["e1-1"].next_state_id is None  # episode boundary, not e2-0
    assert by_id["e2-0"].next_state_id is None
    assert by_id["e1-0"].outcome_success is True
    assert by_id["e2-0"].outcome_success is False
    assert by_id["e1-1"].action_id == "appeal_with_necessity"


def test_unknown_episode_outcome_fails_loudly():
    frame, specs = build_round1_features(tiny_states())
    with pytest.raises(KeyError):
        to_qpsi_records(frame, specs, {"e1": (True, 1.0)})
