"""Form tools: provenance completeness, gap handling, zero invention. Hermetic."""

from cardessa.forms import (
    FIELD_MAPS,
    compose_appeal_letter,
    fill_cms1500,
    fill_prior_auth_request,
    pre_validate,
)


def make_bundle(**state_overrides):
    state = {
        "state_id": "ep-03000-s0", "episode_id": "ep-03000", "cpt": "93229",
        "dx_codes": "I48.0,R55", "carc": "CO-50", "balance": 700.0,
        "allowed_amount": 700.0, "auth_status": "missing",
        "pairing_valid": True, "days_to_filing_deadline": 60,
        "days_since_service": 5, "retro_window_days": 10,
        "cob_position_ok": True,
        "clinical_indication_text": "Recurrent palpitations with syncope.",
        "payer_correspondence_text": "Denied: not medically necessary (CO-50).",
    }
    state.update(state_overrides)
    return {
        "state": state,
        "patients": {
            "patient_id": "pt-0001", "first_name": "Ana", "last_name": "Keller",
            "dob": "1957-03-14", "sex": "F", "address_street": "12 Maple Ave",
            "address_city": "Fairview", "address_state": "IL", "address_zip": "60412",
        },
        "coverage": {
            "patient_id": "pt-0001", "member_id": "M12345678",
            "subscriber_name": "Ana Keller", "subscriber_relationship": "self",
            "secondary_payer_id": None, "aob_on_file": True,
        },
        "clinics": {
            "clinic_id": "cl-001", "name": "Northgate Cardiology Associates",
            "npi": "9123456789",
        },
        "studies": {
            "study_id": "st-03000", "service_start": "2025-06-01",
            "service_end": "2025-06-15",
        },
        "payers": {"payer_id": "meridian", "name": "Meridian Health Plan"},
    }


def test_cms1500_provenance_and_render():
    artifact = fill_cms1500({}, make_bundle())
    assert artifact["fields"]["patient_name"] == "Ana Keller"
    assert artifact["fields"]["service_cpt"] == "93229"
    by_field = {e["field"]: e for e in artifact["field_map"]}
    assert by_field["insured_member_id"]["source_table"] == "coverage"
    assert by_field["insured_member_id"]["row_id"] == "pt-0001"
    # every mapped field appears in the field_map with a source
    assert set(by_field) == set(FIELD_MAPS["cms1500"])
    assert "Ana Keller" in artifact["html"]


def test_cms1500_gaps_stay_empty_never_invented():
    bundle = make_bundle()
    bundle["coverage"]["secondary_payer_id"] = None  # no secondary: gap
    artifact = fill_cms1500({}, bundle)
    assert artifact["fields"]["cob_other_coverage"] == ""
    assert "cob_other_coverage" in artifact["gaps"]
    # auth 'missing' is blanked to a declared gap, not rendered as a value
    assert artifact["fields"]["prior_auth_reference"] == ""
    assert "prior_auth_reference" in artifact["gaps"]
    assert "EMPTY" in artifact["html"]


def test_prior_auth_quotes_indication_verbatim():
    payload = {"justification_code": "auth_not_obtained_before_service"}
    artifact = fill_prior_auth_request(payload, make_bundle())
    assert artifact["fields"]["clinical_indication"] == (
        "Recurrent palpitations with syncope."
    )
    assert artifact["fields"]["retro_justification_code"] == (
        "auth_not_obtained_before_service"
    )
    assert "Recurrent palpitations with syncope." in artifact["html"]


def test_appeal_letter_carries_paragraph_and_facts():
    payload = {
        "denial_carc": "CO-50",
        "necessity_paragraph": (
            "Continuous monitoring (CPT 93229) is medically necessary given "
            "documented I48.0 and recurrent syncope."
        ),
    }
    artifact = compose_appeal_letter(payload, make_bundle())
    assert artifact["fields"]["denial_carc"] == "CO-50"
    assert artifact["necessity_paragraph"].startswith("Continuous monitoring")
    assert "Statement of medical necessity" in artifact["html"]


def test_pre_validate_flags_closed_windows():
    assert pre_validate(make_bundle(), "submit_clean") == []
    closed = make_bundle(days_to_filing_deadline=-5)
    assert "filing window closed" in pre_validate(closed, "submit_clean")
    late = make_bundle(days_since_service=40)
    assert "retro window closed" in pre_validate(late, "request_retro_auth")

