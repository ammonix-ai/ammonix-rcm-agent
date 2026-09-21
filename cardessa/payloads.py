"""Action payload schemas and the deterministic M2 rule predicates (P8).

M1 emits an ActionObject whose payload must validate against these schemas;
M2 then evaluates the named rules against the case record and the payer's
published configuration. The LLM chooses and phrases; code fills and
verifies (agent-UI spec Section 1). Payload schemas live in each Skill's
ExpectedResultSpec so the merged ActionMap v1 stays untouched.
"""

from cardessa.payers import build_payer_configs

RETRO_JUSTIFICATION_CODES = (
    "auth_not_obtained_before_service",
    "urgent_service_no_time_for_auth",
    "payer_error_auth_on_file",
)

# v0.4 s.2b: code-bearing fields carry pattern/maxLength constraints so the
# strict decoder physically cannot emit copy-through garbage (the ep-05041
# family of malformed payloads). validate_payload enforces these keywords
# since v4-fixes; the grammar-constrained decoder enforces them at emit time.
_CPT = {"type": "string", "pattern": "^[0-9]{5}$"}
_DX = {"type": "string", "pattern": r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$",
       "maxLength": 8}
_CARC = {"type": "string", "pattern": "^(CO|PR)-[0-9]{1,3}$", "maxLength": 6}

_COMMON = {
    "episode_id": {"type": "string", "maxLength": 12},
    "cpt": _CPT,
}


def _schema(extra: dict, required_extra: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {**_COMMON, **extra},
        "required": ["episode_id", "cpt", *required_extra],
        "additionalProperties": False,
    }


PAYLOAD_SCHEMAS: dict[str, dict] = {
    "submit_clean": _schema(
        {"dx_codes": {"type": "array", "items": _DX, "maxItems": 8}}, ["dx_codes"]
    ),
    "submit_with_records": _schema(
        {
            "dx_codes": {"type": "array", "items": _DX, "maxItems": 8},
            "attachments": {
                "type": "array", "items": {"enum": ["clinical_notes"]}, "minItems": 1,
            },
        },
        ["dx_codes", "attachments"],
    ),
    "correct_and_resubmit": _schema(
        {
            "dx_codes": {"type": "array", "items": _DX, "maxItems": 8},
            "correction": {"enum": ["cob_order", "claim_fields"]},
            "addresses_carc": _CARC,
        },
        ["dx_codes", "correction", "addresses_carc"],
    ),
    "request_retro_auth": _schema(
        {"justification_code": {"enum": list(RETRO_JUSTIFICATION_CODES)}},
        ["justification_code"],
    ),
    "appeal_with_necessity": _schema(
        {
            "denial_carc": _CARC,
            "necessity_paragraph": {"type": "string", "minLength": 80,
                                    "maxLength": 1200},
        },
        ["denial_carc", "necessity_paragraph"],
    ),
    "request_peer_to_peer": _schema(
        {"denial_carc": _CARC}, ["denial_carc"]
    ),
    "provide_requested_info": _schema(
        {"requested_item": {"type": "string", "maxLength": 80}},
        ["requested_item"]
    ),
    "bill_secondary": _schema({}, []),
    "bill_patient": _schema({"amount": {"type": "number", "minimum": 0}}, ["amount"]),
    "write_off": _schema({}, []),
}

# named rules per action (agent-UI spec Section 3); every rule is a pure
# predicate over (payload, case_row) where case_row is the state's tabular data
RULES: dict[str, list[str]] = {
    "submit_clean": ["cpt_matches_case", "pairing_legal", "filing_window_open"],
    "submit_with_records": ["cpt_matches_case", "pairing_legal", "filing_window_open"],
    "correct_and_resubmit": [
        "cpt_matches_case", "filing_window_open", "addresses_current_carc",
    ],
    "request_retro_auth": ["cpt_matches_case", "retro_window_open", "auth_actually_missing"],
    "appeal_with_necessity": [
        "cpt_matches_case", "cites_denial_carc", "paragraph_grounded_in_case",
    ],
    "request_peer_to_peer": ["cpt_matches_case", "cites_denial_carc", "p2p_offered"],
    "provide_requested_info": [
        "cpt_matches_case", "requested_item_in_record", "letter_names_requested_item",
    ],
    "bill_secondary": ["cpt_matches_case", "secondary_exists"],
    "bill_patient": ["cpt_matches_case", "amount_within_balance"],
    "write_off": ["cpt_matches_case"],
}

_PAYERS = {p.payer_id: p for p in build_payer_configs()}


def _dx_list(case_row: dict) -> list[str]:
    return [d for d in (case_row.get("dx_codes") or "").split(",") if d]


def evaluate_rule(rule: str, payload: dict, case_row: dict) -> tuple[bool, str]:
    """Returns (passed, detail). Every rule reads only the case record and the
    payer's PUBLISHED configuration - nothing invented, nothing post-hoc."""
    if rule == "cpt_matches_case":
        ok = payload.get("cpt") == case_row["cpt"]
        return ok, f"payload cpt {payload.get('cpt')} vs case {case_row['cpt']}"
    if rule == "pairing_legal":
        dx = payload.get("dx_codes", [])
        ok = sorted(dx) == sorted(_dx_list(case_row)) and bool(case_row["pairing_valid"])
        return ok, "dx must equal the case record's and pairing_valid must hold"
    if rule == "filing_window_open":
        ok = case_row["days_to_filing_deadline"] > 0
        return ok, f"days_to_filing_deadline {case_row['days_to_filing_deadline']}"
    if rule == "addresses_current_carc":
        carc = case_row.get("carc") or ""
        cited = payload.get("addresses_carc")
        # pre-denial corrections (wrong COB order caught before submission)
        # carry no CARC; the prompt renders that as "none"
        ok = cited == carc or (carc == "" and cited in ("", "none"))
        return ok, f"payload addresses {cited} vs case carc {carc or 'none'}"
    if rule == "retro_window_open":
        ok = case_row["retro_window_days"] >= case_row["days_since_service"]
        return ok, (
            f"retro window {case_row['retro_window_days']}d vs "
            f"{case_row['days_since_service']}d since service"
        )
    if rule == "auth_actually_missing":
        ok = case_row["auth_status"] != "on_file" and bool(case_row["auth_required"])
        return ok, f"auth_status {case_row['auth_status']}, required {case_row['auth_required']}"
    if rule == "cites_denial_carc":
        cited = payload.get("denial_carc")
        ok = bool(cited) and cited == (case_row.get("carc") or "")
        return ok, f"cited {cited} vs case carc {case_row.get('carc')}"
    if rule == "paragraph_grounded_in_case":
        paragraph = payload.get("necessity_paragraph", "")
        dx = _dx_list(case_row)
        ok = bool(paragraph) and (
            case_row["cpt"] in paragraph or any(code in paragraph for code in dx)
        )
        return ok, "paragraph must reference the case's CPT or a diagnosis code"
    if rule == "p2p_offered":
        payer = _PAYERS[case_row["payer_id"]]
        return payer.p2p_available, f"payer {payer.payer_id} p2p {payer.p2p_available}"
    if rule == "secondary_exists":
        ok = bool(case_row["has_secondary"])
        return ok, f"has_secondary {case_row['has_secondary']}"
    if rule == "amount_within_balance":
        ok = 0 < payload.get("amount", -1) <= case_row["balance"] + 0.01
        return ok, f"amount {payload.get('amount')} vs balance {case_row['balance']}"
    if rule == "requested_item_in_record":
        # the record's only attachable document in this world is the
        # ordering clinician's notes; anything else cannot be provided
        item = payload.get("requested_item", "")
        ok = item == "clinical_notes"
        return ok, f"requested_item {item!r}; the record holds only clinical_notes"
    if rule == "letter_names_requested_item":
        letter = (case_row.get("payer_correspondence_text") or "").lower()
        ok = "clinical note" in letter or "physician's notes" in letter
        return ok, (
            "the payer letter must actually name the notes as the missing item; "
            "if it demands something else (or nothing specific), a human calls"
        )
    raise ValueError(f"unknown rule {rule!r}")
