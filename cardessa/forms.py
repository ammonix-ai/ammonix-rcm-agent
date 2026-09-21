"""The three form tools (agent-UI spec Section 2): deterministic renderers.

M1 never writes into a form. Each tool reads the committed field-source
mapping (cardessa/field_maps.yaml), pulls every field from its fixed
source in the case bundle, records provenance (source table, source column,
row id) per field, leaves sourceless fields EMPTY and lists them as gaps.
The artifact is printable self-contained HTML (the spec's "pdf" realised
without a binary dependency; the field_map is the contract either way).
"""

import html
from pathlib import Path

import yaml

FIELD_MAPS = yaml.safe_load(
    (Path(__file__).parent / "field_maps.yaml").read_text(encoding="utf-8")
)


def _pull(bundle: dict, table: str, column) -> tuple[str, str]:
    """Returns (value, row_id). Multi-column sources join with a space."""
    source = bundle.get(table) or {}
    row_id = str(
        source.get("patient_id")
        or source.get("clinic_id")
        or source.get("study_id")
        or source.get("state_id")
        or source.get("payer_id")
        or ""
    )
    columns = column if isinstance(column, list) else [column]
    parts = []
    for col in columns:
        value = source.get(col)
        if value is None or value == "":
            return "", row_id
        parts.append(str(value))
    return " ".join(parts), row_id


def fill_fields(artifact: str, bundle: dict) -> tuple[dict, list[dict], list[str]]:
    """(fields, field_map, gaps) for one artifact from the committed mapping."""
    fields: dict[str, str] = {}
    field_map: list[dict] = []
    gaps: list[str] = []
    for field, spec in FIELD_MAPS[artifact].items():
        value, row_id = _pull(bundle, spec["table"], spec["column"])
        fields[field] = value
        field_map.append(
            {
                "field": field,
                "value": value,
                "source_table": spec["table"],
                "source_column": spec["column"],
                "row_id": row_id,
            }
        )
        if value == "":
            gaps.append(field)
    return fields, field_map, gaps


def _render(title: str, fields: dict[str, str], gaps: list[str], body: str = "") -> str:
    rows = "\n".join(
        f'<tr class="{"gap" if not value else ""}">'
        f"<td>{html.escape(field)}</td>"
        f"<td>{html.escape(value) or '&mdash; EMPTY &mdash;'}</td></tr>"
        for field, value in fields.items()
    )
    gap_note = (
        f'<p class="gaps">Gaps (no source; left empty): {", ".join(gaps)}</p>'
        if gaps
        else ""
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:sans-serif;max-width:48rem;margin:2rem auto}"
        "table{border-collapse:collapse;width:100%}td{border:1px solid #999;"
        "padding:.3rem .6rem}tr.gap td{background:#fff3cd}.gaps{color:#856404}"
        "blockquote{border-left:3px solid #999;padding-left:1rem}</style></head>"
        f"<body><h1>{html.escape(title)}</h1><table>{rows}</table>{gap_note}{body}"
        "<p><em>Simulated artifact; no real patient, payer or provider exists.</em></p>"
        "</body></html>"
    )


def fill_cms1500(action_payload: dict, bundle: dict) -> dict:
    fields, field_map, gaps = fill_fields("cms1500", bundle)
    # the auth reference renders only what the state record shows
    if fields.get("prior_auth_reference") == "missing":
        fields["prior_auth_reference"] = ""
        gaps = sorted({*gaps, "prior_auth_reference"})
    return {
        "artifact": "cms1500",
        "fields": fields,
        "field_map": field_map,
        "gaps": gaps,
        "html": _render("CMS-1500 Health Insurance Claim Form (simulated)", fields, gaps),
    }


def fill_prior_auth_request(action_payload: dict, bundle: dict) -> dict:
    bundle = {**bundle, "action_object": action_payload}
    fields, field_map, gaps = fill_fields("prior_auth_request", bundle)
    indication = fields.get("clinical_indication", "")
    body = (
        f"<h2>Clinical indication (verbatim)</h2><blockquote>{html.escape(indication)}"
        "</blockquote>"
        if indication
        else ""
    )
    return {
        "artifact": "prior_auth_request",
        "fields": fields,
        "field_map": field_map,
        "gaps": gaps,
        "html": _render("Retroactive Authorization Request (simulated)", fields, gaps, body),
    }


def compose_appeal_letter(action_payload: dict, bundle: dict) -> dict:
    bundle = {**bundle, "action_object": action_payload}
    fields, field_map, gaps = fill_fields("appeal_letter_facts", bundle)
    paragraph = fields.get("necessity_paragraph", "")
    body = (
        "<h2>Statement of medical necessity</h2>"
        f"<blockquote>{html.escape(paragraph)}</blockquote>"
    )
    return {
        "artifact": "appeal_letter",
        "fields": fields,
        "field_map": field_map,
        "gaps": gaps,
        "necessity_paragraph": paragraph,
        "html": _render("Appeal of Claim Denial (simulated)", fields, gaps, body),
    }


TOOLS = {
    "submit_clean": fill_cms1500,
    "submit_with_records": fill_cms1500,
    "correct_and_resubmit": fill_cms1500,
    "bill_secondary": fill_cms1500,
    "request_retro_auth": fill_prior_auth_request,
    "appeal_with_necessity": compose_appeal_letter,
}


def pre_validate(bundle: dict, action_id: str) -> list[str]:
    """Payer-engine pre-validation on the artifact's case: the same published
    facts the engine checks (code pairing, filing window, COB order)."""
    state = bundle["state"]
    failures = []
    if action_id in ("submit_clean", "submit_with_records", "correct_and_resubmit"):
        if not state["pairing_valid"]:
            failures.append("code pairing not legal for this payer")
        if state["days_to_filing_deadline"] <= 0:
            failures.append("filing window closed")
        if not state["cob_position_ok"] and action_id != "correct_and_resubmit":
            failures.append("COB order wrong and not being corrected")
    if action_id == "request_retro_auth":
        if state["retro_window_days"] < state["days_since_service"]:
            failures.append("retro window closed")
    return failures
