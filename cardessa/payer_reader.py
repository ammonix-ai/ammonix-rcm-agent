"""The payer's reviewer: reads an appeal letter against the records the payer
holds and marks every statement the records do not support.

The reviewer is part of the simulated world, not of the agent: one pinned
model, temperature 0, the same reader for every writer ever tested. Its
readings are cached by prompt hash under data/payer_reader so the
demonstration claims replay offline and identically. Only the reviewer's
verdict (supported / not supported) is ever used downstream; its text is
never a training target.
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

READER_MODEL = "claude-haiku-4-5-20251001"
READER_TAG = "payer-reader-v1"

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "basis": {"type": "string"},
                },
                "required": ["statement", "supported", "basis"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["statements"],
    "additionalProperties": False,
}


def reviewer_prompt(letter: str, records: dict[str, str]) -> str:
    listing = "\n".join(f"- {k}: {v}" for k, v in records.items() if v)
    return (
        "You are a utilization-review nurse at a health insurer in a "
        "simulated billing dataset (no real patient exists). A provider has "
        "appealed a denial with the letter below. You hold the records the "
        "provider submitted with the claim, listed after the letter.\n\n"
        "Split the letter into its factual statements about the patient, the "
        "service, the codes, the amounts, and the denial. For each statement, "
        "decide whether the submitted records support it. A statement is "
        "supported only if a record states it or directly implies it; "
        "conclusions of medical necessity, characterisations of the records "
        "(\"the documentation shows\"), requests, and identifiers that appear "
        "in no record are not supported. Name the record you relied on, or "
        "\"none\".\n\n"
        f"LETTER:\n{letter}\n\nSUBMITTED RECORDS:\n{listing}\n\n"
        "Answer with only the JSON object."
    )


def payer_records(case_row: dict) -> dict[str, str]:
    """What the payer holds about this claim: the submitted codes, the
    clinical indication note, its own denial letter, and the balance."""
    fields = {
        "procedure_code": str(case_row.get("cpt") or ""),
        "diagnosis_codes": str(case_row.get("dx_codes") or ""),
        "denial_reason_code": str(case_row.get("carc") or ""),
        "billed_balance": str(case_row.get("balance") or ""),
        "clinical_indication_note": str(case_row.get("clinical_indication_text") or ""),
        "denial_letter": str(case_row.get("payer_correspondence_text") or ""),
    }
    return {k: v for k, v in fields.items() if v}


def _load_api_key(data_dir: Path) -> str:
    return (data_dir / "anthropic_key.txt").read_text(encoding="utf-8-sig").strip()


@dataclass
class PayerReader:
    cache_dir: Path
    model_id: str = READER_MODEL
    max_tokens: int = 1500
    calls_to_provider: int = 0
    cache_hits: int = 0
    _client: object = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _anthropic(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=_load_api_key(self.cache_dir.parent))
        return self._client

    def read(self, letter: str, case_row: dict) -> dict:
        """Returns {holds_up, n_statements, unsupported: [statement, ...]}."""
        prompt = reviewer_prompt(letter, payer_records(case_row))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256((self.model_id + READER_TAG + prompt).encode("utf-8")).hexdigest()
        cached = self.cache_dir / f"{key}.json"
        if cached.is_file():
            answer = json.loads(cached.read_text(encoding="utf-8"))
            with self._lock:
                self.cache_hits += 1
        else:
            response = self._anthropic().messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            )
            if response.stop_reason == "refusal":
                raise RuntimeError("the reviewer model declined the request")
            text = next(b.text for b in response.content if b.type == "text").strip()
            answer = json.loads(text)
            cached.write_text(json.dumps(answer, sort_keys=True), encoding="utf-8")
            with self._lock:
                self.calls_to_provider += 1
        unsupported = [s["statement"] for s in answer["statements"] if not s["supported"]]
        return {
            "holds_up": not unsupported,
            "n_statements": len(answer["statements"]),
            "unsupported": unsupported,
        }


def payer_receives(action: str, payload: dict, facts: dict, reader: "PayerReader | None",
                   records: dict | None = None) -> str | None:
    """What the payer does with the paperwork it receives for `action`.
    Returns the rejection reason, or None when the payer processes it.

    facts: what the payer holds about the claim - cpt (the order), dx_codes
    (list, the patient's diagnoses), carc (the denial on file or None),
    patient_share (the amount the patient may be billed), plus the texts
    for the reviewer (clinical_indication_text, payer_correspondence_text,
    balance). records, if given, override the texts."""
    if not payload:
        return None
    if action in ("submit_clean", "submit_with_records", "correct_and_resubmit"):
        if payload.get("cpt") and payload["cpt"] != facts["cpt"]:
            return "procedure code does not match the order"
        dx = payload.get("dx_codes")
        if dx is not None and (not dx or not set(dx) <= set(facts["dx_codes"])):
            return "diagnosis codes do not match the submitted records"
        if action == "correct_and_resubmit" and facts.get("carc") \
                and payload.get("addresses_carc") != facts["carc"]:
            return (f"correction addresses {payload.get('addresses_carc')}, "
                    f"denial on file is {facts['carc']}")
        return None
    if action == "appeal_with_necessity":
        if facts.get("carc") and payload.get("denial_carc") != facts["carc"]:
            return f"appeal cites {payload.get('denial_carc')}, denial on file is {facts['carc']}"
        if reader is None:
            raise RuntimeError("an appeal letter arrived but the world has no reviewer")
        row = {**facts, **(records or {})}
        row["dx_codes"] = ",".join(facts["dx_codes"])
        if not reader.read(payload.get("necessity_paragraph", ""), row)["holds_up"]:
            return "appeal letter not supported by the records"
        return None
    if action == "request_peer_to_peer":
        if facts.get("carc") and payload.get("denial_carc") != facts["carc"]:
            return (f"peer-to-peer request cites {payload.get('denial_carc')}, "
                    f"denial on file is {facts['carc']}")
        return None
    if action == "provide_requested_info":
        if payload.get("requested_item") != "clinical_notes":
            # the payer's reviewer only ever asks for the clinical notes
            return f"sent {payload.get('requested_item')!r}; the request was for clinical_notes"
        return None
    if action == "bill_patient":
        share = facts.get("patient_share")
        if share is not None and payload.get("amount") is not None \
                and float(payload["amount"]) > share + 0.005:
            return f"patient statement {payload['amount']} exceeds the share {share:.2f}"
        return None
    return None


class OfflinePayerReader(PayerReader):
    """Replay-only reader: serves cached readings and refuses new ones."""

    def _anthropic(self):
        raise RuntimeError(
            "this appeal letter has no cached reading; the payer's reviewer "
            "needs the API key to read it"
        )
