"""Text channel generation (corpus spec Section 2, channel 2).

The pinned local model (Qwen3.8-27B-AWQ-INT4 via vLLM, temperature 0) writes
payer correspondence and clinical indication notes from templates with salt.
Every generation is cached on disk keyed by the sha256 of its prompt, so the
corpus regenerates hash-identically (double-generation equality including the
text channel) and re-runs cost nothing.

The published historical corpus (data/text_cache, ~17,900 texts written by
Qwen3.6-27B before 2026-09-15) is frozen, read-only data and reads back from
the cache model-independently. The pin moved to Qwen3.8-27B on 2026-09-16 when
Qwen3.6 was retired across the products (preregistration
comparison_qwen38_full_swap): any NEW correspondence is now realized by 3.8.

The provider REFUSES to run against any served model other than the pinned
one: generating corpus text with a different model would silently fork the
world.
"""

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ammonix_core.hashing import sha256_text

from cardessa.codes import CARC, DX_NAMES

PINNED_MODEL_SUBSTRING = "Qwen3.8-27B"
# 127.0.0.1, not localhost: wslrelay.exe squats [::1]:8000 and localhost
# resolves IPv6-first on Windows, shadowing Docker's IPv4 port proxy
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# full study names so the model never mis-expands the abbreviations
STUDY_FULL_NAMES = {
    "93229": "mobile cardiac telemetry (MCT) monitoring",
    "93271": "cardiac event monitoring (CEM)",
    "93247": "long-term continuous ECG monitoring",
    "93226": "48-hour Holter monitoring",
}


class TextProvider(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class VllmProvider:
    """OpenAI-compatible vLLM client, pinned-model-only, temperature 0."""

    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = 260
    model_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as response:
            served = json.loads(response.read())["data"]
        ids = [entry["id"] for entry in served]
        matches = [i for i in ids if PINNED_MODEL_SUBSTRING in i]
        if not matches:
            raise RuntimeError(
                f"vLLM at {self.base_url} serves {ids}, not the pinned model "
                f"(*{PINNED_MODEL_SUBSTRING}*). Refusing to generate corpus text: "
                "restart vLLM with the pinned Qwen3.8-27B-AWQ-INT4."
            )
        self.model_id = matches[0]

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": self.max_tokens,
                # Qwen3.6 is a thinking model; corpus text must be the answer
                # only, at deterministic cost
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        body = None
        for attempt in range(3):  # transient server stalls: retry, same request
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    body = json.loads(response.read())
                break
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
        assert body is not None
        content = body["choices"][0]["message"]["content"]
        # belt and braces: strip any thinking block that still leaks through
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        return content.strip()


@dataclass
class CachedTextGenerator:
    """Prompt-hash-keyed disk cache in front of any provider.

    The cache is the determinism anchor for the text channel: a regeneration
    replays identical text regardless of server-side batching effects.
    """

    provider: TextProvider
    cache_dir: Path
    calls_to_provider: int = 0

    def generate(self, prompt: str) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = sha256_text(prompt)
        cached = self.cache_dir / f"{key}.txt"
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        text = self.provider.generate(prompt)
        cached.write_text(text, encoding="utf-8")
        self.calls_to_provider += 1
        return text


def salt_token(seed_material: str) -> str:
    """Six hex chars of styling salt so templated letters vary in wording."""
    return sha256_text(seed_material)[:6]


def indication_prompt(dx_codes: list[str], study_cpt: str, salt: str) -> str:
    dx_lines = ", ".join(f"{c} ({DX_NAMES[c]})" for c in dx_codes)
    study = STUDY_FULL_NAMES[study_cpt]
    return (
        "You write synthetic clinical text for a simulated health-billing dataset; "
        "no real patient exists. Write the clinical indication note (2-3 sentences, "
        f"plain clinical prose, no headers) from a cardiology clinic ordering a {study} "
        f"study (CPT {study_cpt}). Diagnoses: {dx_lines}. Describe the symptoms and "
        "history consistent with these codes. Do not invent identifiers or dates. "
        f"Style variation token (do not print it): {salt}"
    )


def correspondence_prompt(
    kind: str,
    payer_name: str,
    study_cpt: str,
    carc: str | None,
    salt: str,
    names_missing_document: bool = False,
    money: dict | None = None,
) -> str:
    study = STUDY_FULL_NAMES[study_cpt]
    base = (
        "You write synthetic payer correspondence for a simulated health-billing "
        "dataset; no real payer or patient exists. Write only the letter body, "
        "60-120 words, formal payer voice, no letterhead, no signature block, "
        "no dates, no identifiers. "
    )
    if kind == "denial":
        detail = (
            f"{payer_name} is denying a claim for a {study} study (CPT {study_cpt}) "
            f"with reason code {carc}: {CARC[carc]}. State the denial reason and the "
            "member's appeal rights."
        )
        if names_missing_document:
            detail += (
                " The letter must explicitly state that the missing item is the "
                "ordering physician's clinical notes."
            )
    elif kind == "eob_underpaid":
        detail = (
            f"{payer_name} processed a claim for a {study} study (CPT {study_cpt}) "
            "and paid it partially: the remaining balance is member responsibility "
            "(deductible and coinsurance). Explain this as an EOB remark."
        )
        if money:
            detail += (
                f" State the exact amounts: allowed ${money['allowed']:,.2f}, "
                f"plan paid ${money['payer_paid_total']:,.2f}, member "
                f"responsibility ${money['member_responsibility']:,.2f}."
            )
    elif kind == "eob_paid":
        detail = (
            f"{payer_name} processed and paid a claim for a {study} study "
            f"(CPT {study_cpt}) in full per plan benefits. Write a short EOB remark."
        )
        if money:
            detail += (
                f" State the amount paid: ${money['payer_paid_total']:,.2f} of "
                f"${money['allowed']:,.2f} allowed."
            )
    elif kind == "auth_decision_granted":
        detail = (
            f"{payer_name} grants retroactive authorization for a {study} study "
            f"(CPT {study_cpt}). Reference the authorization on future claims."
        )
    elif kind == "auth_decision_denied":
        detail = (
            f"{payer_name} declines a retroactive authorization request for a "
            f"{study} study (CPT {study_cpt}); the request falls outside policy."
        )
    elif kind == "appeal_granted":
        detail = (
            f"{payer_name} overturns a prior denial for a {study} study "
            f"(CPT {study_cpt}) on appeal; the claim will be reprocessed for payment."
        )
        if money:
            detail += (
                f" State the exact amounts: plan payment ${money['payer_paid_total']:,.2f} "
                f"of ${money['allowed']:,.2f} allowed; member responsibility "
                f"${money['member_responsibility']:,.2f}."
            )
    elif kind == "appeal_granted_deductible":
        detail = (
            f"{payer_name} overturns a prior denial for a {study} study "
            f"(CPT {study_cpt}) on appeal. Upon reprocessing NO payment is due "
            "from the plan: the allowed amount of "
            f"${(money or {}).get('allowed', 0):,.2f} applies to the member's "
            "deductible and the balance of "
            f"${(money or {}).get('member_responsibility', 0):,.2f} is member "
            "responsibility. The letter must state clearly that no plan payment "
            "will be issued and must NOT promise or imply any payment."
        )
    elif kind == "appeal_denied":
        detail = (
            f"{payer_name} upholds its prior denial for a {study} study "
            f"(CPT {study_cpt}) after appeal review."
        )
    elif kind == "paperwork_rejected":
        detail = (
            f"{payer_name} is returning, unprocessed, the documents received for a "
            f"{study} study (CPT {study_cpt}) because they do not match the records "
            "on file" + (f"; the denial under reason code {carc} stands" if carc else "")
            + ". Ask the provider to review and resubmit correct documents."
        )
    else:
        raise ValueError(f"unknown correspondence kind: {kind}")
    return base + detail + f" Style variation token (do not print it): {salt}"
