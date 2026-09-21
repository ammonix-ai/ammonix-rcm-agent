"""Q_PSI round 2: LLM text extractor (build package Section 3, round 2).

The pinned model reads the two text columns and answers named booleans under
guided JSON (schema-constrained decoding, temperature 0). Answers are cached
on disk keyed by prompt hash, exactly like corpus generation, so the round
regenerates hash-identically and re-runs cost nothing.

Round 2 features = round 1 features + these booleans; the round's AUROC
delta measures what reading the letters is worth.
"""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from ammonix_core.hashing import sha256_text
from ammonix_core.schema import FeatureSpec

from cardessa.features import build_round1_features
from cardessa.textgen import DEFAULT_BASE_URL, PINNED_MODEL_SUBSTRING

LETTER_BOOLEANS: dict[str, str] = {
    "letter_names_missing_document": (
        "The letter explicitly names a specific missing document or record"
    ),
    "letter_cites_medical_policy": (
        "The letter cites a medical policy, coverage criterion, or medical necessity"
    ),
    "letter_mentions_appeal_rights": (
        "The letter mentions the member's or provider's right to appeal"
    ),
    "letter_mentions_filing_deadline": (
        "The letter mentions a filing deadline, time limit, or expiration window"
    ),
    "letter_mentions_authorization": (
        "The letter discusses prior or retroactive authorization"
    ),
}

INDICATION_BOOLEANS: dict[str, str] = {
    "indication_mentions_syncope": (
        "The note mentions syncope, fainting, or loss of consciousness"
    ),
    "indication_mentions_palpitations": (
        "The note mentions palpitations or irregular heartbeat sensations"
    ),
    "symptom_duration_documented": (
        "The note states how long the symptoms have been present"
    ),
    "indication_mentions_prior_workup": (
        "The note references prior monitoring, ECG, or earlier cardiac workup"
    ),
}


def json_schema(names: dict[str, str]) -> dict:
    return {
        "type": "object",
        "properties": {name: {"type": "boolean"} for name in names},
        "required": list(names),
        "additionalProperties": False,
    }


def extraction_prompt(kind: str, text: str, questions: dict[str, str]) -> str:
    lines = "\n".join(f"- {name}: {desc}" for name, desc in questions.items())
    return (
        f"Read this synthetic {kind} from a simulated billing dataset and answer "
        "with a JSON object of booleans, nothing else. Judge only from the text "
        f"given.\n\nTEXT:\n{text}\n\nFields:\n{lines}"
    )


@dataclass
class VllmJsonExtractor:
    """Pinned-model-only, temperature 0, schema-constrained decoding."""

    schema: dict
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = 150
    model_id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as response:
            served = json.loads(response.read())["data"]
        ids = [entry["id"] for entry in served]
        matches = [i for i in ids if PINNED_MODEL_SUBSTRING in i]
        if not matches:
            raise RuntimeError(
                f"vLLM at {self.base_url} serves {ids}, not the pinned model "
                f"(*{PINNED_MODEL_SUBSTRING}*). Refusing to extract features."
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
                "chat_template_kwargs": {"enable_thinking": False},
                # this vLLM silently ignores the legacy guided_json param;
                # response_format json_schema is the enforced path. The P4
                # extraction answers were key-validated (CachedExtractor
                # raises on missing booleans), so cached results stand.
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction", "schema": self.schema, "strict": True,
                    },
                },
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
                with urllib.request.urlopen(request, timeout=300) as response:
                    body = json.loads(response.read())
                break
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
        assert body is not None
        content = body["choices"][0]["message"]["content"]
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        return content.strip()


@dataclass
class CachedExtractor:
    """Prompt-hash disk cache holding validated JSON answers."""

    provider: VllmJsonExtractor | None
    cache_dir: Path
    expected_keys: tuple[str, ...]
    calls_to_provider: int = 0

    def extract(self, prompt: str) -> dict[str, bool]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self.cache_dir / f"{sha256_text(prompt)}.json"
        if cached.is_file():
            answer = json.loads(cached.read_text(encoding="utf-8"))
        else:
            if self.provider is None:
                raise RuntimeError("cache miss but no provider (offline rebuild)")
            answer = json.loads(self.provider.generate(prompt))
            self.calls_to_provider += 1
            cached.write_text(json.dumps(answer, sort_keys=True), encoding="utf-8")
        missing = [k for k in self.expected_keys if not isinstance(answer.get(k), bool)]
        if missing:
            raise ValueError(f"extraction answer missing booleans {missing}")
        return {k: bool(answer[k]) for k in self.expected_keys}


def warm_cache(extractor: CachedExtractor, prompts: list[str], workers: int = 12) -> int:
    """Fill the cache concurrently; returns provider call count."""
    unique = sorted(set(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(extractor.extract, unique))
    return extractor.calls_to_provider


def build_round2_features(
    states: pl.DataFrame,
    letter_extractor: CachedExtractor,
    indication_extractor: CachedExtractor,
) -> tuple[pl.DataFrame, list[FeatureSpec]]:
    """Round 1 features + guided-JSON booleans over the two text columns."""
    frame, specs = build_round1_features(states)
    specs = list(specs)

    letter_prompt_of: dict[str, str] = {}
    indication_prompt_of: dict[str, str] = {}
    for row in states.select(
        "state_id", "payer_correspondence_text", "clinical_indication_text"
    ).to_dicts():
        letter = (row["payer_correspondence_text"] or "").strip()
        if letter:
            letter_prompt_of[row["state_id"]] = extraction_prompt(
                "payer letter", letter, LETTER_BOOLEANS
            )
        indication = (row["clinical_indication_text"] or "").strip()
        if indication:
            indication_prompt_of[row["state_id"]] = extraction_prompt(
                "clinical indication note", indication, INDICATION_BOOLEANS
            )

    warm_cache(letter_extractor, list(letter_prompt_of.values()))
    warm_cache(indication_extractor, list(indication_prompt_of.values()))

    letter_rows = {
        state_id: letter_extractor.extract(prompt)
        for state_id, prompt in letter_prompt_of.items()
    }
    indication_rows = {
        state_id: indication_extractor.extract(prompt)
        for state_id, prompt in indication_prompt_of.items()
    }

    state_ids = frame["state_id"].to_list()
    columns: dict[str, list[float]] = {"letter_present": []}
    for name in (*LETTER_BOOLEANS, *INDICATION_BOOLEANS):
        columns[name] = []
    for state_id in state_ids:
        letter = letter_rows.get(state_id)
        columns["letter_present"].append(1.0 if letter is not None else 0.0)
        for name in LETTER_BOOLEANS:
            columns[name].append(float(letter[name]) if letter else 0.0)
        indication = indication_rows.get(state_id)
        for name in INDICATION_BOOLEANS:
            columns[name].append(float(indication[name]) if indication else 0.0)

    frame = frame.with_columns(
        *(pl.Series(name, values) for name, values in columns.items())
    )
    specs.append(
        FeatureSpec(
            name="letter_present", dtype="bool",
            description="A payer letter is attached to this touch (false pre-response)",
        )
    )
    for name, description in LETTER_BOOLEANS.items():
        specs.append(
            FeatureSpec(
                name=name, dtype="bool",
                description=f"LLM-extracted from the payer letter: {description}",
            )
        )
    for name, description in INDICATION_BOOLEANS.items():
        specs.append(
            FeatureSpec(
                name=name, dtype="bool",
                description=f"LLM-extracted from the indication note: {description}",
            )
        )
    return frame, specs
