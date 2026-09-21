"""Round-2 extractor: caching, validation, feature assembly. Hermetic (fake LLM)."""

import json

import polars as pl
import pytest

from cardessa.extract import (
    INDICATION_BOOLEANS,
    LETTER_BOOLEANS,
    CachedExtractor,
    build_round2_features,
    extraction_prompt,
    json_schema,
)
from cardessa.tests.test_features import tiny_states


class FakeProvider:
    """Deterministic canned answers; counts calls like the real provider."""

    def __init__(self, keys):
        self.keys = tuple(keys)
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        # syncope letter/note gets True on the first key, else all False
        first_true = "syncope" in prompt or "missing" in prompt
        return json.dumps(
            {k: (first_true and i == 0) for i, k in enumerate(self.keys)}
        )


def make_extractor(tmp_path, keys, subdir="cache"):
    return CachedExtractor(
        provider=FakeProvider(keys),
        cache_dir=tmp_path / subdir,
        expected_keys=tuple(keys),
    )


def test_cache_hit_skips_provider(tmp_path):
    extractor = make_extractor(tmp_path, LETTER_BOOLEANS)
    prompt = extraction_prompt("payer letter", "some text", LETTER_BOOLEANS)
    first = extractor.extract(prompt)
    second = extractor.extract(prompt)
    assert first == second
    assert extractor.provider.calls == 1
    assert extractor.calls_to_provider == 1


def test_offline_rebuild_from_cache(tmp_path):
    extractor = make_extractor(tmp_path, INDICATION_BOOLEANS)
    prompt = extraction_prompt("note", "text", INDICATION_BOOLEANS)
    answer = extractor.extract(prompt)
    offline = CachedExtractor(
        provider=None,
        cache_dir=extractor.cache_dir,
        expected_keys=tuple(INDICATION_BOOLEANS),
    )
    assert offline.extract(prompt) == answer
    with pytest.raises(RuntimeError, match="cache miss"):
        offline.extract("never seen prompt")


def test_missing_boolean_fails_loudly(tmp_path):
    extractor = make_extractor(tmp_path, LETTER_BOOLEANS)
    bad = extractor.cache_dir
    bad.mkdir(parents=True)
    prompt = "p"
    from ammonix_core.hashing import sha256_text

    (bad / f"{sha256_text(prompt)}.json").write_text('{"unrelated": true}')
    with pytest.raises(ValueError, match="missing booleans"):
        extractor.extract(prompt)


def test_json_schema_shape():
    schema = json_schema(LETTER_BOOLEANS)
    assert schema["required"] == list(LETTER_BOOLEANS)
    assert all(v == {"type": "boolean"} for v in schema["properties"].values())
    assert schema["additionalProperties"] is False


def test_round2_features_assemble(tmp_path):
    states = tiny_states()
    letter_ex = make_extractor(tmp_path, LETTER_BOOLEANS, "letters")
    indication_ex = make_extractor(tmp_path, INDICATION_BOOLEANS, "indications")
    frame, specs = build_round2_features(states, letter_ex, indication_ex)
    names = [s.name for s in specs]
    for name in (*LETTER_BOOLEANS, *INDICATION_BOOLEANS, "letter_present"):
        assert name in names
        assert frame[name].dtype == pl.Float64
    # every state in the fixture has both texts -> letter_present all 1
    assert frame["letter_present"].sum() == frame.height
    # descriptions marked as LLM-extracted
    by_name = {s.name: s for s in specs}
    assert "LLM-extracted" in by_name["letter_cites_medical_policy"].description
    # double build from cache is identical and calls the provider once per prompt
    rebuilt, _ = build_round2_features(states, letter_ex, indication_ex)
    assert frame.equals(rebuilt)


def test_round2_zero_fills_missing_letter(tmp_path):
    states = tiny_states().with_columns(
        pl.when(pl.col("state_id") == "e1-0")
        .then(pl.lit(""))
        .otherwise(pl.col("payer_correspondence_text"))
        .alias("payer_correspondence_text")
    )
    letter_ex = make_extractor(tmp_path, LETTER_BOOLEANS, "letters")
    indication_ex = make_extractor(tmp_path, INDICATION_BOOLEANS, "indications")
    frame, _ = build_round2_features(states, letter_ex, indication_ex)
    row = frame.filter(pl.col("state_id") == "e1-0").to_dicts()[0]
    assert row["letter_present"] == 0.0
    assert all(row[name] == 0.0 for name in LETTER_BOOLEANS)
