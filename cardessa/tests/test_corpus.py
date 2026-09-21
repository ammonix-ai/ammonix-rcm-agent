"""Corpus generation: schema shape, determinism, traps — on a small slice,
with a stub text provider (the real corpus uses the pinned vLLM model)."""

from dataclasses import dataclass

import polars as pl
import pytest
from ammonix_core.hashing import sha256_text

from cardessa import MASTER_SEED
from cardessa.audits import audit_poisoned_column, coverage_report
from cardessa.corpus import (
    MAX_DECISIONS,
    CachedTextGenerator,
    CardessaEnvironment,
    generate_tranche,
)
from cardessa.engine import PayerEngine
from cardessa.world import generate_world

N_TEST_EPISODES = 80


@dataclass
class StubProvider:
    """Deterministic text stand-in for unit tests only."""

    def generate(self, prompt: str) -> str:
        return f"[stub-text {sha256_text(prompt)[:10]}]"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    text = CachedTextGenerator(
        provider=StubProvider(), cache_dir=tmp_path_factory.mktemp("text_cache")
    )
    result = generate_tranche(world, engine, MASTER_SEED, text, 0, N_TEST_EPISODES)
    return world, engine, result


def test_factory_schema_shape(built):
    _, _, result = built
    assert len(result.examples) == N_TEST_EPISODES
    for example in result.examples:
        assert example.state_ids, example.example_id
        assert isinstance(example.outcome.success, bool)
        assert 0.0 <= example.outcome.score <= 1.0
    by_id = {s.state_id: s for s in result.states}
    for example in result.examples:
        for i, state_id in enumerate(example.state_ids):
            state = by_id[state_id]
            assert state.seq == i
            expected_next = (
                example.state_ids[i + 1] if i + 1 < len(example.state_ids) else None
            )
            assert state.next_state_id == expected_next


def test_episode_lengths_bounded(built):
    _, _, result = built
    lengths = [len(e.state_ids) for e in result.examples]
    assert max(lengths) <= MAX_DECISIONS + 1
    assert min(lengths) >= 1


def test_double_generation_hash_equality(built, tmp_path):
    world, engine, result = built
    text = CachedTextGenerator(provider=StubProvider(), cache_dir=tmp_path / "cache2")
    again = generate_tranche(world, engine, MASTER_SEED, text, 0, N_TEST_EPISODES)
    assert again.content_sha256() == result.content_sha256()


def test_poisoned_column_matches_posthoc_truth(built):
    _, _, result = built
    states = pl.DataFrame(result.states_rows)
    assert audit_poisoned_column(states).passed
    truth = {r["episode_id"]: r["days_to_payment"] for r in result.episodes_rows}
    for row in result.states_rows:
        assert row["days_to_payment"] == truth[row["episode_id"]]  # leaky by design


def test_posthoc_fields_only_in_meta_not_states(built):
    _, _, result = built
    state_columns = set(result.states_rows[0])
    assert "paid_amount_final" not in state_columns
    assert "total_touches" not in state_columns
    assert "engine_truth_allowed" not in state_columns
    assert "days_to_payment" in state_columns  # the single deliberate poison
    for example in result.examples:
        assert "paid_amount_final" in example.meta
        assert "engine_truth_allowed" in example.meta


def test_text_channel_filled(built):
    _, _, result = built
    for row in result.states_rows:
        assert row["clinical_indication_text"].startswith("[stub-text")
        if row["touch_seq"] > 0 and row["carc"]:
            assert row["payer_correspondence_text"].startswith("[stub-text")


def test_environment_protocol_shape(built):
    world, engine, _ = built
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    case = env.reset(0)
    actions = env.legal_actions(case)
    assert "submit_clean" in actions and "write_off" in actions
    assert env.outcome(case) is None
    case = env.apply(case, "submit_clean")
    while not case.terminal:
        case = env.apply(case, "write_off")
    outcome = env.outcome(case)
    assert outcome is not None


def test_coverage_report_shape(built):
    _, _, result = built
    report = coverage_report(
        pl.DataFrame(result.states_rows), pl.DataFrame(result.episodes_rows)
    )
    assert report["expansion_trigger"] == "armed"
    assert set(report["counts"]) == {
        "submit_clean",
        "submit_with_records",
        "request_retro_auth",
        "correct_and_resubmit",
        "appeal_with_necessity",
        "request_peer_to_peer",
        "provide_requested_info",
        "bill_secondary",
        "bill_patient",
        "write_off",
    }


def test_personas_rotate(built):
    _, _, result = built
    personas = {e.meta["persona_id"] for e in result.examples}
    assert personas == {"diligent", "hasty", "conservative"}
