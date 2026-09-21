"""Toy world: planted truth, oracle, determinism, protocol behaviour."""

import numpy as np
import pytest

from ammonix_core import toy


def test_oracle_matches_planted_regions():
    assert toy.optimal_action({"x1": 0.9, "x2": 0.5, "x3": True, "x4": 0.1}) == "act_a"
    assert toy.optimal_action({"x1": 0.9, "x2": 0.5, "x3": False, "x4": 0.1}) == "act_a"
    assert toy.optimal_action({"x1": 0.1, "x2": 0.5, "x3": True, "x4": 0.1}) == "act_b"
    assert toy.optimal_action({"x1": 0.1, "x2": 0.5, "x3": False, "x4": 0.1}) == "act_c"


def test_true_probs_are_sharp_at_the_extremes():
    deep_a = {"x1": 0.95, "x2": 0.0, "x3": True, "x4": 0.0}
    assert toy.true_success_prob(deep_a, "act_a") > 0.95
    assert toy.true_success_prob(deep_a, "act_b") < 0.05
    assert toy.true_success_prob(deep_a, "act_c") < 0.05


def test_decoys_do_not_move_the_truth():
    base = {"x1": 0.8, "x2": 0.1, "x3": True, "x4": 0.9}
    moved = {"x1": 0.8, "x2": 0.9, "x3": True, "x4": 0.2}
    for action in toy.ACTIONS:
        assert toy.true_success_prob(base, action) == toy.true_success_prob(moved, action)


def test_unknown_action_rejected():
    with pytest.raises(ValueError):
        toy.true_success_prob({"x1": 0.5, "x2": 0.5, "x3": True, "x4": 0.5}, "act_z")


def test_environment_protocol_shape():
    world = toy.ToyWorld()
    s0 = world.reset(seed=7)
    assert set(s0.features) == set(toy.FEATURE_NAMES)
    assert world.legal_actions(s0) == list(toy.ACTIONS)
    assert world.outcome(s0) is None  # no decision yet
    s1 = world.apply(s0, "act_a")
    assert world.legal_actions(s1) == []  # terminal
    outcome = world.outcome(s1)
    assert outcome is not None and isinstance(outcome.success, bool)


def test_corpus_double_generation_hash_equality():
    a = toy.generate_corpus(n_episodes=50, seed=11)
    b = toy.generate_corpus(n_episodes=50, seed=11)
    assert a.content_sha256() == b.content_sha256()
    c = toy.generate_corpus(n_episodes=50, seed=12)
    assert c.content_sha256() != a.content_sha256()


def test_corpus_is_schema_shaped():
    corpus = toy.generate_corpus(n_episodes=20, seed=3)
    assert len(corpus.examples) == len(corpus.states) == 20
    for example, state in zip(corpus.examples, corpus.states, strict=True):
        assert example.state_ids == [state.state_id]
        assert state.next_state_id is None
        assert state.action_raw in toy.ACTIONS


def test_logging_policy_is_biased_but_covers_all_actions():
    rng = np.random.default_rng(0)
    actions = [toy.logging_action({}, rng) for _ in range(3000)]
    counts = {a: actions.count(a) for a in toy.ACTIONS}
    assert all(counts[a] > 0 for a in toy.ACTIONS)
    assert counts["act_a"] > counts["act_b"]  # the imitation trap: act_a dominates


def test_test_states_carry_oracle_labels():
    rows = toy.generate_test_states(n_states=30, seed=5)
    for row in rows:
        assert row["optimal_action"] == toy.optimal_action(row["features"])
