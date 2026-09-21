"""Pipeline contracts: tokenizer, fold integrity, swarm semantics, recommendation."""

import numpy as np
import pytest

from ammonix_core import toy
from ammonix_core.pipeline import (
    build_fold_plan,
    recommend,
    tokenize_tabular,
    train_swarm,
)
from ammonix_core.schema import FeatureSpec

SPECS = [
    FeatureSpec(name="x1", dtype="float", description="planted signal"),
    FeatureSpec(name="x2", dtype="float", description="decoy"),
    FeatureSpec(name="x3", dtype="bool", description="planted selector"),
    FeatureSpec(name="x4", dtype="float", description="decoy"),
]
ACTION_MAP = {a: a for a in toy.ACTIONS}


def small_build(n_episodes=600, seed=5):
    corpus = toy.generate_corpus(n_episodes, seed)
    records = tokenize_tabular(corpus.states, corpus.examples, SPECS, ACTION_MAP)
    fold_plan = build_fold_plan(corpus.examples, n_folds=5, seed=seed)
    return corpus, records, fold_plan


def test_tokenizer_rejects_unmapped_action():
    corpus = toy.generate_corpus(5, seed=1)
    corpus.states[0].action_raw = "act_unknown"
    with pytest.raises(ValueError, match="unmapped raw action"):
        tokenize_tabular(corpus.states, corpus.examples, SPECS, ACTION_MAP)


def test_tokenizer_rejects_missing_feature():
    corpus = toy.generate_corpus(5, seed=1)
    del corpus.states[0].data.inline["x2"]
    with pytest.raises(ValueError, match="missing features"):
        tokenize_tabular(corpus.states, corpus.examples, SPECS, ACTION_MAP)


def test_qpsi_records_inherit_outcome():
    corpus, records, _ = small_build(n_episodes=30)
    outcome_by_example = {e.example_id: e.outcome.success for e in corpus.examples}
    for record in records:
        assert record.outcome_success == outcome_by_example[record.example_id]


def test_fold_plan_covers_every_example_exactly_once():
    corpus, _, fold_plan = small_build(n_episodes=200)
    assert set(fold_plan.assignment) == {e.example_id for e in corpus.examples}
    assert set(fold_plan.assignment.values()) == set(range(5))


def test_fold_plan_is_stratified_by_outcome():
    corpus, _, fold_plan = small_build(n_episodes=600)
    success_by_example = {e.example_id: e.outcome.success for e in corpus.examples}
    rates = []
    for fold in range(5):
        members = [e for e, f in fold_plan.assignment.items() if f == fold]
        rates.append(np.mean([success_by_example[m] for m in members]))
    assert max(rates) - min(rates) < 0.10  # folds carry similar outcome rates


def test_swarm_learns_planted_truth_and_ignores_decoys():
    _, records, fold_plan = small_build(n_episodes=600)
    swarm = train_swarm(records, fold_plan, [s.name for s in SPECS], seed=5)
    # small corpus, loose bar: the real bar (0.95) is enforced by check_toy.py at full size
    assert swarm.mean_oof_auroc > 0.85
    for action_id, labels in swarm.oof_labels.items():
        assert len(labels) == swarm.oof_scores[action_id].shape[0]


def test_recommendation_matches_oracle_away_from_boundary():
    _, records, fold_plan = small_build(n_episodes=600)
    swarm = train_swarm(records, fold_plan, [s.name for s in SPECS], seed=5)
    clear_cases = [
        {"x1": 0.9, "x2": 0.5, "x3": True, "x4": 0.5},   # act_a
        {"x1": 0.9, "x2": 0.5, "x3": False, "x4": 0.5},  # act_a
        {"x1": 0.1, "x2": 0.5, "x3": True, "x4": 0.5},   # act_b
        {"x1": 0.1, "x2": 0.5, "x3": False, "x4": 0.5},  # act_c
    ]
    recommended = recommend(swarm, clear_cases)
    assert recommended == ["act_a", "act_a", "act_b", "act_c"]


def test_success_learning_beats_imitation():
    """The logging policy habitually picks act_a; the swarm must not inherit that."""
    _, records, fold_plan = small_build(n_episodes=600)
    swarm = train_swarm(records, fold_plan, [s.name for s in SPECS], seed=5)
    rows = toy.generate_test_states(n_states=200, seed=9)
    recommended = recommend(swarm, [r["features"] for r in rows])
    act_a_share = recommended.count("act_a") / len(recommended)
    assert act_a_share < 0.7  # imitation of the majority action would give ~1.0
