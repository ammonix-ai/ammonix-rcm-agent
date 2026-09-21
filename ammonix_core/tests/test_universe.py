"""Universe: ECE, cross-action OOF scoring, leak screen, tribes. Hermetic."""

import numpy as np
import polars as pl

from ammonix_core.pipeline import build_fold_plan
from ammonix_core.schema import Example, Outcome, QPsiRecord
from ammonix_core.universe import (
    build_tribes,
    build_universe,
    expected_calibration_error,
    leak_screen_features,
)


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 20000)
    labels = (rng.uniform(0, 1, 20000) < scores).astype(int)
    assert expected_calibration_error(scores, labels) < 0.02


def test_ece_large_for_overconfident_scores():
    scores = np.full(1000, 0.95)
    labels = np.zeros(1000, dtype=int)
    assert expected_calibration_error(scores, labels) > 0.9


def make_records(n_examples=300, seed=7):
    """Two actions; act_a success driven by x0, act_b by x1."""
    rng = np.random.default_rng(seed)
    records, examples = [], []
    for i in range(n_examples):
        example_id = f"e{i:04d}"
        action = "act_a" if i % 2 == 0 else "act_b"
        x0, x1 = rng.uniform(0, 1), rng.uniform(0, 1)
        driver = x0 if action == "act_a" else x1
        success = bool(rng.uniform() < 0.15 + 0.7 * driver)
        examples.append(
            Example(example_id=example_id, state_ids=[f"{example_id}-0"],
                    outcome=Outcome(success=success, score=1.0))
        )
        records.append(
            QPsiRecord(
                state_id=f"{example_id}-0", example_id=example_id, seq=0,
                features={"x0": x0, "x1": x1, "noise": rng.uniform()},
                action_id=action, outcome_success=success, outcome_score=1.0,
                next_state_id=None,
            )
        )
    fold_plan = build_fold_plan(examples, n_folds=5, seed=seed)
    return records, fold_plan


def test_universe_scores_every_state_for_every_action():
    records, fold_plan = make_records()
    build = build_universe(
        records, fold_plan, ["x0", "x1", "noise"],
        trained_actions=["act_a", "act_b"],
        constant_priors={"act_c": 0.25},
        seed=7,
    )
    assert len(build.records) == len(records)
    for rec in build.records[:20]:
        assert set(rec.scores_cal) == {"act_a", "act_b", "act_c"}
        assert rec.scores_cal["act_c"] == 0.25
        assert all(0.0 <= v <= 1.0 for v in rec.scores_cal.values())
        assert rec.top_features  # attribution written for the argmax action
    assert build.ece_overall <= 0.15  # cross-fitted, honest, small sample
    assert set(build.ece_per_action) == {"act_a", "act_b"}
    assert set(build.pooled_metrics) == {"act_a", "act_b"}
    assert build.pooled_metrics["act_a"]["auroc"] > 0.7  # planted signal found


def test_tribes_margins_ignore_constant_priors():
    records, fold_plan = make_records()
    build = build_universe(
        records, fold_plan, ["x0", "x1", "noise"],
        trained_actions=["act_a", "act_b"],
        constant_priors={"act_c": 0.99},  # would dominate a naive argmax
        seed=7,
    )
    tribes, by_state = build_tribes(build, records, min_cluster_size=15, seed=7)
    assert tribes
    for tribe in tribes:
        assert tribe.stats.rival_action_id != "act_c" or not tribe.stats.ambiguous
        assert tribe.stats.n_states == tribe.member_count
    assigned = {r.tribe_id for r in build.records if r.tribe_id is not None}
    assert assigned == {t.tribe_id for t in tribes if t.member_count > 0} & assigned


def test_leak_screen_catches_planted_posthoc_feature():
    rng = np.random.default_rng(3)
    rows = []
    outcome_by_episode = {}
    for i in range(150):
        episode_id = f"e{i:03d}"
        won = i % 3 != 0
        outcome_by_episode[episode_id] = won
        for _seq in range(2):
            rows.append(
                {
                    "episode_id": episode_id,
                    "honest": float(rng.uniform()),
                    "leaky": 1.0 if won else 0.0,  # episode-constant, separates outcome
                    "varying": float(rng.uniform()) + (1.0 if won else 0.0),
                }
            )
    frame = pl.DataFrame(rows)
    flagged = leak_screen_features(
        frame, ["honest", "leaky", "varying"], outcome_by_episode
    )
    assert "leaky" in flagged
    assert "honest" not in flagged
    # varying separates outcome but is NOT constant within episode -> legitimate
    assert "varying" not in flagged
