"""Staged expansion: allocation math, working-only studies, determinism."""

import polars as pl
import pytest

from cardessa import MASTER_SEED
from cardessa.expansion import (
    expansion_studies,
    extend_world,
    plan_allocation,
    working_action_counts,
)
from cardessa.world import generate_world


@pytest.fixture(scope="module")
def world():
    return generate_world(MASTER_SEED)


def synthetic_corpus(counts: dict[str, int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """episodes/states where action `a` appears counts[a] times, all working."""
    state_rows, episode_rows = [], []
    i = 0
    for action, n in counts.items():
        for _ in range(n):
            episode_id = f"e{i:05d}"
            episode_rows.append(
                {"episode_id": episode_id, "split": "working", "family_id": None}
            )
            state_rows.append({"episode_id": episode_id, "action_raw": action})
            i += 1
    return pl.DataFrame(episode_rows), pl.DataFrame(state_rows)


def test_working_action_counts_ignores_other_sides():
    episodes, states = synthetic_corpus({"submit_clean": 3})
    episodes = pl.concat(
        [
            episodes,
            pl.DataFrame(
                [{"episode_id": "q1", "split": "quarantine_a", "family_id": None}]
            ),
        ],
        how="vertical_relaxed",
    )
    states = pl.concat(
        [states, pl.DataFrame([{"episode_id": "q1", "action_raw": "submit_clean"}])]
    )
    counts = working_action_counts(states, episodes)
    assert counts["submit_clean"] == 3


def test_plan_allocation_targets_deficits():
    episodes, states = synthetic_corpus(
        {"request_retro_auth": 90, "request_peer_to_peer": 100, "bill_secondary": 100,
         "correct_and_resubmit": 100, "provide_requested_info": 100}
    )
    alloc = plan_allocation(states, episodes, 500)
    # only retro_auth is deficient: 10 missing at fallback yield 0.25 -> 40 slots
    assert alloc["retro_auth"] == 40
    assert alloc[None] == 460
    assert "p2p" not in alloc


def test_plan_allocation_scales_down_when_oversubscribed():
    episodes, states = synthetic_corpus({"submit_clean": 5})  # everything deficient
    alloc = plan_allocation(states, episodes, 100)
    assert sum(alloc.values()) == 100
    assert all(v >= 0 for v in alloc.values())


def test_expansion_studies_working_patients_only(world):
    alloc = {"retro_auth": 20, "p2p": 10, "cob": 10, None: 10}
    studies = expansion_studies(world, MASTER_SEED, 1, alloc, 1000)
    assert studies.height == 50
    working_ids = set(
        world.patients.filter(pl.col("split") == "working")["patient_id"].to_list()
    )
    assert set(studies["patient_id"].to_list()) <= working_ids
    fam_counts = dict(studies.group_by("family").len().rows())
    assert fam_counts["retro_auth"] == 20 and fam_counts["p2p"] == 10
    retro = studies.filter(pl.col("family") == "retro_auth")
    assert retro["cpt"].unique().to_list() == ["93229"]
    assert not retro["auth_obtained"].any()  # the planted P1 situation preserved


def test_expansion_studies_deterministic(world):
    alloc = {"retro_auth": 5, None: 5}
    a = expansion_studies(world, MASTER_SEED, 1, alloc, 1000)
    b = expansion_studies(world, MASTER_SEED, 1, alloc, 1000)
    assert a.equals(b)
    c = expansion_studies(world, MASTER_SEED, 2, alloc, 1000)
    assert not a.equals(c)  # different tranche -> different seed lineage


def test_extend_world_appends_studies(world):
    alloc = {None: 3}
    extra = expansion_studies(world, MASTER_SEED, 1, alloc, 1000)
    extended = extend_world(world, extra)
    assert extended.studies.height == world.studies.height + 3
    assert extended.studies["study_id"].to_list()[-1] == "st-01002"
    assert extended.patients.equals(world.patients)
