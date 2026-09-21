"""Tranche audits (corpus spec Section 3): family statistics, P1 structure,
action coverage with the expansion trigger, poisoned column, episode shape.

Every audit returns (passed, details); check_corpus.py aggregates them into
the printed verdict. Coverage is checked cumulatively across tranches over
WORKING-split states only and is the primary expansion trigger of Section 2.
"""

from dataclasses import dataclass

import polars as pl

from cardessa.world import FAMILY_TARGETS

MIN_STATES_PER_ACTION = 100  # the 100x rule, working data only
CANONICAL_ACTIONS = (
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
)

EXPECTED_EPISODES = 1000
STATES_RANGE = (2000, 6500)  # "about 4000"
MEAN_DECISIONS_RANGE = (2.0, 5.5)  # "2 to 7 decision states, mean about 4"
FAMILY_TOLERANCE = 0.25  # relative tolerance on curated-family shares


@dataclass
class AuditResult:
    name: str
    passed: bool
    details: dict


def audit_shape(episodes: pl.DataFrame, states: pl.DataFrame) -> AuditResult:
    n_episodes = episodes.height
    n_states = states.height
    mean_states = n_states / max(1, n_episodes)
    max_states = int(episodes["n_states"].max())
    passed = (
        n_episodes == EXPECTED_EPISODES
        and STATES_RANGE[0] <= n_states <= STATES_RANGE[1]
        and MEAN_DECISIONS_RANGE[0] <= mean_states <= MEAN_DECISIONS_RANGE[1]
        and max_states <= 8  # 7 decisions + forced write_off at the cap
    )
    return AuditResult(
        "shape",
        passed,
        {
            "episodes": n_episodes,
            "states": n_states,
            "mean_states_per_episode": round(mean_states, 2),
            "max_states": max_states,
        },
    )


def audit_families(episodes: pl.DataFrame) -> AuditResult:
    counts = dict(
        episodes.filter(pl.col("family_id").is_not_null())
        .group_by("family_id")
        .len()
        .rows()
    )
    deviations = {}
    passed = True
    for family, target in FAMILY_TARGETS.items():
        actual = counts.get(family, 0)
        deviation = abs(actual - target) / target
        deviations[family] = {"target": target, "actual": actual}
        if deviation > FAMILY_TOLERANCE:
            passed = False
    return AuditResult("families", passed, deviations)


def audit_p1_structure(episodes: pl.DataFrame, states: pl.DataFrame) -> AuditResult:
    """The retro-auth family must show the majority/minority trap as specified:
    at the CO-197 decision states, appeal/write_off dominate (the habit),
    request_retro_auth is a real but minority presence at Meridian."""
    family_eps = episodes.filter(pl.col("family_id") == "retro_auth")
    family_ids = set(family_eps["episode_id"].to_list())
    family_states = states.filter(
        pl.col("episode_id").is_in(sorted(family_ids)) & (pl.col("carc") == "CO-197")
    )
    if family_states.height == 0:
        return AuditResult("p1_structure", False, {"error": "no CO-197 family states"})
    action_counts = dict(family_states.group_by("action_raw").len().rows())
    total = sum(action_counts.values())
    majority_share = (
        action_counts.get("appeal_with_necessity", 0) + action_counts.get("write_off", 0)
    ) / total
    meridian_retro = states.filter(
        pl.col("episode_id").is_in(sorted(family_ids))
        & (pl.col("payer_id") == "meridian")
        & (pl.col("action_raw") == "request_retro_auth")
    ).height
    passed = majority_share >= 0.5 and meridian_retro >= 10
    return AuditResult(
        "p1_structure",
        passed,
        {
            "co197_family_states": total,
            "majority_share_appeal_or_writeoff": round(majority_share, 3),
            "meridian_retro_auth_states": meridian_retro,
            "action_counts": action_counts,
        },
    )


def audit_poisoned_column(states: pl.DataFrame) -> AuditResult:
    present = "days_to_payment" in states.columns
    return AuditResult("poisoned_column", present, {"column": "days_to_payment"})


def coverage_report(states: pl.DataFrame, episodes: pl.DataFrame) -> dict:
    """Cumulative action coverage over working-split states; the expansion
    trigger is armed by writing this report every tranche."""
    split_of = dict(episodes.select("episode_id", "split").rows())
    working = states.filter(
        pl.col("episode_id").map_elements(
            lambda e: split_of.get(e) == "working", return_dtype=pl.Boolean
        )
    )
    counts = {a: 0 for a in CANONICAL_ACTIONS}
    counts.update(dict(working.group_by("action_raw").len().rows()))
    violations = [a for a, n in counts.items() if n < MIN_STATES_PER_ACTION]
    return {
        "working_states": working.height,
        "min_states_per_action": MIN_STATES_PER_ACTION,
        "counts": counts,
        "violations": violations,
        "needs_expansion": bool(violations),
        "expansion_trigger": "armed",
    }
