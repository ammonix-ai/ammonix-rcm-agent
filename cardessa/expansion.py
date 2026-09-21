"""Staged corpus expansion (build package Section 1: +500-episode tranches,
cap
TRANCHE_CAP_EPISODES, one master seed lineage, contamination-safe by patient-level split).

The expansion trigger fires when the 100x rule fails on working data. Tranche
composition is data-driven: deficits per rare action are read off the
cumulative working corpus, translated into family slots via measured
per-family yields, and the remainder is a general mix. Expansion studies are
drawn ONLY for working-side patients: the deficit lives on the working side,
and quarantine A / reserve B stay exactly as W1 sealed them.

Every study is a new order for an existing patient (same coverage, same
diagnoses, same split side), drawn from the seed lineage
stable_seed(master_seed, "studies_expansion", tranche_no).
"""

from datetime import date, timedelta

import numpy as np
import polars as pl

from cardessa.audits import MIN_STATES_PER_ACTION
from cardessa.codes import STUDIES
from cardessa.engine import stable_seed
from cardessa.world import World

# rare action -> the curated family that produces it (None = general rate only)
ACTION_FAMILY = {
    "request_retro_auth": "retro_auth",
    "request_peer_to_peer": "p2p",
    "bill_secondary": "cob",
    "correct_and_resubmit": "cob",
    "provide_requested_info": None,
}
# fallback yields (rare-action states per family episode) when the cumulative
# corpus has no episodes of that family yet; measured on tranche 0
FALLBACK_YIELD = {
    ("retro_auth", "request_retro_auth"): 0.25,
    ("p2p", "request_peer_to_peer"): 0.74,
    ("cob", "bill_secondary"): 0.18,
    ("cob", "correct_and_resubmit"): 0.33,
}


def working_action_counts(states: pl.DataFrame, episodes: pl.DataFrame) -> dict[str, int]:
    from cardessa.audits import CANONICAL_ACTIONS

    split_of = dict(episodes.select("episode_id", "split").rows())
    working = states.filter(
        pl.col("episode_id").map_elements(
            lambda e: split_of.get(e) == "working", return_dtype=pl.Boolean
        )
    )
    counts = {a: 0 for a in CANONICAL_ACTIONS}
    counts.update(dict(working.group_by("action_raw").len().rows()))
    return counts


def measured_yield(
    states: pl.DataFrame, episodes: pl.DataFrame, family: str, action: str
) -> float:
    fam_eps = episodes.filter(pl.col("family_id") == family)
    if fam_eps.height == 0:
        return FALLBACK_YIELD.get((family, action), 0.0)
    fam_ids = set(fam_eps["episode_id"].to_list())
    n_action = states.filter(
        pl.col("episode_id").is_in(sorted(fam_ids))
        & (pl.col("action_raw") == action)
    ).height
    measured = n_action / fam_eps.height
    return measured if measured > 0 else FALLBACK_YIELD.get((family, action), 0.0)


def plan_allocation(
    states: pl.DataFrame, episodes: pl.DataFrame, tranche_size: int
) -> dict[str | None, int]:
    """Family slots for one tranche, from deficits and measured yields."""
    counts = working_action_counts(states, episodes)
    slots: dict[str | None, int] = {}
    budget = tranche_size
    for action, family in ACTION_FAMILY.items():
        if family is None or budget <= 0:
            continue
        deficit = MIN_STATES_PER_ACTION - counts.get(action, 0)
        if deficit <= 0:
            continue
        rate = measured_yield(states, episodes, family, action)
        if rate <= 0:
            continue
        need = int(np.ceil(deficit / rate))
        take = min(need, budget)
        slots[family] = max(slots.get(family, 0), take)
    budget -= sum(slots.values())
    if budget < 0:  # families over-subscribed the tranche: scale down evenly
        total = sum(slots.values())
        for family in list(slots):
            slots[family] = int(slots[family] * tranche_size / total)
        budget = tranche_size - sum(slots.values())
    slots[None] = budget  # general mix (also the only lever for CO-16 info requests)
    return slots


def expansion_studies(
    world: World,
    master_seed: int,
    tranche_no: int,
    allocation: dict[str | None, int],
    start_study_index: int,
) -> pl.DataFrame:
    """New studies for WORKING-side patients only, per the allocation."""
    rng = np.random.default_rng(
        stable_seed(master_seed, "studies_expansion", tranche_no)
    )
    working = world.patients.filter(pl.col("split") == "working")
    payer_of = dict(working.select("patient_id", "payer_id").rows())
    stability_of = dict(working.select("patient_id", "stability").rows())
    secondary = {
        r["patient_id"]
        for r in world.coverage.rows(named=True)
        if r["secondary_payer_id"] is not None and r["patient_id"] in payer_of
    }
    heavy = {
        r["patient_id"]
        for r in world.coverage.rows(named=True)
        if r["deductible_remaining"] >= 900 and r["patient_id"] in payer_of
    }
    pools: dict[str | None, list[str]] = {
        "retro_auth": sorted(
            p for p, payer in payer_of.items() if payer in ("meridian", "silverbridge")
        ),
        "p2p": sorted(p for p, payer in payer_of.items() if payer == "cornerstone"),
        "cob": sorted(secondary),
        # v0.4 spec s.5: deductible_remaining >= any allowed amount - the
        # population where findings 2 and 3 (letters, close-out EV) live
        "deductible_heavy": sorted(heavy),
        None: sorted(payer_of),
    }
    payer_map = {p.payer_id: p for p in world.payers}
    clinics = world.clinics
    enrolled = clinics.filter(pl.col("medicaid_enrolled"))["clinic_id"].to_list()
    all_clinics = clinics["clinic_id"].to_list()
    cpts = list(STUDIES)
    cpt_probs = np.array([0.30, 0.25, 0.25, 0.20])

    rows = []
    index = start_study_index
    for family in ("retro_auth", "p2p", "cob", "deductible_heavy", None):  # fixed order
        n = allocation.get(family, 0)
        pool = pools[family]
        if n <= 0:
            continue
        if not pool:
            raise ValueError(f"no working patients eligible for family {family!r}")
        for _ in range(n):
            patient_id = pool[int(rng.integers(len(pool)))]
            payer = payer_map[payer_of[patient_id]]
            if family == "retro_auth":
                cpt = "93229"
            elif family == "p2p":
                cpt = "93271"
            else:
                cpt = cpts[int(rng.choice(len(cpts), p=cpt_probs))]
            if payer.payer_id in ("harborview", "pelican"):
                clinic_id = enrolled[int(rng.integers(len(enrolled)))]
            else:
                clinic_id = all_clinics[int(rng.integers(len(all_clinics)))]
            order_date = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 330)))
            service_start = order_date + timedelta(days=int(rng.integers(2, 15)))
            dur_lo, dur_hi = STUDIES[cpt][2]
            service_end = service_start + timedelta(
                days=int(rng.integers(dur_lo, dur_hi + 1))
            )
            report_delivered = service_end + timedelta(days=int(rng.integers(1, 6)))
            auth_needed = cpt in payer.auth_required_cpts
            if family == "retro_auth":
                auth_obtained = False  # the planted P1 situation, same as W1
            else:
                auth_obtained = bool(auth_needed and rng.random() < 0.65)
            prior_holter = (
                int(rng.integers(10, 26))
                if family == "p2p" or (cpt == "93271" and rng.random() < 0.05)
                else None
            )
            rows.append(
                {
                    "study_id": f"st-{index:05d}",
                    "patient_id": patient_id,
                    "clinic_id": clinic_id,
                    "cpt": cpt,
                    "order_date": order_date,
                    "service_start": service_start,
                    "service_end": service_end,
                    "report_delivered": report_delivered,
                    "auth_required": auth_needed,
                    "auth_obtained": auth_obtained,
                    "eligibility_verified": bool(rng.random() < 0.9),
                    "coverage_lapsed": bool(rng.random() < stability_of[patient_id]),
                    "prior_holter_within_days": prior_holter,
                    "family": family,
                }
            )
            index += 1
    return pl.DataFrame(rows, schema_overrides=world.studies.schema)


def extend_world(world: World, extra_studies: pl.DataFrame) -> World:
    return World(
        payers=world.payers,
        clinics=world.clinics,
        patients=world.patients,
        coverage=world.coverage,
        studies=pl.concat([world.studies, extra_studies]),
    )
