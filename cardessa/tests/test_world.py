"""World snapshot invariants: counts, splits, families, dates, determinism."""

import polars as pl
import pytest

from cardessa import MASTER_SEED
from cardessa.world import (
    FAMILY_TARGETS,
    N_CLINICS,
    N_PATIENTS,
    QUARANTINE_A_TARGET,
    RESERVE_B_TARGET,
    World,
    generate_world,
    load_payers_yaml,
    write_world,
)


@pytest.fixture(scope="module")
def world() -> World:
    return generate_world(MASTER_SEED)


def test_counts(world):
    assert len(world.payers) == 12
    assert sum(p.holdout for p in world.payers) == 2
    assert world.clinics.height == N_CLINICS == 25
    assert world.patients.height == N_PATIENTS == 1000
    assert world.coverage.height == 1000
    assert world.studies.height == 1000


def test_double_generation_hash_equality(world):
    again = generate_world(MASTER_SEED)
    assert again.content_sha256() == world.content_sha256()
    other = generate_world(MASTER_SEED + 1)
    assert other.content_sha256() != world.content_sha256()


def test_holdout_payer_patients_all_in_quarantine_a(world):
    holdout = world.patients.filter(pl.col("payer_id").is_in(["granite", "pelican"]))
    assert holdout.height > 0
    assert set(holdout["split"].to_list()) == {"quarantine_a"}


def test_split_sizes_fixed_at_world_creation(world):
    counts = dict(
        world.patients.group_by("split").len().rows()
    )
    assert counts["quarantine_a"] == QUARANTINE_A_TARGET == 150
    assert counts["reserve_b"] == RESERVE_B_TARGET == 100
    assert counts["working"] == 750


def test_family_targets_met_exactly(world):
    counts = dict(
        world.studies.filter(pl.col("family").is_not_null())
        .group_by("family")
        .len()
        .rows()
    )
    assert counts == FAMILY_TARGETS
    assert sum(counts.values()) == 300  # the 30% curated share


def test_retro_auth_family_is_the_p1_situation(world):
    payer_of = dict(
        zip(
            world.patients["patient_id"].to_list(),
            world.patients["payer_id"].to_list(),
            strict=True,
        )
    )
    rows = world.studies.filter(pl.col("family") == "retro_auth").rows(named=True)
    for row in rows:
        assert row["cpt"] == "93229"  # MCT
        assert row["auth_required"] is True
        assert row["auth_obtained"] is False  # auth missing at first state
        assert payer_of[row["patient_id"]] in ("meridian", "silverbridge")


def test_p2p_family_is_the_cornerstone_bundle(world):
    payer_of = dict(
        zip(
            world.patients["patient_id"].to_list(),
            world.patients["payer_id"].to_list(),
            strict=True,
        )
    )
    rows = world.studies.filter(pl.col("family") == "p2p").rows(named=True)
    for row in rows:
        assert row["cpt"] == "93271"
        assert row["prior_holter_within_days"] is not None
        assert row["prior_holter_within_days"] <= 30
        assert payer_of[row["patient_id"]] == "cornerstone"


def test_medicaid_enrollment_family_uses_non_enrolled_clinics(world):
    enrolled = dict(
        zip(
            world.clinics["clinic_id"].to_list(),
            world.clinics["medicaid_enrolled"].to_list(),
            strict=True,
        )
    )
    rows = world.studies.filter(pl.col("family") == "medicaid_enrollment").rows(named=True)
    for row in rows:
        assert enrolled[row["clinic_id"]] is False


def test_study_dates_are_ordered(world):
    for row in world.studies.rows(named=True):
        assert row["order_date"] < row["service_start"]
        assert row["service_start"] < row["service_end"]
        assert row["service_end"] < row["report_delivered"]


def test_coverage_realism(world):
    rows = world.coverage.rows(named=True)
    secondary = sum(r["secondary_payer_id"] is not None for r in rows)
    assert 140 <= secondary <= 260  # ~20% of 1000
    missing_aob = sum(not r["aob_on_file"] for r in rows)
    assert 5 <= missing_aob <= 50  # ~2%
    dependents = sum(r["subscriber_relationship"] != "self" for r in rows)
    assert 180 <= dependents <= 320  # ~25%


def test_families_only_on_working_side_enough_for_training(world):
    split_of = dict(
        zip(
            world.patients["patient_id"].to_list(),
            world.patients["split"].to_list(),
            strict=True,
        )
    )
    fam = world.studies.filter(pl.col("family").is_not_null()).rows(named=True)
    for family in FAMILY_TARGETS:
        working = sum(
            1 for r in fam if r["family"] == family and split_of[r["patient_id"]] == "working"
        )
        assert working >= 25, f"{family}: only {working} working-side studies"


def test_payers_yaml_round_trip(world, tmp_path):
    write_world(world, tmp_path)
    restored = load_payers_yaml(tmp_path / "payers.yaml")
    assert restored == sorted(world.payers, key=lambda p: p.payer_id)
    for name in ("clinics", "patients", "coverage", "studies"):
        assert (tmp_path / f"{name}.parquet").is_file()
