"""Fresh simulated cases for the toolset check and the demo UI.

Never from training data: demo studies are NEW orders appended AFTER the
full training lineage (original 1000 + every expansion tranche replayed from
its recorded allocation), so demo episode ids start right after the lineage
(right after the lineage) and cannot collide with anything the models
saw. Same master seed, same engine, same text - the same world, fresh cases.
"""

import json
import sys
from pathlib import Path

import polars as pl

from cardessa.corpus import generate_tranche
from cardessa.engine import PayerEngine
from cardessa.expansion import expansion_studies, extend_world
from cardessa.textgen import CachedTextGenerator
from cardessa.world import World, generate_world

DEMO_TRANCHE_NO = 90
DEMO_START_INDEX = None  # derived: demo studies start after the full lineage


def training_world(root: Path, master_seed: int) -> tuple[World, PayerEngine]:
    """The world with the exact expansion lineage replayed (tranches 1-4)."""
    world = generate_world(master_seed)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, master_seed)
    expansion_report = json.loads(
        (root / "runs" / "reports" / "expansion.json").read_text(encoding="utf-8")
    )
    extra = []
    start = 1000
    for tranche in expansion_report["tranches"]:
        allocation = {
            (None if k == "None" else k): v for k, v in tranche["allocation"].items()
        }
        studies = expansion_studies(
            world, master_seed, tranche["tranche"], allocation, start
        )
        extra.append(studies)
        start += studies.height
    world_ext = extend_world(world, pl.concat(extra)) if extra else world
    return world_ext, engine


def demo_allocation() -> dict[str | None, int]:
    """Sample mix: enough denial/retro/appeal paths to exercise every tool."""
    return {"retro_auth": 60, "p2p": 25, "cob": 35, None: 80}


def generate_demo_cases(
    root: Path,
    master_seed: int,
    text: CachedTextGenerator,
    n_episodes: int = 200,
    tranche_no: int = DEMO_TRANCHE_NO,
    allocation: dict[str | None, int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, World, PayerEngine]:
    """(episodes, states, world_ext, engine) for fresh demo episodes."""
    world_ext, engine = training_world(root, master_seed)
    allocation = allocation or demo_allocation()
    assert sum(allocation.values()) == n_episodes
    start_index = world_ext.studies.height  # after the FULL training lineage
    demo_studies = expansion_studies(
        world_ext, master_seed, tranche_no, allocation, start_index
    )
    world_demo = extend_world(world_ext, demo_studies)
    # dry pass + concurrent warm: generate_tranche calls the text provider
    # sequentially, so fill the prompt cache with 12 workers first
    sys.path.insert(0, str(root / "scripts"))
    from build_corpus import PromptCollector, warm_cache

    collector = PromptCollector()
    generate_tranche(world_demo, engine, master_seed, collector, start_index, n_episodes)
    warm_cache(collector.prompts, text)
    result = generate_tranche(
        world_demo, engine, master_seed, text, start_index, n_episodes
    )
    return (
        pl.DataFrame(result.episodes_rows),
        pl.DataFrame(result.states_rows),
        world_demo,
        engine,
    )


def bundle_for_state(
    world: World, engine: PayerEngine, state_row: dict, episode_row: dict
) -> dict:
    """The case bundle every form tool reads: state + source table rows."""
    patient = world.patients.filter(
        pl.col("patient_id") == episode_row["patient_id"]
    ).to_dicts()[0]
    coverage = world.coverage.filter(
        pl.col("patient_id") == episode_row["patient_id"]
    ).to_dicts()[0]
    clinic = world.clinics.filter(
        pl.col("clinic_id") == episode_row["clinic_id"]
    ).to_dicts()[0]
    episode_index = int(state_row["episode_id"].removeprefix("ep-"))
    study = world.studies.row(episode_index, named=True)
    assert study["patient_id"] == episode_row["patient_id"]
    payer = engine.payers[episode_row["payer_id"]]
    return {
        "state": state_row,
        "patients": patient,
        "coverage": coverage,
        "clinics": clinic,
        "studies": study,
        "payers": {"payer_id": payer.payer_id, "name": payer.name},
    }
