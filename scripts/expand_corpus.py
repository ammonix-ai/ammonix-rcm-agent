"""Staged corpus expansion driver (P5): +500-episode tranches until the 100x
rule is met on working data or the 3000-episode cap is reached.

Each tranche: plan family allocation from cumulative deficits -> generate
expansion studies (working patients only) -> dry-pass prompt collection ->
warm the text cache (pinned 27B) -> simulate twice (double-generation hash
equality incl. text) -> write tranche files + pins -> append to the working
side -> recompute cumulative coverage.

After the loop: rebuild the grouped fold plan and dev slice on the final
working set and refresh their manifests (the owning process updates its own
pins; git history records the change).
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.split import build_grouped_fold_plan, pick_dev_slice  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.audits import coverage_report  # noqa: E402
from cardessa.corpus import (  # noqa: E402
    TRANCHE_CAP_EPISODES,
    TRANCHE_SIZE_EXPANSION,
    generate_tranche,
)
from cardessa.engine import PayerEngine  # noqa: E402
from cardessa.expansion import (  # noqa: E402
    expansion_studies,
    extend_world,
    plan_allocation,
)
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from build_corpus import PromptCollector, warm_cache  # noqa: E402

RAW = ROOT / "data" / "raw" / "cardessa_sim"
WORKING = ROOT / "data" / "working" / "cardessa_sim"
DEV_SLICE_TARGET_STATES = 500


def cumulative_raw() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Tranche 0 plus every expansion tranche written so far."""
    episodes = [pl.read_parquet(RAW / "episodes.parquet")]
    states = [pl.read_parquet(RAW / "states.parquet")]
    for tranche_dir in sorted(RAW.glob("tranche_*")):
        episodes.append(pl.read_parquet(tranche_dir / "episodes.parquet"))
        states.append(pl.read_parquet(tranche_dir / "states.parquet"))
    return pl.concat(episodes, how="vertical_relaxed"), pl.concat(
        states, how="vertical_relaxed"
    )


def main() -> int:
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    provider = VllmProvider()  # raises unless the pinned 27B is being served
    print(f"vLLM serving pinned model: {provider.model_id}")
    text = CachedTextGenerator(provider=provider, cache_dir=ROOT / "data" / "text_cache")

    episodes, states = cumulative_raw()
    extra_studies_all: list[pl.DataFrame] = []
    tranche_log: list[dict] = []

    while True:
        coverage = coverage_report(states, episodes)
        print(f"coverage violations: {coverage['violations']}")
        if not coverage["violations"]:
            print("100x rule met; expansion complete")
            break
        if episodes.height >= TRANCHE_CAP_EPISODES:
            print(f"cap {TRANCHE_CAP_EPISODES} reached with violations "
                  f"{coverage['violations']}: they drop before training (schema "
                  "CoverageReport contract)")
            break

        tranche_no = len(tranche_log) + 1
        size = min(TRANCHE_SIZE_EXPANSION, TRANCHE_CAP_EPISODES - episodes.height)
        allocation = plan_allocation(states, episodes, size)
        print(f"tranche {tranche_no}: {size} episodes, allocation "
              f"{ {str(k): v for k, v in allocation.items()} }")

        extra = expansion_studies(
            world, MASTER_SEED, tranche_no,
            allocation, start_study_index=1000 + sum(e.height for e in extra_studies_all),
        )
        extra_studies_all.append(extra)
        world_ext = extend_world(world, pl.concat(extra_studies_all))
        start_index = episodes.height

        collector = PromptCollector()
        generate_tranche(world_ext, engine, MASTER_SEED, collector, start_index, size)
        warm_cache(collector.prompts, text)

        started = time.time()
        result = generate_tranche(world_ext, engine, MASTER_SEED, text, start_index, size)
        tranche_hash = result.content_sha256()
        second = generate_tranche(world_ext, engine, MASTER_SEED, text, start_index, size)
        double_ok = second.content_sha256() == tranche_hash
        print(f"tranche {tranche_no} simulated in {time.time() - started:.0f}s, "
              f"double-generation equal: {double_ok}")
        if not double_ok:
            raise SystemExit("tranche regeneration diverged; refusing to write")

        tranche_eps = pl.DataFrame(result.episodes_rows)
        tranche_sts = pl.DataFrame(result.states_rows)
        non_working = tranche_eps.filter(pl.col("split") != "working")
        if non_working.height:
            raise SystemExit("expansion produced non-working episodes; aborting")

        out_dir = RAW / f"tranche_{tranche_no}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tranche_eps.write_parquet(out_dir / "episodes.parquet")
        tranche_sts.write_parquet(out_dir / "states.parquet")
        pins = [
            {
                "path": f"data/raw/cardessa_sim/tranche_{tranche_no}/{name}",
                "sha256": sha256_file(out_dir / name),
            }
            for name in ("episodes.parquet", "states.parquet")
        ]
        (ROOT / "runs" / "manifests" / f"corpus_tranche_{tranche_no}.json").write_text(
            json.dumps(pins, indent=2), encoding="utf-8"
        )

        episodes = pl.concat([episodes, tranche_eps], how="vertical_relaxed")
        states = pl.concat([states, tranche_sts], how="vertical_relaxed")
        tranche_log.append(
            {
                "tranche": tranche_no,
                "n_episodes": tranche_eps.height,
                "n_states": tranche_sts.height,
                "allocation": {str(k): v for k, v in allocation.items()},
                "content_sha256": tranche_hash,
                "double_generation_equal": double_ok,
            }
        )

    # refresh the working side: original working episodes + all expansion
    working_eps = episodes.filter(pl.col("split") == "working")
    working_ids = set(working_eps["episode_id"].to_list())
    working_sts = states.filter(pl.col("episode_id").is_in(sorted(working_ids)))
    working_eps.write_parquet(WORKING / "episodes.parquet")
    working_sts.write_parquet(WORKING / "states.parquet")

    fold_plan = build_grouped_fold_plan(
        working_eps, example_col="episode_id", outcome_col="success",
        group_column="patient_id", n_folds=5, seed=MASTER_SEED,
    )
    dev_slice = pick_dev_slice(
        working_sts, example_col="episode_id", fold_plan=fold_plan,
        target_states=DEV_SLICE_TARGET_STATES, seed=MASTER_SEED + 8,
    )
    manifests = ROOT / "runs" / "manifests"
    (manifests / "fold_plan.json").write_text(
        fold_plan.model_dump_json(indent=2), encoding="utf-8"
    )
    (manifests / "dev_slice.json").write_text(
        json.dumps(dev_slice, indent=2), encoding="utf-8"
    )
    # the split-build report's side counts are owned by whoever last changed
    # the side files; expansion appends to working, so refresh them here
    report_path = ROOT / "runs" / "reports" / "split_build.json"
    split_report = json.loads(report_path.read_text(encoding="utf-8"))
    split_report["counts"]["working"] = {
        "examples": working_eps.height, "states": working_sts.height,
    }
    split_report["post_expansion"] = True
    report_path.write_text(
        json.dumps(split_report, indent=2), encoding="utf-8", newline="\n"
    )

    split_pins = json.loads((manifests / "split_files.json").read_text(encoding="utf-8"))
    for pin in split_pins:
        if pin["path"].startswith("data/working/"):
            pin["sha256"] = sha256_file(ROOT / pin["path"])
    (manifests / "split_files.json").write_text(
        json.dumps(split_pins, indent=2), encoding="utf-8"
    )

    final_coverage = coverage_report(states, episodes)
    (ROOT / "runs" / "reports" / "coverage_cumulative.json").write_text(
        json.dumps(final_coverage, indent=2), encoding="utf-8"
    )
    (ROOT / "runs" / "reports" / "expansion.json").write_text(
        json.dumps(
            {
                "milestone": "P5",
                "tranches": tranche_log,
                "total_episodes": episodes.height,
                "total_states": states.height,
                "working_episodes": working_eps.height,
                "working_states": working_sts.height,
                "cap": TRANCHE_CAP_EPISODES,
                "final_violations": final_coverage["violations"],
                "dropped_before_training": final_coverage["violations"],
                "llm_calls": text.calls_to_provider,
                "llm_model": provider.model_id,
                "note": (
                    "expansion targets working-side patients only; quarantine A "
                    "and reserve B are untouched"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"expansion done: {episodes.height} episodes total, "
          f"final violations {final_coverage['violations']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
