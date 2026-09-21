"""v0.3 spec 5.1: targeted expansion - double the retro-family volume.

Two tranches of 500 with explicit allocations (not deficit-driven: the
100-floor is already met; this is signal-density work for the weak retro
classifier). Same machinery and discipline as expand_corpus: tranche files
pinned, working side appended, fold plan + dev slice + pins refreshed,
split-build counts updated, coverage report rewritten.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.split import build_grouped_fold_plan, pick_dev_slice  # noqa: E402
from build_corpus import PromptCollector, warm_cache  # noqa: E402
from expand_corpus import cumulative_raw  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.audits import coverage_report  # noqa: E402
from cardessa.corpus import generate_tranche  # noqa: E402
from cardessa.engine import PayerEngine  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

RAW = ROOT / "data" / "raw" / "cardessa_sim"
WORKING = ROOT / "data" / "working" / "cardessa_sim"
V3_TRANCHES = {
    7: {"retro_auth": 240, "p2p": 60, None: 200},
    8: {"retro_auth": 240, "p2p": 60, None: 200},
}


def main() -> int:
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    provider = VllmProvider()
    print(f"vLLM serving pinned model: {provider.model_id}")
    text = CachedTextGenerator(provider=provider, cache_dir=ROOT / "data" / "text_cache")

    episodes, states = cumulative_raw()
    expansion_report = json.loads(
        (ROOT / "runs" / "reports" / "expansion.json").read_text(encoding="utf-8")
    )
    extra = []
    start = 1000
    for tranche in expansion_report["tranches"]:
        allocation = {
            (None if k == "None" else k): v for k, v in tranche["allocation"].items()
        }
        studies = expansion_studies(world, MASTER_SEED, tranche["tranche"], allocation, start)
        extra.append(studies)
        start += studies.height
    world_ext = extend_world(world, pl.concat(extra))
    assert world_ext.studies.height == episodes.height, "lineage replay mismatch"

    for tranche_no, allocation in V3_TRANCHES.items():
        size = sum(allocation.values())
        print(f"tranche {tranche_no}: {size} episodes, allocation "
              f"{ {str(k): v for k, v in allocation.items()} }")
        extra_studies = expansion_studies(
            world_ext, MASTER_SEED, tranche_no, allocation, world_ext.studies.height
        )
        world_ext = extend_world(world_ext, extra_studies)
        start_index = episodes.height

        collector = PromptCollector()
        generate_tranche(world_ext, engine, MASTER_SEED, collector, start_index, size)
        warm_cache(collector.prompts, text, workers=6)
        started = time.time()
        result = generate_tranche(world_ext, engine, MASTER_SEED, text, start_index, size)
        second = generate_tranche(world_ext, engine, MASTER_SEED, text, start_index, size)
        if second.content_sha256() != result.content_sha256():
            raise SystemExit("tranche regeneration diverged; refusing to write")
        print(f"tranche {tranche_no} simulated in {time.time() - started:.0f}s, "
              "double-gen equal")

        tranche_eps = pl.DataFrame(result.episodes_rows)
        tranche_sts = pl.DataFrame(result.states_rows)
        if tranche_eps.filter(pl.col("split") != "working").height:
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
            json.dumps(pins, indent=2), encoding="utf-8", newline="\n"
        )
        episodes = pl.concat([episodes, tranche_eps], how="vertical_relaxed")
        states = pl.concat([states, tranche_sts], how="vertical_relaxed")
        expansion_report["tranches"].append(
            {
                "tranche": tranche_no,
                "n_episodes": tranche_eps.height,
                "n_states": tranche_sts.height,
                "allocation": {str(k): v for k, v in allocation.items()},
                "content_sha256": result.content_sha256(),
                "double_generation_equal": True,
            }
        )

    working_eps = episodes.filter(pl.col("split") == "working")
    working_sts = states.filter(
        pl.col("episode_id").is_in(sorted(set(working_eps["episode_id"].to_list())))
    )
    working_eps.write_parquet(WORKING / "episodes.parquet")
    working_sts.write_parquet(WORKING / "states.parquet")

    fold_plan = build_grouped_fold_plan(
        working_eps, example_col="episode_id", outcome_col="success",
        group_column="patient_id", n_folds=5, seed=MASTER_SEED,
    )
    dev_slice = pick_dev_slice(
        working_sts, example_col="episode_id", fold_plan=fold_plan,
        target_states=500, seed=MASTER_SEED + 8,
    )
    manifests = ROOT / "runs" / "manifests"
    (manifests / "fold_plan.json").write_text(
        fold_plan.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    (manifests / "dev_slice.json").write_text(
        json.dumps(dev_slice, indent=2), encoding="utf-8", newline="\n"
    )
    split_report = json.loads(
        (ROOT / "runs" / "reports" / "split_build.json").read_text(encoding="utf-8")
    )
    split_report["counts"]["working"] = {
        "examples": working_eps.height, "states": working_sts.height,
    }
    split_report["post_expansion"] = True
    (ROOT / "runs" / "reports" / "split_build.json").write_text(
        json.dumps(split_report, indent=2), encoding="utf-8", newline="\n"
    )
    split_pins = json.loads((manifests / "split_files.json").read_text(encoding="utf-8"))
    for pin in split_pins:
        if pin["path"].startswith("data/working/"):
            pin["sha256"] = sha256_file(ROOT / pin["path"])
    (manifests / "split_files.json").write_text(
        json.dumps(split_pins, indent=2), encoding="utf-8", newline="\n"
    )

    expansion_report["total_episodes"] = episodes.height
    expansion_report["total_states"] = states.height
    expansion_report["working_episodes"] = working_eps.height
    expansion_report["working_states"] = working_sts.height
    expansion_report["cap"] = 8000
    coverage = coverage_report(states, episodes)
    expansion_report["final_violations"] = coverage["violations"]
    (ROOT / "runs" / "reports" / "expansion.json").write_text(
        json.dumps(expansion_report, indent=2), encoding="utf-8", newline="\n"
    )
    (ROOT / "runs" / "reports" / "coverage_cumulative.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8", newline="\n"
    )
    print(
        f"v3 expansion done: {episodes.height} episodes; "
        f"working {working_eps.height}/{working_sts.height}; "
        f"violations {coverage['violations']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
