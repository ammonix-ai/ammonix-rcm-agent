"""Sol prompt/retrieval tuning on the 200-claim demo scoreboard (tranche 90).

Limitations item (3), run BEFORE the tranche-101/102 pre-registration so the
registration names one tuned Sol configuration. Tuning data is the demo
scoreboard draw only (tranche 90, the UI's claims); no sealed or fresh
evaluation tranche is touched.

Stage 1: three system-prompt variants at k=25.
Stage 2: the stage-1 winner at k=10 and k=50.
Winner: success_rate_v2, then payer_collected_capped, then fewer mistake
episodes. Each configuration uses its own decision cache directory.

Usage: <factory>/.venv/Scripts/python.exe scripts/tune_sol_demo.py [--stage 1|2|report]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FACTORY_ROOT = Path(
    os.environ.get(
        "AMMONIX_FACTORY_ROOT",
        BASE.parent,  # co-located: baseline lives inside the factory repo
    )
)
for entry in (str(BASE), str(FACTORY_ROOT), str(FACTORY_ROOT / "ammonix_core"),
              str(FACTORY_ROOT / "scripts")):
    sys.path.insert(0, entry)

import run_comparison as rc  # noqa: E402  (the registered play()/summary())
from cardessa import MASTER_SEED  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.livecases import demo_allocation, training_world  # noqa: E402

from agentic.briefing import SYSTEM_PROMPT  # noqa: E402
from agentic.rag_policy import RagPolicy  # noqa: E402

DEMO_TRANCHE = 90  # the UI demo scoreboard draw (livecases.DEMO_TRANCHE_NO)
N_EPISODES = 200
REPORT = BASE / "runs" / "reports" / "sol_tuning_demo90.json"

# P0: the registered v6/rag-arm prompt, unchanged (control).
# P1: corrected success threshold (the registered text says 98%; the code's
#     SUCCESS_FRACTION has been 0.9 throughout - the t99 registration's
#     amendment records this) + explicit close-out guidance targeting the
#     documented write-off reflex.
# P2: P1 + payer-specific weighting of the retrieval tally, targeting the
#     documented payer-specific rework misses.
P1 = SYSTEM_PROMPT.replace("at least 98%", "at least 90%") + (
    "\nWhen the payer path is exhausted, money the patient or a secondary "
    "plan properly owes should be billed, not written off: write off only "
    "when nothing is owed or nothing is collectible."
)
P2 = P1 + (
    "\nPayers differ in which rework succeeds. When several rework actions "
    "are applicable, weight the SIMILAR PAST CLAIMS tally heavily: prefer "
    "the action that actually got similar claims paid at this payer."
)
PROMPTS = {"P0": SYSTEM_PROMPT, "P1": P1, "P2": P2}

STAGE1 = [("P0", 25), ("P1", 25), ("P2", 25)]


def build_world():
    world_ext, engine = training_world(FACTORY_ROOT, MASTER_SEED)
    start = world_ext.studies.height
    studies = expansion_studies(
        world_ext, MASTER_SEED, DEMO_TRANCHE, demo_allocation(), start
    )
    world = extend_world(world_ext, studies)
    indices = list(range(start, start + N_EPISODES))
    return world, engine, indices


def run_config(name: str, prompt_key: str, k: int, world, engine, indices) -> dict:
    from ui.openai_llm import SolAgentLLM

    llm = SolAgentLLM(cache_dir=FACTORY_ROOT / "data" / f"sol_tune_{name}")
    policy = RagPolicy(llm, FACTORY_ROOT, k=k, system_prompt=PROMPTS[prompt_key])
    started = time.perf_counter()
    results = rc.play(f"sol_tune_{name}", world, engine, indices,
                      agentic_policy=policy, route_step=None)
    s = rc.summary(results)
    s["wall_seconds"] = round(time.perf_counter() - started, 1)
    s["prompt_variant"] = prompt_key
    s["retrieval_k"] = k
    s["policy"] = policy.stats()
    s["llm"] = llm.stats()
    (BASE / "runs" / "reports" / f"episodes_sol_tune_{name}.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8", newline="\n"
    )
    return s


def key(s: dict):
    return (s["success_rate_v2"], s["payer_collected_capped"],
            -s["episodes_with_mistakes"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="1", choices=["1", "2", "report"])
    args = parser.parse_args()

    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else {
        "tuning_data": f"demo scoreboard draw only (tranche {DEMO_TRANCHE}, "
                       f"{N_EPISODES} episodes); no evaluation tranche touched",
        "prompts": {k: v for k, v in PROMPTS.items()},
        "configs": {},
    }

    if args.stage in ("1", "2"):
        world, engine, indices = build_world()
        if args.stage == "1":
            todo = [(f"{p}_k{k}", p, k) for p, k in STAGE1]
        else:
            stage1 = {n: s for n, s in report["configs"].items()
                      if n.endswith("_k25")}
            if len(stage1) < 3:
                print("run --stage 1 first")
                return 1
            best = max(stage1.values(), key=key)["prompt_variant"]
            todo = [(f"{best}_k{k}", best, k) for k in (10, 50)]
        for name, p, k in todo:
            if name in report["configs"]:
                print(f"skip {name}: already run")
                continue
            print(f"running {name} (prompt {p}, k={k})...")
            report["configs"][name] = run_config(name, p, k, world, engine, indices)
            REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8",
                              newline="\n")
            print(json.dumps({m: report["configs"][name][m] for m in (
                "success_rate_v2", "payer_collected_capped",
                "episodes_with_mistakes", "wall_seconds")}, indent=1))

    if report["configs"]:
        winner = max(report["configs"].items(), key=lambda kv: key(kv[1]))
        report["winner"] = {
            "config": winner[0],
            "prompt_variant": winner[1]["prompt_variant"],
            "retrieval_k": winner[1]["retrieval_k"],
            "rule": "success_rate_v2, then payer_collected_capped, then fewer "
                    "mistake episodes",
        }
        REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8",
                          newline="\n")
        print("winner so far:", json.dumps(report["winner"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
