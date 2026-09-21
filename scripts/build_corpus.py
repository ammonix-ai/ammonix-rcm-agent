"""`make corpus`: generate the initial tranche of 1000 episodes.

Requires the pinned Qwen3.6-27B served by vLLM (the text provider refuses
anything else). Text generations are cached under data/text_cache/ keyed by
prompt hash, which is what makes double generation hash-identical including
the text channel.

Writes data/raw/cardessa_sim/{episodes,states}.parquet, the tranche audit
and cumulative coverage report, and determinism pins.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.audits import (  # noqa: E402
    audit_families,
    audit_p1_structure,
    audit_poisoned_column,
    audit_shape,
    coverage_report,
)
from cardessa.corpus import TRANCHE_SIZE_INITIAL, generate_tranche, write_tranche  # noqa: E402
from cardessa.engine import PayerEngine  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402
from cardessa.world import generate_world  # noqa: E402


class PromptCollector:
    """Dry-pass provider: records every prompt, emits placeholders, never
    touches the LLM or the real cache. The simulation's decisions do not
    depend on text, so the collected prompt set is exactly what the real
    pass will ask for."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.cache_dir = None

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return ""


def warm_cache(prompts: list[str], text: CachedTextGenerator, workers: int = 12) -> None:
    unique = list(dict.fromkeys(prompts))
    done = 0
    started = time.time()

    def one(prompt: str) -> None:
        nonlocal done
        text.generate(prompt)
        done += 1
        if done % 200 == 0:
            rate = done / max(1.0, time.time() - started)
            remaining = (len(unique) - done) / max(rate, 1e-6)
            print(
                f"  warmed {done}/{len(unique)} prompts "
                f"(~{remaining / 60:.0f} min remaining)",
                flush=True,
            )

    print(f"warming text cache: {len(unique)} unique prompts, {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, unique))


def main() -> int:
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    provider = VllmProvider()  # raises unless the pinned 27B is being served
    print(f"vLLM serving pinned model: {provider.model_id}")
    text = CachedTextGenerator(provider=provider, cache_dir=ROOT / "data" / "text_cache")

    collector = PromptCollector()
    print("dry pass: collecting prompts...")
    generate_tranche(world, engine, MASTER_SEED, collector, 0, TRANCHE_SIZE_INITIAL)
    warm_cache(collector.prompts, text)
    # v0.2 spec 4.2: pin the prompt-hash inventory so cache completeness is
    # verifiable before any regeneration-dependent step (final eval hard gate)
    from ammonix_core.hashing import sha256_text as _sha
    inventory = sorted({_sha(prompt) for prompt in collector.prompts})
    (ROOT / "runs" / "manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "manifests" / "text_cache_tranche_0.json").write_text(
        json.dumps({"tranche": 0, "n_unique_prompts": len(inventory),
                    "prompt_sha256s": inventory}, indent=2),
        encoding="utf-8", newline="\n",
    )

    started = time.time()
    print(f"simulating initial tranche: {TRANCHE_SIZE_INITIAL} episodes...")
    result = generate_tranche(world, engine, MASTER_SEED, text, 0, TRANCHE_SIZE_INITIAL)
    tranche_hash = result.content_sha256()
    print(
        f"tranche done in {time.time() - started:.0f}s; "
        f"{len(result.states)} states, {text.calls_to_provider} LLM calls, "
        f"hash {tranche_hash[:12]}..."
    )

    print("regenerating for double-generation equality (cache-backed text)...")
    second = generate_tranche(world, engine, MASTER_SEED, text, 0, TRANCHE_SIZE_INITIAL)
    double_ok = second.content_sha256() == tranche_hash
    print(f"double-generation hash equality (incl. text): {double_ok}")

    out_dir = ROOT / "data" / "raw" / "cardessa_sim"
    file_hashes = write_tranche(result, out_dir)
    episodes = pl.DataFrame(result.episodes_rows)
    states = pl.DataFrame(result.states_rows)

    audits = [
        audit_shape(episodes, states),
        audit_families(episodes),
        audit_p1_structure(episodes, states),
        audit_poisoned_column(states),
    ]
    coverage = coverage_report(states, episodes)

    reports = ROOT / "runs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "corpus_tranche_0.json").write_text(
        json.dumps(
            {
                "milestone": "W2",
                "tranche": 0,
                "n_episodes": episodes.height,
                "n_states": states.height,
                "tranche_content_sha256": tranche_hash,
                "double_generation_equal": double_ok,
                "llm_calls": text.calls_to_provider,
                "llm_model": provider.model_id,
                "audits": {a.name: {"passed": a.passed, **a.details} for a in audits},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports / "coverage_cumulative.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    manifests = ROOT / "runs" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "corpus_tranche_0.json").write_text(
        json.dumps(
            [
                {"path": f"data/raw/cardessa_sim/{n}", "sha256": h}
                for n, h in file_hashes.items()
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    for audit in audits:
        print(f"audit {audit.name}: {'PASS' if audit.passed else 'FAIL'} {audit.details}")
    print(f"coverage violations (working data): {coverage['violations']}")
    print("reports written: corpus_tranche_0.json, coverage_cumulative.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
