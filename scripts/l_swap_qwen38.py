"""The registered Qwen-seat model swap, L-swap leg
(comparison_qwen38_model_swap): the plain-Qwen writer lane re-runs with
Qwen3.8-27B-AWQ-INT4 in the writer seat on the same 200 demonstration
claims / 525 decision points. The trained-9B, base-9B and Opus lanes and
the demo decisions are untouched.

Additive wrapper: l_swap_experiment is not modified. The 3.8 writer runs
through the same LocalJsonLLM contract as the other swapped writers
(temperature 0, seed 0, JSON-schema-constrained, thinking disabled),
with its own fresh cache (data/m1_qwen38) so no 3.6 draft is reused.

Usage:
  python scripts/l_swap_qwen38.py            # runs the qwen38 lane
  python scripts/l_swap_qwen38.py --compare  # all five lanes, state by state
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import l_swap_experiment as lse  # noqa: E402

QWEN38_MODEL = "/models/Qwen3.8-27B-AWQ-INT4"

lse.WRITERS = dict(lse.WRITERS)
lse.WRITERS["qwen38"] = QWEN38_MODEL

_orig_make_provider = lse.make_provider


def make_provider(model: str):
    if model == "qwen38":
        provider = lse.LocalJsonLLM(
            cache_dir=ROOT / "data" / "m1_qwen38",
            model_id=QWEN38_MODEL,
            base_url="http://127.0.0.1:8000/v1",
        )
        # The 3.8 container advertises the full path "/models/Qwen3.8-27B-AWQ-INT4"
        # as its served-model-name, but LocalJsonLLM.served_id matches on the
        # path tail and would reject it. The served name is exactly QWEN38_MODEL,
        # so pre-resolve it: complete() then sends that name and vLLM matches it.
        provider._served_id = QWEN38_MODEL

        class _M1Shim:
            """LocalJsonLLM speaks complete(messages, schema); the swap
            harness feeds generate(prompt, schema) like M1Provider."""

            model_id = QWEN38_MODEL

            def generate(self, prompt: str, schema: dict) -> str:
                out = provider.complete(
                    [{"role": "user", "content": prompt}], schema)
                self.calls_to_provider = provider.calls_to_provider
                return out

            calls_to_provider = 0

        shim = _M1Shim()
        return shim, lambda: {"model_id": QWEN38_MODEL,
                              "calls_to_provider": provider.calls_to_provider,
                              "cache_hits": provider.cache_hits}
    return _orig_make_provider(model)


lse.make_provider = make_provider

if __name__ == "__main__":
    if "--compare" in sys.argv[1:]:
        # same state-by-state diff as the base script, with the qwen38 lane
        # included next to the four original writers
        lse.compare()
        sys.exit(0)
    sys.argv = [sys.argv[0], "--model", "qwen38"] + sys.argv[1:]
    sys.exit(lse.main())
