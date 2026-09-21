"""The registered full Qwen 3.8 swap (comparison_qwen38_full_swap).

Re-runs the `agentic` and `agentic_rag` arms with Qwen3.8-27B-AWQ-INT4 in
BOTH the agent seat and the world-text channel, on the SAME sealed draws as
the published record. Qwen3.6 is retired: cardessa/textgen.py now pins 3.8,
so the one Qwen server serves both the text channel and the decisions and
there is no Qwen-to-Qwen swap within a step. The only rotation is
Qwen3.8 <-> the 9B writer, exactly as the original published pipeline ran.

The frozen historical corpus (~17,900 Qwen3.6 texts written before
2026-09-15) is read-only cache and is NOT regenerated. Only correspondence
for states not already in that corpus is generated, now by Qwen3.8.

The sealed baseline repo (agentic_baseline/) is NOT modified. Its decision
client pins the served model by substring; this wrapper overrides exactly
two things: the pin substring (to 3.8) and the decision cache directories
(fresh, so no 3.6 decision is reused).

Usage (per sealed draw, mirroring the published pipeline):
  python scripts/rollout_paperwork_qwen38.py --tranche 99 \
      --writer ornith_ft --arms agentic,agentic_rag \
      --report rollout_qwen38_t99.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts"),
              str(ROOT / "agentic_baseline")):
    sys.path.insert(0, entry)

import rollout_paperwork as rp  # noqa: E402
import agentic.llm as al  # noqa: E402

# the registered swap: cyankiwi/Qwen3.8-27B-AWQ-INT4
# revision 6e134bae811fb5adac50ee042ae5f029ac6779aa, served locally with
# the same vLLM flags as the 3.6 pin. textgen.py already pins 3.8 for the
# text channel; this line moves the decision client's pin to match.
al.PINNED_MODEL_SUBSTRING = "Qwen3.8-27B"

# one Qwen server (3.8) now serves both the text channel and the decisions
rp.SERVERS = dict(rp.SERVERS)
rp.SERVERS["qwen"] = ("ammonix-m1-38b", "http://127.0.0.1:8000/v1", "Qwen3.8")

_orig_make_baseline = rp.make_baseline

import threading  # noqa: E402

_RESOLVE_LOCK = threading.Lock()


class AgentLLM(al.AgentLLM):
    """The sealed client, with its served-model check moved from construction
    to the first cache miss. The cache key never includes the model id, so a
    run that is fully cached replays with no model server at all; the moment
    a decision is missing, the pin check runs exactly as in the sealed client
    and refuses any model other than the pinned one."""

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def complete(self, messages: list[dict], schema: dict) -> str:
        key = al.sha256_text(
            al.json.dumps(messages, sort_keys=True) + al.json.dumps(schema, sort_keys=True)
        )
        if not self.model_id and not (self.cache_dir / f"{key}.json").is_file():
            with _RESOLVE_LOCK:
                if not self.model_id:
                    counters_lock = self._lock
                    al.AgentLLM.__post_init__(self)
                    self._lock = counters_lock
        return super().complete(messages, schema)


def make_baseline(arm: str):
    from agentic.policy import AgenticPolicy

    if arm == "agentic":
        return AgenticPolicy(
            AgentLLM(cache_dir=rp.BASELINE / "cache" / "decisions_qwen38_full"))
    if arm == "agentic_rag":
        from agentic.rag_policy import RagPolicy

        return RagPolicy(
            AgentLLM(cache_dir=rp.BASELINE / "cache" / "decisions_rag_qwen38_full"),
            rp.ROOT)
    return _orig_make_baseline(arm)


rp.make_baseline = make_baseline

if __name__ == "__main__":
    sys.exit(rp.main())
