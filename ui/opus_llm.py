"""Claude Opus 4.8 client for the demo UI's LLM-agent lane.

Drop-in for the baseline repo's AgentLLM (same complete()/stats() surface,
same counters) but backed by the Anthropic API instead of the local vLLM.
Demo-only: selected via AMMONIX_LLM_LANE=opus in ui/server.py, writes to its
own cache dir, and never touches the pinned-Qwen caches. Note this turns the
lane into a model+architecture comparison, not the paper's same-model one.

The API key comes from ANTHROPIC_API_KEY or data/anthropic_key.txt (data/ is
gitignored, so the file never reaches a remote).
"""

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = os.environ.get("AMMONIX_OPUS_MODEL", "claude-opus-4-8")  # e.g. claude-opus-5;
# the model id is part of every cache key, so lanes never share decisions
# cache-key tag for the provider configuration; bump when the request shape
# changes so stale decisions are never replayed under a new configuration
CONFIG_TAG = "strong-v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_api_key(data_dir: Path) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        key_file = data_dir / "anthropic_key.txt"
        if key_file.is_file():
            # utf-8-sig: PowerShell writes a BOM that would corrupt the header
            key = key_file.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise RuntimeError(
            "Opus lane needs an Anthropic API key: set ANTHROPIC_API_KEY or "
            f"put the key in {data_dir / 'anthropic_key.txt'}"
        )
    return key


@dataclass
class OpusAgentLLM:
    """Anthropic chat client with the AgentLLM cache/counter contract."""

    cache_dir: Path
    max_tokens: int = 8000  # hard cap on thinking + answer together
    model_id: str = field(default=MODEL_ID, init=False)
    calls_to_provider: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        import anthropic

        self._lock = threading.Lock()
        self._client = anthropic.Anthropic(
            api_key=_load_api_key(self.cache_dir.parent)
        )

    def complete(self, messages: list[dict], schema: dict) -> str:
        """Schema-constrained completion; returns the raw JSON content string."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = _sha256(
            self.model_id
            + CONFIG_TAG
            + json.dumps(messages, sort_keys=True)
            + json.dumps(schema, sort_keys=True)
        )
        cached = self.cache_dir / f"{key}.json"
        if cached.is_file():
            entry = json.loads(cached.read_text(encoding="utf-8"))
            with self._lock:
                self.cache_hits += 1
                self.prompt_tokens += entry["usage"].get("prompt_tokens", 0)
                self.completion_tokens += entry["usage"].get("completion_tokens", 0)
            return entry["content"]

        system = "\n\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        chat = [m for m in messages if m.get("role") != "system"]
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system or None,
            messages=chat,
            # adaptive is NOT the default on Opus 4.8 — must be set explicitly
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        content = next(
            block.text for block in response.content if block.type == "text"
        ).strip()
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }
        cached.write_text(
            json.dumps({"content": content, "usage": usage}), encoding="utf-8"
        )
        with self._lock:
            self.calls_to_provider += 1
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]
        return content

    def stats(self) -> dict:
        return {
            "model_id": self.model_id,
            "calls_to_provider": self.calls_to_provider,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def build_strong_policy(llm):
    """AgenticPolicy with two feedback retries instead of the baseline's one.

    Mirrors agentic.policy.AgenticPolicy.decide with the attempt count
    parameterised; the baseline repo stays byte-identical (it is the
    pre-registered artefact). Imported lazily: Agentic_Baseline must already
    be on sys.path, which ui/server.py guarantees before importing us.
    """
    from agentic.briefing import SYSTEM_PROMPT, build_briefing, decision_schema
    from agentic.policy import AgenticPolicy, Decision, applicability_violation

    class StrongAgenticPolicy(AgenticPolicy):
        ATTEMPTS = 3  # initial + two feedback retries (baseline: initial + one)

        def decide(self, snap: dict, legal_actions: list[str]) -> Decision:
            self._count("decisions")
            allowed = [*legal_actions, "escalate"]
            schema = decision_schema(allowed)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_briefing(snap, legal_actions)},
            ]
            violations: list[str] = []
            for attempt in range(self.ATTEMPTS):
                raw = self.llm.complete(messages, schema)
                action, reasoning, problem = None, "", None
                try:
                    parsed = json.loads(raw)
                    reasoning = str(parsed.get("reasoning", ""))
                    action = parsed.get("action")
                    if action not in allowed:
                        problem = f"'{action}' is not one of the available actions"
                except (json.JSONDecodeError, AttributeError):
                    problem = "the reply was not valid JSON"
                if problem is None:
                    if action == "escalate":
                        self._count("escalations_explicit")
                        return Decision(None, reasoning, attempt > 0, violations)
                    problem = applicability_violation(action, snap)
                    if problem is None:
                        return Decision(action, reasoning, attempt > 0, violations)
                violations.append(f"{action}: {problem}")
                if attempt < self.ATTEMPTS - 1:
                    self._count("retries")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"That action is not applicable: {problem}. "
                                "Choose a different action from the available list."
                            ),
                        }
                    )
            self._count("escalations_forced")
            return Decision(None, "", True, violations)

    return StrongAgenticPolicy(llm)
