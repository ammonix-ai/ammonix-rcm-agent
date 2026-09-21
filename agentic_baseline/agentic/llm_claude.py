"""Claude (Anthropic API) client for the frontier-model agent arm.

Same interface and cache discipline as the local AgentLLM: complete(messages,
schema) -> raw JSON string, prompt-hash disk cache with token usage preserved.
Differences, recorded honestly in the report: Sonnet 5 rejects sampling
parameters (no temperature=0), so API responses are not strictly
deterministic - the cache freezes whatever the first run returned, keeping
the ANALYSIS reproducible. Structured output is enforced server-side via
output_config.format json_schema.
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

import os as _os
CLAUDE_MODEL = _os.environ.get("AMMONIX_CLAUDE_MODEL", "claude-sonnet-5")  # e.g. claude-opus-5


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """The policy builds OpenAI-style messages; Anthropic takes system apart."""
    system = ""
    rest = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        else:
            rest.append(message)
    return system, rest


@dataclass
class ClaudeLLM:
    """Anthropic Messages API client with schema-constrained JSON output."""

    cache_dir: Path
    model: str = CLAUDE_MODEL
    max_tokens: int = 4000  # adaptive thinking spends from this budget too
    calls_to_provider: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    client: anthropic.Anthropic = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.client = anthropic.Anthropic()  # key from env / stored profile

    @property
    def model_id(self) -> str:
        return self.model

    def complete(self, messages: list[dict], schema: dict) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = sha256_text(
            self.model
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
        system, rest = split_system(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=rest,
            thinking={"type": "disabled"},  # parity with the local guided-JSON
            # arm: same agent loop, structured decision only, no extended reasoning
            output_config={
                "format": {"type": "json_schema", "schema": schema}
            },
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude refused the request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Claude output truncated (stop_reason=max_tokens)")
        content = next(
            (block.text for block in response.content if block.type == "text"), ""
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
            "model_id": self.model,
            "calls_to_provider": self.calls_to_provider,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
