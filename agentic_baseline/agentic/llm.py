"""Pinned-model vLLM client for the agentic baseline.

Same discipline as the factory's M1Provider: refuses any served model other
than the pinned Qwen3.6-27B, temperature 0, and response_format json_schema
strict (this vLLM SILENTLY ignores the legacy guided_json param — proven live
during the factory build). Prompt-hash disk cache; each cache file preserves
the token usage of the original call so cost accounting survives reruns.
"""

import hashlib
import json
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PINNED_MODEL_SUBSTRING = "Qwen3.6-27B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class AgentLLM:
    """OpenAI-compatible chat client, pinned-model-only, thread-safe counters."""

    cache_dir: Path
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = 500
    model_id: str = field(default="", init=False)
    calls_to_provider: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as response:
            served = json.loads(response.read())["data"]
        ids = [entry["id"] for entry in served]
        matches = [i for i in ids if PINNED_MODEL_SUBSTRING in i]
        if not matches:
            raise RuntimeError(
                f"vLLM at {self.base_url} serves {ids}, not the pinned model "
                f"(*{PINNED_MODEL_SUBSTRING}*). Refusing: the comparison is only "
                "fair on the same model that powers the Ammonix M1 layer."
            )
        self.model_id = matches[0]

    def complete(self, messages: list[dict], schema: dict) -> str:
        """Schema-constrained completion; returns the raw JSON content string."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = sha256_text(
            json.dumps(messages, sort_keys=True) + json.dumps(schema, sort_keys=True)
        )
        cached = self.cache_dir / f"{key}.json"
        if cached.is_file():
            entry = json.loads(cached.read_text(encoding="utf-8"))
            with self._lock:
                self.cache_hits += 1
                self.prompt_tokens += entry["usage"].get("prompt_tokens", 0)
                self.completion_tokens += entry["usage"].get("completion_tokens", 0)
            return entry["content"]
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "decision", "schema": schema, "strict": True},
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        content, last_error = None, None
        for attempt in range(3):  # transient stalls / malformed bodies: retry
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    body = json.loads(response.read())
                content = body["choices"][0]["message"]["content"]
                break
            except urllib.error.HTTPError as err:
                detail = err.read()[:300].decode("utf-8", "replace")
                if 400 <= err.code < 500:  # our request is wrong; retrying won't fix it
                    raise RuntimeError(f"vLLM rejected the request ({err.code}): {detail}") from err
                last_error = f"HTTP {err.code}: {detail}"
            except (TimeoutError, OSError, ValueError, KeyError, TypeError, IndexError) as err:
                last_error = repr(err)
        if content is None:
            raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        content = content.strip()
        usage = body.get("usage") or {}
        cached.write_text(
            json.dumps({"content": content, "usage": usage}), encoding="utf-8"
        )
        with self._lock:
            self.calls_to_provider += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
        return content

    def stats(self) -> dict:
        return {
            "model_id": self.model_id,
            "calls_to_provider": self.calls_to_provider,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
