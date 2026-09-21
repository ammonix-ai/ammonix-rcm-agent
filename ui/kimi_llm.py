"""Kimi K3 client for the demo UI's LLM-agent lane.

Drop-in for the baseline repo's AgentLLM, backed by Moonshot AI's
OpenAI-compatible API. Same demo-only status and caveats as ui/opus_llm.py
(model+architecture comparison, not the paper's pre-registered same-model
one). Selected via AMMONIX_LLM_LANE=kimi in ui/server.py; own cache dir,
pinned caches untouched.

API key: MOONSHOT_API_KEY or data/moonshot_key.txt (data/ is gitignored).
"""

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "kimi-k3"
BASE_URL = "https://api.moonshot.ai/v1"
# reasoning is always on for kimi-k3; effort left at the vendor default (max),
# which is why max_tokens is larger than the other lanes — reasoning_content
# and the answer share the same max_tokens budget
CONFIG_TAG = "kimi-strong-v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_api_key(data_dir: Path) -> str:
    key = os.environ.get("MOONSHOT_API_KEY", "").strip()
    if not key:
        key_file = data_dir / "moonshot_key.txt"
        if key_file.is_file():
            # utf-8-sig: PowerShell writes a BOM that would corrupt the header
            key = key_file.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise RuntimeError(
            "Kimi lane needs a Moonshot API key: set MOONSHOT_API_KEY or put "
            f"the key in {data_dir / 'moonshot_key.txt'}"
        )
    return key


@dataclass
class KimiAgentLLM:
    """Moonshot chat client with the AgentLLM cache/counter contract."""

    cache_dir: Path
    max_tokens: int = 16000  # reasoning_content + answer together
    model_id: str = field(default=MODEL_ID, init=False)
    calls_to_provider: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._api_key = _load_api_key(self.cache_dir.parent)

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

        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "decision", "schema": schema, "strict": True},
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        content, body, last_error = None, None, None
        for attempt in range(3):  # transient stalls: retry, same request
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    body = json.loads(response.read())
                content = body["choices"][0]["message"]["content"]
                break
            except urllib.error.HTTPError as err:
                detail = err.read()[:300].decode("utf-8", "replace")
                if err.code == 429 or err.code >= 500:
                    last_error = f"HTTP {err.code}: {detail}"
                    continue
                raise RuntimeError(
                    f"Moonshot rejected the request ({err.code}): {detail}"
                ) from err
            except (TimeoutError, OSError, ValueError, KeyError, TypeError, IndexError) as err:
                last_error = repr(err)
        if content is None:
            raise RuntimeError(f"Kimi call failed after 3 attempts: {last_error}")
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
