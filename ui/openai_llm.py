"""GPT-5.6 Sol client for the demo UI's LLM-agent lane.

Drop-in for the baseline repo's AgentLLM, backed by the OpenAI API. Same
demo-only status and caveats as ui/opus_llm.py (model+architecture
comparison, not the paper's pre-registered same-model one). Selected via
AMMONIX_LLM_LANE=sol in ui/server.py; own cache dir, pinned caches untouched.

API key: OPENAI_API_KEY or data/openai_key.txt (data/ is gitignored).
"""

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "gpt-5.6-sol"
BASE_URL = "https://api.openai.com/v1"
# medium reasoning_effort is the model default; part of the cache key
CONFIG_TAG = "sol-strong-v1"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_api_key(data_dir: Path) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        key_file = data_dir / "openai_key.txt"
        if key_file.is_file():
            # utf-8-sig: PowerShell writes a BOM that would corrupt the header
            key = key_file.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise RuntimeError(
            "Sol lane needs an OpenAI API key: set OPENAI_API_KEY or put the "
            f"key in {data_dir / 'openai_key.txt'}"
        )
    return key


@dataclass
class SolAgentLLM:
    """OpenAI chat client with the AgentLLM cache/counter contract."""

    cache_dir: Path
    max_completion_tokens: int = 8000  # reasoning + answer together
    model_id: str = MODEL_ID  # the model id is part of the cache key
    calls_to_provider: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._api_key = ""  # resolved on the first cache miss (keyless replay works)
        self._tls = threading.local()

    # ── per-decision usage capture (thread-local) ──
    # A policy's decide() may issue several complete() calls (feedback
    # retries) on its worker thread. The lane loop brackets each decide with
    # begin/take to attribute every one of those calls - cache hits included,
    # since the cached entry keeps the original usage - to that one move.

    def begin_call_capture(self) -> None:
        self._tls.capture = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def take_call_capture(self) -> dict | None:
        return self._tls.__dict__.pop("capture", None)

    def _record_capture(self, usage: dict) -> None:
        cap = getattr(self._tls, "capture", None)
        if cap is not None:
            cap["prompt_tokens"] += usage.get("prompt_tokens", 0)
            cap["completion_tokens"] += usage.get("completion_tokens", 0)
            cap["calls"] += 1

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
            self._record_capture(entry["usage"])
            return entry["content"]

        if not self._api_key:
            self._api_key = _load_api_key(self.cache_dir.parent)
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "max_completion_tokens": self.max_completion_tokens,
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
                    f"OpenAI rejected the request ({err.code}): {detail}"
                ) from err
            except (TimeoutError, OSError, ValueError, KeyError, TypeError, IndexError) as err:
                last_error = repr(err)
        if content is None:
            raise RuntimeError(f"Sol call failed after 3 attempts: {last_error}")
        content = content.strip()
        usage = body.get("usage") or {}
        cached.write_text(
            json.dumps({"content": content, "usage": usage}), encoding="utf-8"
        )
        with self._lock:
            self.calls_to_provider += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
        self._record_capture(usage)
        return content

    def stats(self) -> dict:
        return {
            "model_id": self.model_id,
            "calls_to_provider": self.calls_to_provider,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
