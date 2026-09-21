"""Content hashing: the reproducibility anchor for every artefact.

All hashes are SHA-256 hex digests. JSON hashing canonicalises (sorted keys,
compact separators, UTF-8) so semantically equal payloads hash identically
regardless of key order or formatting.
"""

import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: object) -> str:
    """Canonical JSON text: sorted keys, compact separators, no NaN."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(payload: object) -> str:
    return sha256_text(canonical_json(payload))


def dataset_fingerprint(example_rows: list[tuple[str, list[str], bool]]) -> str:
    """Hash over all Example and State ids plus Outcomes (BasisManifest field).

    Each row is (example_id, ordered state_ids, outcome_success). Rows are
    sorted by example_id so ingestion order does not affect the fingerprint.
    """
    canonical = sorted(
        (example_id, list(state_ids), bool(success))
        for example_id, state_ids, success in example_rows
    )
    return sha256_json(canonical)
