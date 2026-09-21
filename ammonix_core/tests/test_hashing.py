"""Hashing helpers: determinism is the whole point."""

import pytest

from ammonix_core.hashing import (
    canonical_json,
    dataset_fingerprint,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
)


def test_sha256_known_vector():
    # SHA-256 of the empty string, the classic test vector
    assert (
        sha256_bytes(b"")
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert sha256_text("") == sha256_bytes(b"")


def test_sha256_json_is_key_order_invariant():
    a = {"payer": "Meridian Health", "cpt": "93229", "auth": True}
    b = {"auth": True, "cpt": "93229", "payer": "Meridian Health"}
    assert sha256_json(a) == sha256_json(b)


def test_sha256_json_distinguishes_values():
    assert sha256_json({"x": 1}) != sha256_json({"x": 2})


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sha256_file_matches_bytes(tmp_path):
    payload = b"world snapshot bytes"
    target = tmp_path / "snapshot.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == sha256_bytes(payload)


def test_dataset_fingerprint_is_order_invariant():
    rows = [
        ("e2", ["s3", "s4"], False),
        ("e1", ["s1", "s2"], True),
    ]
    assert dataset_fingerprint(rows) == dataset_fingerprint(list(reversed(rows)))


def test_dataset_fingerprint_sees_outcome_changes():
    base = [("e1", ["s1"], True)]
    flipped = [("e1", ["s1"], False)]
    assert dataset_fingerprint(base) != dataset_fingerprint(flipped)
