"""The KEPT tokenizer, resolved from the round state (v0.3).

Every downstream consumer (swarm, resolution heads, basis, harness, UI,
rollout, final eval) must build features with whatever the tokenizer loop
KEPT - not a hardcoded round. The kept round's code pins are re-verified
against the sources on every call.
"""

import json
from pathlib import Path

import polars as pl
from ammonix_core.hashing import sha256_file, sha256_text

from cardessa.features import build_round1_features
from cardessa.features_rounds import build_with_blocks

ROUND_BLOCK = {3: "derived", 4: "last_event"}


def kept_blocks(root: Path) -> tuple[set[str], int]:
    rounds = json.loads(
        (root / "runs" / "state" / "tokenizer_rounds.json").read_text(encoding="utf-8")
    )
    blocks = {
        ROUND_BLOCK[r["round"]]
        for r in rounds
        if r["kept"] and r["round"] in ROUND_BLOCK
    }
    last_kept = [r["round"] for r in rounds if r["kept"]][-1]
    return blocks, last_kept


def verify_pin(root: Path, last_kept: int) -> str:
    manifest = json.loads(
        (root / "runs" / "manifests" / f"tokenizer_round{last_kept}.json").read_text(
            encoding="utf-8"
        )
    )
    pin = manifest["pin"]["code_sha256"]
    files = [root / "cardessa" / "features.py"]
    if last_kept >= 3:
        files.append(root / "cardessa" / "features_rounds.py")
    hashes = [sha256_file(f) for f in files]
    acceptable = {hashes[0], sha256_text("".join(hashes))}
    if pin not in acceptable:
        raise SystemExit(
            f"feature code no longer matches the kept tokenizer pin (round {last_kept})"
        )
    return pin


def build_kept_features(root: Path, states: pl.DataFrame):
    """(frame, feature_names, kept_round) under the kept tokenizer."""
    blocks, last_kept = kept_blocks(root)
    pin = verify_pin(root, last_kept)
    if blocks:
        frame, specs = build_with_blocks(states, blocks)
    else:
        frame, specs = build_round1_features(states)
    return frame, [s.name for s in specs], last_kept, pin
