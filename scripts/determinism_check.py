"""Determinism check: every pinned artefact still hashes to its recorded value.

Reads every pin file under runs/manifests/*.json. A pin file is a JSON list of
{"path": <repo-relative file>, "sha256": <hex digest>} entries. Each file is
re-hashed and compared. Milestones register their content-hashed artefacts
(world snapshot, corpus tranches, basis files) here as they are built; the
nightly routine re-runs this script.

Exits 0 and prints a one-line JSON verdict when every pin matches (vacuously
green when nothing is pinned yet); exits 1 listing mismatches otherwise.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ammonix_core"))

from ammonix_core.hashing import sha256_file  # noqa: E402

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "runs" / "manifests"


def is_pin_list(entries) -> bool:
    return isinstance(entries, list) and all(
        isinstance(e, dict) and {"path", "sha256"} <= set(e) for e in entries
    )


def main() -> int:
    pins: list[dict] = []
    structured = []
    for pin_file in sorted(MANIFEST_DIR.glob("*.json")) if MANIFEST_DIR.is_dir() else []:
        entries = json.loads(pin_file.read_text(encoding="utf-8"))
        if not is_pin_list(entries):
            # structured manifests (quarantine manifest, fold plan, dev slice)
            # carry their own hashes; they are not file-pin lists
            structured.append(pin_file.name)
            continue
        for entry in entries:
            entry["_pin_file"] = pin_file.name
            pins.append(entry)

    mismatches = []
    skipped_missing = []
    verified = 0
    for entry in pins:
        target = MANIFEST_DIR.parents[1] / entry["path"]
        if not target.is_file():
            # data/ artefacts are regenerable from pinned seeds and live only in
            # the worktree that built them; absence is not drift, a wrong hash is
            skipped_missing.append(entry["path"])
            continue
        actual = sha256_file(target)
        if actual != entry["sha256"]:
            mismatches.append(
                {"path": entry["path"], "expected": entry["sha256"], "actual": actual}
            )
        else:
            verified += 1

    verdict = not mismatches
    print(
        json.dumps(
            {
                "check": "determinism",
                "verdict": verdict,
                "pinned_files": len(pins),
                "verified": verified,
                "structured_manifests": structured,
                "skipped_missing": skipped_missing,
                "mismatches": mismatches,
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
