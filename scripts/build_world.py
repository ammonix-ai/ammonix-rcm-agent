"""`make world`: generate the Cardessa world snapshot from the master seed.

Writes data/world/ (payers.yaml, clinics/patients/coverage/studies parquet),
pins the file hashes in runs/manifests/world.json, and records the snapshot
content hash in runs/reports/world_build.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.world import generate_world, write_world  # noqa: E402


def main() -> int:
    world = generate_world(MASTER_SEED)
    content_hash = world.content_sha256()
    double_ok = generate_world(MASTER_SEED).content_sha256() == content_hash
    print(f"world generated from master seed {MASTER_SEED}")
    print(f"double-generation hash equality: {double_ok} ({content_hash[:12]}...)")

    out_dir = ROOT / "data" / "world"
    file_hashes = write_world(world, out_dir)
    for name, digest in file_hashes.items():
        print(f"  {name}: {digest[:12]}...")

    manifests = ROOT / "runs" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "world.json").write_text(
        json.dumps(
            [{"path": f"data/world/{n}", "sha256": h} for n, h in file_hashes.items()],
            indent=2,
        ),
        encoding="utf-8",
    )
    reports = ROOT / "runs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "world_build.json").write_text(
        json.dumps(
            {
                "milestone": "W1",
                "master_seed": MASTER_SEED,
                "world_content_sha256": content_hash,
                "double_generation_equal": double_ok,
                "n_payers": len(world.payers),
                "n_holdout_payers": sum(p.holdout for p in world.payers),
                "n_clinics": world.clinics.height,
                "n_patients": world.patients.height,
                "n_studies": world.studies.height,
                "split_counts": dict(world.patients.group_by("split").len().rows()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("world snapshot written to data/world/; pins in runs/manifests/world.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
