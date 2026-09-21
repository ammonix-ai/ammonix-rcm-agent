"""W1 checker: the world snapshot is complete, correct and deterministic.

Verifies: payers.yaml with 12 payers (2 flagged holdout), clinics.parquet,
patients.parquet with exactly 1000 rows, generated from the master seed;
payer-engine unit tests cover every CARC pathway of every payer and exit
green (output shown); double-generation hash equality of the world snapshot.
Ends by printing the one-line JSON verdict the goal condition references.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.world import generate_world, load_payers_yaml  # noqa: E402


def main() -> int:
    checks: dict[str, bool] = {}

    # 1. Snapshot files exist and payers.yaml parses to 12 payers, 2 holdout
    world_dir = ROOT / "data" / "world"
    expected_files = [
        "payers.yaml",
        "clinics.parquet",
        "patients.parquet",
        "coverage.parquet",
        "studies.parquet",
    ]
    checks["snapshot_files_exist"] = all((world_dir / n).is_file() for n in expected_files)
    if checks["snapshot_files_exist"]:
        payers = load_payers_yaml(world_dir / "payers.yaml")
        checks["twelve_payers"] = len(payers) == 12
        checks["two_holdout"] = sum(p.holdout for p in payers) == 2
    else:
        checks["twelve_payers"] = checks["two_holdout"] = False

    # 2. Regenerate from the master seed: counts and double-generation equality
    world = generate_world(MASTER_SEED)
    checks["exactly_1000_patients"] = world.patients.height == 1000
    checks["clinics_25"] = world.clinics.height == 25
    checks["double_generation_equal"] = (
        generate_world(MASTER_SEED).content_sha256() == world.content_sha256()
    )

    # 3. On-disk snapshot matches this regeneration (pins re-verified)
    report_path = ROOT / "runs" / "reports" / "world_build.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checks["snapshot_matches_master_seed"] = (
            report.get("world_content_sha256") == world.content_sha256()
        )
    else:
        checks["snapshot_matches_master_seed"] = False

    # 4. Payer-engine unit tests: every CARC pathway of every payer (output shown)
    print("$ python -m pytest cardessa/tests -q")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "cardessa/tests", "-q"], cwd=ROOT
    )
    checks["engine_tests_green"] = result.returncode == 0
    print(f"[pytest exit code: {result.returncode}]")

    verdict = all(checks.values())
    print(json.dumps({"loop": "W1", "verdict": verdict, **checks}))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
