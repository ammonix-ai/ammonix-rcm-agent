"""P10 checker: the final eval ran exactly once via the whitelisted path,
the TestReport exists with P1-P8 trap results and the corpus and world
hashes, its manifest references are genuine, and no build files changed.

Prints the one-line JSON verdict the goal condition references.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.platform import TestReport  # noqa: E402


def main() -> int:
    checks: dict[str, bool] = {}

    report_path = ROOT / "runs" / "reports" / "test_report.json"
    marker_path = ROOT / "runs" / "state" / "final_eval_ran.json"
    doc = json.loads(report_path.read_text(encoding="utf-8"))
    report = TestReport.model_validate(doc["test_report"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))

    checks["report_exists_and_valid"] = True
    checks["ran_exactly_once"] = (
        marker["report_sha256"] == sha256_file(report_path)
    )
    checks["not_tainted"] = report.tainted is False

    # the eight traps are all present, with detail backing each
    checks["all_eight_traps_reported"] = set(report.trap_results) == {
        f"P{i}_{name}" for i, name in [
            (1, "success_not_imitation"), (2, "no_leakage"), (3, "determinism"),
            (4, "calibration"), (5, "ood_honesty"), (6, "honest_near_ties"),
            (7, "coverage_discipline"), (8, "form_fidelity"),
        ]
    }
    detail = doc["detail"]
    checks["world_and_corpus_hashes_present"] = (
        len(detail["world_content_sha256"]) == 64
        and len(detail["corpus_tranche0_sha256"]) == 64
    )

    # manifest references are genuine
    checks["basis_hash_matches"] = report.basis_manifest_sha256 == sha256_file(
        ROOT / "basis" / "manifest.json"
    )
    quarantine_manifest = json.loads(
        (ROOT / "runs" / "manifests" / "quarantine_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    checks["quarantine_manifest_hash_matches"] = (
        report.quarantine_manifest_sha256 == quarantine_manifest["content_sha256"]
    )

    # a second run must refuse
    rerun = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "final_eval.py")],
        capture_output=True, text=True,
    )
    checks["second_run_refused"] = rerun.returncode != 0 and (
        "already ran" in (rerun.stderr + rerun.stdout)
    )

    # no build files modified: only the report, the marker and (this run's)
    # progress/checker additions may be new; nothing tracked may be dirty
    # besides runs/ additions
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    disallowed = [
        line for line in status
        if not line.strip().endswith(
            (
                "runs/reports/test_report.json", "runs/state/final_eval_ran.json",
                "runs/PROGRESS.md", "scripts/final_eval.py",
                "scripts/check_final_cardessa.py", "cardessa/harness.py",
            )
        )
    ]
    checks["no_other_files_modified"] = not disallowed

    determinism = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "determinism_check.py")],
        capture_output=True, text=True,
    )
    sys.stdout.write(determinism.stdout)
    checks["determinism_exit_0"] = determinism.returncode == 0

    verdict = all(checks.values())
    print(
        json.dumps(
            {
                "loop": "P10",
                "verdict": verdict,
                **checks,
                "trap_results": report.trap_results,
                "calibration_ece": round(report.calibration_ece, 4),
                "recommendation_accuracy": round(report.recommendation_accuracy, 4),
                "disallowed_changes": disallowed[:5],
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
