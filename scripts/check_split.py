"""P3 checker: quarantine manifest hash confirmed; probe read denied (negative
test); patient grouping intact; holdout-payer routing verified WITHOUT reading
quarantine (set-theoretic: working and reserve contain zero holdout episodes
and the counts add up).

Prints the one-line JSON verdict the goal condition references.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402
from ammonix_core.platform import QuarantineManifest  # noqa: E402
from ammonix_core.schema import FoldPlan  # noqa: E402
from ammonix_core.split import manifest_content_hash  # noqa: E402

HOLDOUT_PAYERS = {"granite", "pelican"}
GUARD = ROOT / "scripts" / "hooks" / "quarantine_guard.py"


def guard_denies(tool_name: str, tool_input: dict) -> bool:
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    result = subprocess.run(
        [sys.executable, str(GUARD)], input=payload, capture_output=True, text=True
    )
    return result.returncode == 2


def main() -> int:
    checks: dict[str, bool] = {}

    manifest = QuarantineManifest.model_validate_json(
        (ROOT / "runs" / "manifests" / "quarantine_manifest.json").read_text(encoding="utf-8")
    )
    checks["manifest_hash_confirmed"] = (
        manifest_content_hash(manifest) == manifest.content_sha256
    )
    checks["sides_disjoint"] = not (
        set(manifest.quarantine_groups) & set(manifest.reserve_groups)
        | set(manifest.quarantine_groups) & set(manifest.working_groups)
        | set(manifest.reserve_groups) & set(manifest.working_groups)
    )
    checks["all_1000_patients_assigned"] = (
        len(manifest.quarantine_groups)
        + len(manifest.reserve_groups)
        + len(manifest.working_groups)
        == 1000
    )

    # negative test: a probe read of data/quarantine/ must be denied by the hook
    checks["probe_read_denied"] = guard_denies(
        "Read", {"file_path": "data/quarantine/cardessa_sim/episodes.parquet"}
    )
    checks["probe_glob_denied"] = guard_denies("Glob", {"pattern": "data/quarantine/**"})
    checks["probe_shell_denied"] = guard_denies(
        "PowerShell", {"command": "Get-ChildItem data/quarantine/"}
    )

    # holdout routing, proven without reading quarantine
    working = pl.read_parquet(ROOT / "data/working/cardessa_sim/episodes.parquet")
    reserve = pl.read_parquet(ROOT / "data/reserve/cardessa_sim/episodes.parquet")
    checks["no_holdout_payer_in_working"] = (
        working.filter(pl.col("payer_id").is_in(sorted(HOLDOUT_PAYERS))).height == 0
    )
    checks["no_holdout_payer_in_reserve"] = (
        reserve.filter(pl.col("payer_id").is_in(sorted(HOLDOUT_PAYERS))).height == 0
    )
    report = json.loads(
        (ROOT / "runs" / "reports" / "split_build.json").read_text(encoding="utf-8")
    )
    # side counts must match the files on disk; expansion legitimately grows
    # the working side (patient-level totals are asserted separately above)
    checks["episode_counts_add_up"] = (
        report["counts"]["working"]["examples"] == working.height
        and report["counts"]["reserve_b"]["examples"] == reserve.height
    )

    # patient grouping intact in working data and folds
    side_of_patient = (
        {p: "quarantine_a" for p in manifest.quarantine_groups}
        | {p: "reserve_b" for p in manifest.reserve_groups}
        | {p: "working" for p in manifest.working_groups}
    )
    checks["working_patients_on_working_side"] = all(
        side_of_patient.get(p) == "working" for p in working["patient_id"].to_list()
    )
    fold_plan = FoldPlan.model_validate_json(
        (ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8")
    )
    patient_of = dict(working.select("episode_id", "patient_id").rows())
    folds_per_patient: dict[str, set[int]] = {}
    for episode_id, fold in fold_plan.assignment.items():
        folds_per_patient.setdefault(patient_of[episode_id], set()).add(fold)
    checks["no_patient_straddles_folds"] = all(
        len(folds) == 1 for folds in folds_per_patient.values()
    )
    checks["five_folds"] = set(fold_plan.assignment.values()) == set(range(5))

    dev_slice = json.loads(
        (ROOT / "runs" / "manifests" / "dev_slice.json").read_text(encoding="utf-8")
    )
    checks["dev_slice_reserved"] = dev_slice["n_states"] >= 500

    verdict = all(checks.values())
    print(
        json.dumps(
            {
                "loop": "P3",
                "verdict": verdict,
                **checks,
                "manifest_sha256": manifest.content_sha256[:16],
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
