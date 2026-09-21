"""P5 checker: zero unmapped labels; the 100x rule met via staged expansion
(or the cap reached with residual violations formally dropped before
training); no semantic merges (else GATE-P5 approval required in
runs/APPROVALS.md); expansion stayed on the working side; fold plan intact.

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
from ammonix_core.schema import ActionMap, FoldPlan  # noqa: E402

from cardessa.audits import CANONICAL_ACTIONS, MIN_STATES_PER_ACTION  # noqa: E402

TRANCHE_CAP = 8000  # v0.3 spec section 5.1


def main() -> int:
    checks: dict[str, bool] = {}

    action_map = ActionMap.model_validate_json(
        (ROOT / "runs" / "manifests" / "action_map.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (ROOT / "runs" / "reports" / "action_map.json").read_text(encoding="utf-8")
    )
    expansion = json.loads(
        (ROOT / "runs" / "reports" / "expansion.json").read_text(encoding="utf-8")
    )

    # zero unmapped labels: every raw label in the cumulative corpus matches a rule
    raw = ROOT / "data" / "raw" / "cardessa_sim"
    states = [pl.read_parquet(raw / "states.parquet")]
    episodes = [pl.read_parquet(raw / "episodes.parquet")]
    for tranche_dir in sorted(raw.glob("tranche_*")):
        states.append(pl.read_parquet(tranche_dir / "states.parquet"))
        episodes.append(pl.read_parquet(tranche_dir / "episodes.parquet"))
    all_states = pl.concat(states, how="vertical_relaxed")
    all_episodes = pl.concat(episodes, how="vertical_relaxed")
    rule_patterns = {r.pattern: r.action_id for r in action_map.rules}
    inventory = set(all_states["action_raw"].unique().to_list())
    checks["zero_unmapped"] = all(label in rule_patterns for label in inventory)
    checks["rules_cover_all_canonical"] = set(rule_patterns.values()) == set(
        CANONICAL_ACTIONS
    )

    # no semantic merges -> the GATE-P5 approval is not triggered; any merge
    # (pattern != action_id) requires written approval in runs/APPROVALS.md
    merges = [r for r in action_map.rules if r.pattern != r.action_id]
    if merges:
        approvals = (ROOT / "runs" / "APPROVALS.md").read_text(encoding="utf-8")
        checks["merges_human_approved"] = all(
            f"{r.pattern} -> {r.action_id}" in approvals for r in merges
        )
    else:
        checks["no_merges_gate_not_triggered"] = report["merges"] == []

    # the 100x rule on working data, expansion trigger honoured
    working_ids = set(
        all_episodes.filter(pl.col("split") == "working")["episode_id"].to_list()
    )
    working = all_states.filter(pl.col("episode_id").is_in(sorted(working_ids)))
    counts = dict(working.group_by("action_raw").len().rows())
    violations = [
        a for a in CANONICAL_ACTIONS if counts.get(a, 0) < MIN_STATES_PER_ACTION
    ]
    checks["coverage_counts_match_report"] = (
        report["coverage"]["counts"] == {a: counts.get(a, 0) for a in CANONICAL_ACTIONS}
    )
    cap_reached = all_episodes.height >= TRANCHE_CAP
    checks["coverage_met_or_cap_documented"] = (not violations) or (
        cap_reached and violations == expansion["dropped_before_training"]
    )
    checks["cap_respected"] = all_episodes.height <= TRANCHE_CAP

    # expansion stayed on the working side and inside the seed lineage
    expansion_eps = all_episodes.filter(
        ~pl.col("episode_id").is_in(
            sorted(episodes[0]["episode_id"].to_list())
        )
    )
    checks["expansion_working_only"] = (
        expansion_eps.filter(pl.col("split") != "working").height == 0
    )
    checks["episode_counts_consistent"] = (
        all_episodes.height == expansion["total_episodes"]
        and all_episodes.height == all_episodes["episode_id"].n_unique()
    )

    # fold plan rebuilt over the expanded working set, still patient-grouped
    fold_plan = FoldPlan.model_validate_json(
        (ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8")
    )
    working_eps = all_episodes.filter(pl.col("split") == "working")
    checks["fold_plan_covers_working"] = set(fold_plan.assignment) == set(
        working_eps["episode_id"].to_list()
    )
    patient_of = dict(working_eps.select("episode_id", "patient_id").rows())
    folds_per_patient: dict[str, set[int]] = {}
    for episode_id, fold in fold_plan.assignment.items():
        folds_per_patient.setdefault(patient_of[episode_id], set()).add(fold)
    checks["no_patient_straddles_folds"] = all(
        len(f) == 1 for f in folds_per_patient.values()
    )

    dev_slice = json.loads(
        (ROOT / "runs" / "manifests" / "dev_slice.json").read_text(encoding="utf-8")
    )
    checks["dev_slice_reserved"] = dev_slice["n_states"] >= 500

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
                "loop": "P5",
                "verdict": verdict,
                **checks,
                "total_episodes": all_episodes.height,
                "working_states": working.height,
                "violations_dropped": violations,
                "merges": len(merges),
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
