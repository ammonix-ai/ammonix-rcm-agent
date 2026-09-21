"""W2 checker: the initial tranche is complete, audited and deterministic.

Verifies against runs/reports/corpus_tranche_0.json and the on-disk parquet:
1000 episodes (~4000 states) in factory schema shape via the
EnvironmentProtocol adapter; tranche audits green; cumulative coverage report
written with the expansion trigger armed; curated-family statistics within
tolerance; planted P1 family present with majority/minority structure;
poisoned column planted; double-generation hash equality including text.
Prints the one-line JSON verdict the goal condition references.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402


def main() -> int:
    checks: dict[str, bool] = {}
    report_path = ROOT / "runs" / "reports" / "corpus_tranche_0.json"
    coverage_path = ROOT / "runs" / "reports" / "coverage_cumulative.json"
    episodes_path = ROOT / "data" / "raw" / "cardessa_sim" / "episodes.parquet"
    states_path = ROOT / "data" / "raw" / "cardessa_sim" / "states.parquet"

    checks["report_exists"] = report_path.is_file()
    checks["coverage_report_written"] = coverage_path.is_file()
    checks["parquet_files_exist"] = episodes_path.is_file() and states_path.is_file()
    if not all(checks.values()):
        print(json.dumps({"loop": "W2", "verdict": False, **checks}))
        return 1

    report = json.loads(report_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    episodes = pl.read_parquet(episodes_path)
    states = pl.read_parquet(states_path)

    audits = report.get("audits", {})
    checks["episodes_1000"] = episodes.height == 1000 and report["n_episodes"] == 1000
    checks["audit_shape"] = audits.get("shape", {}).get("passed", False)
    checks["audit_families"] = audits.get("families", {}).get("passed", False)
    checks["audit_p1_structure"] = audits.get("p1_structure", {}).get("passed", False)
    checks["audit_poisoned_column"] = audits.get("poisoned_column", {}).get("passed", False)
    checks["poisoned_column_on_disk"] = "days_to_payment" in states.columns
    checks["expansion_trigger_armed"] = coverage.get("expansion_trigger") == "armed"
    checks["double_generation_equal_incl_text"] = bool(
        report.get("double_generation_equal")
    )
    checks["text_channel_present"] = (
        "payer_correspondence_text" in states.columns
        and "clinical_indication_text" in states.columns
        and states["clinical_indication_text"].str.len_chars().min() > 0
    )
    checks["pinned_model_used"] = "Qwen3.6-27B" in report.get("llm_model", "")

    verdict = all(checks.values())
    summary = {
        "loop": "W2",
        "verdict": verdict,
        **checks,
        "n_states": states.height,
        "coverage_violations": coverage.get("violations", []),
    }
    print(json.dumps(summary))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
