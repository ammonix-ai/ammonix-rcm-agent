"""P2 checker: ingest via the DatasetDescriptor; post-hoc quarantine verified.

Runs the generic ingest against the pinned corpus, writes the audit report,
and verifies: the four scope conditions pass; the planted poisoned column
(days_to_payment) is flagged by BOTH the denylist and the statistical screen
and is excluded from Data; no post-hoc episode field appears among the state
Data columns. Prints the one-line JSON verdict.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from ammonix_core.ingest import ingest_dataset  # noqa: E402

from cardessa.descriptor import cardessa_descriptor  # noqa: E402


def main() -> int:
    descriptor = cardessa_descriptor()
    result = ingest_dataset(descriptor, ROOT)
    audit = result.audit

    reports = ROOT / "runs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ingest_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    leakage = audit["leakage_audit"]
    all_data_columns = {
        c for cols in audit["data_columns_by_modality"].values() for c in cols
    }
    checks = {
        "audit_report_written": True,
        **{f"scope_{k}": v for k, v in audit["scope_conditions"].items()},
        "examples_1000": audit["n_examples"] == 1000,
        "poison_flagged_by_denylist": "days_to_payment" in leakage["denylisted"],
        "poison_flagged_statistically": "days_to_payment"
        in leakage["statistically_flagged"],
        "poison_excluded_from_data": "days_to_payment" not in all_data_columns,
        "no_posthoc_in_data": not any(
            c in all_data_columns
            for c in ("paid_amount_final", "total_touches", "engine_truth_allowed")
        ),
        "ten_raw_actions": len(audit["raw_action_inventory"]) == 10,
        "text_columns_mapped": set(
            audit["data_columns_by_modality"].get("text", [])
        )
        == {"payer_correspondence_text", "clinical_indication_text"},
    }
    verdict = all(checks.values())
    print(
        json.dumps(
            {
                "loop": "P2",
                "verdict": verdict,
                **checks,
                "n_states": audit["n_states"],
                "poison_auc": leakage["statistically_flagged"].get("days_to_payment"),
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
