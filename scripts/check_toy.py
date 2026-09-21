"""M1 checker: the toy build recovered the planted truth.

Reads runs/reports/toy_build.json (produced by `make toy`) and verifies
mean_oof_auroc >= 0.95, recommended_matches_optimal >= 0.95 and
double-generation hash equality. Ends by printing the one-line JSON verdict
the goal condition references.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "runs" / "reports" / "toy_build.json"

AUROC_THRESHOLD = 0.95
MATCH_THRESHOLD = 0.95


def main() -> int:
    if not REPORT.is_file():
        print(json.dumps({"loop": "L1", "verdict": False, "error": "report missing"}))
        return 1
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    auroc = report.get("mean_oof_auroc", 0.0)
    match_rate = report.get("recommended_matches_optimal", 0.0)
    double_ok = report.get("double_generation_equal", False)
    verdict = auroc >= AUROC_THRESHOLD and match_rate >= MATCH_THRESHOLD and bool(double_ok)
    print(
        json.dumps(
            {
                "loop": "L1",
                "verdict": verdict,
                "mean_oof_auroc": round(auroc, 4),
                "recommended_matches_optimal": round(match_rate, 4),
                "double_generation_equal": double_ok,
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
