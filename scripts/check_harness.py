"""P8 checker: pass_rate >= 0.90 and mean_iterations <= 2 on the dev slice
under the pinned vLLM model; all 20 seeded impossible cases ended status
"escalated"; the LLM pin is real (weights shards recorded, temperature 0,
constrained decoding); skills exist but are INACTIVE until GATE-P8 written
approval; optimisation rounds changed only files under harness/prompts/.

Prints the one-line JSON verdict the goal condition references.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from ammonix_core.hashing import sha256_text  # noqa: E402
from ammonix_core.schema import HarnessArtefacts  # noqa: E402

PASS_RATE_MIN = 0.90
MEAN_ITERATIONS_MAX = 2.0


def main() -> int:
    checks: dict[str, bool] = {}

    report = json.loads(
        (ROOT / "runs" / "reports" / "harness_eval.json").read_text(encoding="utf-8")
    )
    rounds = json.loads(
        (ROOT / "runs" / "state" / "harness_rounds.json").read_text(encoding="utf-8")
    )
    harness = HarnessArtefacts.model_validate_json(
        (ROOT / "runs" / "manifests" / "harness.json").read_text(encoding="utf-8")
    )
    skills_doc = json.loads(
        (ROOT / "runs" / "manifests" / "skills.json").read_text(encoding="utf-8")
    )

    checks["pass_rate_met"] = report["pass_rate"] >= PASS_RATE_MIN
    checks["pass_rate_definition_recorded"] = "pass_rate_definition" in report
    checks["policy_escalations_reported"] = "dev_escalations_by_reason" in report
    checks["mean_iterations_met"] = report["mean_iterations"] <= MEAN_ITERATIONS_MAX
    checks["dev_slice_full"] = report["n_dev_states"] >= 500

    impossible = report["impossible_results"]
    checks["twenty_impossible_cases"] = len(impossible) == 20
    checks["impossible_all_escalated"] = report["impossible_all_escalated"] and all(
        r["status"] == "escalated" for r in impossible
    )
    categories = {r["category"] for r in impossible}
    checks["impossible_categories_complete"] = categories == {
        "holdout_payer", "dead_end", "missing_document",
    }
    checks["holdout_cases_ood_reason"] = all(
        r["reason"] == "needs_outside_information"
        for r in impossible
        if r["category"] == "holdout_payer"
    )

    # report backs the latest round; the run really used the pinned model
    latest = rounds[-1]
    checks["report_matches_round"] = (
        latest["pass_rate"] == report["pass_rate"]
        and latest["mean_iterations"] == report["mean_iterations"]
        and latest["impossible_all_escalated"] == report["impossible_all_escalated"]
    )
    checks["pinned_model_served"] = "Qwen3.6-27B" in report["llm_model_served"]
    checks["llm_pin_complete"] = (
        harness.llm.name == "Qwen3.6-27B-AWQ-INT4"
        and len(harness.llm.weights_sha256) == 64
        and harness.llm.temperature == 0.0
        and harness.llm.constrained_decoding is True
        and harness.llm.weights_sha256
        == sha256_text(
            "".join(
                skills_doc["weights_shards"][k]
                for k in sorted(skills_doc["weights_shards"])
            )
        )
    )

    # GATE-P8 discipline: skills may be active ONLY with a written approval
    # entry in runs/APPROVALS.md; inactive is always acceptable
    approvals = (ROOT / "runs" / "APPROVALS.md").read_text(encoding="utf-8")
    gate_approved = "## GATE-P8 Skill activation" in approvals and (
        "Decision: approved" in approvals.split("## GATE-P8 Skill activation", 1)[1]
    )
    checks["skills_gate_respected"] = (skills_doc["active"] is False) or gate_approved
    cluster_refs = {
        s["scope"]["ref"] for s in skills_doc["skills"] if s["scope"]["level"] == "cluster"
    }
    checks["skill_per_action"] = len(cluster_refs) == 10 and any(
        s["scope"]["level"] == "default" for s in skills_doc["skills"]
    )

    # optimisation rounds may only change harness/prompts/: every round's
    # recorded prompt hashes must cover the same file set, and the harness
    # manifest embeds the current files verbatim
    file_sets = {tuple(sorted(r["prompt_hashes"])) for r in rounds}
    checks["rounds_touch_only_prompts"] = len(file_sets) == 1
    current = {
        p.name: p.read_text(encoding="utf-8")
        for p in (ROOT / "harness" / "prompts").glob("*.txt")
    }
    checks["manifest_prompts_current"] = all(
        harness.m1_prompts[f"skill-{name.removeprefix('m1_').removesuffix('.txt')}"].text
        == text
        for name, text in current.items()
        if name.startswith("m1_")
    ) and harness.m2_prompt.text == current["m2_feedback.txt"]

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
                "loop": "P8",
                "round": latest["round"],
                "verdict": verdict,
                **checks,
                "pass_rate": round(report["pass_rate"], 4),
                "mean_iterations": round(report["mean_iterations"], 3),
                "impossible_escalated": sum(
                    r["status"] == "escalated" for r in impossible
                ),
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
