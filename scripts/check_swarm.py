"""P6 checker: pooled fold metrics for every trained Action, keep/revert
discipline with the plateau rule, every canonical action accounted for
(trained / constant-prior / dropped), no reads outside working data, and
determinism exit 0 (output shown).

Prints the one-line JSON verdict the goal condition references.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from cardessa.audits import CANONICAL_ACTIONS  # noqa: E402

GAIN_THRESHOLD = 0.005
PLATEAU_ROUNDS = 3


def main() -> int:
    checks: dict[str, bool] = {}

    rounds = json.loads(
        (ROOT / "runs" / "state" / "swarm_rounds.json").read_text(encoding="utf-8")
    )
    latest = rounds[-1]
    gains = [r["gain"] for r in rounds]
    plateau = len(rounds) >= PLATEAU_ROUNDS and all(
        g < GAIN_THRESHOLD for g in gains[-PLATEAU_ROUNDS:]
    )
    checks["gain_or_plateau"] = latest["gain"] >= GAIN_THRESHOLD or plateau
    checks["kept_iff_gained"] = all(
        r["kept"] == (r["gain"] >= GAIN_THRESHOLD) for r in rounds
    )

    last_kept = next((r for r in reversed(rounds) if r["kept"]), None)
    checks["some_round_kept"] = last_kept is not None
    if last_kept is None:
        print(json.dumps({"loop": "P6", "verdict": False, **checks}))
        return 1

    report = json.loads(
        (ROOT / "runs" / "reports" / f"swarm_round{last_kept['round']}.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (
            ROOT / "runs" / "manifests" / f"swarm_round{last_kept['round']}.json"
        ).read_text(encoding="utf-8")
    )
    checks["report_matches_state"] = (
        report["mean_oof_auroc"] == last_kept["mean_oof_auroc"]
        and report["kept"] is True
        and report["classifier_spec"]["hyperparams"] == last_kept["hyperparams"]
    )

    # pooled fold metrics for every trained Action; per-fold metrics present
    trained = report["trained_actions"]
    checks["pooled_metrics_every_trained_action"] = set(report["pooled_auroc"]) == set(
        trained
    ) and all(0.0 <= v <= 1.0 for v in report["pooled_auroc"].values())
    checks["fold_metrics_every_trained_action"] = all(
        report["fold_metrics"].get(a) for a in trained
    )

    # every canonical action accounted for exactly once
    accounted = (
        set(trained) | set(report["constant_prior"]) | set(report["dropped"])
    )
    checks["every_action_accounted"] = accounted == set(CANONICAL_ACTIONS)
    checks["partitions_disjoint"] = (
        len(trained) + len(report["constant_prior"]) + len(report["dropped"])
        == len(accounted)
    )
    checks["constant_priors_have_reasons"] = all(
        info.get("reason") and 0.0 <= info["prior"] <= 1.0
        for info in report["constant_prior"].values()
    )

    # label policy per the platform plan
    checks["label_policy_pinned"] = (
        report["label_policy"]["scope"] == "own_action_states"
        and report["label_policy"]["label"] == "outcome_success"
        and report["label_policy"]["class_weight"] == "balanced"
    )

    # no reads outside working data: the runner touches only data/working
    runner = (ROOT / "scripts" / "run_swarm_round.py").read_text(encoding="utf-8")
    checks["no_reads_outside_working"] = (
        "data/reserve" not in runner
        and "data/raw" not in runner
        and 'ROOT / "data" / "working"' in runner
    )

    # the tokenizer pin still matches the feature code the swarm used
    # (resolve the KEPT round from the tokenizer round state)
    tok_rounds = json.loads(
        (ROOT / "runs" / "state" / "tokenizer_rounds.json").read_text(encoding="utf-8")
    )
    kept_tok = [r["round"] for r in tok_rounds if r["kept"]][-1]
    tokenizer = json.loads(
        (ROOT / "runs" / "manifests" / f"tokenizer_round{kept_tok}.json").read_text(
            encoding="utf-8"
        )
    )
    checks["tokenizer_pin_verified"] = (
        report["tokenizer_code_sha256"] == tokenizer["pin"]["code_sha256"]
        and manifest["tokenizer_code_sha256"] == tokenizer["pin"]["code_sha256"]
    )

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
                "loop": "P6",
                "round": latest["round"],
                "verdict": verdict,
                **checks,
                "mean_oof_auroc": last_kept["mean_oof_auroc"],
                "gain": round(latest["gain"], 4),
                "plateau": plateau,
                "trained": trained,
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
