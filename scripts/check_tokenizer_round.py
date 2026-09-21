"""P4 checker: latest tokenizer round gained >= 0.005 over the previous round
(or a plateau is declared across the last three rounds), determinism check
exits 0 (output shown), features are registered as FeatureSpecs with real
descriptions, and no post-hoc field leaked into the feature set.

Prints the one-line JSON verdict the goal condition references.
"""

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from ammonix_core.schema import Tokenizer  # noqa: E402

GAIN_THRESHOLD = 0.005
PLATEAU_ROUNDS = 3
POSTHOC_PATTERNS = (
    "paid_amount_final", "days_to_payment", "total_touches", "engine_truth_*",
)
TEXT_COLUMNS = ("payer_correspondence_text", "clinical_indication_text")


def main() -> int:
    checks: dict[str, bool] = {}

    rounds = json.loads(
        (ROOT / "runs" / "state" / "tokenizer_rounds.json").read_text(encoding="utf-8")
    )
    latest = rounds[-1]
    gains = [r["gain"] for r in rounds]
    plateau = len(rounds) >= PLATEAU_ROUNDS and all(
        g < GAIN_THRESHOLD for g in gains[-PLATEAU_ROUNDS:]
    )
    checks["gain_or_plateau"] = latest["gain"] >= GAIN_THRESHOLD or plateau
    checks["kept_iff_gained"] = latest["kept"] == (latest["gain"] >= GAIN_THRESHOLD)

    # the round's report must back the state file's numbers
    report = json.loads(
        (ROOT / "runs" / "reports" / f"tokenizer_round{latest['round']}.json").read_text(
            encoding="utf-8"
        )
    )
    checks["report_matches_state"] = (
        report["mean_oof_auroc"] == latest["mean_oof_auroc"]
        and report["kept"] == latest["kept"]
        and report["n_features"] == latest["n_features"]
    )
    checks["double_build_deterministic"] = report["double_build_hash_equal"] is True
    checks["scored_from_working_only"] = report["n_states_scored"] <= 1750

    # FeatureSpecs registered with real descriptions, in the kept tokenizer
    last_kept = next((r for r in reversed(rounds) if r["kept"]), None)
    checks["some_round_kept"] = last_kept is not None
    feature_names: list[str] = []
    if last_kept:
        tokenizer = Tokenizer.model_validate_json(
            (
                ROOT / "runs" / "manifests" / f"tokenizer_round{last_kept['round']}.json"
            ).read_text(encoding="utf-8")
        )
        feature_names = [s.name for s in tokenizer.features]
        checks["feature_count_matches"] = len(feature_names) == last_kept["n_features"]
        checks["real_descriptions"] = all(
            len(s.description.strip()) >= 15 and "todo" not in s.description.lower()
            for s in tokenizer.features
        )
        checks["unique_feature_names"] = len(set(feature_names)) == len(feature_names)
        checks["history_features_declared"] = any(
            s.history is not None for s in tokenizer.features
        )
        checks["pin_present"] = (
            len(tokenizer.pin.code_sha256) == 64 and "master" in tokenizer.pin.seeds
        )
        if last_kept["round"] >= 2:
            # text-extractor rounds must pin the LLM: pinned model, temp 0,
            # schema-constrained decoding
            llm = tokenizer.pin.llm or {}
            checks["llm_pinned"] = (
                "Qwen3.6-27B" in str(llm.get("model_id"))
                and llm.get("temperature") == 0
                and llm.get("decoding") == "guided_json"
            )

    # no post-hoc field, and no text column, may appear as a feature
    leaked = [
        name
        for name in feature_names
        for pattern in POSTHOC_PATTERNS + TEXT_COLUMNS
        if fnmatch.fnmatch(name, pattern)
    ]
    checks["no_posthoc_feature"] = not leaked

    # determinism check gates the round; its output is shown
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
                "loop": "P4",
                "round": latest["round"],
                "verdict": verdict,
                **checks,
                "mean_oof_auroc": latest["mean_oof_auroc"],
                "gain": round(latest["gain"], 4),
                "plateau": plateau,
                "leaked": leaked,
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
