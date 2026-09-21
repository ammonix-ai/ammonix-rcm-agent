"""P7 checker: leakage audit passes; ECE <= 0.05; every working state in the
Universe scored by its hold-out fold; tribe stats and ambiguity flags
written; Basis complete and pinned; determinism exit 0 (output shown).

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
from ammonix_core.schema import BasisManifest, FoldPlan  # noqa: E402

ECE_THRESHOLD = 0.05
AMBIGUITY_MARGIN = 0.05


def _kept_tokenizer_version() -> str:
    rounds = json.loads(
        (ROOT / "runs" / "state" / "tokenizer_rounds.json").read_text(encoding="utf-8")
    )
    return f"round{[r['round'] for r in rounds if r['kept']][-1]}"


def main() -> int:
    checks: dict[str, bool] = {}

    report = json.loads(
        (ROOT / "runs" / "reports" / "universe.json").read_text(encoding="utf-8")
    )
    manifest = BasisManifest.model_validate_json(
        (ROOT / "basis" / "manifest.json").read_text(encoding="utf-8")
    )
    universe = pl.read_parquet(ROOT / "basis" / "universe.parquet")
    tribes = json.loads((ROOT / "basis" / "tribes.json").read_text(encoding="utf-8"))
    working_states = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    fold_plan = FoldPlan.model_validate_json(
        (ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8")
    )

    # leakage audit passed and is recorded
    checks["leakage_audit_passes"] = report["leak_screen_flagged"] == {}

    # ECE within threshold, overall and per trained action
    checks["ece_overall_within_threshold"] = report["ece_overall"] <= ECE_THRESHOLD
    checks["ece_every_action_reported"] = set(report["ece_per_action"]) == set(
        report["trained_actions"]
    )

    # every working state has exactly one UniverseRecord, scored by its fold
    checks["universe_covers_every_working_state"] = (
        universe.height == working_states.height
        and set(universe["state_id"].to_list())
        == set(working_states["state_id"].to_list())
    )
    fold_of_episode = fold_plan.assignment
    checks["fold_matches_plan"] = all(
        fold_of_episode[row["episode_id"]] == row["fold"]
        for row in universe.select("episode_id", "fold").to_dicts()
    )
    sample = universe.head(200).to_dicts()
    all_actions = set(report["trained_actions"]) | set(report["constant_priors"])
    checks["scores_present_and_bounded"] = all(
        set(json.loads(row["scores_cal"])) == all_actions
        and all(0.0 <= v <= 1.0 for v in json.loads(row["scores_cal"]).values())
        and all(0.0 <= v <= 1.0 for v in json.loads(row["scores_raw"]).values())
        for row in sample
    )
    checks["attributions_written"] = all(
        len(json.loads(row["top_features"])) > 0 for row in sample
    )

    # tribe stats and ambiguity flags written
    checks["tribes_written"] = len(tribes) > 0 and report["n_tribes"] == len(tribes)
    checks["tribe_stats_complete"] = all(
        t["stats"]["n_states"] == t["member_count"]
        and 0.0 <= t["stats"]["success_rate"] <= 1.0
        and t["stats"]["ambiguous"] == (t["stats"]["top2_margin"] < AMBIGUITY_MARGIN)
        for t in tribes
    )
    checks["ambiguous_tribes_have_rivals"] = all(
        t["stats"]["rival_action_id"] is not None
        for t in tribes
        if t["stats"]["ambiguous"]
    )
    assigned = universe.filter(pl.col("tribe_id").is_not_null()).height
    checks["tribe_membership_recorded"] = (
        assigned == report["states_in_tribes"]
        and assigned + report["noise_states"] == universe.height
    )

    # Basis complete: manifest fields + artefacts on disk
    artefacts = ROOT / "basis" / "artefacts"
    checks["basis_artefacts_exist"] = all(
        (ROOT / "basis" / c.artefact_uri).is_file() for c in manifest.calibrators
    ) and (artefacts / "scaler.joblib").is_file() and (
        artefacts / "index.joblib"
    ).is_file() and all(
        (artefacts / f"refit_{a}.joblib").is_file()
        for a in report["trained_actions"]
    )
    # v0.4 s.3b: the index must live in label space - one dimension per
    # trained action, and the stored order must match sorted trained actions
    import joblib as _joblib
    _ids = _joblib.load(artefacts / "index_ids.joblib")
    _index = _joblib.load(artefacts / "index.joblib")
    checks["index_in_label_space"] = (
        _ids.get("space") == "label"
        and _ids.get("u_order") == sorted(report["trained_actions"])
        and getattr(_index, "n_features_in_", -1) == len(report["trained_actions"])
    )
    checks["manifest_complete"] = (
        len(manifest.dataset_fingerprint) == 64
        and len(manifest.swarm_config_sha256) == 64
        and manifest.tokenizer.version == _kept_tokenizer_version()
        and set(manifest.metrics) == set(report["trained_actions"])
        and manifest.coverage.min_states_per_action == 100
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
                "loop": "P7",
                "verdict": verdict,
                **checks,
                "ece_overall": round(report["ece_overall"], 4),
                "n_tribes": report["n_tribes"],
                "n_ambiguous_tribes": report["n_ambiguous_tribes"],
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
