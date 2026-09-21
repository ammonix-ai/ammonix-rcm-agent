"""P7: build the OOF Universe, calibrators, index, tribes and the Basis.

Working data only. The swarm spec is the P6 kept round (defaults + balanced);
fold models are rebuilt deterministically from the master seed, so nothing
here depends on pickled state surviving between milestones.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.hashing import dataset_fingerprint, sha256_file, sha256_json  # noqa: E402
from ammonix_core.schema import (  # noqa: E402
    BasisManifest,
    Calibrator,
    CoverageReport,
    FoldMetrics,
    FoldPlan,
    Tokenizer,
    UniverseIndex,
)
from ammonix_core.universe import (  # noqa: E402
    build_tribes,
    build_universe,
    leak_screen_features,
)
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.features import to_qpsi_records  # noqa: E402

WORKING = ROOT / "data" / "working" / "cardessa_sim"
BASIS = ROOT / "basis"


def main() -> int:
    from cardessa.features_kept import kept_blocks as _kept_blocks
    _, _kept_round = _kept_blocks(ROOT)
    tokenizer = Tokenizer.model_validate_json(
        (ROOT / "runs" / "manifests" / f"tokenizer_round{_kept_round}.json").read_text(
            encoding="utf-8"
        )
    )
    rounds = json.loads(
        (ROOT / "runs" / "state" / "swarm_rounds.json").read_text(encoding="utf-8")
    )
    kept_round = [r["round"] for r in rounds if r["kept"]][-1]
    swarm_manifest_path = ROOT / "runs" / "manifests" / f"swarm_round{kept_round}.json"
    print(f"using kept swarm round {kept_round}")
    swarm = json.loads(swarm_manifest_path.read_text(encoding="utf-8"))
    action_report = json.loads(
        (ROOT / "runs" / "reports" / "action_map.json").read_text(encoding="utf-8")
    )

    states = pl.read_parquet(WORKING / "states.parquet")
    episodes = pl.read_parquet(WORKING / "episodes.parquet")
    fold_plan = FoldPlan.model_validate_json(
        (ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8")
    )
    from ammonix_core.schema import FeatureSpec

    from cardessa.features_kept import build_kept_features
    frame, names, _, _ = build_kept_features(ROOT, states)
    specs = [
        FeatureSpec(name=n, dtype="float",
                    description=f"kept tokenizer feature {n}")
        for n in names
    ]  # names drive the build; the real specs live in the kept manifest
    outcome_by_episode = {
        row["episode_id"]: (bool(row["success"]), float(row["outcome_score"]))
        for row in episodes.select("episode_id", "success", "outcome_score").to_dicts()
    }
    records = to_qpsi_records(frame, specs, outcome_by_episode)

    # leakage audit on the Q_PSI features (L7 gate)
    flagged = leak_screen_features(
        frame, names, {e: s for e, (s, _) in outcome_by_episode.items()}
    )
    print(f"leak screen over {len(names)} features: flagged {flagged or 'none'}")
    if flagged:
        raise SystemExit(f"leakage audit failed: {flagged}")

    trained = swarm["trained_actions"]
    priors = {a: info["prior"] for a, info in swarm["constant_prior"].items()}
    print(f"building universe: {len(records)} states x {len(trained)} trained actions")
    build = build_universe(
        records, fold_plan, names, trained, priors,
        seed=MASTER_SEED,
        hyperparams=swarm["classifier_spec"]["hyperparams"],
        balanced=swarm["label_policy"]["class_weight"] == "balanced",
    )
    print(f"ECE per action: { {a: round(v, 4) for a, v in build.ece_per_action.items()} }")
    print(f"ECE overall (own-state weighted): {build.ece_overall:.4f}")

    tribes, tribe_of = build_tribes(build, records, ambiguity_margin=0.05, seed=MASTER_SEED)
    ambiguous = [t for t in tribes if t.stats.ambiguous]
    print(f"tribes: {len(tribes)} ({len(ambiguous)} ambiguous), "
          f"{len(tribe_of)} states assigned")

    # ---- write the Basis ---------------------------------------------------
    artefacts = BASIS / "artefacts"
    artefacts.mkdir(parents=True, exist_ok=True)

    frame.write_parquet(BASIS / "qpsi.parquet")
    universe_rows = [
        {
            "state_id": r.state_id,
            "episode_id": r.example_id,
            "fold": r.fold,
            "true_action_id": r.true_action_id,
            "outcome_success": r.outcome_success,
            "outcome_score": r.outcome_score,
            "tribe_id": r.tribe_id,
            "scores_raw": json.dumps(r.scores_raw, sort_keys=True),
            "scores_cal": json.dumps(r.scores_cal, sort_keys=True),
            "top_features": json.dumps(
                [a.model_dump() for a in r.top_features], sort_keys=True
            ),
        }
        for r in build.records
    ]
    pl.DataFrame(universe_rows).write_parquet(BASIS / "universe.parquet")

    # v0.4 s.3b: the retrieval coordinate is the OOF calibrated score
    # vector over the trained actions (label space), per the architecture
    # paper - NOT the scaled feature vector. The feature scaler is still
    # fitted and shipped (fold-spread and diagnostics consume it).
    x_all = frame.select(names).to_numpy()
    scaler = StandardScaler().fit(x_all)
    u_order = sorted(trained)
    u_all = np.array(
        [[r.scores_cal[a] for a in u_order] for r in build.records]
    )
    index = NearestNeighbors(metric="euclidean", n_neighbors=25).fit(u_all)
    joblib.dump(scaler, artefacts / "scaler.joblib")
    joblib.dump(index, artefacts / "index.joblib")
    joblib.dump(
        {"state_ids": [r.state_id for r in build.records], "space": "label",
         "u_order": u_order},
        artefacts / "index_ids.joblib",
    )
    calibrator_entities = []
    for action, model in build.calibrator_models.items():
        uri = f"artefacts/calibrator_{action}.joblib"
        joblib.dump(model, BASIS / uri)
        calibrator_entities.append(
            Calibrator(action_id=action, method="isotonic", artefact_uri=uri)
        )
    for action, model in build.refit_models.items():
        joblib.dump(model, artefacts / f"refit_{action}.joblib")
    # v0.4: persist the fold models - they are the ensemble whose spread is
    # the swarm's own per-state uncertainty (BasisRuntime.score_spread)
    for (action, fold), model in build.fold_models.items():
        joblib.dump(model, artefacts / f"fold_{action}_{fold}.joblib")
    (BASIS / "tribes.json").write_text(
        json.dumps([t.model_dump() for t in tribes], indent=2), encoding="utf-8", newline="\n"
    )

    coverage = CoverageReport.model_validate(action_report["coverage"])
    manifest = BasisManifest(
        basis_id="cardessa-basis-v1",
        built_at=datetime(2026, 7, 12, tzinfo=UTC),
        dataset_fingerprint=dataset_fingerprint(
            [
                (e["episode_id"], [], bool(e["success"]))
                for e in episodes.select("episode_id", "success").to_dicts()
            ]
        ),
        tokenizer=tokenizer,
        action_map_version="cardessa-v1",
        swarm_config_sha256=sha256_json(
            {k: swarm[k] for k in ("classifier_spec", "label_policy", "seed")}
        ),
        calibrators=calibrator_entities,
        index=UniverseIndex(
            metric="euclidean",
            scaler_uri="artefacts/scaler.joblib",
            index_uri="artefacts/index.joblib",
        ),
        coverage=coverage,
        metrics={
            a: FoldMetrics(**m) for a, m in build.pooled_metrics.items()
        },
    )
    (BASIS / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )

    pins = [
        {
            "path": f"basis/{name}",
            "sha256": sha256_file(BASIS / name),
        }
        for name in ("manifest.json", "qpsi.parquet", "universe.parquet", "tribes.json")
    ]
    (ROOT / "runs" / "manifests" / "basis_files.json").write_text(
        json.dumps(pins, indent=2), encoding="utf-8", newline="\n"
    )

    (ROOT / "runs" / "reports" / "universe.json").write_text(
        json.dumps(
            {
                "milestone": "P7",
                "n_states": len(build.records),
                "trained_actions": trained,
                "constant_priors": priors,
                "leak_screen_flagged": flagged,
                "ece_per_action": build.ece_per_action,
                "ece_overall": build.ece_overall,
                "n_tribes": len(tribes),
                "n_ambiguous_tribes": len(ambiguous),
                "ambiguous_tribes": [
                    {
                        "tribe_id": t.tribe_id,
                        "action_id": t.action_id,
                        "top2_margin": t.stats.top2_margin,
                        "rival_action_id": t.stats.rival_action_id,
                        "n_states": t.stats.n_states,
                    }
                    for t in ambiguous
                ],
                "states_in_tribes": len(tribe_of),
                "noise_states": len(build.records) - len(tribe_of),
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "swarm_config_sha256": manifest.swarm_config_sha256,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("basis written: manifest, qpsi, universe, tribes, artefacts; pins recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
