"""v0.3: train and persist the resolution heads (spec section 4)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import joblib  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.schema import FoldPlan  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.engine import PayerEngine  # noqa: E402
from cardessa.resolution import train_resolution_heads  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

WORKING = ROOT / "data" / "working" / "cardessa_sim"


def main() -> int:
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    states = pl.read_parquet(WORKING / "states.parquet")
    episodes = pl.read_parquet(WORKING / "episodes.parquet")
    fold_plan = FoldPlan.model_validate_json(
        (ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8")
    )
    from cardessa.features_kept import build_kept_features
    frame, names, _, _ = build_kept_features(ROOT, states)

    build = train_resolution_heads(
        engine, states, episodes, frame, names, fold_plan, MASTER_SEED
    )
    artefacts = ROOT / "basis" / "artefacts"
    artefacts.mkdir(parents=True, exist_ok=True)
    for action, model in build.models.items():
        joblib.dump(model, artefacts / f"resolution_{action}.joblib")
        joblib.dump(
            build.calibrators[action], artefacts / f"resolution_cal_{action}.joblib"
        )
    manifest = {
        "milestone": "v0.3-resolution",
        "actions": sorted(build.models),
        "n_per_action": build.n_per_action,
        "ece_per_action": build.ece_per_action,
        "success_given_granted": build.success_given_granted,
        "seed": MASTER_SEED,
    }
    (ROOT / "runs" / "manifests" / "resolution_heads.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
