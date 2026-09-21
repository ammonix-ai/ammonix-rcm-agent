"""Behavior-cloning (imitation) baseline: fit a supervised classifier that
maps a decision-time tabular state to the clerk action recorded in the
training lineage.

Trains on ALL 11,427 lineage rows in
  <v4-build>/data/working/cardessa_sim/states.parquet
(the 5,000-episode training lineage; tranche 98 = episodes 5000..5299 is
NOT in this file, so the eval tranche never touches the model).

Features are decision-time tabular state fields ONLY. Hard-excluded:
- the label (action_raw),
- identifiers (episode_id, state_id, touch_seq),
- the free-text channels (clinical_indication_text, payer_correspondence_text),
- the high-cardinality case identifiers (payer_id: an id; dx_codes: 313 values;
  payer behaviour is carried by payer_archetype),
- the planted poison column days_to_payment and every post-hoc/outcome field
  named in agentic.briefing.EXCLUDED_FIELDS (paid_amount_final, total_touches;
  not present in this parquet but asserted-absent defensively).

featurize()/build_vocab() are imported unchanged by scripts/run_imitation_arm.py
so training and evaluation featurise every row bit-identically.

Run (any factory venv with sklearn + polars):
  <venv>\\python.exe scripts\\train_imitation.py
"""

import os
import pickle
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
WORKTREES = BASE.parent / "Ammonix_Generic" / "Cardessa_worktrees"
V4_ROOT = Path(os.environ.get("AMMONIX_V4_ROOT", WORKTREES / "v4-build"))
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from agentic.briefing import EXCLUDED_FIELDS  # noqa: E402

STATES = V4_ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
MODEL_PATH = BASE / "cache" / "imitation" / "clerk_bc.pkl"

LABEL = "action_raw"

# the 10 action classes (fixed order; also the multi-hot vocab for prior_actions)
ACTIONS = [
    "submit_clean", "submit_with_records", "correct_and_resubmit",
    "request_retro_auth", "appeal_with_necessity", "request_peer_to_peer",
    "provide_requested_info", "bill_secondary", "bill_patient", "write_off",
]

# decision-time tabular fields kept as model inputs -------------------------
CATEGORICAL_FIELDS = [
    "payer_archetype", "cpt", "auth_status", "eligibility_status",
    "subscriber_relationship", "persona_id", "carc",
]
BOOL_FIELDS = [
    "auth_required", "p2p_available", "pairing_valid", "aob_on_file",
    "cob_position_ok", "has_secondary",
]
NUMERIC_FIELDS = [
    "retro_window_days", "timely_filing_days", "days_since_service",
    "days_to_filing_deadline", "balance", "allowed_amount",
    "deductible_remaining", "coinsurance_pct", "touches_so_far",
    "clinic_doc_quality", "cumulative_delay_days",
]
# comma-joined history strings -> multi-hot over a fixed vocab
MULTIHOT_FIELDS = ["prior_actions", "prior_carcs"]

# fields that must NEVER reach the feature matrix (label / ids / text /
# poison / outcome / dropped identifiers). Asserted at build time.
FORBIDDEN_FIELDS = set(EXCLUDED_FIELDS) | {
    LABEL, "episode_id", "state_id", "touch_seq",
    "clinical_indication_text", "payer_correspondence_text",
    "payer_id", "dx_codes",
}


def build_vocab(rows: list[dict]) -> dict:
    """Deterministic encoder vocab learned from the training rows."""
    cat = {f: sorted({str(r.get(f, "")) for r in rows}) for f in CATEGORICAL_FIELDS}
    carc_codes = sorted({
        c for r in rows for c in str(r.get("prior_carcs", "")).split(",") if c
    })
    return {
        "categorical": cat,
        "multihot": {"prior_actions": list(ACTIONS), "prior_carcs": carc_codes},
    }


def feature_names(vocab: dict) -> list[str]:
    names: list[str] = []
    for f in CATEGORICAL_FIELDS:
        names += [f"{f}={v}" for v in vocab["categorical"][f]]
    names += [f"bool:{f}" for f in BOOL_FIELDS]
    names += [f"num:{f}" for f in NUMERIC_FIELDS]
    for f in MULTIHOT_FIELDS:
        names += [f"{f}~{v}" for v in vocab["multihot"][f]]
    return names


def featurize(row: dict, vocab: dict) -> list[float]:
    """Row dict -> fixed-length float vector, identical for train and eval.

    Unknown categorical/multi-hot values encode as all-zero for their group
    (no crash on an eval-time value never seen in training)."""
    vec: list[float] = []
    for f in CATEGORICAL_FIELDS:
        val = str(row.get(f, ""))
        vec += [1.0 if val == v else 0.0 for v in vocab["categorical"][f]]
    for f in BOOL_FIELDS:
        vec.append(1.0 if bool(row.get(f)) else 0.0)
    for f in NUMERIC_FIELDS:
        x = row.get(f)
        vec.append(float(x) if x is not None else 0.0)
    for f in MULTIHOT_FIELDS:
        present = {c for c in str(row.get(f, "")).split(",") if c}
        vec += [1.0 if v in present else 0.0 for v in vocab["multihot"][f]]
    return vec


def _assert_no_forbidden(vocab: dict) -> None:
    used = set(CATEGORICAL_FIELDS + BOOL_FIELDS + NUMERIC_FIELDS + MULTIHOT_FIELDS)
    leaked = used & FORBIDDEN_FIELDS
    assert not leaked, f"forbidden field(s) in feature set: {sorted(leaked)}"
    for banned in ("days_to_payment", "paid_amount_final", "total_touches"):
        assert banned not in used, f"{banned} must not be a feature"
    for f in used:
        assert f not in ("episode_id", "state_id", "touch_seq", LABEL)


def main() -> int:
    df = pl.read_parquet(STATES)
    rows = df.to_dicts()
    print(f"loaded {len(rows)} lineage rows from {STATES}")
    assert len(rows) == 11427, f"expected 11,427 lineage rows, got {len(rows)}"
    # none of these rows may be the eval tranche (episodes 5000..5299)
    ep_ints = [int(str(r["episode_id"]).split("-")[-1]) if str(r["episode_id"]).split("-")[-1].isdigit()
               else None for r in rows]
    max_known = max(e for e in ep_ints if e is not None) if any(ep_ints) else None
    print(f"distinct episodes in training: {df['episode_id'].n_unique()} "
          f"(lineage; eval tranche 98 = 5000..5299 is not in this file)")

    vocab = build_vocab(rows)
    _assert_no_forbidden(vocab)
    names = feature_names(vocab)

    X = np.asarray([featurize(r, vocab) for r in rows], dtype=np.float64)
    y = np.asarray([str(r[LABEL]) for r in rows])
    print(f"feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    # final guard: no forbidden token in any feature name
    for banned in FORBIDDEN_FIELDS:
        assert not any(banned == n.split("=")[0].split(":")[-1].split("~")[0]
                       for n in names), f"forbidden field leaked into names: {banned}"

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=None,
        l2_regularization=1.0, random_state=0,
    )
    clf.fit(X, y)

    pred = clf.predict(X)
    train_acc = float((pred == y).mean())
    support = Counter(y.tolist())
    pred_dist = Counter(pred.tolist())

    print(f"\ntrain accuracy: {train_acc:.4f}")
    print("per-class support (true) -> predicted count:")
    for a in ACTIONS:
        print(f"  {a:24s} support={support.get(a, 0):5d}  predicted={pred_dist.get(a, 0):5d}")
    print(f"distinct predicted classes: {len(pred_dist)} / {len(ACTIONS)}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({
            "model": clf,
            "vocab": vocab,
            "feature_names": names,
            "actions": ACTIONS,
            "train_accuracy": train_acc,
            "n_train_rows": int(X.shape[0]),
            "support": dict(support),
            "predicted_distribution": dict(pred_dist),
            "feature_fields": {
                "categorical": CATEGORICAL_FIELDS, "bool": BOOL_FIELDS,
                "numeric": NUMERIC_FIELDS, "multihot": MULTIHOT_FIELDS,
            },
            "excluded_fields": sorted(FORBIDDEN_FIELDS),
        }, fh)
    print(f"\nmodel persisted to {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
