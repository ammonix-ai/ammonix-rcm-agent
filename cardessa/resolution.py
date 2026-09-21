"""Resolution heads (v0.3 spec section 4): per rework action, a second
classifier answering "will this specific request be GRANTED?" - a different
question from episode success, on a different probability scale.

Labels are the historically observed payer answers, reconstructed exactly by
replaying the engine's deterministic per-episode draws (identical to what
the recorded next state shows; this is reading the past, not the oracle:
at decision time the answer was in the future, at training time it is on
file). Resolution scores own the ambiguity flag and the P6 trap; episode-
success scores keep owning the recommendation objective, except the
path-value substitution of spec section 5.2.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl
from ammonix_core.schema import FoldPlan
from ammonix_core.universe import expected_calibration_error
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

REWORK_ACTIONS = (
    "request_retro_auth", "appeal_with_necessity",
    "request_peer_to_peer", "provide_requested_info",
)


def resolution_label(engine, row: dict) -> bool:
    """The payer's answer to this state's rework request, as recorded by the
    simulation (deterministic per-episode draws; equal to the next state)."""
    action = row["action_raw"]
    payer_id = row["payer_id"]
    episode_id = row["episode_id"]
    if action == "request_retro_auth":
        return engine.retro_auth_granted(
            payer_id, row["days_since_service"], episode_id
        )
    if action == "appeal_with_necessity":
        return engine.appeal_granted(
            payer_id, row["clinic_doc_quality"] >= 0.6, episode_id
        )
    if action == "request_peer_to_peer":
        return engine.p2p_reversed(payer_id, episode_id)
    if action == "provide_requested_info":
        return engine.info_request_resolved(payer_id, episode_id)
    raise ValueError(f"no resolution semantics for action {action!r}")


@dataclass
class ResolutionBuild:
    models: dict[str, HistGradientBoostingClassifier]
    calibrators: dict[str, IsotonicRegression]
    success_given_granted: dict[str, float]
    ece_per_action: dict[str, float]
    n_per_action: dict[str, int]


def train_resolution_heads(
    engine,
    working_states: pl.DataFrame,
    working_episodes: pl.DataFrame,
    frame: pl.DataFrame,
    feature_names: list[str],
    fold_plan: FoldPlan,
    seed: int,
) -> ResolutionBuild:
    success_of = dict(working_episodes.select("episode_id", "success").rows())
    models, calibrators, constants, eces, ns = {}, {}, {}, {}, {}
    feat_by_state = {
        r["state_id"]: [r[n] for n in feature_names]
        for r in frame.filter(
            pl.col("state_id").is_in(working_states["state_id"].to_list())
        ).to_dicts()
    }
    for action in REWORK_ACTIONS:
        own = working_states.filter(pl.col("action_raw") == action).to_dicts()
        if len(own) < 30:
            continue
        x = np.array([feat_by_state[r["state_id"]] for r in own])
        y = np.array([int(resolution_label(engine, r)) for r in own])
        folds = np.array([fold_plan.assignment[r["episode_id"]] for r in own])
        if len(set(y)) < 2:
            continue
        oof = np.full(len(own), np.nan)
        for fold in range(fold_plan.n_folds):
            holdout = folds == fold
            if not holdout.any() or len(set(y[~holdout])) < 2:
                continue
            model = HistGradientBoostingClassifier(random_state=seed).fit(
                x[~holdout], y[~holdout]
            )
            oof[holdout] = model.predict_proba(x[holdout])[:, 1]
        scored = ~np.isnan(oof)
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(oof[scored], y[scored])
        # honest cross-fitted ECE, same discipline as the universe build
        crossfit = np.full(len(own), np.nan)
        for fold in range(fold_plan.n_folds):
            eval_mask = (folds == fold) & scored
            fit_mask = scored & ~eval_mask
            if not eval_mask.any() or len(set(y[fit_mask])) < 2:
                continue
            fold_cal = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(oof[fit_mask], y[fit_mask])
            crossfit[eval_mask] = fold_cal.predict(oof[eval_mask])
        cf = ~np.isnan(crossfit)
        eces[action] = expected_calibration_error(crossfit[cf], y[cf])
        models[action] = HistGradientBoostingClassifier(random_state=seed).fit(x, y)
        calibrators[action] = calibrator
        granted = [r for r, label in zip(own, y, strict=True) if label]
        constants[action] = (
            float(np.mean([success_of[r["episode_id"]] for r in granted]))
            if granted
            else 0.0
        )
        ns[action] = len(own)
    return ResolutionBuild(
        models=models, calibrators=calibrators,
        success_given_granted=constants, ece_per_action=eces, n_per_action=ns,
    )
