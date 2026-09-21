"""Stage B3 + regions: the OOF Universe, calibration, index and tribes.

Every state is scored by its HOLD-OUT fold model for every trained action
(the permanent OOF Universe); constant-prior actions contribute their prior.
Per-action isotonic calibrators are fitted on the action's own OOF scores
against Outcome — the only place ground truth for P(success | state, action)
exists. Attributions are occlusion-style local contributions of the argmax
action's classifier. Tribes are HDBSCAN clusters inside each action's own
states, in scaled feature space; ambiguity is a build-time property.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ammonix_core.ingest import LEAK_AUC_THRESHOLD
from ammonix_core.pipeline import (
    balanced_weights,
    build_classifier,
    fit_classifier,
    matrix_from_rows,
)
from ammonix_core.schema import (
    ActionId,
    AmmonixConfig,
    Attribution,
    ClassifierSpec,
    FoldPlan,
    QPsiRecord,
    Tribe,
    TribeStats,
    UniverseRecord,
)

_CONFIG = AmmonixConfig()  # single source for shared thresholds (schema defaults)

ECE_BINS = 10
TOP_ATTRIBUTIONS = 5
OCCLUSION_FEATURES = 8


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard binned ECE: weighted |accuracy - confidence| over equal-width bins."""
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(labels[mask].mean() - scores[mask].mean())
    return float(ece)


@dataclass
class UniverseBuild:
    records: list[UniverseRecord]
    calibrator_models: dict[ActionId, IsotonicRegression]
    refit_models: dict[ActionId, HistGradientBoostingClassifier]
    constant_priors: dict[ActionId, float]
    ece_per_action: dict[ActionId, float]
    ece_overall: float
    pooled_metrics: dict[ActionId, dict]
    feature_names: list[str]
    fold_models: dict[tuple[ActionId, int], object] = field(default_factory=dict)


def build_universe(
    records: list[QPsiRecord],
    fold_plan: FoldPlan,
    feature_names: list[str],
    trained_actions: list[str],
    constant_priors: dict[str, float],
    seed: int,
    hyperparams: dict | None = None,
    balanced: bool = True,
    classifier: ClassifierSpec | None = None,
) -> UniverseBuild:
    """Score EVERY state by its hold-out fold model for every trained action.

    Besides the hold-out score, every state is scored by ALL fold models per
    action; the spread (std) is recorded as UniverseRecord.scores_std - the
    swarm's own per-state uncertainty, the honest companion to any margin.
    """
    if classifier is None:
        classifier = ClassifierSpec(model="gbdt", hyperparams=hyperparams or {})
    x_all = matrix_from_rows([r.features for r in records], feature_names)
    folds_all = np.array([fold_plan.assignment[r.example_id] for r in records])
    medians = np.median(x_all, axis=0)

    scores_raw = {a: np.full(len(records), np.nan) for a in trained_actions}
    fold_raw_all: dict[str, np.ndarray] = {}  # (n_folds, n_states) raw scores
    calibrators: dict[str, IsotonicRegression] = {}
    refit_models: dict[str, HistGradientBoostingClassifier] = {}
    ece_per_action: dict[str, float] = {}
    pooled_metrics: dict[str, dict] = {}
    global_importance: dict[str, np.ndarray] = {}
    fold_models: dict[tuple[str, int], object] = {}

    for action in trained_actions:
        own = np.array([r.action_id == action for r in records])
        y_own = np.array([r.outcome_success for r in records], dtype=int)[own]
        if len(np.unique(y_own)) < 2:
            raise ValueError(
                f"trained action {action!r} has a single outcome class over "
                f"its {int(own.sum())} states: route it to constant_priors"
            )
        x_own = x_all[own]
        folds_own = folds_all[own]

        fold_raw_all[action] = np.full((fold_plan.n_folds, len(records)), np.nan)
        for fold in range(fold_plan.n_folds):
            train_mask = folds_own != fold
            y_train = y_own[train_mask]
            if len(np.unique(y_train)) < 2:
                raise ValueError(
                    f"action {action!r}, fold {fold}: training partition is "
                    "single-class - too few states per fold for this action"
                )
            model = fit_classifier(
                build_classifier(seed, classifier), x_own[train_mask], y_train,
                sample_weight=balanced_weights(y_train) if balanced else None,
            )
            fold_models[(action, fold)] = model
            holdout_all = folds_all == fold  # EVERY state of this fold, any action
            scores_raw[action][holdout_all] = model.predict_proba(x_all[holdout_all])[:, 1]
            fold_raw_all[action][fold] = model.predict_proba(x_all)[:, 1]

        # calibrator on the action's OWN OOF scores vs outcome
        own_oof = scores_raw[action][own]
        calibrators[action] = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(own_oof, y_own)
        # honest ECE: cross-fitted — fold f's scores calibrated by an isotonic
        # fitted on the OTHER folds' OOF scores (in-sample isotonic ECE is ~0
        # by construction and proves nothing)
        crossfit_cal = np.full(len(y_own), np.nan)
        for fold in range(fold_plan.n_folds):
            eval_mask = folds_own == fold
            fit_mask = ~eval_mask
            if not eval_mask.any() or len(set(y_own[fit_mask])) < 2:
                continue
            fold_calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(own_oof[fit_mask], y_own[fit_mask])
            crossfit_cal[eval_mask] = fold_calibrator.predict(own_oof[eval_mask])
        scored_mask = ~np.isnan(crossfit_cal)
        ece_per_action[action] = expected_calibration_error(
            crossfit_cal[scored_mask], y_own[scored_mask]
        )
        pooled_metrics[action] = {
            "auroc": float(roc_auc_score(y_own, own_oof)),
            "auprc": float(average_precision_score(y_own, own_oof)),
            "n_pos": int(y_own.sum()),
            "n_neg": int((1 - y_own).sum()),
        }
        refit_models[action] = fit_classifier(
            build_classifier(seed, classifier), x_own, y_own,
            sample_weight=balanced_weights(y_own) if balanced else None,
        )

        # global occlusion importance (for picking per-state attribution candidates)
        base = refit_models[action].predict_proba(x_own)[:, 1]
        deltas = np.zeros(len(feature_names))
        for j in range(len(feature_names)):
            x_occ = x_own.copy()
            x_occ[:, j] = medians[j]
            deltas[j] = np.abs(
                base - refit_models[action].predict_proba(x_occ)[:, 1]
            ).mean()
        global_importance[action] = deltas

    unscored = [a for a, v in scores_raw.items() if np.isnan(v).any()]
    if unscored:
        raise ValueError(
            f"unscored states for actions {unscored}: a fold produced no model"
        )

    scores_cal = {
        a: calibrators[a].predict(scores_raw[a]) for a in trained_actions
    }
    # fold-ensemble spread on the calibrated scale: the swarm's per-state
    # uncertainty (how much the five fold models disagree about this state)
    scores_std = {
        a: np.vstack(
            [calibrators[a].predict(row) for row in fold_raw_all[a]]
        ).std(axis=0)
        for a in trained_actions
    }
    n_own = {a: int(sum(r.action_id == a for r in records)) for a in trained_actions}
    ece_overall = float(
        sum(ece_per_action[a] * n_own[a] for a in trained_actions)
        / max(1, sum(n_own.values()))
    )

    # per-state attributions for the argmax action (occlusion of top global features)
    stacked_cal = np.column_stack([scores_cal[a] for a in trained_actions])
    argmax_action = [trained_actions[i] for i in stacked_cal.argmax(axis=1)]
    attributions: list[list[Attribution]] = [[] for _ in records]
    for action in trained_actions:
        rows = np.array([a == action for a in argmax_action])
        if not rows.any():
            continue
        candidates = np.argsort(global_importance[action])[::-1][:OCCLUSION_FEATURES]
        base = np.full(len(records), np.nan)
        for fold in range(fold_plan.n_folds):
            m = rows & (folds_all == fold)
            if m.any():
                base[m] = fold_models[(action, fold)].predict_proba(x_all[m])[:, 1]
        contrib = np.zeros((int(rows.sum()), len(candidates)))
        idx = np.where(rows)[0]
        for c, j in enumerate(candidates):
            x_occ = x_all[rows].copy()
            x_occ[:, j] = medians[j]
            occ = np.full(len(idx), np.nan)
            for fold in range(fold_plan.n_folds):
                m = folds_all[rows] == fold
                if m.any():
                    occ[m] = fold_models[(action, fold)].predict_proba(x_occ[m])[:, 1]
            contrib[:, c] = base[rows] - occ
        for row_i, record_i in enumerate(idx):
            order = np.argsort(np.abs(contrib[row_i]))[::-1][:TOP_ATTRIBUTIONS]
            attributions[record_i] = [
                Attribution(
                    feature=feature_names[candidates[c]],
                    value=float(x_all[record_i, candidates[c]]),
                    contribution=float(contrib[row_i, c]),
                )
                for c in order
            ]

    universe_records = []
    for i, r in enumerate(records):
        raw = {a: float(scores_raw[a][i]) for a in trained_actions}
        cal = {a: float(scores_cal[a][i]) for a in trained_actions}
        std = {a: float(scores_std[a][i]) for a in trained_actions}
        for a, prior in constant_priors.items():
            raw[a] = prior
            cal[a] = prior
            std[a] = 0.0  # a constant prior carries no model uncertainty
        universe_records.append(
            UniverseRecord(
                state_id=r.state_id,
                example_id=r.example_id,
                fold=int(folds_all[i]),
                scores_raw=raw,
                scores_cal=cal,
                scores_std=std,
                true_action_id=r.action_id,
                outcome_success=r.outcome_success,
                outcome_score=r.outcome_score,
                top_features=attributions[i],
            )
        )
    return UniverseBuild(
        records=universe_records,
        calibrator_models=calibrators,
        refit_models=refit_models,
        constant_priors=dict(constant_priors),
        ece_per_action=ece_per_action,
        ece_overall=ece_overall,
        pooled_metrics=pooled_metrics,
        feature_names=list(feature_names),
        fold_models=fold_models,
    )


def build_tribes(
    build: UniverseBuild,
    records: list[QPsiRecord],
    ambiguity_margin: float = _CONFIG.ambiguity_margin,
    min_cluster_size: int = 15,
    seed: int = 0,
) -> tuple[list[Tribe], dict[str, str]]:
    """HDBSCAN inside each action's own states, in LABEL space (v0.4 s.3b).

    The clustering coordinate is the OOF calibrated score vector over the
    trained actions - the paper's u = Lambda(phi(x)) - so tribes are regions
    of "cases whose estimated action profiles look alike", not regions of
    raw-feature similarity. Returns (tribes, state_id -> tribe_id); noise
    points keep tribe_id None.
    """
    by_state: dict[str, str] = {}
    tribes: list[Tribe] = []
    trained = sorted(build.calibrator_models)
    x_scaled = np.array(
        [[rec.scores_cal[a] for a in trained] for rec in build.records]
    )

    for action in trained:
        own_idx = np.array([i for i, r in enumerate(records) if r.action_id == action])
        if len(own_idx) < 2 * min_cluster_size:
            labels = np.zeros(len(own_idx), dtype=int)  # one tribe: too few to split
        else:
            labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(
                x_scaled[own_idx]
            )
            if not (set(labels) - {-1}):
                # no density structure found: the action's own parent cluster
                # is the single tribe rather than leaving every state unassigned
                labels = np.zeros(len(own_idx), dtype=int)
        for label in sorted(set(labels) - {-1}):
            member_idx = own_idx[labels == label]
            tribe_id = f"{action}::t{label}"
            margins, rivals = [], []
            successes = 0
            for i in member_idx:
                rec = build.records[i]
                # margin over TRAINED actions only: constant-prior actions carry
                # no state-dependent signal, so they cannot make a state ambiguous
                ranked = sorted(
                    ((a, rec.scores_cal[a]) for a in trained), key=lambda kv: -kv[1]
                )
                margins.append(ranked[0][1] - ranked[1][1])
                rivals.append(ranked[1][0])
                successes += int(rec.outcome_success)
                by_state[rec.state_id] = tribe_id
            top2_margin = float(np.mean(margins))
            ambiguous = top2_margin < ambiguity_margin
            rival = (
                max(set(rivals), key=rivals.count) if ambiguous and rivals else None
            )
            tribes.append(
                Tribe(
                    tribe_id=tribe_id,
                    action_id=action,
                    centroid={
                        a: float(v)
                        for a, v in zip(
                            trained, x_scaled[member_idx].mean(axis=0), strict=True
                        )
                    },
                    member_count=len(member_idx),
                    stats=TribeStats(
                        n_states=len(member_idx),
                        success_rate=successes / len(member_idx),
                        top2_margin=top2_margin,
                        ambiguous=ambiguous,
                        rival_action_id=rival,
                    ),
                )
            )
    for record in build.records:
        record.tribe_id = by_state.get(record.state_id)
    return tribes, by_state


def leak_screen_features(
    frame: pl.DataFrame,
    feature_names: list[str],
    outcome_by_episode: dict[str, bool],
    flag_auc: float = LEAK_AUC_THRESHOLD,  # one threshold with the ingest screen
    episode_col: str = "episode_id",
) -> dict[str, float]:
    """P2-style statistical screen on the Q_PSI feature columns: a feature that
    is constant within every episode AND near-perfectly separates the outcome
    is post-hoc leakage. Returns the flagged features with their AUC."""
    labels = np.array(
        [int(outcome_by_episode[e]) for e in frame[episode_col].to_list()]
    )
    flagged: dict[str, float] = {}
    for name in feature_names:
        values = frame[name].to_numpy()
        per_episode = frame.select(episode_col, pl.col(name)).group_by(episode_col)
        constant_within = per_episode.n_unique()[name].max() == 1
        if not constant_within or len(set(values)) < 2:
            continue
        auc = float(roc_auc_score(labels, values))
        if auc >= flag_auc or auc <= 1 - flag_auc:
            flagged[name] = auc
    return flagged
