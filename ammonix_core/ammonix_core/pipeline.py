"""Minimal generic pipeline: tokenize -> folds -> swarm -> OOF -> calibrate -> recommend.

First exercised end to end at M1 on the toy world; later milestones extend it
(text extractors, regions, skills) without changing these contracts. Semantics
follow the schema's pinned decisions: each per-Action classifier estimates
P(success | state, action) trained on that Action's own states (LabelPolicy
default), the Universe is out-of-fold permanently, live scoring uses
refit-full models, and scores are only compared after per-Action calibration.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ammonix_core.schema import (
    ActionId,
    ClassifierSpec,
    Example,
    FeatureSpec,
    FoldMetrics,
    FoldPlan,
    QPsiRecord,
    RawState,
)


def tokenize_tabular(
    states: list[RawState],
    examples: list[Example],
    feature_specs: list[FeatureSpec],
    action_map: dict[str, ActionId],
) -> list[QPsiRecord]:
    """Passthrough tabular tokenizer: inline Data columns become features 1:1.

    Unmatched raw action labels fail the build, per the ActionMap contract.
    """
    outcome_by_example = {e.example_id: e.outcome for e in examples}
    names = [spec.name for spec in feature_specs]
    records = []
    for state in states:
        if state.action_raw not in action_map:
            raise ValueError(f"unmapped raw action label: {state.action_raw!r}")
        inline = state.data.inline or {}
        missing = [n for n in names if n not in inline]
        if missing:
            raise ValueError(f"state {state.state_id}: missing features {missing}")
        outcome = outcome_by_example[state.example_id]
        records.append(
            QPsiRecord(
                state_id=state.state_id,
                example_id=state.example_id,
                seq=state.seq,
                features={n: inline[n] for n in names},
                action_id=action_map[state.action_raw],
                outcome_success=outcome.success,
                outcome_score=outcome.score if outcome.score is not None else 1.0,
                next_state_id=state.next_state_id,
            )
        )
    return records


def build_fold_plan(examples: list[Example], n_folds: int, seed: int) -> FoldPlan:
    """Grouped, outcome-stratified fold assignment: one fold per Example."""
    example_ids = [e.example_id for e in examples]
    outcomes = [int(e.outcome.success) for e in examples]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assignment: dict[str, int] = {}
    dummy_x = np.zeros(len(example_ids))
    for fold, (_, test_idx) in enumerate(
        splitter.split(dummy_x, outcomes, groups=example_ids)
    ):
        for i in test_idx:
            assignment[example_ids[i]] = fold
    return FoldPlan(n_folds=n_folds, seed=seed, assignment=assignment)


def _feature_value(value: object, name: str) -> float:
    """Numeric encoding of one feature value: None -> NaN (gbdt handles NaN
    natively; logreg imputes), bool -> 0/1. Raw strings are refused loudly:
    categorical features must be one-hot encoded by the tokenizer
    (FeatureSpec dtype 'category'), never cast to float here."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        raise ValueError(
            f"feature {name!r} carries a string value {value!r}: encode "
            "categoricals as one-hot FeatureSpecs at the tokenizer stage"
        )
    return float(value)


def matrix_from_rows(rows: list[dict], names: list[str]) -> np.ndarray:
    """Shared feature-matrix builder for QPsiRecord.features dicts and live rows."""
    return np.array(
        [[_feature_value(row[n], n) for n in names] for row in rows], dtype=float
    )


def feature_matrix(records: list[QPsiRecord], names: list[str]) -> np.ndarray:
    return matrix_from_rows([rec.features for rec in records], names)


@dataclass
class SwarmResult:
    """Everything the swarm learns, per Action: OOF scores, metrics, refit model."""

    oof_scores: dict[ActionId, np.ndarray]  # aligned with oof_labels rows
    oof_labels: dict[ActionId, np.ndarray]
    fold_metrics: dict[ActionId, list[FoldMetrics]]
    pooled_auroc: dict[ActionId, float]
    refit_models: dict[ActionId, HistGradientBoostingClassifier]
    calibrators: dict[ActionId, IsotonicRegression]
    feature_names: list[str]
    fold_models: dict[ActionId, dict[int, object]] = field(default_factory=dict)

    @property
    def mean_oof_auroc(self) -> float:
        return float(np.mean(list(self.pooled_auroc.values())))


def build_classifier(seed: int, spec: ClassifierSpec | None = None):
    """ClassifierSpec.model -> estimator. Supported: 'gbdt' (default), 'logreg'.

    gbdt tolerates NaN features natively; logreg gets median imputation and
    standardisation so it holds the same feature contract.
    """
    model = spec.model if spec else "gbdt"
    hyperparams = dict(spec.hyperparams) if spec else {}
    if model == "gbdt":
        return HistGradientBoostingClassifier(random_state=seed, **hyperparams)
    if model == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(random_state=seed, max_iter=1000, **hyperparams),
        )
    raise ValueError(f"ClassifierSpec.model {model!r} not supported (gbdt, logreg)")


def _new_model(seed: int, hyperparams: dict | None = None, model: str = "gbdt"):
    return build_classifier(
        seed, ClassifierSpec(model=model, hyperparams=hyperparams or {})
    )


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """LabelPolicy class_weight='balanced': n / (2 * n_class) per sample.

    The single implementation; universe.py imports this one.
    """
    weights = len(y) / (2.0 * np.bincount(y, minlength=2).clip(min=1))
    return weights[y]


_balanced_weights = balanced_weights  # backward-compatible alias


def fit_classifier(model, x: np.ndarray, y: np.ndarray, sample_weight=None):
    """fit() with sample_weight routed correctly for plain estimators and
    sklearn Pipelines (which need it addressed to their final step)."""
    if sample_weight is None:
        return model.fit(x, y)
    if hasattr(model, "steps"):
        final_step = model.steps[-1][0]
        return model.fit(x, y, **{f"{final_step}__sample_weight": sample_weight})
    return model.fit(x, y, sample_weight=sample_weight)


def train_swarm(
    records: list[QPsiRecord],
    fold_plan: FoldPlan,
    feature_names: list[str],
    seed: int = 0,
    hyperparams: dict | None = None,
    balanced: bool = False,
    classifier: ClassifierSpec | None = None,
) -> SwarmResult:
    """Per-Action fold models on own-action states, labelled by Outcome.

    Every state is scored only by its hold-out fold model (the OOF Universe);
    a refit-full model per Action serves live scoring; isotonic calibrators
    are fitted on the OOF scores, the only place ground truth exists.
    `classifier` selects the model family (ClassifierSpec; default gbdt with
    `hyperparams`); `balanced` applies the LabelPolicy's class weighting.
    Fold models are kept on the result: they are the ensemble whose spread
    is the swarm's own uncertainty.
    """
    if classifier is None:
        classifier = ClassifierSpec(model="gbdt", hyperparams=hyperparams or {})
    action_ids = sorted({r.action_id for r in records})
    oof_scores: dict[ActionId, np.ndarray] = {}
    oof_labels: dict[ActionId, np.ndarray] = {}
    fold_metrics: dict[ActionId, list[FoldMetrics]] = {}
    pooled_auroc: dict[ActionId, float] = {}
    refit_models = {}
    calibrators = {}
    fold_models: dict[ActionId, dict[int, object]] = {}

    for action_id in action_ids:
        rows = [r for r in records if r.action_id == action_id]
        x = feature_matrix(rows, feature_names)
        y = np.array([r.outcome_success for r in rows], dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"action {action_id!r} has a single outcome class over its "
                f"{len(rows)} states: route it to a constant prior instead of "
                "training a classifier"
            )
        folds = np.array([fold_plan.assignment[r.example_id] for r in rows])

        scores = np.full(len(rows), np.nan)
        metrics: list[FoldMetrics] = []
        fold_models[action_id] = {}
        for fold in range(fold_plan.n_folds):
            holdout = folds == fold
            if not holdout.any():
                continue
            train_y = y[~holdout]
            if len(np.unique(train_y)) < 2:
                raise ValueError(
                    f"action {action_id!r}, fold {fold}: training partition is "
                    f"single-class ({len(train_y)} rows) - too few states per "
                    "fold for this action"
                )
            model = fit_classifier(
                build_classifier(seed, classifier), x[~holdout], train_y,
                sample_weight=balanced_weights(train_y) if balanced else None,
            )
            fold_models[action_id][fold] = model
            scores[holdout] = model.predict_proba(x[holdout])[:, 1]
            if len(np.unique(y[holdout])) == 2:
                metrics.append(
                    FoldMetrics(
                        auroc=float(roc_auc_score(y[holdout], scores[holdout])),
                        auprc=float(average_precision_score(y[holdout], scores[holdout])),
                        n_pos=int(y[holdout].sum()),
                        n_neg=int((1 - y[holdout]).sum()),
                    )
                )
        if np.isnan(scores).any():
            raise ValueError(
                f"unscored OOF rows for {action_id!r}: some states fell in "
                "folds where the action never appears in training"
            )

        oof_scores[action_id] = scores
        oof_labels[action_id] = y
        fold_metrics[action_id] = metrics
        pooled_auroc[action_id] = float(roc_auc_score(y, scores))
        refit_models[action_id] = fit_classifier(
            build_classifier(seed, classifier), x, y,
            sample_weight=balanced_weights(y) if balanced else None,
        )
        calibrators[action_id] = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(scores, y)

    return SwarmResult(
        oof_scores=oof_scores,
        oof_labels=oof_labels,
        fold_metrics=fold_metrics,
        pooled_auroc=pooled_auroc,
        refit_models=refit_models,
        calibrators=calibrators,
        feature_names=feature_names,
        fold_models=fold_models,
    )


def score_live(
    swarm: SwarmResult,
    features_rows: list[dict],
    inference_model: str = "refit_full",
) -> tuple[dict[ActionId, np.ndarray], dict[ActionId, np.ndarray]]:
    """Calibrated live scores per action, plus the fold-ensemble spread.

    RecommendationPolicy.inference_model: 'refit_full' scores with the
    all-data model (schema default); 'fold_ensemble' averages the fold
    models - whose score distribution is what the calibrators were fitted
    on. The spread (std over fold models) is returned either way when fold
    models are present: it is the swarm's own per-state uncertainty.
    """
    if inference_model not in ("refit_full", "fold_ensemble"):
        raise ValueError(f"unknown inference_model {inference_model!r}")
    x = matrix_from_rows(features_rows, swarm.feature_names)
    calibrated: dict[ActionId, np.ndarray] = {}
    spread: dict[ActionId, np.ndarray] = {}
    for action_id, refit in swarm.refit_models.items():
        per_fold = swarm.fold_models.get(action_id, {})
        if per_fold:
            fold_raw = np.vstack(
                [m.predict_proba(x)[:, 1] for m in per_fold.values()]
            )
            fold_cal = np.vstack(
                [swarm.calibrators[action_id].predict(r) for r in fold_raw]
            )
            spread[action_id] = fold_cal.std(axis=0)
        if inference_model == "fold_ensemble" and per_fold:
            calibrated[action_id] = fold_cal.mean(axis=0)
        else:
            raw = refit.predict_proba(x)[:, 1]
            calibrated[action_id] = swarm.calibrators[action_id].predict(raw)
    return calibrated, spread


def recommend(
    swarm: SwarmResult,
    features_rows: list[dict],
    inference_model: str = "refit_full",
) -> list[ActionId]:
    """Argmax over per-Action calibrated live scores (RecommendationPolicy
    defaults: action_source swarm_scores, inference_model refit_full)."""
    calibrated, _ = score_live(swarm, features_rows, inference_model)
    action_ids = sorted(calibrated)
    stacked = np.column_stack([calibrated[a] for a in action_ids])
    return [action_ids[i] for i in stacked.argmax(axis=1)]
