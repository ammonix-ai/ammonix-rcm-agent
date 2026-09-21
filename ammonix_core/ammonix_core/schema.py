"""Schema v0.1: every entity the Ammonix pipeline reads or writes, stage by stage.

Transcribed from ammonix_schema_v0.1.md, which is the implementation contract.
All identifiers are string ULIDs. Every build artefact carries a version and a
content hash, and the BasisManifest pins the complete set. Raw Data never
crosses Stage A: everything downstream refers to it only by state_id and sha256.
"""

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

StateId = str
ExampleId = str
ActionId = str
TribeId = str
SkillId = str
FeatureValue = float | int | bool | str

# ---------------------------------------------------------------------------
# Stage 0. Ingest: Example, State, Data, Action, Outcome
# ---------------------------------------------------------------------------


class Outcome(BaseModel):
    success: bool  # the core reward signal, one per Example
    score: float | None = None  # optional graded outcome in [0, 1]
    label: str | None = None  # optional domain code, e.g. a discharge diagnosis


class RawData(BaseModel):
    modality: Literal["signal", "image", "text", "tabular", "composite"]
    uri: str | None = None  # blob pointer; raw Data never enters the Basis
    inline: dict | None = None  # small tabular payloads only
    media_type: str | None = None
    sha256: str  # content hash; the reproducibility anchor


class RawState(BaseModel):
    state_id: StateId
    example_id: ExampleId
    seq: int  # position within the Example, 0-based
    data: RawData
    action_raw: str  # the Action exactly as logged
    next_state_id: StateId | None  # None marks the terminal state
    occurred_at: datetime | None = None


class Example(BaseModel):
    example_id: ExampleId
    state_ids: list[StateId]  # ordered
    outcome: Outcome  # exactly one per Example
    meta: dict = Field(default_factory=dict)  # never used as features unless promoted


# ---------------------------------------------------------------------------
# Stage A. Q_PSI tokenisation
# ---------------------------------------------------------------------------


class HistorySpec(BaseModel):
    lookback_states: int  # how many prior states of the same Example feed this feature
    aggregation: Literal["last", "mean", "max", "min", "delta", "count", "custom"]


class FeatureSpec(BaseModel):
    name: str  # unique; the name must explain what it measures
    dtype: Literal["float", "int", "bool", "category", "ordinal"]
    unit: str | None = None
    description: str
    categories: list[str] | None = None
    valid_range: tuple[float, float] | None = None
    missing_policy: Literal["error", "constant", "median", "flag"] = "error"
    history: HistorySpec | None = None  # set when the feature is derived from earlier states


class DeterminismPin(BaseModel):
    code_sha256: str
    seeds: dict[str, int] = Field(default_factory=dict)
    # for LLM extractors: weights hash, temperature 0, schema-constrained
    # decoding; pinned, not assumed
    llm: dict | None = None


class Tokenizer(BaseModel):
    tokenizer_id: str
    version: str
    kind: Literal["algorithm", "neural_net", "llm_extractor", "composite"]
    features: list[FeatureSpec]
    pin: DeterminismPin  # same Data must always yield the same features


class QPsiRecord(BaseModel):
    """One row per State; these rows are the Q_PSI file. No raw Data field, by design."""

    state_id: StateId
    example_id: ExampleId
    seq: int
    features: dict[str, FeatureValue]  # keys come from the Tokenizer's FeatureSpecs
    action_id: ActionId  # canonical, after the Action map
    outcome_success: bool  # inherited from the Example
    outcome_score: float = 1.0
    next_state_id: StateId | None  # keeps the episodic connectivity


# ---------------------------------------------------------------------------
# Stage B, step 1. The Action map
# ---------------------------------------------------------------------------


class CanonicalAction(BaseModel):
    action_id: ActionId
    name: str
    description: str
    # JSON Schema of the executed Action's payload; used by M1 (constrained
    # decoding) and M2 (validation)
    payload_schema: dict


class ActionMapRule(BaseModel):
    pattern: str  # exact string or regex over action_raw
    action_id: ActionId
    note: str | None = None  # provenance of the merge


class ActionMap(BaseModel):
    version: str
    actions: list[CanonicalAction]
    rules: list[ActionMapRule]  # first match wins; unmatched raw labels fail the build


class CoverageReport(BaseModel):
    """The 100x check."""

    states_total: int
    counts: dict[ActionId, int]
    min_states_per_action: int  # from AmmonixConfig, default 100
    violations: list[ActionId]  # merge further or drop before training


# ---------------------------------------------------------------------------
# Stage B, step 2. The AMX swarm
# ---------------------------------------------------------------------------


class FoldPlan(BaseModel):
    n_folds: int = 5
    group_by: Literal["example", "state"] = "example"
    stratify_by: Literal["outcome", "action", "none"] = "outcome"
    seed: int
    assignment: dict[ExampleId, int]  # every Example lands in exactly one fold


class LabelPolicy(BaseModel):
    scope: Literal["own_action_states", "all_states"] = "own_action_states"
    label: Literal["outcome_success", "action_taken"] = "outcome_success"
    outcome_weighting: bool = False  # use outcome_score as a sample weight
    class_weight: Literal["balanced", "none"] = "balanced"


class ClassifierSpec(BaseModel):
    model: str  # e.g. 'gbdt', 'logreg', 'mlp'
    hyperparams: dict = Field(default_factory=dict)
    feature_subset: list[str] | None = None  # None = all registered features


class SwarmConfig(BaseModel):
    fold_plan: FoldPlan
    label_policy: LabelPolicy
    default_classifier: ClassifierSpec
    overrides: dict[ActionId, ClassifierSpec] = Field(default_factory=dict)


class FoldMetrics(BaseModel):
    auroc: float
    auprc: float
    n_pos: int
    n_neg: int


class TrainedFoldModel(BaseModel):
    action_id: ActionId
    fold: int
    artefact_uri: str
    artefact_sha256: str
    metrics: FoldMetrics  # measured on the hold-out fold


# ---------------------------------------------------------------------------
# Stage B, step 3. The Universe
# ---------------------------------------------------------------------------


class Attribution(BaseModel):
    feature: str
    value: FeatureValue  # the feature's value in this state
    contribution: float  # signed importance, e.g. SHAP


class UniverseRecord(BaseModel):
    """One row per State, always scored by its hold-out model."""

    state_id: StateId
    example_id: ExampleId
    fold: int
    scores_raw: dict[ActionId, float]  # independent per classifier; never compare uncalibrated
    scores_cal: dict[ActionId, float]  # after per-Action calibration
    scores_std: dict[ActionId, float] | None = None  # fold-ensemble spread: the
    # model's own uncertainty per action; None on bases built before v0.4
    true_action_id: ActionId
    outcome_success: bool
    outcome_score: float
    top_features: list[Attribution]  # for the attribution_target classifier (config)
    tribe_id: TribeId | None = None  # filled by the region step


class Calibrator(BaseModel):
    action_id: ActionId
    method: Literal["isotonic", "platt", "beta"] = "isotonic"
    artefact_uri: str
    # fitted on the out-of-fold scores of the Action's own states against
    # Outcome, which is the only place ground truth for P(success | state, a) exists


class UniverseIndex(BaseModel):
    """Nearest-state retrieval for live cases."""

    metric: Literal["euclidean", "cosine", "mahalanobis"] = "euclidean"
    scaler_uri: str  # standardisation fitted on the Q_PSI features
    index_uri: str  # ANN artefact (HNSW or FAISS)


# ---------------------------------------------------------------------------
# Basis storage
# ---------------------------------------------------------------------------


class BasisManifest(BaseModel):
    basis_id: str
    built_at: datetime
    dataset_fingerprint: str  # hash over all Example and State ids plus Outcomes
    tokenizer: Tokenizer
    action_map_version: str
    swarm_config_sha256: str
    calibrators: list[Calibrator]
    index: UniverseIndex
    coverage: CoverageReport
    metrics: dict[ActionId, FoldMetrics]  # pooled over folds


# ---------------------------------------------------------------------------
# Regions: clusters and tribes
# ---------------------------------------------------------------------------


class TribeStats(BaseModel):
    n_states: int
    success_rate: float
    top2_margin: float  # mean gap between best and second-best calibrated scores
    ambiguous: bool  # top2_margin < config.ambiguity_margin
    rival_action_id: ActionId | None  # the nearly tied Action, when ambiguous


class Tribe(BaseModel):
    tribe_id: TribeId
    action_id: ActionId  # the parent cluster is the canonical Action itself
    centroid: dict[str, float]  # in scaled feature space
    member_count: int  # membership itself lives in UniverseRecord.tribe_id
    stats: TribeStats
    skill_id: SkillId | None = None  # None = inherit the cluster's skill


class RegionModelSpec(BaseModel):
    algorithm: Literal["kmeans", "hdbscan", "gmm"] = "hdbscan"
    feature_space: Literal["full", "top_attributed", "embedding"] = "full"
    params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# The Skill layer
# ---------------------------------------------------------------------------


class Scope(BaseModel):
    level: Literal["tribe", "cluster", "default"]
    ref: str | None = None  # tribe_id or action_id; None for the default skill


class ContextSource(BaseModel):
    """Where each Context Data field comes from at runtime."""

    field: str
    source: Literal["case_record", "external_api", "user_input", "document"]
    locator: str  # path, endpoint or query


class DecisionQuestion(BaseModel):
    """The ask-before-deciding structure."""

    text_template: str  # rendered with live features and Context Data
    answer_schema: dict  # JSON Schema of admissible answers
    branches: dict[str, ActionId]  # answer value maps to the Action to execute
    on_unavailable: Literal["escalate"] = "escalate"  # outside information means a human


class ExpectedResultSpec(BaseModel):
    """Decision 1 of the guide; the loop terminates on this."""

    kind: Literal["universe_outcome", "reference_exemplar", "schema", "rules", "composite"]
    payload_schema: dict | None = None  # defaults to the Action's payload_schema
    rules: list[str] | None = None  # named validation predicates, evaluated by M2
    exemplar_from_neighbours: bool = False  # source exemplars from the retrieved tribe
    tolerance: dict = Field(default_factory=dict)


class Skill(BaseModel):
    skill_id: SkillId
    version: str
    scope: Scope
    kind: Literal["execute", "ask_before_deciding", "escalate"]
    context_schema: dict  # JSON Schema of the required Context Data
    context_sources: list[ContextSource]
    m1_prompt_ref: str  # produced offline by harness optimisation
    expected_result: ExpectedResultSpec
    question: DecisionQuestion | None = None  # required when kind = ask_before_deciding


# ---------------------------------------------------------------------------
# The harness (offline, once)
# ---------------------------------------------------------------------------


class LLMPin(BaseModel):
    """The frozen local model."""

    name: str
    weights_sha256: str
    quantisation: str | None = None
    temperature: float = 0.0
    constrained_decoding: bool = True  # grammar or JSON-schema mode


class PromptTemplate(BaseModel):
    template_id: str
    version: str
    text: str
    output_schema: dict | None = None  # for M1 prompts: the Action's payload_schema


class HarnessArtefacts(BaseModel):
    """Designed once, never modified at runtime."""

    harness_id: str
    llm: LLMPin
    m1_prompts: dict[SkillId, PromptTemplate]
    m2_prompt: PromptTemplate
    optimisation_log_uri: str


# ---------------------------------------------------------------------------
# Inference: live-state retrieval
# ---------------------------------------------------------------------------


class Neighbour(BaseModel):
    state_id: StateId
    distance: float


class RecommendationPolicy(BaseModel):
    action_source: Literal["swarm_scores", "neighbour_vote", "blend"] = "swarm_scores"
    inference_model: Literal["refit_full", "fold_ensemble"] = "refit_full"
    k_neighbours: int = 25


class LiveCase(BaseModel):
    case_id: str
    data: RawData
    context: dict = Field(default_factory=dict)  # filled against the Skill's context_schema


class RetrievalResult(BaseModel):
    """Everything the loop needs, resolved before M1 runs."""

    case_id: str
    features: dict[str, FeatureValue]  # same Tokenizer version as the Basis, enforced
    scores_cal: dict[ActionId, float]
    neighbours: list[Neighbour]
    tribe_id: TribeId
    recommended_action_id: ActionId  # the ToDo action
    ambiguous: bool  # live margin below threshold, or the tribe is flagged
    skill_id: SkillId  # after tribe, cluster, default resolution
    expected_result: dict  # ExpectedResultSpec resolved with live values


# ---------------------------------------------------------------------------
# The verifiable reward loop
# ---------------------------------------------------------------------------


class ActionObject(BaseModel):
    """What M1 emits; structure, not prose."""

    action_id: ActionId
    payload: dict  # must validate against CanonicalAction.payload_schema
    llm: LLMPin


class CheckResult(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)  # schema paths or rule ids that failed


class IterationRecord(BaseModel):
    n: int
    m1_prompt_sha256: str
    action: ActionObject | None = None  # None when M1 output failed to parse
    check: CheckResult
    adjustment: str | None = None  # what M2 changed before the retry


class EscalationRecord(BaseModel):
    reason: Literal["iteration_cap", "needs_outside_information", "unresolvable"]
    question: str | None = None  # for ask-before-deciding hand-offs
    handed_to: str | None = None


class ExecutionTrace(BaseModel):
    """One per live case; the audit record."""

    case_id: str
    basis_id: str
    harness_id: str
    retrieval: RetrievalResult
    iterations: list[IterationRecord]
    status: Literal["executed", "escalated"]
    escalation: EscalationRecord | None = None
    final: ActionObject | None = None
    started_at: datetime
    finished_at: datetime


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AmmonixConfig(BaseModel):
    """The defaults to confirm, in one place."""

    n_folds: int = 5
    min_states_per_action: int = 100
    ambiguity_margin: float = 0.05
    attribution_target: Literal["argmax", "true_action"] = "argmax"
    calibration: Literal["isotonic", "platt", "beta"] = "isotonic"
    recommendation: RecommendationPolicy = Field(default_factory=RecommendationPolicy)
    max_iterations: int = 3
    region_model: RegionModelSpec = Field(default_factory=RegionModelSpec)


class StateHandle(Protocol):
    """Opaque handle to an environment state; concrete type is the environment's own."""
