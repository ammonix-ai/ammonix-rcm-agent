# Ammonix generic engine: data schema v0.1

Draft for review. This document defines every entity the Ammonix pipeline reads or writes, stage by stage, following the Engineering Guide. Types are given in Pydantic notation so the document doubles as the implementation contract; storage is deliberately neutral (Parquet, SQLite and Postgres all fit the layout at the end). Nothing here is domain-specific: an application is fully defined by one Tokenizer, one Action map and its Skills.

## Pipeline map

| Guide stage | Entities written |
|---|---|
| Ingest | `Example`, `RawState`, `RawData`, `Outcome` |
| Q_PSI (Stage A) | `Tokenizer`, `FeatureSpec`, `QPsiRecord` |
| Action map (Stage B, step 1) | `CanonicalAction`, `ActionMap`, `CoverageReport` |
| AMX swarm (Stage B, step 2) | `FoldPlan`, `LabelPolicy`, `SwarmConfig`, `TrainedFoldModel` |
| Universe (Stage B, step 3) | `UniverseRecord`, `Calibrator`, `UniverseIndex` |
| Basis storage | `BasisManifest` |
| Regions | `Tribe`, `RegionModelSpec` |
| Skill layer | `Skill`, `ExpectedResultSpec`, `DecisionQuestion` |
| Harness (offline) | `HarnessArtefacts`, `PromptTemplate`, `LLMPin` |
| Inference | `LiveCase`, `RetrievalResult`, `RecommendationPolicy` |
| Verifiable reward loop | `ActionObject`, `IterationRecord`, `ExecutionTrace` |
| Configuration | `AmmonixConfig` |

## Conventions

All identifiers are string ULIDs. Every build artefact carries a version and a content hash, and the `BasisManifest` pins the complete set, so any Universe row is traceable to the exact tokenizer, action map, fold seed and model that produced it. Raw Data never crosses Stage A: everything downstream refers to it only by `state_id` and `sha256`.

```python
StateId = ExampleId = ActionId = TribeId = SkillId = str
FeatureValue = float | int | bool | str
```

## Stage 0. Ingest: Example, State, Data, Action, Outcome

```python
class Outcome(BaseModel):
    success: bool                     # the core reward signal, one per Example
    score: float | None = None        # optional graded outcome in [0, 1]
    label: str | None = None          # optional domain code, e.g. a discharge diagnosis

class RawData(BaseModel):
    modality: Literal['signal', 'image', 'text', 'tabular', 'composite']
    uri: str | None = None            # blob pointer; raw Data never enters the Basis
    inline: dict | None = None        # small tabular payloads only
    media_type: str | None = None
    sha256: str                       # content hash; the reproducibility anchor

class RawState(BaseModel):
    state_id: StateId
    example_id: ExampleId
    seq: int                          # position within the Example, 0-based
    data: RawData
    action_raw: str                   # the Action exactly as logged
    next_state_id: StateId | None     # None marks the terminal state
    occurred_at: datetime | None = None

class Example(BaseModel):
    example_id: ExampleId
    state_ids: list[StateId]          # ordered
    outcome: Outcome                  # exactly one per Example
    meta: dict = {}                   # site, cohort, device; never used as features unless promoted
```

`Outcome.success` stays boolean because the swarm's reward is binary in the guide; `score` exists so a generic build can weight samples by a graded outcome without changing the schema.

## Stage A. Q_PSI tokenisation

```python
class HistorySpec(BaseModel):
    lookback_states: int              # how many prior states of the same Example feed this feature
    aggregation: Literal['last', 'mean', 'max', 'min', 'delta', 'count', 'custom']

class FeatureSpec(BaseModel):
    name: str                         # unique; the name must explain what it measures
    dtype: Literal['float', 'int', 'bool', 'category', 'ordinal']
    unit: str | None = None
    description: str
    categories: list[str] | None = None
    valid_range: tuple[float, float] | None = None
    missing_policy: Literal['error', 'constant', 'median', 'flag'] = 'error'
    history: HistorySpec | None = None    # set when the feature is derived from earlier states

class DeterminismPin(BaseModel):
    code_sha256: str
    seeds: dict[str, int] = {}
    llm: dict | None = None           # for LLM extractors: weights hash, temperature 0,
                                      # schema-constrained decoding; pinned, not assumed

class Tokenizer(BaseModel):
    tokenizer_id: str
    version: str
    kind: Literal['algorithm', 'neural_net', 'llm_extractor', 'composite']
    features: list[FeatureSpec]
    pin: DeterminismPin               # same Data must always yield the same features

class QPsiRecord(BaseModel):          # one row per State; these rows are the Q_PSI file
    state_id: StateId
    example_id: ExampleId
    seq: int
    features: dict[str, FeatureValue] # keys come from the Tokenizer's FeatureSpecs
    action_id: ActionId               # canonical, after the Action map
    outcome_success: bool             # inherited from the Example
    outcome_score: float = 1.0
    next_state_id: StateId | None     # keeps the episodic connectivity
    # no raw Data field, by design
```

History features are computed by the tokenizer at build time and stored flat on the state's row, so the Universe and the retrieval index never need to walk the next-state chain themselves.

## Stage B, step 1. The Action map

```python
class CanonicalAction(BaseModel):
    action_id: ActionId
    name: str
    description: str
    payload_schema: dict              # JSON Schema of the executed Action's payload;
                                      # used by M1 (constrained decoding) and M2 (validation)

class ActionMapRule(BaseModel):
    pattern: str                      # exact string or regex over action_raw
    action_id: ActionId
    note: str | None = None           # provenance of the merge

class ActionMap(BaseModel):
    version: str
    actions: list[CanonicalAction]
    rules: list[ActionMapRule]        # first match wins; unmatched raw labels fail the build

class CoverageReport(BaseModel):      # the 100x check
    states_total: int
    counts: dict[ActionId, int]
    min_states_per_action: int        # from AmmonixConfig, default 100
    violations: list[ActionId]        # merge further or drop before training
```

The `payload_schema` sits here rather than in the Skill layer on purpose: it is the single definition of what an executed instance of this Action looks like, and both the harness prompts and the M2 checks reference it.

## Stage B, step 2. The AMX swarm

```python
class FoldPlan(BaseModel):
    n_folds: int = 5
    group_by: Literal['example', 'state'] = 'example'    # see Decisions, point 2
    stratify_by: Literal['outcome', 'action', 'none'] = 'outcome'
    seed: int
    assignment: dict[ExampleId, int]  # every Example lands in exactly one fold

class LabelPolicy(BaseModel):         # see Decisions, point 1
    scope: Literal['own_action_states', 'all_states'] = 'own_action_states'
    label: Literal['outcome_success', 'action_taken'] = 'outcome_success'
    outcome_weighting: bool = False   # use outcome_score as a sample weight
    class_weight: Literal['balanced', 'none'] = 'balanced'

class ClassifierSpec(BaseModel):
    model: str                        # e.g. 'gbdt', 'logreg', 'mlp'
    hyperparams: dict = {}
    feature_subset: list[str] | None = None    # None = all registered features

class SwarmConfig(BaseModel):
    fold_plan: FoldPlan
    label_policy: LabelPolicy
    default_classifier: ClassifierSpec
    overrides: dict[ActionId, ClassifierSpec] = {}

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
    metrics: FoldMetrics              # measured on the hold-out fold
```

With the default `LabelPolicy`, the classifier for Action a is trained on the states where a was actually taken, labelled by the Example's Outcome. Its score therefore estimates P(success | state, a), which is the "Outcome is the reward, not imitation" reading of the guide. The imitation setting (`all_states` + `action_taken`) is kept only for ablation.

## Stage B, step 3. The Universe

```python
class Attribution(BaseModel):
    feature: str
    value: FeatureValue               # the feature's value in this state
    contribution: float               # signed importance, e.g. SHAP

class UniverseRecord(BaseModel):      # one row per State, always scored by its hold-out model
    state_id: StateId
    example_id: ExampleId
    fold: int
    scores_raw: dict[ActionId, float] # independent per classifier; never compare uncalibrated
    scores_cal: dict[ActionId, float] # after per-Action calibration
    true_action_id: ActionId
    outcome_success: bool
    outcome_score: float
    top_features: list[Attribution]   # for the attribution_target classifier (config)
    tribe_id: TribeId | None = None   # filled by the region step

class Calibrator(BaseModel):
    action_id: ActionId
    method: Literal['isotonic', 'platt', 'beta'] = 'isotonic'
    artefact_uri: str
    # fitted on the out-of-fold scores of the Action's own states against Outcome,
    # which is the only place ground truth for P(success | state, a) exists

class UniverseIndex(BaseModel):       # nearest-state retrieval for live cases
    metric: Literal['euclidean', 'cosine', 'mahalanobis'] = 'euclidean'
    scaler_uri: str                   # standardisation fitted on the Q_PSI features
    index_uri: str                    # ANN artefact (HNSW or FAISS)
```

## Basis storage

```python
class BasisManifest(BaseModel):
    basis_id: str
    built_at: datetime
    dataset_fingerprint: str          # hash over all Example and State ids plus Outcomes
    tokenizer: Tokenizer
    action_map_version: str
    swarm_config_sha256: str
    calibrators: list[Calibrator]
    index: UniverseIndex
    coverage: CoverageReport
    metrics: dict[ActionId, FoldMetrics]   # pooled over folds
```

On disk a Basis is three files plus an artefact directory:

```
basis/
  manifest.json          # BasisManifest
  qpsi.parquet           # QPsiRecord rows
  universe.parquet       # UniverseRecord rows
  artefacts/             # fold models, refit models, calibrators, scaler, ANN index
```

## Regions: clusters and tribes

The action cluster needs no entity of its own: it is the set of Universe rows sharing one `action_id`. Tribes are materialised.

```python
class TribeStats(BaseModel):
    n_states: int
    success_rate: float
    top2_margin: float                # mean gap between best and second-best calibrated scores
    ambiguous: bool                   # top2_margin < config.ambiguity_margin
    rival_action_id: ActionId | None  # the nearly tied Action, when ambiguous

class Tribe(BaseModel):
    tribe_id: TribeId
    action_id: ActionId               # the parent cluster is the canonical Action itself
    centroid: dict[str, float]        # in scaled feature space
    member_count: int                 # membership itself lives in UniverseRecord.tribe_id
    stats: TribeStats
    skill_id: SkillId | None = None   # None = inherit the cluster's skill

class RegionModelSpec(BaseModel):
    algorithm: Literal['kmeans', 'hdbscan', 'gmm'] = 'hdbscan'
    feature_space: Literal['full', 'top_attributed', 'embedding'] = 'full'
    params: dict = {}
```

`TribeStats.ambiguous` is what routes a region to an ask-before-deciding skill: it is a property of the Universe, computed at build time, not something decided at runtime.

## The Skill layer

```python
class Scope(BaseModel):
    level: Literal['tribe', 'cluster', 'default']
    ref: str | None = None            # tribe_id or action_id; None for the default skill

class ContextSource(BaseModel):       # where each Context Data field comes from at runtime
    field: str
    source: Literal['case_record', 'external_api', 'user_input', 'document']
    locator: str                      # path, endpoint or query

class DecisionQuestion(BaseModel):    # the ask-before-deciding structure
    text_template: str                # rendered with live features and Context Data
    answer_schema: dict               # JSON Schema of admissible answers
    branches: dict[str, ActionId]     # answer value maps to the Action to execute
    on_unavailable: Literal['escalate'] = 'escalate'   # outside information means a human

class ExpectedResultSpec(BaseModel):  # decision 1 of the guide; the loop terminates on this
    kind: Literal['universe_outcome', 'reference_exemplar', 'schema', 'rules', 'composite']
    payload_schema: dict | None = None      # defaults to the Action's payload_schema
    rules: list[str] | None = None          # named validation predicates, evaluated by M2
    exemplar_from_neighbours: bool = False  # source exemplars from the retrieved tribe
    tolerance: dict = {}

class Skill(BaseModel):
    skill_id: SkillId
    version: str
    scope: Scope
    kind: Literal['execute', 'ask_before_deciding', 'escalate']
    context_schema: dict              # JSON Schema of the required Context Data
    context_sources: list[ContextSource]
    m1_prompt_ref: str                # produced offline by harness optimisation
    expected_result: ExpectedResultSpec
    question: DecisionQuestion | None = None   # required when kind = ask_before_deciding
```

Resolution at runtime is tribe, then cluster, then default; the first Skill found wins. Neither the Skill nor the Context Data is stored in the Universe, matching the guide.

## The harness (offline, once)

```python
class LLMPin(BaseModel):              # the frozen local model
    name: str
    weights_sha256: str
    quantisation: str | None = None
    temperature: float = 0.0
    constrained_decoding: bool = True # grammar or JSON-schema mode

class PromptTemplate(BaseModel):
    template_id: str
    version: str
    text: str
    output_schema: dict | None = None # for M1 prompts: the Action's payload_schema

class HarnessArtefacts(BaseModel):    # designed once, never modified at runtime
    harness_id: str
    llm: LLMPin
    m1_prompts: dict[SkillId, PromptTemplate]
    m2_prompt: PromptTemplate
    optimisation_log_uri: str
```

## Inference: live-state retrieval

```python
class Neighbour(BaseModel):
    state_id: StateId
    distance: float

class RecommendationPolicy(BaseModel):
    action_source: Literal['swarm_scores', 'neighbour_vote', 'blend'] = 'swarm_scores'  # Decisions, point 4
    inference_model: Literal['refit_full', 'fold_ensemble'] = 'refit_full'              # Decisions, point 3
    k_neighbours: int = 25

class LiveCase(BaseModel):
    case_id: str
    data: RawData
    context: dict = {}                # filled against the resolved Skill's context_schema

class RetrievalResult(BaseModel):     # everything the loop needs, resolved before M1 runs
    case_id: str
    features: dict[str, FeatureValue] # same Tokenizer version as the Basis, enforced
    scores_cal: dict[ActionId, float]
    neighbours: list[Neighbour]
    tribe_id: TribeId
    recommended_action_id: ActionId   # the ToDo action
    ambiguous: bool                   # live margin below threshold, or the tribe is flagged
    skill_id: SkillId                 # after tribe, cluster, default resolution
    expected_result: dict             # ExpectedResultSpec resolved with live values
```

## The verifiable reward loop

```python
class ActionObject(BaseModel):        # what M1 emits; structure, not prose
    action_id: ActionId
    payload: dict                     # must validate against CanonicalAction.payload_schema
    llm: LLMPin

class CheckResult(BaseModel):
    passed: bool
    failures: list[str] = []          # schema paths or rule ids that failed

class IterationRecord(BaseModel):
    n: int
    m1_prompt_sha256: str
    action: ActionObject | None = None    # None when M1 output failed to parse
    check: CheckResult
    adjustment: str | None = None     # what M2 changed before the retry

class EscalationRecord(BaseModel):
    reason: Literal['iteration_cap', 'needs_outside_information', 'unresolvable']
    question: str | None = None       # for ask-before-deciding hand-offs
    handed_to: str | None = None

class ExecutionTrace(BaseModel):      # one per live case; the audit record
    case_id: str
    basis_id: str
    harness_id: str
    retrieval: RetrievalResult
    iterations: list[IterationRecord]
    status: Literal['executed', 'escalated']
    escalation: EscalationRecord | None = None
    final: ActionObject | None = None
    started_at: datetime
    finished_at: datetime
```

The `ExecutionTrace` is deliberately complete: basis and harness ids, retrieval evidence, every iteration, and the escalation record give a full audit trail per case, which matters for clinical deployments and for QMS documentation.

## Configuration

```python
class AmmonixConfig(BaseModel):       # the defaults to confirm, in one place
    n_folds: int = 5
    min_states_per_action: int = 100
    ambiguity_margin: float = 0.05
    attribution_target: Literal['argmax', 'true_action'] = 'argmax'
    calibration: Literal['isotonic', 'platt', 'beta'] = 'isotonic'
    recommendation: RecommendationPolicy = RecommendationPolicy()
    max_iterations: int = 3
    region_model: RegionModelSpec = RegionModelSpec()
```

## Decisions this schema pins, for your confirmation

The guide leaves a few points open that a generic tool has to fix. The schema makes each one an explicit field with a default, so changing your mind is a config edit, not a refactor.

1. **Classifier semantics.** Each AMX model estimates P(success | state, action), trained only on states where that Action was taken and labelled by Outcome. This is the natural reading of "Outcome is the reward, not imitation", and it is the only labelling for which ground truth exists without counterfactuals. Consequence: Actions with near-perfect historical success yield few negatives, which is why `class_weight` and `outcome_weighting` exist on `LabelPolicy`.

2. **Fold grouping.** `FoldPlan.group_by` defaults to `example`, not `state`. The guide says to split the states, but for multi-state Examples that would put consecutive, correlated states of one episode on both sides of a fold boundary and quietly break the no-leakage guarantee. Grouping by Example preserves it; single-state Examples make the two settings identical.

3. **Deployment models.** The Universe stays out-of-fold permanently. For scoring live cases you need a model that saw everything: `refit_full` (default) retrains each classifier on all data with the fold-validated hyperparameters, `fold_ensemble` averages the five fold models instead.

4. **Source of the ToDo action.** Slide 20 can be read two ways: score the live features with the swarm, or read the recommendation from the nearest tribe. Default is `swarm_scores` for the Action, with the neighbours supplying the tribe, the ambiguity check and the expected result; `neighbour_vote` and `blend` are available.

5. **Attribution target.** "Top features" needs a classifier to attribute against. Default is the argmax Action's classifier, switchable to the true Action for retrospective analysis.
