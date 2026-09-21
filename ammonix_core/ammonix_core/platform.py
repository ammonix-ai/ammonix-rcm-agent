"""Platform entities (schema v0.2), pinned at milestone M0 per platform plan v0.3.

These are the factory-side entities of v0.2 Appendix B as merged by plan v0.3
Section 3: DatasetDescriptor, SplitPolicy, QuarantineManifest, LoopSpec,
LoopIteration, LoopRun, TestReport, plus the EnvironmentProtocol for
self-generated corpora. Field sets follow the stage-by-stage contract table of
plan v0.3 Section 3 and the concrete usage in the Cardessa build package.
"""

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from ammonix_core.schema import ActionId, FoldMetrics, Outcome, StateHandle

# ---------------------------------------------------------------------------
# Stage 0. Onboarding
# ---------------------------------------------------------------------------


class DatasetDescriptor(BaseModel):
    """One descriptor fully defines how a raw episodic dataset maps onto the schema.

    Onboarding a new domain is configuration, not code.
    """

    dataset_id: str
    root_uri: str
    example_table: str
    state_table: str
    id_fields: dict[str, str]  # schema id -> source column, e.g. {'example_id': 'episode_id'}
    action_field: str
    outcome_field: str
    outcome_positive: bool = True
    # source column (or '*' default) -> modality name
    modality_map: dict[str, Literal["signal", "image", "text", "tabular", "composite"]]
    outcome_score_field: str | None = None  # optional graded outcome column
    # post-hoc quarantine: columns that exist only after the episode resolves;
    # never eligible as state Data / features ('prefix*' wildcards allowed)
    posthoc_fields: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Stage 1. Split and quarantine
# ---------------------------------------------------------------------------


class SplitPolicy(BaseModel):
    test_fraction: float = 0.15
    reserve_fraction: float = 0.10
    n_folds: int = 5
    group_by: str = "example"  # grouping key; 'example' or a meta field like 'patient_id'
    stratify_by: Literal["outcome", "action", "none"] = "outcome"
    seed: int


class QuarantineManifest(BaseModel):
    """The vault record: which groups are sealed where, fixed once at split time.

    Group ids only; no Data. The manifest hash is the negative-test anchor:
    any later change to the assignment invalidates every downstream report.
    """

    manifest_id: str
    created_at: datetime
    policy: SplitPolicy
    group_field: str  # e.g. 'patient_id'
    quarantine_groups: list[str]  # test set A
    reserve_groups: list[str]  # sealed reserve B
    working_groups: list[str]  # everything foldable
    forced_quarantine_rules: list[str] = Field(default_factory=list)  # e.g. holdout payers
    content_sha256: str


# ---------------------------------------------------------------------------
# Build loops (factory bookkeeping)
# ---------------------------------------------------------------------------


class LoopSpec(BaseModel):
    loop_id: str  # e.g. 'L0', 'P4-round2'
    purpose: str
    goal_condition: str  # ends with 'as printed by <script>' per plan v0.3 Section 2
    checker_script: str  # the script whose printed one-line JSON verdict stops the loop
    max_turns: int


class LoopIteration(BaseModel):
    n: int
    started_at: datetime
    finished_at: datetime | None = None
    summary: str  # one unit of work, described for the next fresh-context round
    verdict: bool | None = None  # the checker's printed verdict for this iteration
    metrics: dict[str, float] = Field(default_factory=dict)


class LoopRun(BaseModel):
    run_id: str
    spec: LoopSpec
    worktree: str | None = None
    iterations: list[LoopIteration] = Field(default_factory=list)
    verdict: bool | None = None  # final printed verdict
    cost_tokens: int | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Stage 11. Evaluation gate
# ---------------------------------------------------------------------------


class TestReport(BaseModel):
    """Written exactly once, by the whitelisted final-eval path, on quarantine A.

    Any later build change marks the report tainted, forcing acceptance of the
    shipped state or a fresh draw from reserve B.
    """

    report_id: str
    evaluated_at: datetime
    basis_manifest_sha256: str
    quarantine_manifest_sha256: str
    per_action_metrics: dict[ActionId, FoldMetrics]
    calibration_ece: float
    recommendation_accuracy: float
    harness_stats: dict = Field(default_factory=dict)
    trap_results: dict[str, bool] = Field(default_factory=dict)  # e.g. P1..P8 for Cardessa
    tainted: bool = False


# ---------------------------------------------------------------------------
# Self-generated corpora
# ---------------------------------------------------------------------------


class EnvironmentProtocol(Protocol):
    """For self-generated corpora (toy world, Skat, text games): any domain that
    can produce its own episodes implements this, and the recorder emits
    schema-shaped Examples so the DatasetDescriptor is trivial."""

    def reset(self, seed: int) -> StateHandle: ...

    def legal_actions(self, s: StateHandle) -> list[str]: ...

    def apply(self, s: StateHandle, action_raw: str) -> StateHandle: ...

    def outcome(self, s: StateHandle) -> Outcome | None: ...
