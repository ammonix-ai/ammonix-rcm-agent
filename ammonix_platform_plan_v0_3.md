# Ammonix Loop Engineering Platform: plan v0.3

Consolidates and supersedes `ammonix_loop_factory_plan_v0_2.md` where they differ; the schema remains `ammonix_schema_v0_1.md` plus the v0.2 additions in its Appendix B. Written to be executed by Fable 5 under Claude Code. Section 2 records the July 2026 state of loop engineering practice as verified by web research; Section 3 is the generic data contract stage by stage; Section 7 is the test dataset roadmap answering the "what do we test on" question.

## 1. Objective

A generic platform where a user introduces a raw episodic dataset (many Examples, each an ordered sequence of States carrying signal, image, text or tabular Data plus a logged Action, with one gold Outcome per Example) and the platform, governed by Claude at build time only, autonomously constructs the full Ammonix system: Q_PSI tokenisation, canonical Action map, AMX swarm on a five-fold grouped split, out-of-fold Universe with calibration and regions, Skill layer with the M1/M2 harness on a pinned local LLM, and a three-surface inference UI. Final quality is proven exactly once on a quarantined hold-out test set that no build loop can read. An application is fully defined by one DatasetDescriptor, one Tokenizer, one Action map and its Skills; onboarding a new domain is configuration, not code.

## 2. Loop engineering: verified state of practice, July 2026

The v0.2 plan's account holds up against current sources. Verified points and refinements:

**The pattern and its parts.** Loop engineering (named by Addy Osmani, 7 June 2026, after Boris Cherny's "my job is to write loops" and Peter Steinberger's "design loops that prompt your agents") is designing the control system that prompts the agent: automations on a schedule, git worktrees for parallel isolation, skills holding project knowledge, MCP connectors to real tools, sub-agents splitting maker from checker, and an on-disk memory that survives between runs because the model forgets and the repo does not. All six primitives now ship natively in Claude Code; nothing here requires custom bash orchestration.

**/goal is the load-bearing primitive.** Claude Code's `/goal` (native since v2.1.139, May 2026) keeps an agent working across turns until a condition holds, with a separate small model grading the transcript after every turn. Two practical consequences the v0.2 plan did not state explicitly:

1. **The evaluator reads the transcript, not the filesystem.** A condition like "runs/reports/toy_build.json shows mean_oof_auroc >= 0.95" only stops the loop if that evidence appears in the transcript. Therefore every checker script must print its verdict and key metrics to stdout as its final act, and every goal condition in Appendix C is amended to end with "as printed by <script>". This is the single most common silent failure mode reported in the field.
2. **Compound goals fail.** One measurable end state, one stated check, constraints, and a cap per goal. Sequential small goals beat one large one.

**Fresh context per iteration (the Ralph pattern).** For long optimisation loops, the reliable shape is: same prompt against a written spec, one unit of work, commit, then a fresh instance with clean context reads the on-disk state and repeats. The intelligence lives in the spec, the check and the state file, not in one long session. A long-running session accumulates stale tool output and dead ends ("context rot"), and decision quality drops as the pile grows. The defences are to compact long runs, offload big outputs to files, and delegate messy subtasks to subagents so only the clean result returns to the parent context. L4 (tokenizer rounds) and L6 (hyperparameter search) adopt this explicitly: each round is a fresh goal run that begins by reading `runs/PROGRESS.md` and the round's state JSON.

**Constrain, don't only review.** Every part of a loop is a standing capability grant. The current guidance is to narrow what the loop can touch (deny-hooks, pruned connector lists, read-only checkers) rather than relying on post-hoc review. Our quarantine PreToolUse hook is exactly this principle; we extend it with a read-only verifier subagent (no Write or Edit tools) so the checker physically cannot fix the work it grades.

**Budgets and unattended runs.** Turn caps (`max_turns`) and spend caps (`max_budget_usd`) exist at the SDK level; every goal carries a turn cap in its condition text, and `LoopRun.cost_tokens` is recorded. Unattended scheduled runs (routines / `/schedule`) are available but not needed for v0.3: milestones are launched interactively and run unattended inside each goal (auto mode), with Agent View for monitoring. Scheduled nightly re-runs of the determinism CI are the one routine worth installing from day one.

**Two loop levels, kept separate.** Build-time loops (Claude in the factory, creative, terminated by numeric script-computed checks, budget-capped) versus the run-time loop (the shipped product: M1/M2 on the pinned local model, deterministic, hash-pinned, no Claude and no external API anywhere in runtime code paths). This preserves the air-gapped deployment story. The symmetry stands: every build loop is logged in Example/State/Action/Outcome shape, so the factory's own history becomes a future Ammonix dataset.

## 3. The generic data contract: structures and variables at every step

Schema v0.1 defines the entities; this table is the stage-by-stage input/output map that a generic tool must honour. Each row names what the stage reads, what it writes, and the variables that parameterise it. All identifiers are ULIDs; every artefact carries a version and content hash pinned in the BasisManifest.

| # | Stage | Reads | Writes | Key variables and defaults |
|---|---|---|---|---|
| 0 | Onboarding | user files + `DatasetDescriptor` (id fields, action field, outcome field, modality map) | `Example`, `RawState`, `RawData`, `Outcome`; ingest audit report | four scope conditions (volume, states, Data+Action per state, one Outcome per Example); missing-value profile; raw action inventory |
| 1 | Split | Example table | `QuarantineManifest`, `FoldPlan`, dev-slice manifest | `SplitPolicy`: test_fraction 0.15, reserve_fraction 0.10, n_folds 5, group_by example, stratify_by outcome, seed |
| 2 | Q_PSI (Stage A) | `RawData` per state, working data only | `Tokenizer` (+`FeatureSpec` list, `DeterminismPin`), `QPsiRecord` per state | dtype, unit, valid_range, missing_policy per feature; `HistorySpec` (lookback_states, aggregation) for episodic features; LLM extractors pinned at temperature 0, constrained decoding |
| 3 | Action map (B1) | `action_raw` inventory | `ActionMap` (`CanonicalAction` with payload_schema, `ActionMapRule`), `CoverageReport` | min_states_per_action 100 (the 100x rule); unmatched raw labels fail the build; semantic merges are a human gate |
| 4 | AMX swarm (B2) | `QPsiRecord` + `FoldPlan` | `TrainedFoldModel` per (Action, fold) with `FoldMetrics` | `LabelPolicy`: scope own_action_states, label outcome_success, class_weight balanced (imitation setting kept for ablation only); `ClassifierSpec` default gbdt, per-Action overrides |
| 5 | Universe (B3) | fold models + hold-out states | `UniverseRecord` per state (scores_raw, scores_cal, true_action_id, outcome, top_features, tribe_id), `Calibrator` per Action, `UniverseIndex` | calibration isotonic; attribution_target argmax; index metric euclidean over standardised features; every state scored only by its hold-out fold model |
| 6 | Basis | all of the above | `BasisManifest` + `basis/` directory (manifest.json, qpsi.parquet, universe.parquet, artefacts/) | dataset_fingerprint; refit_full models for live scoring alongside the permanent OOF Universe |
| 7 | Regions | `UniverseRecord` | `Tribe` (centroid, `TribeStats`: success_rate, top2_margin, ambiguous, rival_action_id) | `RegionModelSpec` hdbscan on full feature space; ambiguity_margin 0.05; ambiguity is a build-time Universe property, not a runtime decision |
| 8 | Skill layer | Tribes, clusters, `CanonicalAction.payload_schema` | `Skill` (scope tribe/cluster/default, kind execute / ask_before_deciding / escalate, context_schema, `ContextSource`s, `ExpectedResultSpec`, optional `DecisionQuestion`) | resolution order tribe then cluster then default; ambiguous tribes auto-propose ask_before_deciding; activation is a human gate |
| 9 | Harness (offline, once) | Skills, dev slice, pinned model | `HarnessArtefacts` (`LLMPin`, m1_prompts per Skill, m2_prompt, optimisation log) | pass_rate >= 0.90, mean_iterations <= 2 on dev slice; seeded impossible cases must end escalated |
| 10 | Inference (runtime) | `LiveCase` + Basis + Harness | `RetrievalResult` (features, scores_cal, neighbours, tribe, recommended_action_id, ambiguous flag, resolved skill and expected result), then `ExecutionTrace` (per-iteration `ActionObject`, `CheckResult`, adjustment; status executed or escalated) | `RecommendationPolicy`: action_source swarm_scores, inference_model refit_full, k_neighbours 25; max_iterations 3 |
| 11 | Evaluation gate | quarantine (once, whitelisted script only) | `TestReport` (per-Action metrics, calibration ECE, recommendation_accuracy, harness stats, tainted flag) | any later build change marks the report tainted, forcing acceptance of the shipped state or a fresh draw from reserve |

Platform-side entities (`DatasetDescriptor`, `SplitPolicy`, `QuarantineManifest`, `LoopSpec`, `LoopIteration`, `LoopRun`, `TestReport`) are as drafted in v0.2 Appendix B and are merged into the schema as v0.2 at milestone M0. One addition for genericity:

```python
class EnvironmentProtocol(Protocol):
    """For self-generated corpora (toy world, Skat, text games): any domain that
    can produce its own episodes implements this, and the recorder emits
    schema-shaped Examples so the DatasetDescriptor is trivial."""
    def reset(self, seed: int) -> StateHandle: ...
    def legal_actions(self, s: StateHandle) -> list[str]: ...
    def apply(self, s: StateHandle, action_raw: str) -> StateHandle: ...
    def outcome(self, s: StateHandle) -> Outcome | None: ...
```

## 4. Architecture and stack

Unchanged from v0.2 Section 3 (repo layout, three UI surfaces: build monitor, Universe explorer, inference console). Stack: Python 3.12, Pydantic v2, Polars + Parquet, scikit-learn plus a GBDT library, SHAP, hnswlib, vLLM serving the pinned local model, FastAPI, React, Playwright. Runs on the existing RTX 5090. One addition: `adapters/` gains a plugin registry keyed by modality so third-party tokenizers register `FeatureSpec`s through a single interface, which is what makes the tool generic rather than a rebuilt pipeline per dataset.

## 5. The build loops

L0 to L10 as in v0.2 Section 6, with these amendments from Section 2 above:

1. **Transcript evidence.** Every checker script ends by printing a one-line JSON verdict (`{"loop": "L4", "verdict": true, "metric": 0.873}`) to stdout; every goal condition references that printed line. Appendix C conditions are amended accordingly.
2. **Fresh-context rounds.** L4 and L6 run as repeated single-round goals, not one long session. Each round: read PROGRESS.md and the round state file, do one propose-score-keep/revert cycle, commit, append state, stop. The plateau rule (< 0.005 gain over 3 rounds) is evaluated by the checker script across state files, not by the worker's memory.
3. **Read-only verifier.** The verifier subagent (L4 leakage/contamination audit, L7 bookkeeping audit) is configured without Write/Edit tools and in its own worktree, so grading and fixing are physically separated.
4. **Negative tests are first-class.** L3's quarantine probe (a deliberate blocked read) and L8's seeded impossible cases (escalation must be the passing behaviour) stay in; they are the two places where the loop is rewarded for refusing, and current practice confirms these are exactly the checks that get silently trained away when omitted.

| Loop | Purpose | Stop condition (summary, printed by checker) | Budget |
|---|---|---|---|
| L0 | Scaffold, schema package, CI, hooks | pytest and ruff exit 0 on ammonix_core | 30 turns |
| L1 | Toy world recovers planted truth end to end | mean OOF AUROC >= 0.95 and optimal action recommended in >= 95% of toy test states | 40 turns |
| L2 | Ingest and audit via DatasetDescriptor | audit report generated; four scope conditions pass or stop with repair list | 30 turns |
| L3 | Splits, quarantine, hook enforcement | manifest written; probe read of quarantine blocked (negative test) | 15 turns |
| L4 | Tokenizer engineering (AutoML-shaped) | determinism check exits 0; AUROC improves per round until plateau | 40 turns/round |
| L5 | Action map, 100x rule | zero unmapped labels; coverage met; merges human-approved | 20 turns |
| L6 | Swarm and hyperparameters | pooled fold metrics for every Action; plateau; no reads outside working data | 40 turns |
| L7 | Universe, calibration, regions | leakage audit passes; ECE <= threshold; tribe stats and ambiguity flags written | 25 turns |
| L8 | Harness optimisation (RLVR at prompt level) | dev-slice pass rate >= 0.90, mean iterations <= 2, impossible cases escalate | 30 turns |
| L9 | Inference UI | Playwright end-to-end suite exits 0 including full case to exported trace | 40 turns |
| L10 | Evaluation gate | final_eval.py run exactly once via whitelisted path; TestReport committed | 5 turns |

## 6. Governance

As v0.2 Section 7: PreToolUse deny on `data/quarantine/**` for all agents and subagents; PostToolUse ruff plus schema validators; Stop-level check refusing completion on a stale PROGRESS.md; turn caps in every goal; maker/checker split with /goal's evaluator as the third, transcript-level check; one worktree per loop, merge only on green; memory spine in `runs/`; determinism CI as a nightly routine; and exactly three human gates (Action merges, Skill activation, release sign-off). CLAUDE.md draft as in v0.2 Appendix A, unchanged except golden rule 6 gains: "each optimisation round is a fresh goal run; do not carry a round in one long session".

## 7. Test datasets: the roadmap

The scope conditions are the filter: hundreds to millions of Examples; each Example one or more consecutive States; each State carrying Data and a logged human Action; one gold Outcome per Example. The best candidates also stress the platform's distinguishing claims: success-not-imitation (S1-type traps), honest calibration, ambiguity flagging, multimodality, and the escalation mechanics. Recommended sequence:

**Tier 0, pipeline validation (already planned).**
1. **Toy world** (milestone M1): synthetic generator with planted feature-Action-Outcome structure and a known optimal policy. The factory's permanent regression fixture.
2. **Lighthouse** (existing spec): the acceptance instrument with planted traps, full multimodality including the image channel, escalation mechanics, contamination-clean, regenerable from a master seed.

**Tier 1, first real public data, fast to obtain, tabular plus text.**
3. **MLB Statcast pitch sequencing (public via pybaseball).** Example = plate appearance (about 190,000 per season); State = one pitch decision with count, runners, batter/pitcher profiles, and genuine signal-channel data on prior pitches (release speed, spin, movement vectors); Action = pitch type by zone (canonical 10 to 20); Outcome = plate appearance result. Millions of states, a real signal modality, and strong sequential structure (history features earn their keep).

**Tier 2, controlled synthetic with psychology and chance (already spec'd).**
5. **Skat** (companion spec): three pinned-M1 personas, hesitation as a signal channel, bluff discounting, PIMC oracle, and the self-improvement headline (Ammonix outplays its own teachers). Runs in parallel from M0 per milestone M7.
6. Optional: **text-adventure corpus** (Jericho/TextWorld games played by the pinned local model), a pure-text stress test of the LLM-extractor tokenizer path at near-zero data cost.

**Tier 3, the flagship clinical demonstration (credentialed access).**
7. **MIMIC-IV ICU treatment episodes** (PhysioNet, credentialed, DUA). Example = ICU stay (about 65,000+); State = a 4-hour window with vitals (signal), laboratory values (tabular), clinical notes (text), and optionally paired chest radiographs from MIMIC-CXR (image), giving all four modalities on one dataset; Action = discretised treatment decision, e.g. the sepsis fluid-by-vasopressor grid of roughly 25 canonical actions from the AI Clinician literature, or ventilation and antibiotic decisions; Outcome = survival to 90 days or discharge disposition. This is the closest public analogue to the intended clinical product: irreducible outcome noise, genuine class imbalance, ask-before-deciding tribes with clinical meaning, and escalation as a safety requirement rather than a metric. MIMIC-IV-ED (about 425,000 emergency visits, triage-to-disposition) is the lighter on-ramp with fewer states per Example; HiRID (Bern) is the high-resolution European alternative.

**Tier 4, the target.**

Sequencing rationale: Tier 0 proves the machinery, Tier 1 proves it on messy public reality within days of M2, Tier 2 proves the self-generated-corpus path plus the psychology and calibration claims, Tier 3 proves the clinical story, and Tier 4 is the product. Each tier reuses the previous tier's Basis-building code unchanged; only the DatasetDescriptor, the Tokenizer plugins and the Action map differ, which is the genericity claim made testable.

## 8. Milestones for Fable 5 under Claude Code

M0 scaffold, M1 toy world, M2 ingest and splits, M3 basis build, M4 skill layer and harness, M5 UI, M6 evaluation gate, M7 Skat parallel track: as v0.2 Section 8, with the amended goal conditions below. Run each in its own worktree in auto mode, monitor from Agent View, one goal per milestone stage, never compound.

```
/goal `pytest -q` exits 0 and `ruff check .` exits 0 in ammonix_core, with both
command outputs shown; do not modify or skip tests to pass; max 30 turns.

/goal scripts/check_toy.py prints {"verdict": true} with mean_oof_auroc >= 0.95
and recommended_matches_optimal >= 0.95, produced by `make toy`; never touch
data/quarantine/; max 40 turns.

/goal scripts/check_split.py prints {"verdict": true} confirming the quarantine
manifest hash and that a probe read of data/quarantine/ was denied by the hook;
max 15 turns.

/goal scripts/check_tokenizer_round.py prints {"verdict": true} showing either
mean_oof_auroc gain >= 0.005 over the previous round or a declared plateau, and
scripts/determinism_check.py exits 0 (output shown); one round only; features
registered as FeatureSpecs with real descriptions; max 40 turns.

/goal scripts/check_harness.py prints {"verdict": true} with pass_rate >= 0.90
and mean_iterations <= 2 on the dev slice under the pinned vLLM model, and all
seeded impossible cases ending status "escalated"; only files under
harness/prompts/ changed; max 30 turns.

/goal scripts/check_final.py prints {"verdict": true} confirming
runs/reports/test_report.json exists, produced by `make final-eval` run exactly
once, committed with the Basis manifest hash; no other files modified;
max 5 turns.
```

## 9. Open decisions for the designated approver

1. Confirm the dataset sequence in Section 7, in particular whether MIMIC-IV credentialing should be started now so access is ready by M3.
2. Confirm Skat open parameters (spec Section 9: Ramsch exclusion, 30,000 deals, hesitation exposure, p_bluff levels, oracle grid, fourth scripted player).
3. Confirm the v0.2 Appendix B entities as schema v0.2 at M0.
4. Confirm the three human gates as the only three.
5. Confirm the nightly determinism routine and the turn budgets per loop.
