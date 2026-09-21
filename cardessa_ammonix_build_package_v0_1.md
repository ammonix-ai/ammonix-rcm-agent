# Cardessa Ammonix build package v0.1

File 2 of 3. Consumes the corpus specified in `cardessa_world_corpus_spec_v0_1.md` (file 1); the agent toolset and demo UI are file 3. The factory, loops, governance, hooks and goal mechanics are exactly those of `ammonix_platform_plan_v0_3.md`; this package closes the dataset-specific parameters, in the familiar package shape. This is a STANDALONE build: the complete Ammonix system is constructed from scratch inside this project, independent of any other dataset or repository.

## 1. Closed decisions

| # | Decision | Value |
|---|---|---|
| 1 | Corpus | 1,000 patients; STAGED corpus per file 1 Section 2: initial 1,000 episodes (about 4,000 states), automatic +500-episode tranches when coverage or data-limited calibration gates fail, cap 3,000 episodes; all tranches from one master seed lineage, contamination-safe by patient-level split assignment |
| 2 | Example granularity | One claim episode (study report delivered to final financial resolution) |
| 3 | Outcome | Binary: greater than or equal to 90 percent of allowed collected within 120 days; outcome_score = fraction collected minus 0.05 per extra touch |
| 4 | Actions | The 10 canonical Actions of file 1 Section 2; payload schemas are the form field sets of file 3 |
| 5 | Split grouping | By patient_id (all episodes of a patient on one side of every boundary) |
| 6 | Quarantine | 15 percent of patients (test set A), including ALL episodes involving the two holdout payers; 10 percent sealed reserve (B); five folds on the rest, grouped by patient, stratified by outcome |
| 7 | Modalities | Tabular + text from the start; text features via the pinned local extractor at temperature 0, guided JSON |
| 8 | Pinned model | PLACEHOLDER-CONFIRM: Qwen3-14B, FP8 or AWQ, vLLM, temperature 0, guided JSON decoding; hash-pinned in LLMPin before the harness milestone; one-line edit if changed |
| 9 | Oracle | The payer engine's true P(success given state, action), evaluation-only interface, 2,000-state grid |
| 10 | min_states_per_action | 100, enforced via the staged-corpus expansion trigger rather than oversized initial generation; ambiguity_margin 0.05; isotonic calibration; k_neighbours 25; max_iterations 3; all other config at platform-plan defaults |
| 11 | Human gates | Same three; sole approver: the designated company approver |
| 12 | Sequencing | Standalone, one continuous run: platform foundation (M0 scaffold, M1 toy world per the platform plan), then world (W1, W2), then the dataset milestones P2 to P10 below, then toolset and UI |

## 2. DatasetDescriptor

```python
DatasetDescriptor(
    dataset_id='cardessa_sim_v1',
    root_uri='data/raw/cardessa_sim/',
    example_table='episodes.parquet',
    state_table='states.parquet',
    id_fields={'example_id': 'episode_id',
               'state_id': 'state_id',
               'seq': 'touch_seq'},
    action_field='action_raw',          # already canonical-ready from the simulator
    outcome_field='success',
    outcome_positive=True,
    modality_map={'*': 'tabular',
                  'payer_correspondence_text': 'text',
                  'clinical_indication_text': 'text'},
    notes='Grouping key meta.patient_id. meta carries payer_id, payer_archetype, persona_id, family_id, clinic_id. Post-hoc quarantine: paid_amount_final, days_to_payment, total_touches, engine_truth_* fields. Holdout payers Granite Shield and Pelican Care route to quarantine A by construction.'
)
```

## 3. Q_PSI feature plan

Round 1, deterministic tabular: payer archetype one-hots and the payer's published rule facts as the admin would know them (auth_required, retro_window_days, timely_filing_days, p2p_available), study CPT, diagnosis-pool membership flags and CPT x ICD pairing validity per the payer's public criteria, days_since_service and days_to_filing_deadline, auth_status, eligibility_status, cob_position, current CARC code one-hot, balance and allowed_amount, touches_so_far, clinic documentation-quality trait, persona_id (a legitimate feature: temperaments are stable and learnable), secondary-coverage flag. History via HistorySpec within the episode: prior actions taken, prior CARC codes seen, cumulative delay.

Round 2, text extractor (pinned model, temperature 0, guided JSON) over payer_correspondence_text and clinical_indication_text: named booleans such as letter_names_missing_document, letter_cites_medical_policy, indication_mentions_syncope, symptom_duration_documented. The before/after AUROC delta of this round is reported: it measures what reading the letters is worth, which is itself a demo talking point.

Determinism check gates every round; no feature may appear in the post-hoc manifest (P2).

## 4. Build milestones and goal conditions

World milestones (run after M1, before the dataset milestones):

```
/goal scripts/check_world.py prints {"verdict": true}: payers.yaml (12 payers,
2 flagged holdout), clinics.parquet, patients.parquet (exactly 1000) generated
from the master seed; payer-engine unit tests cover every CARC pathway of
every payer and exit green (output shown); double-generation hash equality of
the world snapshot; max 35 turns.

/goal scripts/check_corpus.py prints {"verdict": true}: the initial tranche of 1000 episodes
(about 4000 states) in factory schema shape via the EnvironmentProtocol adapter;
tranche audits green and the cumulative coverage report written (expansion trigger armed); curated-family statistics
within design tolerance; planted P1 family present with majority/minority
structure as specified; poisoned column planted; double-generation hash
equality including text channel; max 40 turns.
```

Full standalone sequence. First the platform foundation per the platform plan: M0 (scaffold, schema package incl. v0.2 platform entities, CI, hooks, nightly determinism routine) and M1 (toy world proving the pipeline end to end on planted truth), using the M0 and M1 goal texts of the platform plan verbatim. Then the world milestones W1 and W2 (goals above). Then the dataset milestones: P2 ingest (descriptor above; post-hoc quarantine verified), P3 splits (grouped by patient; holdout-payer routing verified by a negative test), P4 tokenizer rounds (two rounds as in Section 3, fresh-context, one round per goal), P5 action map gate (trivial mapping, human approval of any merges), P6 swarm, P7 universe and regions (calibration against the engine oracle grid), P8 harness (shared with file 3 toolset; dev slice 500 states + 20 impossible cases incl. holdout-payer cases that must escalate), P9 toolset and demo UI (the two goals of file 3 Section 5), P10 evaluation gate on quarantine A producing the TestReport with P1 to P8 trap results.

```
/goal scripts/check_final_cardessa.py prints {"verdict": true}: make
final-eval run exactly once on quarantine A via the whitelisted path;
TestReport committed with P1-P8 results and the corpus and world hashes;
no other files modified; max 5 turns.
```

## 5. Evaluation headlines (what the demo will claim)

1. P1: on the retro-auth family, the system recommends the payer-conditional correct action where the simulated admins habitually chose wrong, with the imitation ablation failing the same test: learned what works, not what people did.
2. P4: calibrated probabilities within ECE 0.05 of engine truth: the confidence numbers shown in the UI mean what they say.
3. P5: claims from never-seen payers are escalated, not guessed: the system knows what it does not know.
4. P8 (from file 3): every filled form field traces to source data, zero fabricated values: the anti-hallucination guarantee that the document's "zero tolerance" section demands.
