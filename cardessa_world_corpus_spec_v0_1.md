# Cardessa reimbursement world and corpus: specification v0.1

File 1 of 3. Companion to `cardessa_ammonix_build_package_v0_1.md` (file 2) and `cardessa_agent_ui_spec_v0_1.md` (file 3). Reuses the factory built per `ammonix_platform_plan_v0_3.md` and the schema `ammonix_schema_v0_1.md`. This file specifies the simulated world and the corpus Fable generates BEFORE the Ammonix build starts. Everything is synthetic: every name, address, member ID, NPI, phone number and dollar amount is invented. Diagnosis and procedure codes are real public code sets (ICD-10, CPT, CARC) so the demo is credible to healthcare insiders; no real patient, provider or payer identity appears anywhere.

Honest framing, stated once and carried into the demo: the trained system learns the SIMULATOR'S payer behaviour, not real payers'. The demo claim is "this is what the factory does when pointed at real claims history", never "this system knows real payer policy".

## 1. The world

### 1.1 The business being simulated

An IDTF (independent diagnostic testing facility) performs ambulatory cardiac monitoring studies ordered by cardiology clinics. After a study is completed and the clinical report is delivered, the reimbursement admin must get the study paid by the patient's insurance. The admin's decisions from "report back" to "money resolved" are the expertise being captured.

Study types and real CPT codes (technical/professional components as billed by an IDTF):

| Study | CPT (primary technical) | Typical allowed amount range (invented) |
|---|---|---|
| Mobile cardiac telemetry (MCT), up to 30 days | 93229 (93228 professional) | 550 to 950 USD |
| Cardiac event monitoring (CEM), up to 30 days | 93271 (93270 recording, 93272 review) | 210 to 420 USD |
| Long-term ECG monitoring, 7 to 15 days | 93247 (93245 to 93248 family) | 240 to 480 USD |
| Holter, up to 48 hours | 93226 (93224 global, 93227 review) | 90 to 180 USD |

Diagnosis pool (real ICD-10, each patient carries 2 to 3): I48.0 paroxysmal AF, I48.91 AF unspecified, I47.1 SVT, I49.5 sick sinus syndrome, I49.9 arrhythmia unspecified, R00.2 palpitations, R55 syncope, R42 dizziness, G45.9 TIA (cryptogenic stroke workup), I25.10 CAD, I34.0 mitral insufficiency, Z86.73 personal history of TIA/stroke.

### 1.2 Payers: 12 synthetic, built on real archetypes

Invented names, realistic behaviour. Two are excluded from the training corpus entirely (OOD holdout).

| # | Invented name | Archetype | Key behaviours |
|---|---|---|---|
| 1 | Meridian Health | National commercial | Prior auth required for MCT; retro-auth window 30 days; appeals succeed with strong documentation |
| 2 | Atlas Mutual | National commercial | No prior auth, but aggressive medical-necessity denials on MCT for palpitations-only |
| 3 | Cornerstone United | National commercial | Bundles event monitoring with recent Holter (CO-97); peer-to-peer reverses 70 percent |
| 4 | Federal Medicare (sim) | Medicare | No prior auth; strict LCD-style coverage rules per CPT x ICD pair; timely filing 365 days |
| 5 | SilverBridge Advantage | Medicare Advantage | Prior auth for MCT and long-term ECG; retro-auth never granted; appeals slow but fair |
| 6 | Lakeshore Advantage | Medicare Advantage | Delegates to radiology-benefit manager; frequent CO-16 missing-info denials |
| 7 | Prairie State Medicaid | Medicaid MCO | Low allowed amounts; secondary to Medicare rules; 90-day timely filing |
| 8 | Harborview Medicaid | Medicaid MCO | Requires ordering-physician Medicaid enrolment; else PR-204 |
| 9 | Blue Summit Plan | Regional Blues-type | Generous but strict on code pairing; corrected claims succeed |
| 10 | Keystone TPA | Self-funded TPA | Plan-document exclusions; external review possible |
| 11 | Granite Shield (HOLDOUT) | Regional commercial | Never appears in training; OOD trap |
| 12 | Pelican Care (HOLDOUT) | Medicaid MCO | Never appears in training; OOD trap |

### 1.3 The payer policy engine (the oracle)

One deterministic rules engine, parameterised per payer, adjudicates every submission. Inputs: the claim as submitted (payer, plan, CPT, ICD list, auth reference, attachments, days since service, ordering NPI status, COB position). Outputs: pay in full, underpay, or deny with a real CARC code (CO-197 no precert, CO-50 not medically necessary, CO-16 missing information, CO-97 bundled, CO-29 timely filing expired, PR-204 not covered, CO-22 COB error), plus response delay in days. Where real payers are genuinely stochastic (appeal outcomes, reviewer strictness), the engine draws from seeded per-payer distributions, so chance is real but regenerate-identical from the master seed. The engine also exposes, for evaluation only, the true probability of success for every legal action in any state: this is the corpus's perfect-information oracle, a perfect answer key at zero compute cost. Engine rules live in one YAML per payer; unit tests assert every CARC pathway per payer.

### 1.4 Providers and patients

25 invented cardiology clinics (name, invented NPI, Medicaid enrolment flag, documentation quality trait from 0.3 to 0.95 affecting attachment completeness). Exactly 1,000 invented patients: name, DOB, invented address, sex, 2 to 3 diagnoses from the pool, and a stability trait (probability of coverage having lapsed by service date). Each patient carries a coverage record with the fields real adjudication and the CMS-1500 actually require: primary payer, plan variant, member ID, SUBSCRIBER identity and relationship to patient (self, spouse, child; about 25 percent are dependents), assignment-of-benefits flag (missing in about 2 percent of patients, a realistic denial pathway), patient financial responsibility parameters (annual deductible, deductible remaining at service date, coinsurance percent, copay if any), and for 20 percent a secondary payer with COB order. Each STUDY record carries explicit dates: order date, service start and end dates (monitoring spans days to weeks), and report-delivered date; all deadline logic (timely filing, retro-auth window, appeal deadline) counts from these dates, and the payer engine computes patient responsibility (deductible and coinsurance) from the coverage record so underpayments and the bill_patient action have realistic amounts. Patient generation from the master seed; patients.parquet plus coverage.parquet plus studies.parquet plus payers.yaml plus clinics.parquet constitute the world snapshot, content-hashed.

## 2. Episodes and the decision process

An Example = one claim episode: study completed and report delivered (the "Holter is back" moment in the domain-expert note) through final financial resolution (paid, partially paid and accepted, patient-billed, or written off). Pre-study facts (whether eligibility was verified, whether prior auth was obtained, ordering clinic traits) are Data of the first state, not in-episode actions: a missing prior auth is a situation the admin discovers, and what to do about it is the decision.

**Staged corpus (generate more if needed, per the designated approver's decision).** The corpus starts small and expands automatically only when the build's own quality gates demand it. Initial tranche: 1,000 episodes drawn from the 1,000 patients (about 4,000 states; each episode contains 2 to 7 decision states, mean about 4: the initial submission decision, then one decision per payer response requiring action). Expansion rule: if any canonical Action falls below 100 working-data states, or a calibration or trap gate fails for lack of data (not for model error), the generator produces one additional tranche of 500 episodes and the affected stage re-runs; hard cap 3,000 episodes. Expansion is deterministic and contamination-safe by construction: every patient's split assignment (quarantine A, reserve B, or working fold) is fixed once at world creation, episodes are generated in indexed order from the master seed lineage, each tranche is separately content-hashed, and a new episode inherits its patient's split side automatically, so the vault wall is never crossed. Curated-family shares apply within every tranche. Patients may accumulate additional studies across tranches (sequential, clinically motivated: inconclusive Holter leads to MCT order).

**State Data channels:**
1. Tabular: payer identity and archetype features (auth-required flag, retro window, timely filing days as the admin's reference knowledge), plan type, study CPT, diagnosis list, days since service and days to filing deadline (both computed from the study's explicit dates), auth status, eligibility status at service, assignment-of-benefits status, subscriber relationship, COB position, denial CARC code if any, allowed vs billed amounts, patient responsibility so far (deductible remaining, coinsurance percent), balance, touches so far, clinic documentation-quality trait, patient secondary-coverage flag.
2. Text: the payer correspondence for this touch (denial letter or EOB remark text) and the clinical indication note from the ordering clinic. Generated by the pinned local model at temperature 0 from templates with salt; occasionally the letter carries decision-relevant information beyond the CARC code (low intensity), e.g. a denial letter that names the exact missing document.
3. No image or signal channel; that machinery is exercised elsewhere.

**Canonical Actions (10):**

| ActionId | Meaning | Payload schema (what the executed form needs) |
|---|---|---|
| submit_clean | Standard claim submission | CMS-1500 field set |
| submit_with_records | Submission with clinical documentation attached | CMS-1500 + attachment list |
| request_retro_auth | Obtain retroactive authorisation before (re)submitting | Prior-auth request field set |
| correct_and_resubmit | Fix coding, COB or data error and resubmit | Corrected CMS-1500 + correction note |
| appeal_with_necessity | Formal appeal with medical-necessity letter | Appeal letter field set |
| request_peer_to_peer | Request reviewer-to-physician call | P2P request fields |
| provide_requested_info | Answer a CO-16 information request | Attachment cover sheet |
| bill_secondary | Route balance to secondary payer | CMS-1500 (COB) |
| bill_patient | Transfer balance to patient responsibility | Statement fields |
| write_off | Close as uncollectable | Write-off memo |

**Outcome (per confirmed default):** binary success = collected at least 90 percent of the allowed amount within 120 days of first submission; graded outcome_score = fraction of allowed amount collected, minus 0.05 per touch beyond the first (each rework has labour cost; this encodes the 73-FTE story). Post-hoc fields (final paid amount, days to payment, touch count) are quarantined from state Data (the post-hoc quarantine rule of the platform plan).

## 3. Admin personas and corpus generation

Scripted decision policies (per confirmed default), with the local LLM used only to write text artifacts. Three personas, rotated per episode, identity recorded as a legitimate state feature:

1. **Diligent** follows payer rules correctly with small error rate 0.05.
2. **Hasty** submits clean without checking auth or eligibility; on denial, resubmits unchanged once before escalating actions; misses timely-filing windows with probability 0.1.
3. **Conservative** avoids appeals; writes off or patient-bills balances above a low effort threshold; requests records excessively (extra touches).

**Planted suboptimal habits (the success-not-imitation trap, P1):** in the family "MCT, Meridian or SilverBridge, prior auth missing at first state", Hasty and Conservative (together the majority of family states by construction) choose appeal_with_necessity or write_off, which the engine grants low success; request_retro_auth within the window succeeds greater than or equal to 80 percent at Meridian (30-day retro window) while at SilverBridge (retro never granted) the optimal is appeal, so the correct policy is payer-conditional and cannot be imitated from the majority. Curated scenario families (about 30 percent of episodes) guarantee minimum state counts for rare actions: peer-to-peer families (Cornerstone bundling denials), COB families, Medicaid enrolment families, timely-filing near-miss families.

**Generation harness:** pure-Python simulation loop (world snapshot + persona policies + payer engine), one master seed, regenerate-identical; the only LLM calls are text generation (about 6,000 calls per 1,000 episodes, minutes on the 5090). Corpus written directly in factory schema shape (Example, RawState, RawData, Outcome) via the EnvironmentProtocol adapter, content-hashed. Generator audits per tranche: family statistics match design targets within tolerance; engine unit tests green; double-generation hash equality. Action coverage (greater than or equal to 100 working-data states per canonical Action) is checked cumulatively across tranches and is the primary expansion trigger of Section 2.

## 4. Trap table (corpus-level guarantees the build must face)

| # | Claim under test | Construction | Acceptance |
|---|---|---|---|
| P1 | Success, not imitation | Retro-auth family above | Platform recommends the payer-conditional minority action in greater than or equal to 85 percent of family test states; imitation ablation fails |
| P2 | No leakage | Post-hoc fields quarantined; one poisoned column (days_to_payment) planted in raw tabular data | Leakage audit flags and excludes it |
| P3 | Determinism | Double generation and double tokenisation from master seed | Hash equality, all channels |
| P4 | Calibration against truth | Engine's true P(success) on an evaluation grid of 1,000 states (grows with tranches, 500 per expansion, max 2,000) | ECE less than or equal to 0.05 against engine truth |
| P5 | OOD honesty | Granite Shield and Pelican Care claims appear only in quarantine | Out-of-tribe flagged; silent-confident-error rate less than or equal to 5 percent; escalation is the passing behaviour |
| P6 | Honest near-ties | Families where engine truth makes appeal vs peer-to-peer nearly equal | Tribes flagged ambiguous; agreement with engine flatness greater than or equal to 0.7 |
| P7 | Coverage discipline | bill_secondary appears only in secondary-coverage episodes | Below-threshold actions merged or refused via CoverageReport only |
| P8 | Form fidelity (no hallucination) | Every filled form field must trace to source case data | 100 percent of required fields correct against source; zero fabricated values; validated by the payer engine and a field-level diff (specified in file 3) |

## 5. Open parameters (defaults applied, listed for the record)

Patients 1,000 (confirmed); staged corpus per Section 2 (initial 1,000 episodes about 4,000 states, +500-episode tranches on demand, cap 3,000); curated-family share 30 percent; payer count 12 with 2 held out; persona mix Diligent 0.4, Hasty 0.35, Conservative 0.25; text channel included from the start (cheap here); splits grouped by PATIENT, not episode (a patient's episodes share coverage facts and must not straddle fold boundaries).
