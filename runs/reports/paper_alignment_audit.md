# Alignment audit: Cardessa build vs Ammonix_Architecture_June_20_2026.pdf

Audited 2026-07-21 at owner request. Each paper component checked against
the implemented system (ammonix_core + cardessa instantiation, v0.3
lineage + v0.4 in build). Verification was against code and artefacts, not
docs: basis/manifest.json, universe.parquet schema, tribes.json centroids,
harness.py route_case, skills.json, the factory round records.

## Aligned (verified)

| Paper component | Implementation | Evidence |
|---|---|---|
| phi mechanistic world model (fixed, expert-coded, traceable primitives) | the kept tokenizer: ~80 deterministic features, hash-pinned, no learning | features.py + verify_pin |
| Lambda calibrated classifier swarm | per-action GBDT swarm + isotonic calibration, ECE-gated | swarm_round1.json, pooled AUROCs |
| U~ experience ledger records (u, phi, a*, y*) | universe.parquet: state_id, features ref, true_action_id, outcome_success/score, tribe, scores_raw/cal, top_features | 11,427 records |
| E experience field (action-conditioned success context) | calibrated P(success/action,state) + tribe success rates + v0.3 resolution/path-value | implemented parametrically (models) rather than by kernel smear - same role, stronger estimator, paper explicitly allows domain-specific ranking |
| Drive gradient (discrete case: differences among E(u,a)) | argmax over calibrated action scores + path-value substitution | route_case |
| Skill layer: versioned artifacts, rules, escalation, safe-set removal BEFORE ranking | skills.json (versioned, scoped, rules), applicability filter before argmax, M2 checks, confidence floor, out-of-tribe escalation, iteration cap | harness + P8 gate |
| L frozen local LM: structured output only, never overrides rules, rationale + dialog | pinned Qwen 27B: M1 ActionObjects (guided JSON), M2-checked prose, and (since 2026-07-21) the grounded case dialog panel | dialog was the one missing L duty; closed today |
| RHO-VR | the factory itself: retrospective propose-score-keep rounds, deterministic verifiable rewards on held-out folds, sealed evals, harness optimized while phi/Lambda/L change only by versioned rebuild | tokenizer/swarm round records |
| Temporal firewall | post-hoc quarantine (outcome fields never features), leak screens, planted poisoned column caught, live case carries no outcome fields | golden rule 9 |
| Full provenance chain | ExecutionTrace: rationale -> neighbours -> labels -> outcomes -> field -> rules; field-level form provenance | UI case view |
| Local frozen-LLM operation | everything on-prem: vLLM local, no cloud calls | - |

## Deviations (ranked)

1. LATTICE GEOMETRY (significant, architectural). The paper defines the
   Universe coordinate as u = Lambda(phi(x)) - cases retrievable by
   similarity in LABEL-SPACE geometry ("two cases may differ in raw
   observations but occupy nearby regions if their estimated label/action
   vectors have similar structure"). The implementation indexes the
   Universe by euclidean distance over SCALED phi FEATURES
   (basis/manifest.json index.metric=euclidean over the 81-feature scaler;
   tribes.json centroids are keyed by feature names). Neighbours and
   tribes therefore live in phi-space, not Lambda-space. The data to
   re-index per the paper already exists (universe records carry
   scores_raw/scores_cal per state). Candidate v0.5 change: build the
   retrieval index and tribes over the calibrated score vectors (or a
   concatenation), and re-validate out-of-tribe escalation thresholds.
2. LOCAL LABEL PRIOR pi_U (moderate). The paper separates "what was
   correct nearby" (pi_U, kernel-weighted label density) from "what
   succeeded nearby" (E). The implementation has no explicit pi_U: ranking
   uses success models + applicability only; the UI shows neighbour
   actions but no density prior enters the decision. Candidate: compute
   and display pi_U from the neighbour set (cheap), and evaluate whether
   it adds decision value beyond the swarm.
3. SKILL REGIONALIZATION (mild). Paper: a Skill attaches to a
   NEIGHBOURHOOD of the ledger. Implementation: skills scope by action
   cluster (skill-submit_clean etc.), i.e. regions of the action map, not
   regions of the Universe. Same governance properties; coarser
   regionality. Candidate: tribe-scoped skill variants when evidence
   shows payer/region-conditional rules (the SilverBridge-vs-Meridian
   retro distinction is currently learned by the models, not by skills).
4. EPISODIC LEDGER REFRESH (mild). Paper's learning episode inserts new
   adjudicated records and refreshes pi_U/E continuously. Implementation
   freezes the ledger between versioned builds; new cases accumulate in
   caches but never enter the Universe until a rebuild. This matches the
   paper's "controlled retraining" stance for phi/Lambda/L but is
   stricter than its ledger-refresh step. Deliberate (determinism gates);
   record as a policy choice, not an accident.

## Note

The v4-fixes fold-ensemble uncertainty (dormant until the v0.4 basis
build) implements the same signal family as the paper's Figure-4
"outcome-spread among nearest-neighbor clusters" detector - it will
activate naturally in the current v0.4 build.

## Amendment 2026-07-22: allowed_amount static-leak disclosure

Audit finding (c). corpus.py writes `allowed_amount` into every state
from touch 0 (state_tabular, corpus.py:411), byte-identical to the meta
field `engine_truth_allowed` (corpus.py:631) that golden rule 9 denylists
from features. Meanwhile features.py described the feature as "Payer's
allowed amount if adjudicated, else -1" - false: it is never -1 in
corpus data and never gated on adjudication.

Classification: STATIC, not temporal. The value is the exogenous
contracted/allowed rate, drawn once per (payer, cpt, member) at episode
start (engine.allowed_amount is deliberately stable across
resubmissions) and constant across every state of the episode. It
encodes no outcome, no future information, and no post-hoc field; the
temporal firewall (golden rule 9: paid_amount_final, days_to_payment,
total_touches, engine_truth_* never features) is about POST-HOC
knowledge, and this value is decision-time knowledge under the modeling
assumption "the contracted rate is known to the billing admin" (fee
schedules are contract documents; real billing systems load them).

The golden-rule-9 tension, stated explicitly: `engine_truth_allowed` is
on the denylist BY NAME, and a byte-identical copy of it has been a
feature in every basis since round 1. The rule's intent (no outcome
leakage) is not violated - success is defined as a FRACTION of
allowed_amount collected, so knowing the denominator reveals nothing
about the numerator - but the letter of the rule and the shipped feature
description were both dishonest about it. Resolution chosen: honest
documentation, not silent removal. The feature stays (shipped v0.4
behavior and every hash-pinned basis depend on it); features.py now
carries the true description and this note records the tension for the
owner. Removal, if ever desired, is a v0.5 basis-rebuild decision.

Mechanical consequence: the description fix changes features.py's byte
hash, so the kept-tokenizer code pin (runs/manifests/
tokenizer_round3.json pin.code_sha256) was re-pinned with an in-manifest
amendment record (previous hash preserved). Feature VALUES and
tokenisation are unchanged; historical manifests (tokenizer_round1.json,
basis/manifest.json) retain the shipped description as build records.
