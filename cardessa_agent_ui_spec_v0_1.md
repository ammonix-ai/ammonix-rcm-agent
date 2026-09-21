# Cardessa agent toolset and demo UI: specification v0.1

File 3 of 3. The step that makes this project distinctive: the Skill layer's execute actions produce real artifacts, filled by tools from original case data, checked deterministically, and shown in a demo UI. Depends on files 1 and 2 and the factory runtime (M1/M2 harness, RetrievalResult, ExecutionTrace) from `ammonix_platform_plan_v0_3.md`.

## 1. Principle

M1 (the pinned local model) never writes into a form directly. M1 emits a structured ActionObject (JSON conforming to the canonical Action's payload_schema); deterministic form tools render the artifact from the ActionObject plus the case record; M2 checks the result against the expected result spec; the payer engine provides final validation. The LLM chooses and phrases; code fills and verifies. This is the architecture answer to the document's zero-tolerance-for-hallucination requirement, and P8 is its test.

## 2. The toolset (three artifacts, per confirmed default)

### 2.1 fill_cms1500(action_object, case) -> pdf + field_map
Renders the CMS-1500 claim form. Field sources are fixed by a mapping table (committed as YAML): patient demographics from patients.parquet, insured/member ID from coverage, diagnosis pointers from the episode's ICD list, service line CPT and charges from the study record, ordering-provider NPI from clinics.parquet, auth reference from the episode auth record, COB fields from coverage. Every field carries provenance (source table, source column, row id) in the field_map. Fields with no source stay EMPTY and are listed as gaps; the tool never invents a value.

### 2.2 fill_prior_auth_request(action_object, case) -> pdf + field_map
Payer-specific retro/prior-auth request: member, provider, study CPT, diagnoses, date of service, clinical indication text (quoted verbatim from the ordering clinic's note, never paraphrased into the form), retro justification (structured reason code chosen by M1 from an enumerated list, not free text).

### 2.3 compose_appeal_letter(action_object, case) -> docx + field_map
The one artifact with generated prose. Fixed skeleton (facts block rendered by code from the field_map) plus a medical-necessity paragraph written by M1 under constraints: may reference only facts present in the case record and the denial letter; every clinical claim in the paragraph must be traceable to a source field; M2 runs a claim-extraction check (pinned model, temperature 0) that lists each factual assertion in the paragraph and its source field, failing the iteration if any assertion lacks one.

## 3. Expected-result checks (M2) per Action

| Action | Checks |
|---|---|
| submit_clean / submit_with_records / correct_and_resubmit / bill_secondary | Schema-valid ActionObject; CMS-1500 field_map 100 percent provenance-complete for required fields; payer engine pre-validation passes (code pairing legal, filing window open, COB order correct); for corrections, the changed fields address the denial CARC |
| request_retro_auth | Retro window still open per payer rules; justification code consistent with the gap found in state Data |
| appeal_with_necessity | Claim-extraction check green; appeal deadline open; letter cites the denial CARC and the payer's own criterion |
| request_peer_to_peer / provide_requested_info | Requested artifact matches what the correspondence asked for |
| bill_patient / write_off | Balance thresholds and prior-step requirements met (no write_off while a live appeal path with engine-known success probability above threshold exists in this tribe) |

Iteration budget max_iterations 3; failures escalate with the ExecutionTrace. The 20 seeded impossible cases of the dev slice include: holdout-payer claims (no tribe), retro window expired at every path (correct end state: escalate with write_off recommendation flagged low-confidence), and a case whose denial letter requests a document that does not exist in the record (correct end state: escalate, not fabricate).

## 4. Demo UI (three screens, extends the factory's inference console)

### 4.1 Case inbox
List of incoming episodes (fresh simulated cases generated on demand from the world, never from training data). Each row: patient, payer, study, balance, status, and the system's triage: green (execute recommended), amber (ask-before-deciding with the DecisionQuestion shown), red (escalated, reason shown: out-of-tribe payer, ambiguous tribe, failed checks).

### 4.2 Case view (the money screen)
Left: the case facts and correspondence. Centre: the recommendation panel: recommended Action, calibrated success probability, the rival action and its probability when the tribe is ambiguous, and the tribe's plain-language description with the k nearest historical cases (clickable, showing their situations and outcomes). Right: the artifact preview: the filled CMS-1500 / auth request / appeal letter, with every field highlighted on hover showing its provenance (P8 made visible), gaps highlighted in amber, and the M2 check log underneath. Buttons: approve and execute (submits to the payer engine, which adjudicates live and the case advances), edit first, escalate to human.

### 4.3 Story mode (for presenting)
A scripted 6-case walkthrough selectable from the inbox: (1) clean Meridian claim executed straight through; (2) the P1 moment: missing auth at Meridian, admins historically appealed, system recommends retro-auth, shows the neighbour statistics that justify it; (3) same situation at SilverBridge, system recommends appeal because retro is never granted there: payer-conditional expertise; (4) ambiguous Cornerstone bundling case: system asks the DecisionQuestion instead of guessing; (5) Granite Shield (never-seen payer): system escalates, shows out-of-tribe distance; (6) the impossible-document case: system escalates rather than fabricates. Each step displays the one-line claim it demonstrates. This sequence IS the pitch of the source document, made clickable.

## 5. Build milestones and goals (after F8 of file 2; UI is F9)

```
/goal scripts/check_toolset.py prints {"verdict": true}: three form tools
implemented with committed field-mapping YAML; on a 200-case sample every
required field provenance-complete, zero invented values (field-level diff vs
source), payer-engine pre-validation green, claim-extraction check green on
all generated appeal letters; max 40 turns.

/goal scripts/check_demo_ui.py prints {"verdict": true}: Playwright end-to-end
suite green covering inbox triage, case view with provenance hover, live
adjudication round trip, and all six story-mode cases reaching their scripted
end states; max 40 turns.
```

## 6. Out of scope for v0.1 (recorded so nobody builds them by accident)

Real payer connectivity or clearinghouse formats (X12 837/835), real RPA execution into third-party systems (the document's Microsoft RPA layer is the production story, not the demo), eligibility-verification agents (a later domain), and any real patient or payer data. The demo's closing slide states the substitution rule: replace the simulator with the real claims history and the same factory produces the production system.
