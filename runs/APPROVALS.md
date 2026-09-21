# Human gate approvals

Sole approver: the designated approver. The build stops at each gate below and waits for a
written approval entry in this file. Format per entry:

```
## <gate id> — <date>
Approved by: <name>
Decision: <approved / approved with changes / rejected>
Notes: <anything binding on the build>
```

The three gates (the only three):

1. **GATE-P5 Action-map merges** — any semantic merge of raw action labels.
2. **GATE-P8 Skill activation** — activation of the proposed Skill layer.
3. **GATE-P10 Release sign-off** — acceptance of the TestReport from the
   single quarantine-A evaluation.

## GATE-P5 Action-map merges — 2026-07-11

Not triggered: the action map is a 1:1 exact mapping with zero semantic
merges (runs/reports/action_map.json, "merges": []). Nothing to approve.

## GATE-P8 Skill activation — 2026-07-13

Approved by: the project owner (session chat, transcribed verbatim by the
build agent on their instruction: "You have my approval to actívate skills")
Decision: approved
Notes: activates the 11 skills of runs/manifests/skills.json exactly as
verified at P8 (pass_rate 0.9660, mean_iterations 1.038, 20/20 impossible
cases escalated; skills content sha unchanged from the verified commit
ca3bea9). Any change to skill definitions after this entry requires a fresh
gate.

### Addendum — 2026-07-13 (under the blanket build approval)

The user granted blanket approval in session chat ("you dont need to ask for
my explicit approvl every time, please build the system until the end ...
i give you full approval"). Under it, one skill-definition change at P9:
skill-appeal_with_necessity gains the rule claims_grounded_llm (the
agent-UI spec 2.3 claim-extraction check moved INSIDE the M2 iteration
loop, failing iterations whose paragraph makes unsourced assertions).
No other skill changed.

## GATE-P10 Release sign-off — 2026-07-13

Approved by: the project owner (session chat, transcribed by the build
agent on their instruction: "i agree, you can write the sign-off",
agreeing to the recommendation presented with the TestReport)
Decision: approved with changes — accepted as v0.1 with failures on record
Notes binding on the build:
- TestReport runs/reports/test_report.json (sha256 pinned in
  runs/state/final_eval_ran.json) is accepted AS IS: traps P2, P3, P4
  (ECE 0.0495), P5 (0 silent OOD errors), P7, P8 (zero invented values)
  PASS; P1 FAILS at 35.7% vs the 85% bar (beats the 21.4% imitation
  ablation); P6 is unmeasurable (near-tie family never planted in the
  W1 world). Integrity caveat accepted: the eval's text channel was
  regenerated (cache-chain gap); the simulation channel is identical to
  the sealed set by construction.
- The quarantine-A draw is SPENT. Reserve B is held for a single v0.2
  evaluation and may only be drawn after: (1) a decision on the
  recommendation objective (episode-success argmax vs rework-first
  policy — the P1 root cause), (2) more rework-action training data
  (raise the expansion cap or re-weight the initial corpus),
  (3) planting the P6 near-tie in the world parameters, (4) fixing the
  text-cache regeneration discipline.
- Any post-sign-off build change marks the v0.1 TestReport tainted per
  the platform plan.

## v0.2 change specification — 2026-07-13

Approved by: the project owner (session chat: "i approve the v0.2 spec,
start the build")
Decision: approved
Notes: ammonix_v0_2_change_spec.md governs the v0.2 build. Outcome v2
(payer-valid collections only), process-mistake ledger, cornerstone
near-tie planted, cap 5000, text-cache hard gate. The three human gates
apply to the v0.2 lineage; its final eval draws RESERVE B, exactly once.
The v0.1 TestReport is superseded-in-progress but remains the v0.1
record.

## GATE-P8 Skill activation (v0.2 lineage) — 2026-07-13

Approved by: the project owner (session chat: "go ahead", responding to the
two decisions presented with the P8v2 report)
Decision: approved
Notes: (1) skill-provide_requested_info flips escalate -> execute (trained,
coverage 108, AUROC 0.89) under two grounding rules (requested_item_in_record,
letter_names_requested_item); fabrication-bait cases verified to escalate.
(2) The harness pass-rate definition is split: 0.9639 on policy-routed cases
(374/388, real M1 failures counted against), with the 74.8% all-states rate
and the 112 policy escalations recorded alongside — the owner consciously
accepts that ~22% of dev states hand to a human under outcome v2. Skills
content as verified at commit 9d7bfff; any further change needs a fresh gate.

## GATE-P10 Release sign-off (v0.2) — 2026-07-14

Approved by: the project owner (session chat: "i approve the v0.2 sign-off")
Decision: approved — v0.2 is the shipped state
Notes binding on the record:
- TestReport runs/reports/test_report_v2.json (sha pinned in
  runs/state/final_eval_reserve_ran.json) accepted AS IS: P2, P3, P5
  (0/12 silent OOD errors), P7, P8 (130 artifacts, 0 failures) PASS;
  P1 57.1% on n=7 (bar 85%; up from v0.1's 35.7%, recommendation
  accuracy 0.24 = 22x v0.1); P4 ECE 0.1025 on thin reserve slices
  (working-side cross-fitted 0.0234); P6 population exists (83 states)
  but the ambiguity detector measures episode-success margins where the
  trap defines resolution-probability ties (v0.3 design item);
  no-self-mistakes missed by one case.
- The hard cache gate PASSED: this evaluation is byte-faithful to the
  sealed lineage. Exactly-once enforced; rerun refuses.
- ALL SEALED DATA IS SPENT (quarantine A by v0.1, reserve B by v0.2).
  Any v0.3 evaluation uses fresh simulated draws only. Recorded v0.3
  findings: align the ambiguity detector's scale with resolution
  probabilities; allocate larger evaluation reserves at world design;
  close the last P1 mile (retro classifier data or policy tier).
- Any post-sign-off build change marks the v0.2 TestReport tainted.

## v0.3 change specification — 2026-07-14

Approved by: the project owner (session chat: "i approve")
Decision: approved
Notes: ammonix_v0_3_change_spec.md governs the v0.3 build. Measurement
decomposition (portfolio / worker / clinic), decision regret,
doomed-at-intake clinic report, resolution-scale ambiguity, P1 path-value
scoring, expansion cap 8000, pre-registered fresh-draw evaluation
(tranche 95, size 500, allocation frozen in the spec) plus the policy
rollout as a first-class exam. The v0.2 TestReport becomes the tainted-
superseded record once the build begins; v0.3 is a fresh lineage on the
UNCHANGED world.

## GATE-P10 Release sign-off (v0.3) — 2026-07-15

Approved by: the project owner (session chat: "We should keep v3")
Decision: approved — v0.3 is the shipped state, superseding v0.2
Notes binding on the record:
- test_report_v3.json + amendment accepted AS IS: 7/13 pre-registered
  bars. Passing: calibration (ECE 0.0468, n>=400), rollout collections
  +52% and mistakes 32-vs-84, OOD honesty, form fidelity, coverage,
  determinism. Failing, with root causes on record: P1 55% on n=124
  (capability plateau across three cycles), P6 9.5% (tie width below
  model precision - mechanism limit), oracle regret 0.38, 10
  self-mistakes, doomed-flag 91.4% (0.05 runtime floor vs 0.10 doom
  floor - threshold alignment item).
- Worker and clinic scorecards are accepted as v0.3 deliverables.
- The simulated-data research loop CLOSES here: further P1/P6/regret
  progress requires real claims data or new mechanisms. Any build change
  taints the v0.3 report.

## GATE-P5 Action-map merges (v0.4 lineage) — 2026-07-21

Not triggered: the v0.4 action map is a 1:1 exact mapping with zero
semantic merges (runs/reports/action_map.json, "merges": 0), matching the
v0.1 precedent. The project owner additionally approved proceeding in
session chat ("i approve") with the P5 checker verdict true on record
(10 rules, zero unmapped, coverage met, determinism green).
Decision: approved / not triggered
Notes: retro_auth coverage 239 states (vs 122 in v0.3) after the targeted
doubling; the swarm/basis phases may proceed on the v0.4 lineage.

## GATE-P8 Skill activation (v0.4 lineage) — 2026-07-21

Approved by: the project owner (session chat: "approve", responding to the
P8 report presented with the gate summary)
Decision: approved
Notes: activates the 20 skills of runs/manifests/skills.json (sha256
c88088ce109870e4) exactly as verified at P8 round 5: the 11 v0.3-carried
skills under v0.4-hardened payload schemas (pattern/maxLength on
code-bearing fields, spec s.2b), plus 9 NEW tribe-scoped
ask_before_deciding skills attached to the label-space Universe's
ambiguous tribes - firing only when the live case's own margin/noise
confirms the tie (territory-vs-trigger rule). Verified: pass_rate 0.9594
on 345 policy-routed cases (bar 0.90), mean_iterations 1.24, 20/20
impossible cases escalated with correct reasons, zero invented values,
pinned model confirmed. The owner consciously accepts the more cautious
autonomy profile (66.1% all-states execution vs v0.3's 74.8%) as the
cost of live uncertainty detection. Any change to skill definitions
after this entry requires a fresh gate.

## GATE-P10 Release sign-off (v0.4) — 2026-07-21

Approved by: the project owner (session chat: "approve", responding to the
recommendation presented with the sealed TestReport)
Decision: approved — v0.4 is the shipped state, superseding v0.3
Notes binding on the record:
- test_report_v4.json (sha pinned in runs/state/final_exam_v4_ran.json)
  accepted AS IS: 13/18 pre-registered bars. Passing includes EVERY v0.4
  mechanism bar: no payer overpayment across all three rollout lanes,
  zero policy resubmit-after-paid (the duplicate-payment exploit is
  unlearned), 100% letter/money consistency, abandoned patient share 0.0%
  (bar 10%), rollout collections +62% vs personas (141,673 vs 87,632),
  mistakes not worse, doomed-flag 100%, ECE 0.0261, determinism, OOD
  honesty, form fidelity, no leakage.
- Failing, with root causes on record: P1 63.1% (bar 85; up from v0.3's
  55% via the retro doubling; capability plateau persists), P6 33% (bar
  70; up from 9.5% via the now-active uncertainty machinery; tie-width
  mechanism limit), oracle regret 0.335 (bar 0.05; v0.3: 0.38),
  self-mistakes (same class as v0.3's accepted ten), deductible_heavy EV
  uplift 1.264x (bar 1.5x; companion abandonment bar passed at 0.0).
- The pre-registered draws (tranches 96 exam, 97 rollout) are SPENT for
  this lineage; the exactly-once marker refuses reruns.
- v0.5 candidates carried on record: patient-payment model (flat 0.7/0.4
  coin), bill_secondary applicability on nothing-pending, further P1/P6/
  regret progress requires real claims data or new mechanisms.
- Any post-sign-off build change taints the v0.4 TestReport.

## v0.6 kernel decision field: activation (P8-class gate)

Date: 2026-08-14. Approver: the designated approver.

Scope approved: switch the SHIPPED default decision field from the fitted
per-action models ("model") to the kernel decision field ("kernel"):
neighbour-count success estimates over the stored label-space Knowledge
Universe at the live coordinate (Lambda places, the Ether decides), the
canonical form of the Foundation paper. Evidence on record:
- runs/reports/kernel_ether_probe.json: estimator tie (AUROC 0.8739 vs
  0.8751); raw-feature counting clearly wors
- runs/reports/kernel_ether_rollout.json and kernel_rollout_inharness.json:
  300 fresh tranche-92 episodes, identical d
  3 mistake episodes vs model $90,619 / 52.0% / 6; personas $66,863.
- Checkers V6-M1 (flag off byte-identical; f
  retrain-free) and V6-M2 (temporal firewall on consolidation, snapshot
  round-trip, recency window, runtime switch
Determinism posture: the field is content-hashed; deployments pin one
snapshot hash (basis/kernel_snapshots/manife
ordinal-stamped and windowed (cardessa/consolidation.py). Every decision
attributes to "snapshot H plus N consolidate
Consequences accepted: the v0.4/v0.5 sealed TestReport describes the
"model" default and is NOT re-graded; any v0
fresh tranche and its own pre-registration. Rollback: set
AMMONIX_DECISION_FIELD=model or basis/decisi

Approved: the designated approver, 2026-08-14

## GATE-P8 Skill re-activation (scrubber-only rules) — 2026-08-24

Approved by: the project owner (session chat, transcribed verbatim by the
build agent on their instruction: "I approve the P8")
Decision: approved
Notes: re-activates the 20 skills of runs/manifests/skills.json after the
check change: skill-appeal_with_necessity drops the model-read rule
claims_grounded_llm; its rules are cpt_matches_case, cites_denial_carc,
paragraph_grounded_in_case (the deterministic scrubber). The appeal prompt
now states that the payer's reviewer checks every statement against the
submitted records. No other skill changed. Skills content sha256 at
approval: 5478d85d64bce400 (before the activation stamp below).
