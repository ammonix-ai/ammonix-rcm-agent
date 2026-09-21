# DRAFT — GATE-P10 v0.4 sign-off amendment (awaiting owner approval)

Status: DRAFT ONLY. Not an approval record. This text is proposed for
appending to runs/APPROVALS.md under the GATE-P10 v0.4 entry, by the
sole approver (the designated approver), in writing. Do not copy it into APPROVALS.md
without that written approval.

Proposed amendment text:

---

### GATE-P10 amendment (v0.4) — 2026-07-22 audit corrections

The 2026-07-21 sign-off notes recorded "doomed-flag 100%" among the
passing bars. That bar was vacuous: final_exam_v4.py counted
`status == "escalated" or best < DOOM_FLOOR` inside the
`best < DOOM_FLOOR` branch, so ROLLOUT_doomed_flagged_100pct was 1.0 by
construction (same defect v0.3 carried and amended). Honest recompute
from the same pinned tranche-96 draw (states hash verified identical to
the sealed report; runs/reports/test_report_v4_amendment.json):

- 213 doomed intakes (touch-0, known payer, best calibrated score < 0.10)
- 207 escalated by the runtime (best < confidence floor 0.05, all
  close-out EVs zero) — honest flag rate 97.18%, not 100%
- 0 EV close-outs (no close-out action is applicable at touch 0)
- 6 executed normally at scores in [0.05, 0.10) — no flag of any kind

The honest bar verdict is FALSE (rate < 1.0). The exam script itself is
sealed and unmodified; the amendment report is the corrective record.
v0.4's bar tally therefore reads 12/18 passing with the doomed-flag bar
restated at 97.18%, alongside the other corrections on record:

- EV close-out constants (harness.py expected_values): documented as
  copied from simulator parameters, not "the world's published
  constants"; empirically recoverable from recorded episodes (all three
  inside Wilson 95% CIs, verdict data_recoverable,
  runs/reports/ev_constants_estimate.json). Values unchanged.
- allowed_amount feature: description corrected (present from touch 0,
  identical to meta engine_truth_allowed; static, not temporal, leak;
  golden-rule-9 tension recorded in
  runs/reports/paper_alignment_audit.md). Feature and behavior
  unchanged.
- UI: inbox mistake definitions completed (fifth category
  resubmitted_paid_claim); learning-page sample-efficiency crossing now
  computed against both reference lines (human and LLM agent).

No build artefact, model, basis, or sealed draw changed; the shipped
v0.4 system's behavior is byte-identical except for the corrected
feature-description pin (tokenizer_round3.json, description-only,
previous hash preserved).

Approved by: ____________________ (the designated approver) Date: ____________

---
