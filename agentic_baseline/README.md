# Agentic baseline for the Cardessa Ammonix comparison

The question this repo answers: **how does the Ammonix platform (learned
swarm + calibration + skills) compare against what most teams build today —
an LLM agent loop with a prompt, tools-style context, and a retry guardrail?**

It is a completely independent decision system that shares only the
*environment* with the factory: the same Cardessa world, payer engine,
fresh-episode generator and outcome-v2 scoring. No trained models, no
tokenizer, no calibration, no tribes, no skills — one LLM reasoning over the
case facts at every touch.

## Design

- **Agent** (`agentic/`): at each claim touch the agent receives a briefing —
  the admin-visible state snapshot, the payer's published rule sheet, the
  standard CARC meanings, the action catalog — and must return strict JSON
  `{reasoning, action}` (schema-constrained decoding). Applicability
  guardrails (can't appeal without a live denial, can't request retro-auth
  when the window is closed, …) give ONE feedback retry, mirroring the
  factory's M2 verifiable-reward loop. A second violation, or an explicit
  `escalate`, hands the touch to the human (persona policy) — the same
  escalation semantics the Ammonix arm gets.
- **Model**: the same pinned Qwen3.6-27B (vLLM, temperature 0, strict
  `response_format: json_schema`) that powers Ammonix's M1 layer, so the
  comparison measures **architecture, not model quality**. The client refuses
  any other served model. Prompt-hash disk cache (`cache/decisions/`) makes
  reruns deterministic and free. The paper's Qwen rows come from the registered
  rerun of this baseline with Qwen3.8-27B-AWQ-INT4 on the same sealed draws
  (`preregistration/comparison_qwen38_full_swap.json`, runner
  `scripts/rollout_paperwork_qwen38.py` in the repository root, decision caches
  `cache/decisions_qwen38_full/` and `cache/decisions_rag_qwen38_full/`, which
  ship). This sealed client is unchanged; the runner overrides only the pin
  and the cache directories.
- **Evaluation** (`scripts/run_comparison.py`): the exact 300 fresh episodes
  of the factory's policy-rollout analysis (tranche 92, appended after the
  full training lineage — never seen by any system), played step-synchronously
  by three arms on identical engine randomness:
  1. `agentic` — this repo's agent on the pinned local 27B,
  2. `claude` — the same agent loop on `claude-sonnet-5` via the Anthropic
     API (the frontier-model baseline; needs `ANTHROPIC_API_KEY`; not
     sampling-deterministic — the decision cache freezes the run),
  3. `system` — the shipped v0.2 Ammonix runtime (escalations fall back to
     the persona, as in the original rollout),
  4. `personas` — the scripted human baseline.
  Scoring is outcome v2 with the process-mistake ledger, identical to
  `scripts/policy_rollout.py` in the factory.

## Fairness contract

- Same information surface: the agent sees only `state_tabular` facts
  (published payer rules + case state + touch history). The poisoned
  `days_to_payment` column and every post-hoc/engine-truth field are
  excluded. Parity is bidirectional (v0.4 audit): every decision-time fact
  the shipped Ammonix basis' 83 features read is rendered in the briefing —
  including `persona_id` and `subscriber_relationship`, which the tranche-92
  run wrongly withheld, and the CO-18 duplicate-claim legend/rule the v0.4
  world added.
- The prompt states the published objective (≥98 % of allowed collected
  within 120 days, touch penalty, mistakes penalised) but contains **no
  strategy rules**. Guardrails enforce *applicability only*, never strategy —
  working out payer-conditional rework (the P1 finding) is the agent's job,
  exactly as it was the platform's job to learn it.
- Same LLM, temperature 0, same escalation fallback, same episodes, same
  scoring. The report also records LLM cost (calls, tokens) per arm —
  Ammonix decides with classifiers (zero LLM calls at decision time); the
  agent pays an LLM call per touch. That cost asymmetry is part of the
  comparison's point.

## Running

Needs the factory's shipped v0.2 worktree (default
`..\Ammonix_Generic\Cardessa_worktrees\v2-agent`, override with
`AMMONIX_FACTORY_ROOT`) and the pinned model serving:

```
docker start ammonix-m1-27b        # wait until /v1/models answers
<v2-agent>\.venv\Scripts\python.exe scripts\run_comparison.py
```

Options: `--arms agentic,system,personas` (default all), `--episodes N`
(default 300). Output: `runs/reports/comparison.json` + stdout summary.

Offline tests (fake LLM, no server): `<v2-agent>\.venv\Scripts\python.exe -m pytest tests -q`

## v0.4 rerun (tranche 98, pre-registered)

`preregistration/comparison_v4_tranche98.json` (committed before any
tranche-98 data existed) pins tranche, allocation, arms, metrics and the
win/tie/lose rule. All arms run under the v4-build worktree code (the CO-18
duplicate-claim mechanic changed episode dynamics), system python:

```
set AMMONIX_FACTORY_ROOT=..\Ammonix_Generic\Cardessa_worktrees\v4-build
python scripts\run_comparison.py --arms personas,agentic --tranche 98 ^
    --report comparison_v4_tranche98.json --tag _v4t98
python scripts\run_v4_arm.py --report comparison_v4_tranche98.json --tag _v4t98
```

`run_v4_arm.py` replays the personas arm as a bit-exact canary before playing
the shipped v0.4 runtime (route_case + confidence floor + EV close-out) as
`ammonix_v4`.
