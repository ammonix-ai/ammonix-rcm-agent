# Reproducing the paper

Paper: *The Ammonix RCM Agent: Learning to Collect Healthcare Claims from Recorded Outcomes* (https://doi.org/10.5281/zenodo.22871212).
Everything below is regenerated from artifacts shipped in this repository. All data is synthetic — there is
no dataset to obtain and no credentialing step.

> Verified 2026-08-26 against the frozen paper (Overleaf 274cb24): test suite 106 pass; every
> figure regenerates byte-identical from shipped artifacts and matches the copy in the paper;
> digit check vs source reports — tab:headtohead 63/63 cells, tab:lswap 4 writers, tab:classifier
> 8/8 rows, tab:skills 9/9 rows, prose margins/SEs/paired counts/oracle gap, SFT pair counts
> (1,503 → 1,339) — zero mismatches; four-writer decision identity (525/525) in
> `l_swap_comparison_demo.json`.
>
> Re-verified 2026-09-11 against the paper as pushed on 2026-09-10 (Overleaf 20b8325), which added
> the GPT-6 token comparison (sec:tokens, tab:tokens): all 16 table cells and the prose (per-claim
> tokens, token ratios, the 2.7-point margin, the one-point Ornith gap) recomputed from the shipped
> tranche-103 reports by `scripts/gpt6_tokens_verdict.py` — zero mismatches; the four tranche-103
> runs replay keyless from the shipped caches with zero provider calls and identical summaries;
> test suite 106 pass; figures unchanged and md5-identical to the paper's copies.
>
> Re-verified 2026-09-21 against the paper of 2026-09-17 (Overleaf cd4b9fc and later), which reports
> the Qwen baseline rows from the registered Qwen 3.8 rerun: the two Qwen rows of tab:headtohead,
> the Qwen row of tab:lswap and the prose around them recompute from the `rollout_qwen38_full_t*`
> and `l_swap_qwen38_demo.json` reports with zero mismatches. With no model server running and no
> API key, the Qwen 3.8 writer lane and head-to-head draw 99 replay from the shipped caches with
> zero provider calls and results identical to the recorded reports; test suite 106 pass.

## The Qwen 3.6 to 3.8 rerun

Qwen3.6-27B built this system: it is the model hash-pinned in `runs/manifests/harness.json`, it wrote
the frozen historical correspondence in `data/text_cache` (about 17,900 texts), and it sat in the
agentic-LLM baseline seat for the comparisons registered in tranches 98 to 102. When Qwen3.6 was
retired, the baseline was rerun with Qwen3.8-27B-AWQ-INT4 on the same sealed draws, under two
registrations made before the runs: `agentic_baseline/preregistration/comparison_qwen38_full_swap.json`
(head-to-head) and `comparison_qwen38_model_swap.json` (writer lane). The paper reports the rerun.
The earlier Qwen 3.6 reports stay in the repository unchanged as the historical record, so
`comparison_paperwork_t*.json` and `l_swap_qwen_demo.json` carry the 3.6 numbers and the
`*_qwen38_*` reports carry the numbers in the paper. The agent's decisions are identical under
every writer, including both Qwen versions (525 of 525, `l_swap_comparison_demo.json`).

## Environment

- Python ≥ 3.12; install with `pip install -e ./ammonix_core -e .` (add extras as noted: `.[figures]`, `.[llm-arms]`, `.[test]`).
- The world, training, and evaluation are seed-deterministic (`MASTER_SEED` in `cardessa/`); artifacts are
  content-hashed and pinned in `runs/manifests/` (`scripts/determinism_check.py` re-verifies every pin).
- LLM comparison arms are stochastic (API models, live endpoints) — reruns reproduce conclusions, not digits.
- The shipped agent serves two local models via vLLM, one on the GPU at a time: the trained 9B writer
  (`runs/manifests/writer_pin.json`) fills the paperwork, and the Qwen 27B (hash-pinned in `LLMPin`) runs the
  dialog and the world's text channel. The payer's reviewer readings ship as a cache (`data/payer_reader`).
  The replay UI needs no model at all.
- Hosted demo option: the operator dialog (and only the dialog) can be pointed at any OpenAI-compatible
  endpoint serving stock Qwen with `DEMO_DIALOG_BASE`, `DEMO_DIALOG_MODEL`, `DEMO_DIALOG_KEY` and, for
  gateways whose control fields differ, `DEMO_DIALOG_EXTRA` (JSON merged into the request body). Off by
  default; the chat widget discloses the served model when a remote backend is configured. Nothing else
  in the demo (caches, replay, live runs) changes.
- The scoreboard's comparison lane is chosen at startup with `AMMONIX_LLM_LANE` (`qwen` default = the
  paper's same-model comparison; `gpt6`, `sol`, `opus` = a frontier model given the whole job by prompt, a
  model+architecture comparison). Every lane's results ship as a cache (`data/ui_llm_lane*.json` +
  its decision cache), so switching lanes needs no key. The public demo runs `AMMONIX_LLM_LANE=gpt6`.

## Result classes

- **exact** — regenerates bit-for-bit (or visually identically) from shipped artifacts.
- **statistical** — rerun draws fresh stochastic samples (LLM calls); expect the paper's conclusions and
  intervals, not identical digits.
- **evidence-only** — cannot be rerun from this release (e.g. arms that replayed historical development
  builds); the producing run's full report ships, and verification means auditing it.

## Figures

| Paper figure | Regenerate with | Source artifact | Class |
|---|---|---|---|
| Sample-efficiency curve (fig2) | same | `runs/reports/universe.json` + curve reports | exact |
| Per-action AUROC (fig3) | same | classifier reports | exact |
| Knowledge Universe (fig4) | same | `runs/reports/universe.json` | exact |
| Calibration (fig5) | same | calibration report | exact |
| Safeguard ablation (fig6) | `python paper/make_fig6_v7.py` | `runs/reports/ablation_paperwork_analysis.json` | exact |
| Case-based fallback case study | `python paper/make_figc2_v7.py` | `runs/reports/c2_case_detail.json` | exact |
| Build pipeline schematic (fig9) | `python paper/make_pipeline.py` | none (drawn) | exact |
| Interface figures (audit view, dialog, universe view) | `python paper/shoot_audit.py` / `shoot_ui.py` / `shoot_universe.py` against the running UI | UI in replay mode | exact (same claims replayed) |

## Tables

| Paper table | Source artifact | Rerun | Class |
|---|---|---|---|
| Components (tab:components) | descriptive | — | n/a |
| Main comparison, three draws (tab:headtohead) | All rows except the two Qwen rows: `runs/reports/comparison_paperwork_t99.json`, `_t100`, `_t101` + `extra_arms_paperwork_t*.json`. The two Qwen rows (LLM agent, LLM agent with retrieval): `runs/reports/rollout_qwen38_full_t99.json`, `_t100`, `_t101` (+ per-episode `episodes_agentic{,_rag}_rollout_qwen38_full_t*.json`), registration `agentic_baseline/preregistration/comparison_qwen38_full_swap.json`; see "The Qwen 3.6 to 3.8 rerun" above | `scripts/rollout_paperwork.py --tranche 99\|100\|101`. Qwen rows: `python scripts/rollout_paperwork_qwen38.py --tranche 99 --writer ornith_ft --arms agentic,agentic_rag --no-swap --report <name>.json` (likewise 100, 101) replays **keyless and offline** from `agentic_baseline/cache/decisions_qwen38_full`, `decisions_rag_qwen38_full`, `data/m1_ornith_ft`, `data/text_cache`, `data/payer_reader` — verified 2026-09-21 on draw 99: no model server, results identical to the recorded report. A from-scratch rerun needs the local vLLM serving Qwen3.8-27B-AWQ-INT4 and the 9B writer | statistical (LLM arms), exact replays from caches |
| Writer training (sec:finetune) | `runs/sft/ornith_r2/` (prompts, pairs, scores) | `scripts/finetune_chain.sh` | The trained writer's outputs replay exactly from shipped caches (tab:lswap). Retraining the adapter from the shipped `pairs.jsonl` needs no API key — one consumer GPU (~3 h) and the pinned base Ornith 1.5 9B; result is statistically equivalent (GPU training is not bit-deterministic). Regenerating the pairs from scratch (8 drafts at T=0.8 + the payer's reviewer) needs local vLLM and an Anthropic key — statistical. The trained writer itself — merged weights (shard SHA-256s = `runs/manifests/writer_pin.json`) and the un-merged adapter — is freely downloadable on Hugging Face: `Ammonix/AmmonixRCM-Writer-9B` (https://huggingface.co/Ammonix/AmmonixRCM-Writer-9B). |
| Safeguard ablation | `runs/reports/ablation_paperwork_analysis.json` | `scripts/rollout_paperwork.py --arms ammonix_ablated` | exact from caches |
| World parameters (tab:world) | `cardessa_world_corpus_spec_v0_1.md` + `cardessa/` constants | — | exact (read off) |
| Classifier swarm (tab:classifier) | classifier reports | full training rerun: `scripts/` pipeline, seed-deterministic | exact |
| Skills (tab:skills) | `runs/manifests/skills.json` | — | exact (read off) |
| GPT-6 token comparison (tab:tokens) | `runs/reports/rollout_{gpt6_tuned,gpt6_rag_tuned,ammonix_gpt6,ammonix_ornith}_t103.json` (+ per-episode `episodes_*_t103.json`), verdict `gpt6_harness_verdict_t103.json`, registration `agentic_baseline/preregistration/comparison_gpt6_tranche103.json` | `python scripts/gpt6_tokens_verdict.py` rebuilds the table and the registered decision rule from the reports. The runs replay **keyless and offline**: `python scripts/rollout_paperwork.py --tranche 103 --no-swap --writer gpt6 --arms gpt6_tuned` (likewise `gpt6_rag_tuned`, `personas,ammonix`) and `--writer ornith_ft --arms ammonix`, from `data/gpt6_decisions_*`, `data/m1_gpt6`, `data/m1_ornith_ft`, `data/text_cache`, `data/payer_reader` — verified 2026-09-11: zero provider calls, summaries and token counts identical. A from-scratch rerun needs an OpenAI key (GPT-6), an Anthropic key (the payer's reviewer), and the local vLLM models. | exact from cache; statistical from scratch (LLM outputs) |
| Language-model swap (tab:lswap) | `runs/reports/l_swap_{qwen38,ornith,ornith_ft,opus5}_demo.json` + `l_swap_comparison_demo.json` (the Qwen row of the table is `qwen38`; `l_swap_qwen_demo.json` is the earlier Qwen 3.6 lane, kept as the record) | `python scripts/l_swap_experiment.py --model {ornith|ornith_ft|opus5}`, `python scripts/l_swap_qwen38.py` for the Qwen lane, then `python scripts/l_swap_qwen38.py --compare` for all lanes state by state. Replays **keyless and offline** from the shipped caches (`data/m1_qwen38`, `m1_cache`, `m1_ornith`, `m2_ornith`, `m1_opus5`, `m2_opus5`, `m2_opus5_judge`, `payer_reader`) — verified 2026-08-22 (four original lanes) and 2026-09-21 (Qwen 3.8 lane): summaries identical, zero provider calls. A from-scratch rerun needs the local vLLM (Qwen 27B, Ornith 1.5 9B) and an Anthropic key. | exact from cache; statistical from scratch (LLM outputs) |

## The pre-registered baseline (`agentic_baseline/`)

The comparison system the paper evaluates against ships in this repo for audit. Its registrations were
committed and pushed (public timestamp) **before** the corresponding episodes existed; each registration
JSON carries its own hash and the paper prints the SHA-256 prefixes. Historical arms (`run_v3_arm.py`,
`run_ablation_v4.py`, `run_imitation_arm.py`, `run_oracle_arm.py`) replayed intermediate development
builds that do not ship — set `AMMONIX_FACTORY_ROOT`/`CARDESSA_WORKTREES_ROOT` only if you have such a
build; otherwise treat their reports as evidence-only.

## Rebuilding the factory itself

The full build (world generation → training → harness assembly → sealed evaluation) is scripted under
`scripts/` with per-milestone checkers and determinism pins. Requires a local vLLM-served Qwen 27B for the harness
stages. This is the long path; auditing the shipped artifacts (`runs/manifests/`, `runs/reports/`) is the
short one.
