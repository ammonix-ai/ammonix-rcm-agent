"""Imitation / behavior-cloning arm on tranche 98 (POST-HOC, NOT pre-registered).

A supervised classifier (scripts/train_imitation.py) trained purely on the
5,000-episode lineage clerk actions is replayed as a policy on the SAME 300
tranche-98 episodes (5000..5299) and SAME engine dice as the sealed
comparison. At each touch it featurises the harness's decision-time state row
exactly as at training time and predicts the next action; if that action is
not legal for the case it falls back to the highest-probability LEGAL action,
and only if no legal action exists does it hand the touch to the persona
(human) policy — mirroring the escalate hand-off the other arms use.

Scoring reuses run_oracle_arm.outcome_v2 / summary UNCHANGED, so the arm is
apples-to-apples with the sealed report. It does NOT touch
comparison_v4_tranche98.json or preregistration/. It writes:
  runs/reports/episodes_imitation_v4t98.json      (per episode)
  runs/reports/comparison_posthoc_v4t98.json      (sealed 4 arms + imitation_bc)

Run (any factory venv with sklearn + polars; train first):
  <venv>\\python.exe scripts\\train_imitation.py
  <venv>\\python.exe scripts\\run_imitation_arm.py
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
WORKTREES = BASE.parent / "Ammonix_Generic" / "Cardessa_worktrees"
V4_ROOT = Path(os.environ.get("AMMONIX_V4_ROOT", WORKTREES / "v4-build"))
for entry in (str(BASE), str(SCRIPTS), str(V4_ROOT),
              str(V4_ROOT / "ammonix_core"), str(V4_ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import CardessaEnvironment, state_tabular  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.livecases import training_world  # noqa: E402
from cardessa.personas import choose_action  # noqa: E402

from agentic.fallback import persona_action  # noqa: E402

# reuse the EXACT scoring + canary helpers from the oracle arm (do not
# reimplement): identical outcome_v2 / summary / case_view keeps every arm
# apples-to-apples.
from run_oracle_arm import (  # noqa: E402
    ALLOCATION,
    LINEAGE_EPISODES,
    N_EPISODES,
    ROLLOUT_TRANCHE,
    case_view,
    outcome_v2,
    summary,
)
from train_imitation import MODEL_PATH, featurize  # noqa: E402

import os as _os
_TAG = _os.environ.get("AMMONIX_TAG", "_v4t98")
_CMP = _os.environ.get("AMMONIX_REPORT", "comparison_v4_tranche98.json")
_ARM = _os.environ.get("AMMONIX_SYSTEM_ARM", "ammonix_v4")
SEALED_ARMS = ("personas", "agentic", _ARM, "oracle_greedy")


def load_bc():
    with open(MODEL_PATH, "rb") as fh:
        bundle = pickle.load(fh)
    model = bundle["model"]
    classes = list(model.classes_)
    col = {a: i for i, a in enumerate(classes)}
    return model, bundle["vocab"], col


def bc_action(model, vocab, col, case, payer, legal):
    """Featurise the harness state row identically to training and predict.

    Predicted-but-illegal -> highest-probability LEGAL action; no legal action
    at all -> None (persona hand-off, like the other arms' abstain path)."""
    if not legal:
        return None
    row = state_tabular(case, payer, 0.0)
    x = np.asarray([featurize(row, vocab)], dtype=np.float64)
    proba = model.predict_proba(x)[0]
    pred = model.classes_[int(np.argmax(proba))]
    if pred in legal:
        return pred
    # fall back to the highest-probability legal action
    ranked = sorted(legal, key=lambda a: proba[col[a]] if a in col else -1.0,
                    reverse=True)
    return ranked[0] if ranked else None


def play(policy_name, world, engine, indices, bc=None):
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    live = {}
    for index in indices:
        case = env.reset(index)
        live[case.episode_id] = {"case": case, "pending": [], "escalated_touches": 0}
    results = {}
    step = 0
    while live and step < 8:
        step += 1
        finished = []
        for episode_id, slot in list(live.items()):
            case = slot["case"]
            payer = engine.payers[case.payer_id]
            legal = env.legal_actions(case)
            if policy_name == "imitation":
                model, vocab, col = bc
                action = bc_action(model, vocab, col, case, payer, legal)
                if action is None:  # abstain -> the human works this touch
                    slot["escalated_touches"] += 1
                    action = persona_action(
                        choose_action, case.persona, case_view(case, payer),
                        MASTER_SEED, case.episode_id,
                        case.patient_share_pending, legal,
                    )
            else:  # personas canary (bit-identical to run_oracle_arm)
                action = persona_action(
                    choose_action, case.persona, case_view(case, payer),
                    MASTER_SEED, case.episode_id, case.patient_share_pending,
                    legal,
                )
            snapshot = state_tabular(case, payer, 0.0)
            slot["pending"].append((snapshot, action))
            case = env.apply(case, action)
            slot["case"] = case
            if case.terminal:
                success, score, n_mistakes = outcome_v2(env, case, slot["pending"])
                results[episode_id] = {
                    "success": success, "score": score, "mistakes": n_mistakes,
                    "touches": len(case.action_history),
                    "collected_payer": case.collected_payer,
                    "collected_patient": case.collected_patient,
                    "escalated_touches": slot["escalated_touches"],
                    "resolution": case.resolution,
                    "allowed": case.allowed,
                    "actions": list(case.action_history),
                }
                finished.append(episode_id)
        for episode_id in finished:
            del live[episode_id]
    return results


def main() -> int:
    sealed_path = BASE / "runs" / "reports" / _CMP
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))

    world_ext, engine = training_world(V4_ROOT, MASTER_SEED)
    start = world_ext.studies.height
    assert start == LINEAGE_EPISODES, f"lineage height {start} != {LINEAGE_EPISODES}"
    rollout_studies = expansion_studies(
        world_ext, MASTER_SEED, ROLLOUT_TRANCHE, ALLOCATION, start
    )
    world = extend_world(world_ext, rollout_studies)
    indices = list(range(start, start + N_EPISODES))
    print(f"imitation (behavior-cloning) arm on tranche {ROLLOUT_TRANCHE} "
          f"episodes {start}..{start + N_EPISODES - 1}")

    # CANARY: personas replay must match the sealed summary exactly (same
    # world, dice and identity as every other arm) or we abort.
    canary = {k: v for k, v in summary(play("personas", world, engine,
                                            indices)).items()
              if k not in ("wall_seconds", "escalated_touches")}
    stored = {k: v for k, v in sealed["personas"].items()
              if k not in ("wall_seconds", "escalated_touches")}
    if canary != stored:
        print("CANARY MISMATCH (personas) - aborting; NOT writing any report.")
        for k in sorted(set(canary) | set(stored)):
            if canary.get(k) != stored.get(k):
                print(f"  {k}: replay={canary.get(k)} sealed={stored.get(k)}")
        return 1
    print("canary ok: personas replay matches the sealed summary exactly "
          f"(episodes {start}..{start + N_EPISODES - 1})")

    bc = load_bc()
    started = time.perf_counter()
    results = play("imitation", world, engine, indices, bc=bc)
    wall = round(time.perf_counter() - started, 1)

    (BASE / "runs" / "reports" / f"episodes_imitation{_TAG}.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8", newline="\n"
    )

    arm = summary(results)
    # summary() (borrowed from the oracle arm) hardcodes escalated_touches=0;
    # report the true hand-off count honestly.
    arm["escalated_touches"] = sum(r["escalated_touches"] for r in results.values())
    arm["wall_seconds"] = wall
    arm["_provenance"] = (
        "post-hoc, NOT pre-registered; behavior-cloning baseline trained on "
        "lineage clerk actions (scripts/train_imitation.py, HistGradientBoosting "
        "on 11,427 lineage rows, decision-time tabular features only, poison "
        "column days_to_payment and all outcome fields excluded); same tranche "
        "98 world+dice and same outcome_v2/summary scoring as the sealed arms"
    )

    posthoc = {
        "analysis": "post-hoc imitation / behavior-cloning baseline vs the "
                    "sealed tranche-98 arms",
        "_provenance": {
            "note": "NOT pre-registered. The four reference arms are copied "
                    "verbatim from runs/reports/comparison_v4_tranche98.json "
                    "for context; only imitation_bc is new. The sealed report "
                    "and preregistration/ are untouched.",
            "tranche": ROLLOUT_TRANCHE,
            "n_episodes": N_EPISODES,
            "source": "runs/reports/comparison_v4_tranche98.json",
        },
    }
    for name in SEALED_ARMS:
        posthoc[name] = sealed[name]
    posthoc["imitation_bc"] = arm

    out = BASE / "runs" / "reports" / f"comparison_posthoc{_TAG}.json"
    out.write_text(json.dumps(posthoc, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({
        "imitation_bc": {
            "success_rate_v2": arm["success_rate_v2"],
            "payer_collected_capped": arm["payer_collected_capped"],
            "episodes_with_mistakes": arm["episodes_with_mistakes"],
            "mean_touches": arm["mean_touches"],
            "escalated_touches": arm["escalated_touches"],
            "resolutions": arm["resolutions"],
        }
    }, indent=1))
    print(f"per-episode written to episodes_imitation_v4t98.json")
    print(f"combined post-hoc report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
