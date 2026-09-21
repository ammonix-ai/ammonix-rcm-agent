"""Amendment: recompute the v0.4 doom-flag bar HONESTLY (audit finding).

test_report_v4.json's ROLLOUT_doomed_flagged_100pct bar is vacuous: inside
the doomed branch (`best < DOOM_FLOOR`) the exam counted
`traces[s].status == "escalated" or best < DOOM_FLOOR` - the second
disjunct restates the branch condition, so the flag rate is 1.0 by
construction. Same defect as v0.3, amended there by recompute_doom_flag.py.

Despite the ROLLOUT_ prefix, the exam computed this bar on the EXAM draw
(tranche 96), not the rollout draw (tranche 97): the doom loop iterates
exam_ids from generate_demo_cases(tranche 96). This script mirrors that.

Honest re-derivation from the same pinned draw, all pre-LLM and
deterministic (route_case + the v0.4 EV close-out in resolve_and_run).
Of the touch-0 states with no forced action, a known payer, and best
calibrated score under DOOM_FLOOR (0.10):
- best < CONFIDENCE_FLOOR (0.05) and every close-out EV zero -> the
  runtime escalates: honestly handed to a human;
- best < CONFIDENCE_FLOOR with a positive-EV close-out -> the runtime
  executes that close-out INSTEAD of escalating (v0.4 s.3): the case is
  worked, not flagged;
- CONFIDENCE_FLOOR <= best < DOOM_FLOOR -> the runtime executes its best
  action normally: no flag of any kind.

Usage: python recompute_doom_flag_v4.py [OUT_JSON]
Regenerates the exam draw deterministically (text via the prompt-hash
cache); never reads or writes the exactly-once exam markers.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.harness import (  # noqa: E402
    CONFIDENCE_FLOOR,
    BasisRuntime,
    close_out_choice,
    expected_values,
    route_case,
)
from cardessa.impossible import holdout_cases  # noqa: E402
from cardessa.livecases import generate_demo_cases  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402

DOOM_FLOOR = 0.10
EXAM_TRANCHE = 96
EXAM_ALLOCATION = {
    "retro_auth": 100, "deductible_heavy": 100, "p2p": 50, "cob": 50, None: 200,
}


def main() -> int:
    text = CachedTextGenerator(
        provider=VllmProvider(), cache_dir=ROOT / "data" / "text_cache"
    )
    episodes, states, world, engine = generate_demo_cases(
        ROOT, MASTER_SEED, text, n_episodes=500,
        tranche_no=EXAM_TRANCHE, allocation=EXAM_ALLOCATION,
    )
    exam_hash = str(pl.DataFrame(states).hash_rows().sum())
    report_path = ROOT / "runs" / "reports" / "test_report_v4.json"
    report_hash = None
    if report_path.is_file():
        report_hash = json.loads(report_path.read_text(encoding="utf-8")).get(
            "exam_states_hash"
        )
    rt = BasisRuntime.load(ROOT)
    working = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    # mirror the exam's tokenisation frame EXACTLY (holdout rows included)
    fresh_holdout = pl.DataFrame(
        [{c: r.get(c) for c in working.columns}
         for r in holdout_cases(world, engine, MASTER_SEED, 12)],
        schema_overrides=working.schema,
    )
    all_eval = pl.concat(
        [states.select(working.columns), fresh_holdout], how="vertical_relaxed"
    )
    frame, _names, _kept, _ = build_kept_features(
        ROOT, pl.concat([working, all_eval], how="vertical_relaxed")
    )
    names = rt.feature_names
    t0 = states.filter(pl.col("touch_seq") == 0)
    feats = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame.filter(
            pl.col("state_id").is_in(t0["state_id"].to_list())
        ).to_dicts()
    }
    doomed = escalated = ev_closed = neither = 0
    closeout_actions: Counter[str] = Counter()
    for row in t0.to_dicts():
        scores, known, forced, _ = route_case(rt, row, feats[row["state_id"]])
        best = max(scores.values()) if scores else 0.0
        if forced is not None or not known or best >= DOOM_FLOOR:
            continue
        doomed += 1
        if not scores:
            # no applicable scored action: the default escalate skill fires
            escalated += 1
        elif best < CONFIDENCE_FLOOR:
            choice = close_out_choice(expected_values(row, scores), scores)
            if choice is None:
                escalated += 1
            else:
                ev_closed += 1
                closeout_actions[choice] += 1
        else:
            neither += 1
    rate = escalated / max(1, doomed)
    amendment = {
        "amends": "runs/reports/test_report_v4.json ROLLOUT_doomed_flagged_100pct",
        "finding": (
            "original bar counted `status == 'escalated' or best < DOOM_FLOOR` "
            "inside the `best < DOOM_FLOOR` branch: vacuously 1.0 by "
            "construction (same defect as v0.3; despite the ROLLOUT_ prefix "
            "the bar ran on the exam draw, tranche 96)"
        ),
        "honest_definition": (
            "of touch-0 states with forced None, known payer, and best "
            "calibrated score under the doom floor (0.10), how many does the "
            "v0.4 runtime actually hand to a human (best under its confidence "
            "floor 0.05 AND every close-out expected value zero); a "
            "positive-EV close-out is executed instead of escalating (v0.4 "
            "s.3), and a doomed intake scoring in [0.05, 0.10) is executed "
            "normally with no flag"
        ),
        "exam_draw": {
            "tranche": EXAM_TRANCHE, "n_episodes": 500,
            "allocation": {k or "general": v for k, v in EXAM_ALLOCATION.items()},
            "recomputed_states_hash": exam_hash,
            "report_states_hash": report_hash,
            "draw_matches_report": exam_hash == report_hash,
        },
        "n_doomed_intakes": doomed,
        "n_escalated_by_system": escalated,
        "n_ev_closed_out": ev_closed,
        "ev_closeout_actions": dict(closeout_actions),
        "n_executed_between_floors": neither,
        "honest_flag_rate": rate,
        "bar_100pct": rate >= 1.0,
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "runs" / "reports" / "test_report_v4_amendment.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(amendment, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(amendment, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
