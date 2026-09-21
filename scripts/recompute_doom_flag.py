"""Amendment: recompute the doom-flag bar HONESTLY (verifier finding).

The exam's original bar was vacuous (tautologically 1.0). The honest
question: of the doomed-at-intake episodes, how many did the SYSTEM
actually hand to a human (escalate) at touch 0? Routing is deterministic
and pre-LLM, so this is re-derivation from the same pinned draw.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.harness import CONFIDENCE_FLOOR, BasisRuntime, route_case  # noqa: E402
from cardessa.livecases import generate_demo_cases  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402

DOOM_FLOOR = 0.10


def main() -> int:
    text = CachedTextGenerator(
        provider=VllmProvider(), cache_dir=ROOT / "data" / "text_cache"
    )
    episodes, states, world, engine = generate_demo_cases(
        ROOT, MASTER_SEED, text, n_episodes=500, tranche_no=95,
        allocation={"retro_auth": 120, "p2p": 60, "cob": 60, None: 260},
    )
    rt = BasisRuntime.load(ROOT)
    working = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    frame, _n, _, _ = build_kept_features(
        ROOT, pl.concat([working, states.select(working.columns)],
                        how="vertical_relaxed")
    )
    names = rt.feature_names
    t0 = states.filter(pl.col("touch_seq") == 0)
    feats = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame.filter(
            pl.col("state_id").is_in(t0["state_id"].to_list())
        ).to_dicts()
    }
    doomed = handed_off = 0
    for row in t0.to_dicts():
        scores, known, forced, _ = route_case(rt, row, feats[row["state_id"]])
        if forced is not None or not known or not scores:
            continue
        best = max(scores.values())
        if best < DOOM_FLOOR:
            doomed += 1
            # the system hands off when the best score is under ITS floor
            if best < CONFIDENCE_FLOOR:
                handed_off += 1
    rate = handed_off / max(1, doomed)
    amendment = {
        "amends": "runs/reports/test_report_v3.json ROLLOUT_doomed_flagged_100pct",
        "verifier_finding": "original bar was vacuously true by construction",
        "honest_definition": (
            "of touch-0 states whose best calibrated score is under the doom "
            "floor (0.10), how many does the runtime actually escalate "
            "(its own confidence floor is 0.05)"
        ),
        "n_doomed_intakes": doomed,
        "n_escalated_by_system": handed_off,
        "honest_flag_rate": rate,
        "bar_100pct": rate >= 1.0,
    }
    (ROOT / "runs" / "reports" / "test_report_v3_amendment.json").write_text(
        json.dumps(amendment, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(amendment, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
