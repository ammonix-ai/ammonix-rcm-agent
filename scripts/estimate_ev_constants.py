"""Audit: are the EV close-out constants recoverable from training data?

cardessa/harness.py expected_values() hardcodes three constants -
P(patient pays | billed, amount <= 300) = 0.7,
P(patient pays | billed, amount > 300) = 0.4, and
P(secondary accepts | billed secondary) = 0.9955 - which its docstring
called "the world's published constants". They were in fact copied from
the simulator's hidden parameters (corpus.py bill_patient branch). This
script estimates all three empirically from WORKING data only (the
recorded raw tranches; quarantine untouched) and reports whether the
hardcoded values fall inside Wilson 95% CIs.

Method: the recorded episodes are replayed deterministically (same master
seed, same recorded action sequences - the audit_letters_and_money
pattern from the exam), because the tabular state channel does not record
the billed patient amount for unpaid bills. Replay fidelity is validated
per episode against the recorded paid_amount_final and resolution; the
replay only re-reads what the corpus already recorded.

Usage: python estimate_ev_constants.py [OUT_JSON]
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import CardessaEnvironment  # noqa: E402
from cardessa.livecases import training_world  # noqa: E402

RAW = ROOT / "data" / "raw" / "cardessa_sim"


def wilson95(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_recorded() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """All recorded raw tranches (episodes, states); never quarantine."""
    sources = ["data/raw/cardessa_sim"]
    episodes = [pl.read_parquet(RAW / "episodes.parquet")]
    states = [pl.read_parquet(RAW / "states.parquet")]
    for tranche in sorted(RAW.glob("tranche_*")):
        if (tranche / "episodes.parquet").is_file():
            sources.append(f"data/raw/cardessa_sim/{tranche.name}")
            episodes.append(pl.read_parquet(tranche / "episodes.parquet"))
            states.append(pl.read_parquet(tranche / "states.parquet"))
    return (
        pl.concat(episodes, how="vertical_relaxed"),
        pl.concat(states, how="vertical_relaxed"),
        sources,
    )


def main() -> int:
    world, engine = training_world(ROOT, MASTER_SEED)
    episodes, states, sources = load_recorded()
    actions_of: dict[str, list[str]] = {}
    for r in states.sort("touch_seq").select(
        "episode_id", "touch_seq", "action_raw"
    ).to_dicts():
        actions_of.setdefault(r["episode_id"], []).append(r["action_raw"])
    recorded = {e["episode_id"]: e for e in episodes.to_dicts()}

    env = CardessaEnvironment(world, engine, MASTER_SEED)
    bp_small = [0, 0]  # [billed, paid] for amount <= 300
    bp_large = [0, 0]  # [billed, paid] for amount > 300
    sec = [0, 0]  # [billed secondary, accepted]
    replay_mismatches = []
    n_replayed = 0
    for episode_id, actions in actions_of.items():
        case = env.reset(int(episode_id.removeprefix("ep-")))
        for action in actions:
            if not action or case.terminal:
                break
            if action == "bill_patient":
                amount = case.patient_share_pending or case.allowed
                before = case.collected_patient
                case = env.apply(case, action)
                bucket = bp_small if amount <= 300 else bp_large
                bucket[0] += 1
                bucket[1] += int(case.collected_patient > before + 0.005)
            elif action == "bill_secondary":
                owed = case.patient_share_pending
                before = case.collected_payer
                case = env.apply(case, action)
                sec[0] += 1
                sec[1] += int(case.collected_payer >= before + owed - 0.005)
            else:
                case = env.apply(case, action)
        n_replayed += 1
        rec = recorded[episode_id]
        replayed_paid = round(case.collected_payer + case.collected_patient, 2)
        if (
            abs(replayed_paid - rec["paid_amount_final"]) > 0.01
            or case.resolution != rec["resolution"]
        ):
            replay_mismatches.append(episode_id)

    def entry(name: str, hardcoded: float, billed: int, paid: int) -> dict:
        lo, hi = wilson95(paid, billed)
        return {
            "constant": name,
            "hardcoded_in_harness": hardcoded,
            "n_billed": billed,
            "n_paid": paid,
            "p_hat": round(paid / billed, 4) if billed else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "hardcoded_inside_ci": bool(billed) and lo <= hardcoded <= hi,
        }

    estimates = [
        entry("P(patient pays | billed, amount <= 300)", 0.7, *bp_small),
        entry("P(patient pays | billed, amount > 300)", 0.4, *bp_large),
        entry("P(secondary accepts | billed secondary)", 0.9955, *sec),
    ]
    report = {
        "audits": "cardessa/harness.py expected_values() hardcoded constants",
        "method": (
            "deterministic replay of every recorded raw-tranche episode "
            "(same master seed, recorded action sequences), observing each "
            "bill_patient / bill_secondary event; replay validated per "
            "episode against recorded paid_amount_final and resolution"
        ),
        "sources": sources,
        "n_episodes_replayed": n_replayed,
        "replay_mismatches": replay_mismatches[:10],
        "replay_validated": not replay_mismatches,
        "estimates": estimates,
        "verdict": (
            "data_recoverable"
            if all(e["hardcoded_inside_ci"] for e in estimates)
            else "oracle_knowledge"
        ),
        "note": (
            "the simulator's generative values are 0.7/0.4 (corpus.py "
            "bill_patient) and exactly 1.0 for bill_secondary (corpus.py "
            "pays the pending share unconditionally); the harness's 0.9955 "
            "matches no simulator constant"
        ),
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "runs" / "reports" / "ev_constants_estimate.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": report["verdict"],
                      "replay_validated": report["replay_validated"],
                      "n_episodes": n_replayed,
                      "estimates": estimates}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
