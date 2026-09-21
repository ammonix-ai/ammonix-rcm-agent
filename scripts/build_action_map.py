"""P5: build the ActionMap (trivial 1:1 mapping) and the CoverageReport.

The simulator already emits canonical labels, so every rule is an exact
match with no merges; a semantic merge would be a human gate (GATE-P5), a
1:1 map triggers none. Unmatched raw labels fail the build.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402
from ammonix_core.schema import (  # noqa: E402
    ActionMap,
    ActionMapRule,
    CanonicalAction,
    CoverageReport,
)

from cardessa.audits import CANONICAL_ACTIONS, MIN_STATES_PER_ACTION  # noqa: E402
from cardessa.expansion import working_action_counts  # noqa: E402

ACTION_DESCRIPTIONS = {
    "submit_clean": "Submit the claim as-is, no attachments",
    "submit_with_records": "Submit the claim with clinical notes attached",
    "request_retro_auth": "Ask the payer for retroactive authorization",
    "correct_and_resubmit": "Fix claim errors (e.g. COB order) and resubmit",
    "appeal_with_necessity": "Appeal the denial with a medical-necessity letter",
    "request_peer_to_peer": "Request a peer-to-peer review with the payer's medical director",
    "provide_requested_info": "Send the information the payer asked for (CO-16)",
    "bill_secondary": "Bill the patient's secondary coverage for the remaining balance",
    "bill_patient": "Bill the patient for their share of the balance",
    "write_off": "Write the remaining balance off and close the claim",
}


def cumulative_raw() -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = ROOT / "data" / "raw" / "cardessa_sim"
    episodes = [pl.read_parquet(raw / "episodes.parquet")]
    states = [pl.read_parquet(raw / "states.parquet")]
    for tranche_dir in sorted(raw.glob("tranche_*")):
        episodes.append(pl.read_parquet(tranche_dir / "episodes.parquet"))
        states.append(pl.read_parquet(tranche_dir / "states.parquet"))
    return pl.concat(episodes, how="vertical_relaxed"), pl.concat(
        states, how="vertical_relaxed"
    )


def main() -> int:
    episodes, states = cumulative_raw()
    inventory = sorted(states["action_raw"].unique().to_list())
    print(f"raw action inventory ({len(inventory)}): {inventory}")

    unmapped = [label for label in inventory if label not in CANONICAL_ACTIONS]
    if unmapped:
        raise SystemExit(f"unmapped raw labels fail the build: {unmapped}")

    action_map = ActionMap(
        version="cardessa-v1",
        actions=[
            CanonicalAction(
                action_id=action,
                name=action.replace("_", " "),
                description=ACTION_DESCRIPTIONS[action],
                payload_schema={
                    "type": "object",
                    "properties": {"episode_id": {"type": "string"}},
                    "required": ["episode_id"],
                    "additionalProperties": False,
                },
            )
            for action in CANONICAL_ACTIONS
        ],
        rules=[
            ActionMapRule(pattern=action, action_id=action, note=None)
            for action in CANONICAL_ACTIONS
        ],
    )
    counts = working_action_counts(states, episodes)
    coverage = CoverageReport(
        states_total=sum(counts.values()),
        counts=counts,
        min_states_per_action=MIN_STATES_PER_ACTION,
        violations=[a for a, n in counts.items() if n < MIN_STATES_PER_ACTION],
    )

    manifests = ROOT / "runs" / "manifests"
    (manifests / "action_map.json").write_text(
        action_map.model_dump_json(indent=2), encoding="utf-8"
    )
    (ROOT / "runs" / "reports" / "action_map.json").write_text(
        json.dumps(
            {
                "milestone": "P5",
                "raw_inventory": inventory,
                "unmapped": [],
                "merges": [],  # 1:1 map: the GATE-P5 merge approval is not triggered
                "coverage": coverage.model_dump(),
                "dropped_before_training": coverage.violations,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"action map: {len(action_map.rules)} exact rules, 0 merges")
    print(f"coverage: {counts}")
    print(f"violations (drop before training): {coverage.violations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
