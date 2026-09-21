"""Q_PSI proposal blocks for tokenizer rounds 3+ (deterministic, no LLM).

Each block is a named, self-contained set of derived features layered on the
round-1 base; a round proposes exactly one new block on top of every block
kept so far, so keep/revert stays a clean per-block decision.

Block "derived": pressure and ratio numerics the admin could compute from
the same public facts (how much of the filing window is burned, whether the
payer's retro-auth window is still open, balance vs allowed, expected
patient share, delay per touch).

Block "last_event": the most recent prior action and prior CARC as one-hots
(the round-1 history counts lose order; the last event is what an admin
actually reacts to), plus a first-touch flag.
"""

import polars as pl
from ammonix_core.schema import FeatureSpec, HistorySpec

from cardessa.audits import CANONICAL_ACTIONS
from cardessa.codes import CARC
from cardessa.features import build_round1_features

LAST = HistorySpec(lookback_states=1, aggregation="last")

BLOCK_NAMES = ("derived", "last_event")


def _derived(states: pl.DataFrame) -> tuple[list[pl.Expr], list[FeatureSpec]]:
    exprs = [
        (
            pl.col("days_since_service").cast(pl.Float64)
            / pl.col("timely_filing_days").cast(pl.Float64).clip(1.0)
        ).alias("filing_window_burned"),
        (
            pl.col("retro_window_days").cast(pl.Float64)
            - pl.col("days_since_service").cast(pl.Float64)
        ).alias("retro_window_remaining"),
        (
            (pl.col("retro_window_days") >= pl.col("days_since_service"))
            & pl.col("auth_required")
        ).cast(pl.Float64).alias("retro_still_possible"),
        (
            pl.col("balance").cast(pl.Float64)
            / pl.col("allowed_amount").cast(pl.Float64).clip(1.0)
        ).fill_null(-1.0).alias("balance_to_allowed"),
        (
            pl.col("deductible_remaining").cast(pl.Float64).fill_null(0.0)
            + pl.col("coinsurance_pct").cast(pl.Float64).fill_null(0.0)
            * pl.col("balance").cast(pl.Float64)
        ).alias("expected_patient_share"),
        (
            pl.col("cumulative_delay_days").cast(pl.Float64)
            / pl.col("touches_so_far").cast(pl.Float64).clip(1.0)
        ).alias("delay_per_touch"),
    ]
    specs = [
        FeatureSpec(
            name="filing_window_burned", dtype="float", unit="fraction",
            description="Share of the payer's timely-filing window already elapsed",
        ),
        FeatureSpec(
            name="retro_window_remaining", dtype="float", unit="days",
            description="Days left in the payer's retro-auth window (negative = closed)",
        ),
        FeatureSpec(
            name="retro_still_possible", dtype="bool",
            description="Auth is required and the payer's retro window is still open",
        ),
        FeatureSpec(
            name="balance_to_allowed", dtype="float", unit="ratio",
            description="Billed balance over the payer's allowed amount, -1 if unknown",
        ),
        FeatureSpec(
            name="expected_patient_share", dtype="float", unit="USD",
            description="Deductible remaining plus coinsurance share of the balance",
        ),
        FeatureSpec(
            name="delay_per_touch", dtype="float", unit="days",
            description="Cumulative delay divided by touches taken so far",
            history=HistorySpec(lookback_states=99, aggregation="custom"),
        ),
    ]
    return exprs, specs


def _last_event(states: pl.DataFrame) -> tuple[list[pl.Expr], list[FeatureSpec]]:
    prior_actions = pl.col("prior_actions").fill_null("").str.split(",")
    last_action = prior_actions.list.last().fill_null("")
    prior_carcs = pl.col("prior_carcs").fill_null("").str.split(",")
    last_carc = prior_carcs.list.last().fill_null("")

    exprs = [
        (pl.col("prior_actions").fill_null("") == "").cast(pl.Float64)
        .alias("is_first_touch")
    ]
    specs = [
        FeatureSpec(
            name="is_first_touch", dtype="bool",
            description="No action has been taken on this claim before this touch",
        )
    ]
    for action in CANONICAL_ACTIONS:
        name = f"last_action={action}"
        exprs.append((last_action == action).cast(pl.Float64).alias(name))
        specs.append(
            FeatureSpec(
                name=name, dtype="bool",
                description=f"The immediately preceding action on this claim was {action}",
                history=LAST,
            )
        )
    for code in sorted(CARC):
        name = f"last_carc={code}"
        exprs.append((last_carc == code).cast(pl.Float64).alias(name))
        specs.append(
            FeatureSpec(
                name=name, dtype="bool",
                description=f"The most recent prior denial on this claim was {code}",
                history=LAST,
            )
        )
    return exprs, specs


_BLOCKS = {"derived": _derived, "last_event": _last_event}


def build_with_blocks(
    states: pl.DataFrame, blocks: set[str]
) -> tuple[pl.DataFrame, list[FeatureSpec]]:
    """Round-1 base plus the named proposal blocks."""
    unknown = blocks - set(BLOCK_NAMES)
    if unknown:
        raise ValueError(f"unknown proposal blocks: {sorted(unknown)}")
    frame, specs = build_round1_features(states)
    specs = list(specs)
    for block in BLOCK_NAMES:  # fixed order, deterministic column layout
        if block not in blocks:
            continue
        exprs, block_specs = _BLOCKS[block](states)
        # block features come from raw state columns; the round-1 frame kept
        # row order from `states`, so attach horizontally
        frame = frame.hstack(states.select(*exprs))
        specs.extend(block_specs)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate feature names across blocks")
    return frame, specs
