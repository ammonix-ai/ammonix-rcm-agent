"""Q_PSI round 1: deterministic tabular features (build package Section 3).

Every feature is a pure function of the state row plus the payer's PUBLISHED
rule facts as the admin would know them; nothing here reads episode outcomes,
post-hoc fields, or the text channel (that is round 2). The poisoned
days_to_payment column is excluded by name and asserted absent.

The same Data must always yield the same features: no randomness, category
sets fixed from the public code tables (CPT, ICD-10 pool, CARC, canonical
actions) or recorded in the FeatureSpec when derived from the working data.
"""

import polars as pl
from ammonix_core.schema import FeatureSpec, HistorySpec, QPsiRecord

from cardessa.audits import CANONICAL_ACTIONS
from cardessa.codes import CARC, CPT_CODES, DX_POOL

POSTHOC_FORBIDDEN = ("paid_amount_final", "days_to_payment", "total_touches")

NUMERIC_COLUMNS: dict[str, tuple[str, str | None]] = {
    # column -> (description, unit)
    "retro_window_days": ("Payer's published retro-authorization window", "days"),
    "timely_filing_days": ("Payer's published timely-filing limit", "days"),
    "days_since_service": ("Elapsed time from date of service to this touch", "days"),
    "days_to_filing_deadline": ("Time left before the payer's filing deadline", "days"),
    "balance": ("Outstanding billed balance on the claim", "USD"),
    # honest description (audit 2026-07-22): the corpus writes this into
    # EVERY state from touch 0 (corpus.py state_tabular), byte-identical to
    # the denylisted meta field engine_truth_allowed. It is NOT revealed by
    # adjudication; it is the exogenous contracted rate, fixed at episode
    # start. Modeling assumption: the contracted rate is known to the
    # billing admin (fee schedules are contract documents). Not a temporal
    # leak - the value never changes and encodes no outcome - but the old
    # description ("if adjudicated, else -1") was false.
    "allowed_amount": (
        "Contracted allowed rate, known from touch 0 (exogenous, fixed at "
        "episode start; assumed known to the billing admin; -1 if absent)",
        "USD",
    ),
    "deductible_remaining": ("Patient deductible remaining, -1 if unknown", "USD"),
    "coinsurance_pct": ("Patient coinsurance share under the plan", "fraction"),
    "touches_so_far": ("How many touches this claim has already taken", "count"),
    "clinic_doc_quality": ("Ordering clinic's documentation-quality trait", "score 0-1"),
}

BOOL_COLUMNS: dict[str, str] = {
    "auth_required": "Payer publishes prior authorization as required for this CPT",
    "p2p_available": "Payer publishes a peer-to-peer review pathway",
    "pairing_valid": "CPT x ICD pairing is valid per the payer's public criteria",
    "aob_on_file": "Assignment of benefits is on file for the patient",
    "cob_position_ok": "This payer is being billed in correct COB position",
    "has_secondary": "Patient carries secondary coverage",
}

# categoricals whose category set is fixed by public code tables
FIXED_CATEGORICALS: dict[str, tuple[tuple[str, ...], str]] = {
    "cpt": (tuple(sorted(CPT_CODES)), "Study CPT code being billed"),
    "carc": (
        tuple(sorted(CARC)) + ("none",),
        "Current CARC denial code on the claim, 'none' before any denial",
    ),
}

# categoricals whose category set is read off the working data and recorded
DATA_CATEGORICALS: dict[str, str] = {
    "payer_archetype": "Payer's behavioural archetype (public reputation class)",
    "auth_status": "Authorization status of the claim at this touch",
    "eligibility_status": "Patient eligibility status with the payer",
    "subscriber_relationship": "Patient's relationship to the plan subscriber",
    "persona_id": "Admin persona working the claim (temperaments are stable)",
}

HISTORY = HistorySpec(lookback_states=99, aggregation="count")


def _hot(column: str, value: str) -> str:
    return f"{column}={value}"


def build_round1_features(
    states: pl.DataFrame,
) -> tuple[pl.DataFrame, list[FeatureSpec]]:
    """Return (feature frame keyed by state_id, FeatureSpec list).

    The frame carries only id/action passthrough columns plus float feature
    columns named exactly after the FeatureSpecs.
    """
    for forbidden in POSTHOC_FORBIDDEN:
        if forbidden in NUMERIC_COLUMNS or forbidden in BOOL_COLUMNS:
            raise AssertionError(f"post-hoc field {forbidden} in feature plan")

    specs: list[FeatureSpec] = []
    exprs: list[pl.Expr] = []

    for column, (description, unit) in NUMERIC_COLUMNS.items():
        exprs.append(pl.col(column).cast(pl.Float64).fill_null(-1.0).alias(column))
        specs.append(
            FeatureSpec(
                name=column, dtype="float", unit=unit, description=description,
                missing_policy="constant",
            )
        )

    for column, description in BOOL_COLUMNS.items():
        exprs.append(pl.col(column).cast(pl.Float64).fill_null(0.0).alias(column))
        specs.append(FeatureSpec(name=column, dtype="bool", description=description))

    categorical_sets: dict[str, tuple[str, ...]] = {}
    for column, (categories, description) in FIXED_CATEGORICALS.items():
        categorical_sets[column] = categories
        for value in categories:
            name = _hot(column, value)
            base = pl.col(column).fill_null("none") if column == "carc" else pl.col(column)
            exprs.append((base == value).cast(pl.Float64).alias(name))
            specs.append(
                FeatureSpec(
                    name=name, dtype="bool",
                    description=f"{description}: one-hot for {value}",
                    categories=list(categories),
                )
            )

    for column, description in DATA_CATEGORICALS.items():
        categories = tuple(sorted(states[column].fill_null("none").unique().to_list()))
        categorical_sets[column] = categories
        for value in categories:
            name = _hot(column, value)
            exprs.append(
                (pl.col(column).fill_null("none") == value).cast(pl.Float64).alias(name)
            )
            specs.append(
                FeatureSpec(
                    name=name, dtype="bool",
                    description=f"{description}: one-hot for {value}",
                    categories=list(categories),
                )
            )

    # diagnosis-pool membership flags + count
    dx_list = pl.col("dx_codes").fill_null("").str.split(",")
    exprs.append(
        dx_list.list.eval(pl.element().filter(pl.element() != "")).list.len()
        .cast(pl.Float64).alias("n_dx_codes")
    )
    specs.append(
        FeatureSpec(
            name="n_dx_codes", dtype="int", unit="count",
            description="Number of diagnosis codes attached to the claim",
        )
    )
    for code in DX_POOL:
        name = f"dx_has_{code.replace('.', '_')}"
        exprs.append(dx_list.list.contains(code).cast(pl.Float64).alias(name))
        specs.append(
            FeatureSpec(
                name=name, dtype="bool",
                description=f"Claim carries diagnosis {code} from the study dx pool",
            )
        )

    # history within the episode (precomputed by the recorder from earlier
    # states only; declared with HistorySpec as episodic features)
    prior_actions = pl.col("prior_actions").fill_null("").str.split(",")
    for action in CANONICAL_ACTIONS:
        name = f"prior_action_count_{action}"
        exprs.append(
            prior_actions.list.eval(pl.element().filter(pl.element() == action))
            .list.len().cast(pl.Float64).alias(name)
        )
        specs.append(
            FeatureSpec(
                name=name, dtype="int", unit="count",
                description=f"How often {action} was already tried on this claim",
                history=HISTORY,
            )
        )
    prior_carcs = pl.col("prior_carcs").fill_null("").str.split(",")
    for code in sorted(CARC):
        name = f"prior_carc_seen_{code.replace('-', '_')}"
        exprs.append(prior_carcs.list.contains(code).cast(pl.Float64).alias(name))
        specs.append(
            FeatureSpec(
                name=name, dtype="bool",
                description=f"A {code} denial was already seen earlier in this episode",
                history=HistorySpec(lookback_states=99, aggregation="max"),
            )
        )
    exprs.append(
        pl.col("cumulative_delay_days").cast(pl.Float64).fill_null(0.0)
        .alias("cumulative_delay_days")
    )
    specs.append(
        FeatureSpec(
            name="cumulative_delay_days", dtype="int", unit="days",
            description="Delay accumulated across all prior touches of this episode",
            history=HistorySpec(lookback_states=99, aggregation="custom"),
        )
    )

    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate feature names in round-1 plan")
    frame = states.select(
        pl.col("state_id"), pl.col("episode_id"), pl.col("touch_seq"),
        pl.col("action_raw"), *exprs,
    )
    return frame, specs


def to_qpsi_records(
    features: pl.DataFrame,
    specs: list[FeatureSpec],
    outcome_by_episode: dict[str, tuple[bool, float]],
) -> list[QPsiRecord]:
    """Assemble QPsiRecords (action_id = raw label until the P5 action map)."""
    names = [spec.name for spec in specs]
    ordered = features.sort("episode_id", "touch_seq")
    records = []
    rows = ordered.to_dicts()
    for i, row in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        success, score = outcome_by_episode[row["episode_id"]]
        records.append(
            QPsiRecord(
                state_id=row["state_id"],
                example_id=row["episode_id"],
                seq=row["touch_seq"],
                features={n: row[n] for n in names},
                action_id=row["action_raw"],
                outcome_success=success,
                outcome_score=score,
                next_state_id=(
                    nxt["state_id"]
                    if nxt is not None and nxt["episode_id"] == row["episode_id"]
                    else None
                ),
            )
        )
    return records
