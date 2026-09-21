"""Stage 0: generic ingest via a DatasetDescriptor (platform plan Section 3).

One descriptor fully defines how a raw episodic dataset maps onto the schema;
this module contains no domain knowledge. It reads the example and state
tables, emits schema-shaped Examples and RawStates, and writes the ingest
audit: the four scope conditions, a missing-value profile, the raw action
inventory, and the leakage audit.

Leakage audit (trap P2): a state-table column is excluded from Data when it
is (a) denylisted by the descriptor's posthoc_fields, or (b) statistically
post-hoc — constant within every episode while separating outcomes almost
perfectly (pooled AUC outside [1 - threshold, threshold]). Both paths are
reported; excluded columns never reach the feature inventory.
"""

import fnmatch
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ammonix_core.hashing import sha256_json
from ammonix_core.platform import DatasetDescriptor
from ammonix_core.schema import Example, Outcome, RawData, RawState

SCOPE_VOLUME_MIN = 100  # "hundreds to millions of Examples"
SCOPE_VOLUME_MAX = 10_000_000
LEAK_AUC_THRESHOLD = 0.98


@dataclass
class IngestResult:
    examples: list[Example]
    states: list[RawState]
    audit: dict


def _leak_screen(
    states: pl.DataFrame, outcome_by_example: dict, example_col: str
) -> dict[str, float]:
    """Pooled AUC per numeric/bool column vs outcome; near-perfect separation
    on a column that never varies within an episode is post-hoc information."""
    flagged: dict[str, float] = {}
    y = np.array(
        [outcome_by_example[e] for e in states[example_col].to_list()], dtype=int
    )
    if len(np.unique(y)) < 2:
        return flagged
    for column in states.columns:
        dtype = states[column].dtype
        if not (dtype.is_numeric() or dtype == pl.Boolean):
            continue
        all_values = states[column].cast(pl.Float64, strict=False).to_numpy()
        # missing values must not exempt a column from the screen: evaluate
        # on the non-null rows (a leaking column with one NaN still leaks)
        present = ~np.isnan(all_values)
        values, y_col = all_values[present], y[present]
        if len(values) == 0 or len(np.unique(y_col)) < 2:
            continue
        constant_within = (
            states.group_by(example_col)
            .agg(pl.col(column).drop_nulls().n_unique().alias("n"))["n"]
            .max()
            <= 1
        )
        if not constant_within:
            continue
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1)
        n_pos, n_neg = int(y_col.sum()), int((1 - y_col).sum())
        auc = (ranks[y_col == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        if auc >= LEAK_AUC_THRESHOLD or auc <= 1 - LEAK_AUC_THRESHOLD:
            flagged[column] = round(float(auc), 4)
    return flagged


def _denylisted(column: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(column, pattern) for pattern in patterns)


def ingest_dataset(descriptor: DatasetDescriptor, base_dir: str | Path) -> IngestResult:
    root = Path(base_dir) / descriptor.root_uri
    example_df = pl.read_parquet(root / descriptor.example_table)
    state_df = pl.read_parquet(root / descriptor.state_table)

    example_col = descriptor.id_fields["example_id"]
    state_col = descriptor.id_fields["state_id"]
    seq_col = descriptor.id_fields["seq"]

    # -- outcomes: exactly one per Example ---------------------------------
    outcome_by_example: dict[str, bool] = {}
    score_by_example: dict[str, float | None] = {}
    for row in example_df.rows(named=True):
        raw = bool(row[descriptor.outcome_field])
        outcome_by_example[row[example_col]] = (
            raw if descriptor.outcome_positive else not raw
        )
        score_by_example[row[example_col]] = (
            float(row[descriptor.outcome_score_field])
            if descriptor.outcome_score_field
            else None
        )
    scope_one_outcome = (
        example_df[example_col].n_unique() == example_df.height
        and example_df[descriptor.outcome_field].null_count() == 0
    )

    # -- leakage audit ------------------------------------------------------
    id_like = {example_col, state_col, seq_col, descriptor.action_field}
    denylisted = [
        c
        for c in state_df.columns
        if c not in id_like and _denylisted(c, descriptor.posthoc_fields)
    ]
    statistically_flagged = _leak_screen(
        state_df.drop([c for c in id_like if c in state_df.columns and c != example_col]),
        outcome_by_example,
        example_col,
    )
    excluded = sorted(set(denylisted) | set(statistically_flagged))

    # -- states -------------------------------------------------------------
    data_columns = [
        c for c in state_df.columns if c not in id_like and c not in excluded
    ]
    modalities = {
        c: descriptor.modality_map.get(c, descriptor.modality_map.get("*", "tabular"))
        for c in data_columns
    }
    states: list[RawState] = []
    state_df_sorted = state_df.sort([example_col, seq_col])
    rows_by_example: dict[str, list[dict]] = {}
    for row in state_df_sorted.rows(named=True):
        rows_by_example.setdefault(row[example_col], []).append(row)

    scope_states = True
    scope_data_action = True
    for example_id, rows in rows_by_example.items():
        seqs = [r[seq_col] for r in rows]
        if seqs != list(range(len(rows))):
            scope_states = False
        for i, row in enumerate(rows):
            inline = {c: row[c] for c in data_columns}
            if not inline or row[descriptor.action_field] in (None, ""):
                scope_data_action = False
            states.append(
                RawState(
                    state_id=row[state_col],
                    example_id=example_id,
                    seq=int(row[seq_col]),
                    data=RawData(
                        modality="composite" if len(set(modalities.values())) > 1 else "tabular",
                        inline=inline,
                        sha256=sha256_json({k: inline[k] for k in sorted(inline)}),
                    ),
                    action_raw=str(row[descriptor.action_field]),
                    next_state_id=rows[i + 1][state_col] if i + 1 < len(rows) else None,
                )
            )

    # -- examples -------------------------------------------------------------
    meta_columns = [
        c
        for c in example_df.columns
        if c not in (example_col, descriptor.outcome_field, descriptor.outcome_score_field)
    ]
    examples: list[Example] = []
    for row in example_df.rows(named=True):
        example_id = row[example_col]
        state_rows = rows_by_example.get(example_id, [])
        examples.append(
            Example(
                example_id=example_id,
                state_ids=[r[state_col] for r in state_rows],
                outcome=Outcome(
                    success=outcome_by_example[example_id],
                    score=score_by_example[example_id],
                ),
                meta={c: row[c] for c in meta_columns},
            )
        )

    # -- audit ----------------------------------------------------------------
    action_inventory = dict(
        state_df.group_by(descriptor.action_field).len().rows()
    )
    missing_profile = {
        c: round(state_df[c].null_count() / max(1, state_df.height), 4)
        for c in state_df.columns
        if state_df[c].null_count() > 0
    }
    scope_volume = SCOPE_VOLUME_MIN <= len(examples) <= SCOPE_VOLUME_MAX
    scope_states = scope_states and all(e.state_ids for e in examples)

    audit = {
        "dataset_id": descriptor.dataset_id,
        "n_examples": len(examples),
        "n_states": len(states),
        "scope_conditions": {
            "volume": scope_volume,
            "states_consecutive_per_example": scope_states,
            "data_and_action_per_state": scope_data_action,
            "one_outcome_per_example": scope_one_outcome,
        },
        "raw_action_inventory": action_inventory,
        "missing_value_profile": missing_profile,
        "leakage_audit": {
            "denylisted": sorted(denylisted),
            "statistically_flagged": statistically_flagged,
            "excluded_from_data": excluded,
        },
        "data_columns_by_modality": {
            m: sorted(c for c, mm in modalities.items() if mm == m)
            for m in sorted(set(modalities.values()))
        },
        "example_meta_columns": sorted(meta_columns),
    }
    return IngestResult(examples=examples, states=states, audit=audit)
