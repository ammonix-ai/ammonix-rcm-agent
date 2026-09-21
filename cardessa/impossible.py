"""The 20 seeded impossible cases (P8): every one must end status=escalated.

- 8 holdout-payer cases: fresh simulated first-touch states for Granite
  Shield / Pelican Care patients, generated on demand from the world (never
  from stored training data; the sealed quarantine files are never read);
- 6 dead-end cases: retro window expired, filing deadline past, the 120-day
  success clock already blown - every path structurally dead or hopeless;
- 6 missing-document cases: a CO-16 denial letter requesting a document
  that does not exist in the record - the responsive action is
  provide_requested_info, which is below coverage, so the system must hand
  off instead of fabricating.
"""

import polars as pl

from cardessa.corpus import CardessaEnvironment, state_tabular
from cardessa.engine import PayerEngine
from cardessa.world import World

HOLDOUT_PAYERS = ("granite", "pelican")

MISSING_DOC_LETTER = (
    "This claim lacks information needed for adjudication (CO-16). To "
    "complete review, submit the original device implantation operative "
    "report signed by the implanting surgeon, including the implant lot "
    "number. No other documentation will be accepted."
)


def holdout_cases(world: World, engine: PayerEngine, master_seed: int, n: int) -> list[dict]:
    """Fresh first-touch states for holdout-payer patients, straight from the
    simulator; no corpus file is involved."""
    env = CardessaEnvironment(world, engine, master_seed)
    payer_of_patient = dict(world.patients.select("patient_id", "payer_id").rows())
    rows = []
    for index, study in enumerate(world.studies.rows(named=True)):
        if payer_of_patient[study["patient_id"]] not in HOLDOUT_PAYERS:
            continue
        case = env.reset(index)
        payer = engine.payers[case.payer_id]
        row = state_tabular(case, payer, 0.0)
        row.update(
            {
                "episode_id": f"imp-holdout-{len(rows):02d}",
                "state_id": f"imp-holdout-{len(rows):02d}-s0",
                "touch_seq": 0,
                "action_raw": "",
                "clinical_indication_text": "",
                "payer_correspondence_text": "",
                "_impossible_category": "holdout_payer",
            }
        )
        rows.append(row)
        if len(rows) == n:
            break
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} holdout studies found, need {n}")
    return rows


def dead_end_cases(template_row: dict, n: int) -> list[dict]:
    """Every path dead: filing window closed, retro window long expired, the
    120-day clock blown, a timely-filing denial on file."""
    rows = []
    for i in range(n):
        row = dict(template_row)
        row.update(
            {
                "episode_id": f"imp-deadend-{i:02d}",
                "state_id": f"imp-deadend-{i:02d}-s0",
                "touch_seq": 3,
                "action_raw": "",
                "carc": "CO-29",
                "days_since_service": 400 + i,
                "days_to_filing_deadline": -220 - i,
                "auth_status": "missing",
                "auth_required": True,
                "retro_window_days": 10,
                "touches_so_far": 3,
                "cumulative_delay_days": 300 + i,
                "prior_actions": "submit_clean,submit_clean,appeal_with_necessity",
                "prior_carcs": "CO-29,CO-29,CO-29",
                "balance": 480.0,
                "has_secondary": False,
                "payer_correspondence_text": (
                    "The time limit for filing this claim has expired (CO-29). "
                    "The appeal has been reviewed and the original decision is upheld."
                ),
                "_impossible_category": "dead_end",
            }
        )
        rows.append(row)
    return rows


def missing_document_cases(template_row: dict, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        row = dict(template_row)
        row.update(
            {
                "episode_id": f"imp-missingdoc-{i:02d}",
                "state_id": f"imp-missingdoc-{i:02d}-s0",
                "touch_seq": 1,
                "action_raw": "",
                "carc": "CO-16",
                "touches_so_far": 1,
                "prior_actions": "submit_clean",
                "prior_carcs": "CO-16",
                "payer_correspondence_text": MISSING_DOC_LETTER,
                "_impossible_category": "missing_document",
            }
        )
        rows.append(row)
    return rows


def build_impossible_cases(
    world: World,
    engine: PayerEngine,
    master_seed: int,
    working_states: pl.DataFrame,
) -> pl.DataFrame:
    """20 cases, aligned to the working states schema plus a category column."""
    template = working_states.filter(
        (pl.col("payer_id") == "meridian") & (pl.col("touch_seq") == 0)
    ).to_dicts()[0]
    rows = (
        holdout_cases(world, engine, master_seed, 8)
        + dead_end_cases(template, 6)
        + missing_document_cases(template, 6)
    )
    schema_cols = [*working_states.columns, "_impossible_category"]
    normalised = [{c: row.get(c) for c in schema_cols} for row in rows]
    return pl.DataFrame(normalised, schema_overrides=working_states.schema)
