"""The Cardessa DatasetDescriptor (build package Section 2, verbatim values,
plus the structured post-hoc quarantine list from its notes)."""

from ammonix_core.platform import DatasetDescriptor


def cardessa_descriptor() -> DatasetDescriptor:
    return DatasetDescriptor(
        dataset_id="cardessa_sim_v1",
        root_uri="data/raw/cardessa_sim/",
        example_table="episodes.parquet",
        state_table="states.parquet",
        id_fields={
            "example_id": "episode_id",
            "state_id": "state_id",
            "seq": "touch_seq",
        },
        action_field="action_raw",  # already canonical-ready from the simulator
        outcome_field="success",
        outcome_positive=True,
        modality_map={
            "*": "tabular",
            "payer_correspondence_text": "text",
            "clinical_indication_text": "text",
        },
        outcome_score_field="outcome_score",
        posthoc_fields=[
            "paid_amount_final",
            "days_to_payment",
            "total_touches",
            "engine_truth_*",
            "mistake_*",
            "n_process_mistakes",
        ],
        notes=(
            "Grouping key meta.patient_id. meta carries payer_id, payer_archetype, "
            "persona_id, family_id, clinic_id. Post-hoc quarantine: "
            "paid_amount_final, days_to_payment, total_touches, engine_truth_* "
            "fields. Holdout payers Granite Shield and Pelican Care route to "
            "quarantine A by construction."
        ),
    )
