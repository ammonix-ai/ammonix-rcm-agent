"""P9 toolset checker (the goal condition runs THIS script): on a 200-case
sample of fresh simulated cases, every required artifact field is
provenance-complete, zero values are invented (field-level diff against the
source tables), payer-engine pre-validation passes, and the claim-extraction
check is green on every generated appeal letter.

Prints the one-line JSON verdict the goal condition references.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.schema import HarnessArtefacts, Skill  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.forms import TOOLS, compose_appeal_letter, pre_validate  # noqa: E402
from cardessa.harness import (  # noqa: E402
    BasisRuntime,
    M1Provider,
    prompt_fields,
)
from cardessa.livecases import bundle_for_state, generate_demo_cases  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_harness_eval import resolve_and_run  # noqa: E402

MIN_ARTIFACTS = 200

REQUIRED_FIELDS = {
    "cms1500": [
        "patient_name", "patient_dob", "patient_sex", "patient_address",
        "insured_member_id", "subscriber_name", "subscriber_relationship",
        "payer_name", "diagnosis_codes", "service_cpt", "service_charge",
        "date_of_service_start", "date_of_service_end",
        "ordering_provider_npi", "ordering_clinic_name",
    ],
    "prior_auth_request": [
        "member_id", "member_name", "payer_name", "provider_npi",
        "provider_name", "study_cpt", "diagnosis_codes",
        "date_of_service_start", "clinical_indication",
        "retro_justification_code",
    ],
    "appeal_letter": [
        "member_id", "patient_name", "payer_name", "study_cpt",
        "diagnosis_codes", "denial_carc", "balance", "clinical_indication",
        "denial_letter", "necessity_paragraph",
    ],
}


def independent_repull(bundle: dict, entry: dict) -> str:
    """Re-read a field_map entry's value straight from the source tables -
    the field-level diff proving nothing was invented."""
    source = bundle.get(entry["source_table"]) or {}
    columns = (
        entry["source_column"]
        if isinstance(entry["source_column"], list)
        else [entry["source_column"]]
    )
    parts = []
    for col in columns:
        value = source.get(col)
        if value is None or value == "":
            return ""
        parts.append(str(value))
    return " ".join(parts)


def main() -> int:
    checks: dict[str, bool] = {}
    rt = BasisRuntime.load(ROOT)
    harness = HarnessArtefacts.model_validate_json(
        (ROOT / "runs" / "manifests" / "harness.json").read_text(encoding="utf-8")
    )
    skills = [
        Skill.model_validate(s)
        for s in json.loads(
            (ROOT / "runs" / "manifests" / "skills.json").read_text(encoding="utf-8")
        )["skills"]
    ]
    text = CachedTextGenerator(
        provider=VllmProvider(), cache_dir=ROOT / "data" / "text_cache"
    )
    provider = M1Provider(cache_dir=ROOT / "data" / "m1_cache")
    from cardessa.payloads import evaluate_rule as _rule

    def paragraph_check(paragraph: str, row: dict) -> dict:
        """The appeal paragraph under the Ammonix-side rules (code only)."""
        failures = []
        for rule in ("paragraph_grounded_in_case",):
            ok, detail = _rule(rule, {"necessity_paragraph": paragraph}, row)
            if not ok:
                failures.append(f"{rule}: {detail}")
        return {"passed": not failures, "failures": failures}

    episodes, states, world, engine = generate_demo_cases(ROOT, MASTER_SEED, text)
    print(f"fresh demo cases: {episodes.height} episodes, {states.height} states")
    episode_of = {e["episode_id"]: e for e in episodes.to_dicts()}

    working_states = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    combined = pl.concat(
        [working_states, states.select(working_states.columns)],
        how="vertical_relaxed",
    )
    from cardessa.features_kept import build_kept_features
    frame, _kept_names, _, _ = build_kept_features(ROOT, combined)
    names = rt.feature_names
    demo_ids = set(states["state_id"].to_list())
    eval_frame = frame.filter(pl.col("state_id").is_in(sorted(demo_ids)))
    features_by_state = {
        row["state_id"]: {n: row[n] for n in names} for row in eval_frame.to_dicts()
    }
    rows_by_state = {r["state_id"]: r for r in states.to_dicts()}
    ordered = sorted(demo_ids)
    scaled = rt.scaler.transform(
        np.array([[features_by_state[s][n] for n in names] for s in ordered])
    )
    scaled_by_state = dict(zip(ordered, scaled, strict=True))

    def one(state_id):
        return resolve_and_run(
            rt, skills, harness, provider,
            rows_by_state[state_id], features_by_state[state_id],
            scaled_by_state[state_id],
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        traces = dict(zip(ordered, pool.map(one, ordered), strict=True))

    artifacts = []
    provenance_failures: list[str] = []
    invented: list[str] = []
    prevalidation_failures: list[str] = []
    appeal_results = []
    for state_id, trace in traces.items():
        if trace.status != "executed" or trace.final is None:
            continue
        action = trace.final.action_id
        tool = TOOLS.get(action)
        if tool is None:  # bill_patient / write_off produce no artifact
            continue
        row = rows_by_state[state_id]
        bundle = bundle_for_state(world, engine, row, episode_of[row["episode_id"]])
        artifact = tool(trace.final.payload, bundle)
        bundle_with_action = {**bundle, "action_object": trace.final.payload}
        artifacts.append((state_id, action, artifact))

        for field in REQUIRED_FIELDS[artifact["artifact"]]:
            if not artifact["fields"].get(field):
                provenance_failures.append(f"{state_id}:{artifact['artifact']}:{field}")
        for entry in artifact["field_map"]:
            expected = independent_repull(bundle_with_action, entry)
            rendered = entry["value"]
            if artifact["artifact"] == "cms1500" and entry["field"] == "prior_auth_reference":
                continue  # 'missing' is blanked to a declared gap by design
            if rendered != expected:
                invented.append(f"{state_id}:{entry['field']}: {rendered!r} != {expected!r}")
        failures = pre_validate(bundle, action)
        if failures:
            prevalidation_failures.append(f"{state_id}:{action}: {failures}")
        if artifact["artifact"] == "appeal_letter":
            result = paragraph_check(artifact["necessity_paragraph"], row)
            appeal_results.append({"state_id": state_id, **result})

    # appeal audit set: the recommendation policy rarely picks appeals (other
    # paths usually score higher), so exercise the appeal tool + claim check
    # deliberately on appealable denial states, labelled as an audit
    from ammonix_core.runtime import run_case  # noqa: E402
    from ammonix_core.schema import RetrievalResult  # noqa: E402

    from cardessa.harness import make_rule_evaluator  # noqa: E402

    evaluator = make_rule_evaluator()
    appeal_skill = next(s for s in skills if s.scope.ref == "appeal_with_necessity")
    appealable = [
        s for s in ordered
        if rows_by_state[s].get("carc") in ("CO-50", "CO-97", "CO-197", "CO-29")
    ][:30]

    def audit_one(state_id):
        row = rows_by_state[state_id]
        retrieval = RetrievalResult(
            case_id=state_id, features={}, scores_cal={}, neighbours=[],
            tribe_id="appeal_with_necessity::cluster",
            recommended_action_id="appeal_with_necessity", ambiguous=False,
            skill_id=appeal_skill.skill_id,
            expected_result=appeal_skill.expected_result.model_dump(),
        )
        return state_id, run_case(
            retrieval, appeal_skill, row,
            harness.m1_prompts[appeal_skill.skill_id], harness.m2_prompt,
            harness.llm, provider.generate, evaluator,
            rt.manifest.basis_id, harness.harness_id,
            prompt_fields=prompt_fields,
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        audit_traces = list(pool.map(audit_one, appealable))
    n_audit_executed = 0
    for state_id, trace in audit_traces:
        if trace.status != "executed":
            continue
        n_audit_executed += 1
        row = rows_by_state[state_id]
        bundle = bundle_for_state(world, engine, row, episode_of[row["episode_id"]])
        artifact = compose_appeal_letter(trace.final.payload, bundle)
        result = paragraph_check(artifact["necessity_paragraph"], row)
        appeal_results.append({"state_id": state_id, "audit": True, **result})
    print(f"appeal audit set: {len(appealable)} denial states, "
          f"{n_audit_executed} letters composed")

    n_appeals = len(appeal_results)
    checks["sample_large_enough"] = len(artifacts) >= MIN_ARTIFACTS
    checks["required_fields_provenance_complete"] = not provenance_failures
    checks["zero_invented_values"] = not invented
    checks["engine_prevalidation_green"] = not prevalidation_failures
    checks["appeal_letters_present"] = n_appeals >= 10
    checks["claim_extraction_green"] = n_appeals > 0 and all(
        r["passed"] for r in appeal_results
    )
    checks["field_maps_committed"] = (
        ROOT / "cardessa" / "field_maps.yaml"
    ).is_file()

    report = {
        "milestone": "P9-toolset",
        "n_demo_episodes": episodes.height,
        "n_demo_states": states.height,
        "n_artifacts": len(artifacts),
        "artifacts_by_kind": {
            kind: sum(1 for _, _, a in artifacts if a["artifact"] == kind)
            for kind in ("cms1500", "prior_auth_request", "appeal_letter")
        },
        "provenance_failures": provenance_failures[:20],
        "invented_values": invented[:20],
        "prevalidation_failures": prevalidation_failures[:20],
        "n_appeal_letters": n_appeals,
        "appeal_failures": [r for r in appeal_results if not r["passed"]][:10],
        "m1_calls": provider.calls_to_provider,
        "text_calls": text.calls_to_provider,
    }
    (ROOT / "runs" / "reports" / "toolset_check.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )

    verdict = all(checks.values())
    print(
        json.dumps(
            {
                "loop": "P9T",
                "verdict": verdict,
                **checks,
                "n_artifacts": len(artifacts),
                "n_appeal_letters": n_appeals,
            }
        )
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
