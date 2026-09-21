"""P8: build the Skill layer, the LLMPin and the HarnessArtefacts.

Skills are written INACTIVE: activation is GATE-P8, the designated approver's written approval
in runs/APPROVALS.md. Initial M1 prompt templates are seeded here; the
optimisation rounds may change only files under harness/prompts/.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

from ammonix_core.hashing import sha256_file, sha256_text  # noqa: E402
from ammonix_core.schema import (  # noqa: E402
    HarnessArtefacts,
    PromptTemplate,
    Tribe,
)

from cardessa.harness import COVERED_ACTIONS, build_skills, make_llm_pin  # noqa: E402

PROMPTS = ROOT / "harness" / "prompts"

# sha256 of each weights shard, computed inside the docker volume
# (docker exec ammonix-m1-27b sha256sum /models/Qwen3.6-27B-AWQ-INT4/*.safetensors)
WEIGHT_SHARD_HASHES = {
    "model-00001-of-00004.safetensors":
        "01c4b57fae6db15daf177e5a96b3e30cd687fc60d178cdc8a31661b4fc0c917d",
    "model-00002-of-00004.safetensors":
        "b2dda3fdb3865495e28e119fabc1d9962667732939dd35364379ffc89b6edff9",
    "model-00003-of-00004.safetensors":
        "a209a59e2c98d2e34979dbdc8a481b7b9a143b246979cbd62a9ff7cc0ec7b8ef",
    "model-00004-of-00004.safetensors":
        "5332e9bc4f3770771304197e7adf225b2979ce884bcd93d28099b8b47b3d0b06",
}

TASK_LINES = {
    "submit_clean": (
        "Submit the claim as-is. dx_codes must be exactly the case's diagnosis "
        "codes as a JSON array (split the comma-separated list; keep each code "
        "verbatim)."
    ),
    "submit_with_records": (
        "Submit the claim with clinical notes attached. dx_codes must be exactly "
        "the case's diagnosis codes as a JSON array. attachments must be "
        '["clinical_notes"].'
    ),
    "correct_and_resubmit": (
        "Fix the claim and resubmit. dx_codes must be exactly the case's "
        "diagnosis codes as a JSON array. correction is \"cob_order\" if the "
        "denial concerns coordination of benefits (CO-22), otherwise "
        "\"claim_fields\". addresses_carc must be exactly the current denial "
        "code shown in the case facts."
    ),
    "request_retro_auth": (
        "Request retroactive authorization. justification_code: pick "
        "\"auth_not_obtained_before_service\" when the study ran without an "
        "auth on file; \"urgent_service_no_time_for_auth\" only if the facts "
        "say the service was urgent; \"payer_error_auth_on_file\" only if the "
        "auth status shows one on file."
    ),
    "appeal_with_necessity": (
        "Appeal the denial on medical necessity. denial_carc must be exactly "
        "the current denial code. necessity_paragraph: 2-4 sentences (at "
        "least 80 characters) stating the clinical facts from the clinical "
        "indication text and the case facts; it must mention the CPT code or "
        "at least one of the case's diagnosis codes verbatim; the payer's "
        "reviewer will check every statement against the submitted records, "
        "so never invent symptoms, dates, documents or findings."
    ),
    "request_peer_to_peer": (
        "Request a peer-to-peer review. denial_carc must be exactly the "
        "current denial code."
    ),
    "provide_requested_info": (
        "Send the payer the information its letter asked for. requested_item "
        "must name what the record can actually provide (the ordering "
        "clinician's notes are the only attachable document); if the letter "
        "demands something the record does not contain, that cannot be "
        "satisfied here."
    ),
    "bill_secondary": "Bill the patient's secondary coverage for the pending share.",
    "bill_patient": (
        "Bill the patient. amount must be the case's outstanding balance, as a "
        "number, never more."
    ),
    "write_off": "Write the remaining balance off and close the claim.",
}

M1_TEMPLATE = """You are the billing agent of a simulated claims system. No real patient exists.
Action to execute: {action_id}

Task: __TASK__

Case facts (use ONLY these; never invent a value):
- episode_id: {episode_id}
- CPT: {cpt}
- diagnosis codes (comma-separated): {dx_codes}
- payer: {payer_id}
- current denial code: {carc}
- outstanding balance: {balance}
- days since service: {days_since_service}
- days to filing deadline: {days_to_filing_deadline}
- payer retro-auth window (days): {retro_window_days}
- auth status: {auth_status}

Latest payer letter (may be empty):
{correspondence}

Clinical indication note:
{indication}

{feedback}
Answer with ONLY the JSON payload."""

M2_TEMPLATE = (
    "Your previous payload failed these checks: {failures}. "
    "Fix exactly these issues using only the case facts above and answer "
    "with the corrected JSON payload."
)


def main() -> int:
    tribes = [
        Tribe.model_validate(t)
        for t in json.loads(
            (ROOT / "basis" / "tribes.json").read_text(encoding="utf-8")
        )
    ]
    skills = build_skills(tribes)
    llm_pin = make_llm_pin(WEIGHT_SHARD_HASHES)
    print(f"LLM pin: {llm_pin.name} weights {llm_pin.weights_sha256[:16]}...")

    PROMPTS.mkdir(parents=True, exist_ok=True)
    m1_prompts = {}
    for action in COVERED_ACTIONS:
        path = PROMPTS / f"m1_{action}.txt"
        if not path.is_file():  # optimisation rounds own these files afterwards
            path.write_text(
                M1_TEMPLATE.replace("__TASK__", TASK_LINES[action]),
                encoding="utf-8", newline="\n",
            )
        skill_id = f"skill-{action}"
        m1_prompts[skill_id] = PromptTemplate(
            template_id=f"m1_{action}",
            version=sha256_file(path)[:12],
            text=path.read_text(encoding="utf-8"),
            output_schema=next(
                s for s in skills if s.skill_id == skill_id
            ).expected_result.payload_schema,
        )
    m2_path = PROMPTS / "m2_feedback.txt"
    if not m2_path.is_file():
        m2_path.write_text(M2_TEMPLATE, encoding="utf-8", newline="\n")
    m2_prompt = PromptTemplate(
        template_id="m2_feedback",
        version=sha256_file(m2_path)[:12],
        text=m2_path.read_text(encoding="utf-8"),
    )

    artefacts = HarnessArtefacts(
        harness_id="cardessa-harness-v1",
        llm=llm_pin,
        m1_prompts=m1_prompts,
        m2_prompt=m2_prompt,
        optimisation_log_uri="runs/state/harness_rounds.json",
    )
    manifests = ROOT / "runs" / "manifests"
    (manifests / "harness.json").write_text(
        artefacts.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    (manifests / "skills.json").write_text(
        json.dumps(
            {
                "activation": "PENDING GATE-P8 (designated approver, runs/APPROVALS.md)",
                "active": False,
                "ask_before_deciding_proposals": [
                    s.skill_id for s in skills
                    if s.kind == "ask_before_deciding"
                ],
                "skills": [s.model_dump() for s in skills],
                "weights_shards": WEIGHT_SHARD_HASHES,
            },
            indent=2,
        ),
        encoding="utf-8", newline="\n",
    )
    print(f"skills: {len(skills)} written (INACTIVE, gate pending); "
          f"harness manifest written; prompts under harness/prompts/")
    skills_text = (manifests / "skills.json").read_text(encoding="utf-8")
    print(f"skills sha256: {sha256_text(skills_text)[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
