"""The payer adjudicates the paperwork it receives, and the Ammonix-side
appeal check is code only."""

from dataclasses import dataclass

from cardessa import MASTER_SEED
from cardessa.corpus import CardessaEnvironment
from cardessa.engine import PayerEngine
from cardessa.payer_reader import PayerReader, payer_records, reviewer_prompt
from cardessa.payloads import RULES
from cardessa.world import generate_world


@dataclass
class StubReader(PayerReader):
    """A reviewer whose verdict is fixed by the test."""

    verdict: bool = True

    def read(self, letter: str, case_row: dict) -> dict:
        return {"holds_up": self.verdict, "n_statements": 1,
                "unsupported": [] if self.verdict else [letter[:40]]}


def _env(tmp_path, verdict=True):
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    reader = StubReader(cache_dir=tmp_path, verdict=verdict)
    return CardessaEnvironment(world, engine, MASTER_SEED, reader=reader)


def _first_denied(env, attempts=200):
    """An episode whose first clean submission comes back denied."""
    for i in range(attempts):
        case = env.apply(env.reset(i), "submit_clean")
        if case.carc and not case.terminal:
            return case
    raise AssertionError("no denied first submission found")


def test_wrong_codes_are_returned_unprocessed(tmp_path):
    env = _env(tmp_path)
    case = env.reset(0)
    before = case.touch_seq
    case = env.apply(case, "submit_clean", payload={
        "episode_id": case.episode_id, "cpt": "00000", "dx_codes": list(case.patient["diagnoses"]),
    })
    assert case.paperwork_rejections == 1
    assert case.correspondence_kind == "paperwork_rejected"
    assert not case.submitted_once  # the payer never adjudicated it
    assert case.touch_seq == before + 1  # the touch was spent
    assert "procedure code" in case.paperwork_rejection_log[0]


def test_correct_codes_are_adjudicated(tmp_path):
    env = _env(tmp_path)
    case = env.reset(0)
    case = env.apply(case, "submit_clean", payload={
        "episode_id": case.episode_id, "cpt": case.study["cpt"],
        "dx_codes": list(case.patient["diagnoses"]),
    })
    assert case.paperwork_rejections == 0
    assert case.submitted_once


def test_no_payload_means_correct_form(tmp_path):
    env = _env(tmp_path)
    case = env.apply(env.reset(0), "submit_clean")
    assert case.paperwork_rejections == 0 and case.submitted_once


def test_appeal_citing_wrong_denial_is_returned(tmp_path):
    env = _env(tmp_path)
    case = _first_denied(env)
    case = env.apply(case, "appeal_with_necessity", payload={
        "episode_id": case.episode_id, "cpt": case.study["cpt"],
        "denial_carc": "CO-999", "necessity_paragraph": "x" * 90,
    })
    assert case.paperwork_rejections == 1 and not case.appealed


def test_unsupported_letter_is_returned(tmp_path):
    env = _env(tmp_path, verdict=False)
    case = _first_denied(env)
    case = env.apply(case, "appeal_with_necessity", payload={
        "episode_id": case.episode_id, "cpt": case.study["cpt"],
        "denial_carc": case.carc, "necessity_paragraph": "x" * 90,
    }, records={"clinical_indication_text": "note", "payer_correspondence_text": "letter"})
    assert case.paperwork_rejections == 1 and not case.appealed
    assert "not supported" in case.paperwork_rejection_log[0]


def test_supported_letter_reaches_appeal_review(tmp_path):
    env = _env(tmp_path, verdict=True)
    case = _first_denied(env)
    case = env.apply(case, "appeal_with_necessity", payload={
        "episode_id": case.episode_id, "cpt": case.study["cpt"],
        "denial_carc": case.carc, "necessity_paragraph": "x" * 90,
    })
    assert case.paperwork_rejections == 0 and case.appealed


def test_appeal_without_reviewer_is_an_error(tmp_path):
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    case = _first_denied(env)
    try:
        env.apply(case, "appeal_with_necessity", payload={
            "episode_id": case.episode_id, "cpt": case.study["cpt"],
            "denial_carc": case.carc, "necessity_paragraph": "x" * 90,
        })
    except RuntimeError as exc:
        assert "reviewer" in str(exc)
    else:
        raise AssertionError("an appeal letter needs the payer's reviewer")


def test_requested_item_must_be_what_the_payer_asked(tmp_path):
    env = _env(tmp_path)
    case = _first_denied(env)
    case = env.apply(case, "provide_requested_info", payload={
        "episode_id": case.episode_id, "cpt": case.study["cpt"],
        "requested_item": "operative_report",
    })
    assert case.paperwork_rejections == 1


def test_appeal_rules_are_the_scrubber_only():
    assert RULES["appeal_with_necessity"] == [
        "cpt_matches_case", "cites_denial_carc", "paragraph_grounded_in_case",
    ]


def test_reviewer_prompt_lists_only_held_records():
    records = payer_records({"cpt": "93229", "dx_codes": "I48.0", "carc": "",
                             "clinical_indication_text": "note", "payer_correspondence_text": ""})
    assert "denial_reason_code" not in records and "denial_letter" not in records
    prompt = reviewer_prompt("A letter.", records)
    assert "clinical_indication_note: note" in prompt and "LETTER:" in prompt
