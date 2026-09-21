"""V4-M1 checker: engine no-duplicate-payment + money-aware letters.

Prints the one-line JSON verdict (transcript-evidence rule). Read-only:
runs tests/lint in subprocesses and inspects code/behaviour; writes nothing.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)


def run(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-400:]


def main() -> None:
    checks: dict[str, bool] = {}

    ok, out = run([sys.executable, "-m", "pytest", "cardessa/tests", "-q"], ROOT)
    checks["cardessa_tests_green"] = ok and " passed" in out
    ok, out = run([sys.executable, "-m", "pytest", "-q"], ROOT / "ammonix_core")
    checks["core_tests_green"] = ok and " passed" in out
    ok, _ = run(
        [sys.executable, "-m", "ruff", "check", "--config",
         "ammonix_core/pyproject.toml", "cardessa", "scripts"], ROOT,
    )
    checks["ruff_clean"] = ok

    from cardessa import MASTER_SEED
    from cardessa.codes import CARC
    from cardessa.corpus import Case, CardessaEnvironment, compute_mistakes
    from cardessa.engine import PayerEngine, patient_share
    from cardessa.textgen import correspondence_prompt
    from cardessa.world import generate_world

    checks["co18_in_code_table"] = "CO-18" in CARC
    checks["case_has_resubmitted_after_paid"] = (
        "resubmitted_after_paid" in Case.__dataclass_fields__
    )
    dummy = Case.__new__(Case)
    dummy.resubmitted_after_paid = True
    ledger_keys = None
    try:
        ledger_keys = compute_mistakes([], dummy, False).keys()
    except Exception:
        pass
    checks["ledger_has_duplicate_flag"] = bool(
        ledger_keys and "mistake_resubmitted_paid_claim" in ledger_keys
    )

    p = correspondence_prompt(
        "appeal_granted_deductible", "P", "93229", None, "s",
        money={"allowed": 1.0, "payer_paid_total": 0.0,
               "member_responsibility": 1.0},
    )
    checks["deductible_kind_no_promise"] = (
        "NO payment is due" in p and "reprocessed for payment" not in p
    )
    checks["legacy_prompts_unchanged"] = "exact amounts" not in (
        correspondence_prompt("eob_underpaid", "P", "93229", None, "s")
    )

    # live property: payer never exceeds obligation; duplicate submit -> CO-18
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({pl.payer_id: pl for pl in world.payers}, MASTER_SEED)
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    over, dup_ok = 0, False
    for index in range(80):
        case = env.reset(index)
        for action in ("submit_with_records", "appeal_with_necessity",
                       "submit_clean", "submit_clean"):
            if case.terminal:
                break
            paid_before = case.collected_payer
            case = env.apply(case, action)
            if (action.startswith("submit") and paid_before > 0
                    and case.carc == "CO-18"
                    and case.collected_payer == paid_before):
                dup_ok = True
        share = patient_share(
            case.allowed, case.coverage["deductible_remaining"],
            case.coverage["coinsurance_pct"],
        )
        if case.collected_payer > round(case.allowed - share, 2) + 0.01:
            over += 1
    checks["no_payer_overpayment_in_80_episodes"] = over == 0
    checks["duplicate_submit_denied_co18"] = dup_ok

    verdict = all(checks.values())
    print(json.dumps({"loop": "V4-M1", "verdict": verdict, **checks}))
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
