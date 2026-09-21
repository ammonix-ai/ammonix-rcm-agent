"""Recompute the GPT-6 token comparison (paper tab:tokens, sec:tokens) from the
shipped tranche-103 rollout reports, and check it against the recorded verdict.

Registered in agentic_baseline/preregistration/comparison_gpt6_tranche103.json:
ammonix_gpt6 wins or ties both GPT-6 arms on success_rate_v2 under the
tranche-99 band (5.64 points) AND spends fewer total GPT-6 tokens than each.
The GPT-6 tokens of an arm are its decision calls plus its writer calls; the
Ammonix arm decides without a language model, so its whole bill is the writer.
The Ornith row is the shipped system on the same draw, added after the
registered arms (descriptive, not part of the rule).

Usage: python scripts/gpt6_tokens_verdict.py   (prints the table; exit 1 on
any mismatch with runs/reports/gpt6_harness_verdict_t103.json)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "runs" / "reports"
BAND = 5.64  # tranche-99 band, points of success rate

ARMS = [  # (label, report file, arm key in the report)
    ("RCM Agent, GPT-6 writing", "rollout_ammonix_gpt6_t103.json", "ammonix"),
    ("RCM Agent, Ornith 1.5 writing", "rollout_ammonix_ornith_t103.json", "ammonix"),
    ("GPT-6 agent, tuned, retrieval", "rollout_gpt6_rag_tuned_t103.json", "gpt6_rag_tuned"),
    ("GPT-6 agent, tuned", "rollout_gpt6_tuned_t103.json", "gpt6_tuned"),
]


def tokens(block: dict) -> int:
    """prompt + completion tokens; the client counters use *_tokens keys, the
    recorded verdict's local-writer block the short prompt/completion keys."""
    return (int(block.get("prompt_tokens", block.get("prompt", 0)))
            + int(block.get("completion_tokens", block.get("completion", 0))))


def row(report_file: str, arm: str) -> dict:
    r = json.loads((REPORTS / report_file).read_text(encoding="utf-8"))
    writer = r["llm"]["writer"]
    decisions = r["policies"].get(arm, {}).get("llm", {})
    frontier = writer["model_id"].startswith("gpt-6")
    return {
        "success_rate": r[arm]["success_rate_v2"],
        "success_pct": round(100 * r[arm]["success_rate_v2"], 1),
        "collected": r[arm]["payer_collected_capped"],
        "gpt6_tokens": (tokens(decisions) + tokens(writer)) if frontier else 0,
        "local_tokens": 0 if frontier else tokens(writer),
        "decision_tokens": tokens(decisions),
        "writer_tokens": tokens(writer),
    }


def main() -> int:
    rows = {label: row(f, arm) for label, f, arm in ARMS}
    print(f"{'Policy':<32}{'Success %':>10}{'Collected':>12}{'GPT-6 tokens':>14}{'Local tokens':>14}")
    for label, v in rows.items():
        print(f"{label:<32}{v['success_pct']:>10.1f}{v['collected']:>12,.0f}"
              f"{v['gpt6_tokens']:>14,}{v['local_tokens']:>14,}")

    agent = rows["RCM Agent, GPT-6 writing"]
    verdict = {}
    for label in ("GPT-6 agent, tuned", "GPT-6 agent, tuned, retrieval"):
        margin = 100 * (agent["success_rate"] - rows[label]["success_rate"])
        verdict[label] = {
            "success": "WIN" if margin > BAND else ("TIE" if margin >= -BAND else "LOSS"),
            "margin_points": round(margin, 2),
            "fewer_tokens": agent["gpt6_tokens"] < rows[label]["gpt6_tokens"],
            "token_ratio": round(rows[label]["gpt6_tokens"] / agent["gpt6_tokens"], 1),
        }
    holds = all(v["success"] in ("WIN", "TIE") and v["fewer_tokens"] for v in verdict.values())
    print()
    for label, v in verdict.items():
        print(f"vs {label}: {v['success']} ({v['margin_points']:+.2f} points, band {BAND}); "
              f"GPT-6 tokens {'fewer' if v['fewer_tokens'] else 'NOT fewer'} ({v['token_ratio']}x)")
    print("registered claim:", "HOLDS" if holds else "DOES NOT HOLD")

    # cross-check against the recorded verdict file
    rec = json.loads((REPORTS / "gpt6_harness_verdict_t103.json").read_text(encoding="utf-8"))
    mismatches = []
    for label, f, arm in ARMS:
        key = {"rollout_ammonix_gpt6_t103.json": "ammonix_gpt6",
               "rollout_gpt6_rag_tuned_t103.json": "gpt6_rag_tuned",
               "rollout_gpt6_tuned_t103.json": "gpt6_tuned"}.get(f)
        v = rows[label]
        if key is None:  # the descriptive Ornith addition
            add = rec["descriptive_addition_not_registered"]
            exp = (round(100 * add["success_rate_v2"], 1), add["payer_collected_capped"],
                   tokens(add["local_writer_tokens"]))
            got = (v["success_pct"], v["collected"], v["local_tokens"])
        else:
            exp = (round(100 * rec["success_rate_v2"][key], 1), rec["payer_collected_capped"][key],
                   rec["gpt6_tokens"][key]["total"])
            got = (v["success_pct"], v["collected"], v["gpt6_tokens"])
        if exp != got:
            mismatches.append((label, exp, got))
    if not holds or mismatches:
        for m in mismatches:
            print("MISMATCH", m)
        return 1
    print("matches runs/reports/gpt6_harness_verdict_t103.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
