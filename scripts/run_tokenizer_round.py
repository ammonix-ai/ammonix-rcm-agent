"""One tokenizer optimisation round (P4): propose, score, keep/revert, stop.

Round number = next entry in runs/state/tokenizer_rounds.json. Round 1
proposes the deterministic tabular plan of build package Section 3; round 2
adds LLM-extracted booleans over the two text columns (pinned model,
temperature 0, guided JSON). Scoring is mean pooled OOF AUROC of the
per-action swarm over the P3 fold plan on WORKING data only. The round is
kept when it beats the previous round's score by >= 0.005 (round 1 is
scored against chance, 0.5); otherwise the round is recorded as reverted
and its artefacts are not pinned. Either way the delta is reported: for
round 2 it measures what reading the letters is worth.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core")):
    sys.path.insert(0, entry)

import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_file, sha256_text  # noqa: E402
from ammonix_core.pipeline import train_swarm  # noqa: E402
from ammonix_core.schema import DeterminismPin, FoldPlan, Tokenizer  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.extract import (  # noqa: E402
    INDICATION_BOOLEANS,
    LETTER_BOOLEANS,
    CachedExtractor,
    VllmJsonExtractor,
    build_round2_features,
    json_schema,
)
from cardessa.features import build_round1_features, to_qpsi_records  # noqa: E402
from cardessa.features_rounds import build_with_blocks  # noqa: E402

GAIN_THRESHOLD = 0.005
MIN_STATES_TO_SCORE = 30
WORKING = ROOT / "data" / "working" / "cardessa_sim"
STATE_FILE = ROOT / "runs" / "state" / "tokenizer_rounds.json"


def frame_hash(frame: pl.DataFrame) -> str:
    return sha256_text(frame.write_csv())


def scoreable_actions(records, fold_plan: FoldPlan) -> tuple[list[str], dict[str, str]]:
    """Actions where every fold's TRAINING partition holds both classes."""
    eligible, excluded = [], {}
    for action in sorted({r.action_id for r in records}):
        rows = [r for r in records if r.action_id == action]
        if len(rows) < MIN_STATES_TO_SCORE:
            excluded[action] = f"only {len(rows)} states (< {MIN_STATES_TO_SCORE})"
            continue
        labels = {r.outcome_success for r in rows}
        if len(labels) < 2:
            excluded[action] = "single outcome class"
            continue
        bad_fold = None
        for fold in range(fold_plan.n_folds):
            train_labels = {
                r.outcome_success
                for r in rows
                if fold_plan.assignment[r.example_id] != fold
            }
            if len(train_labels) < 2:
                bad_fold = fold
                break
        if bad_fold is not None:
            excluded[action] = f"fold {bad_fold} training partition single-class"
            continue
        eligible.append(action)
    return eligible, excluded


PROPOSALS = {
    1: "deterministic tabular features per build package Section 3 round 1",
    2: "round 1 + LLM text extractor booleans (pinned model, guided JSON)",
    3: "kept blocks + derived pressure/ratio numerics (block: derived)",
    4: "kept blocks + last prior action/CARC one-hots (block: last_event)",
}
ROUND_BLOCK = {3: "derived", 4: "last_event"}


def make_extractors() -> tuple[CachedExtractor, CachedExtractor]:
    letter = CachedExtractor(
        provider=VllmJsonExtractor(schema=json_schema(LETTER_BOOLEANS)),
        cache_dir=ROOT / "data" / "text_extract_cache" / "letters",
        expected_keys=tuple(LETTER_BOOLEANS),
    )
    indication = CachedExtractor(
        provider=VllmJsonExtractor(schema=json_schema(INDICATION_BOOLEANS)),
        cache_dir=ROOT / "data" / "text_extract_cache" / "indications",
        expected_keys=tuple(INDICATION_BOOLEANS),
    )
    return letter, indication


def main() -> int:
    rounds = (
        json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if STATE_FILE.is_file()
        else []
    )
    round_no = len(rounds) + 1
    if round_no not in PROPOSALS:
        raise SystemExit(
            f"no proposal implemented for round {round_no}; state file has "
            f"{len(rounds)} round(s)"
        )
    print(f"tokenizer round {round_no}: {PROPOSALS[round_no]}")

    states = pl.read_parquet(WORKING / "states.parquet")
    episodes = pl.read_parquet(WORKING / "episodes.parquet")
    fold_plan = FoldPlan.model_validate_json(
        (ROOT / "runs" / "manifests" / "fold_plan.json").read_text(encoding="utf-8")
    )

    llm_calls = 0
    extractor_model_id = None
    if round_no == 1:
        features, specs = build_round1_features(states)
        rebuilt, _ = build_round1_features(states)
    elif round_no == 2:
        letter_ex, indication_ex = make_extractors()
        extractor_model_id = letter_ex.provider.model_id
        features, specs = build_round2_features(states, letter_ex, indication_ex)
        llm_calls = letter_ex.calls_to_provider + indication_ex.calls_to_provider
        print(f"LLM extraction calls this run: {llm_calls} (rest from cache)")
        rebuilt, _ = build_round2_features(states, letter_ex, indication_ex)
    else:
        kept_blocks = {
            ROUND_BLOCK[r["round"]]
            for r in rounds
            if r["kept"] and r["round"] in ROUND_BLOCK
        }
        blocks = kept_blocks | {ROUND_BLOCK[round_no]}
        print(f"building with blocks: {sorted(blocks)}")
        features, specs = build_with_blocks(states, blocks)
        rebuilt, _ = build_with_blocks(states, blocks)
    double_equal = frame_hash(features) == frame_hash(rebuilt)
    print(f"features: {len(specs)} specs, double-build equal: {double_equal}")
    if not double_equal:
        raise SystemExit("feature build is not deterministic; refusing to score")

    outcome_by_episode = {
        row["episode_id"]: (bool(row["success"]), float(row["outcome_score"]))
        for row in episodes.select("episode_id", "success", "outcome_score").to_dicts()
    }
    records = to_qpsi_records(features, specs, outcome_by_episode)

    eligible, excluded = scoreable_actions(records, fold_plan)
    for action, reason in excluded.items():
        print(f"  not scored: {action} ({reason})")
    scored_records = [r for r in records if r.action_id in eligible]
    names = [s.name for s in specs]
    swarm = train_swarm(scored_records, fold_plan, names, seed=MASTER_SEED)
    mean_auroc = swarm.mean_oof_auroc
    per_action = {a: round(v, 4) for a, v in sorted(swarm.pooled_auroc.items())}
    print(f"mean OOF AUROC {mean_auroc:.4f} over {len(eligible)} actions: {per_action}")

    previous = rounds[-1]["mean_oof_auroc"] if rounds else 0.5
    gain = mean_auroc - previous
    kept = gain >= GAIN_THRESHOLD
    print(f"previous {previous:.4f} -> gain {gain:+.4f} -> {'KEEP' if kept else 'REVERT'}")

    report = {
        "milestone": "P4",
        "round": round_no,
        "proposal": PROPOSALS[round_no],
        "n_features": len(specs),
        "llm_calls_this_run": llm_calls,
        "mean_oof_auroc": mean_auroc,
        "previous_auroc": previous,
        "gain": gain,
        "kept": kept,
        "per_action_auroc": per_action,
        "actions_excluded": excluded,
        "n_states_scored": len(scored_records),
        "double_build_hash_equal": double_equal,
    }
    (ROOT / "runs" / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "reports" / f"tokenizer_round{round_no}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if kept:
        code_files = [ROOT / "cardessa" / "features.py"]
        llm_pin = None
        if round_no == 2:
            code_files.append(ROOT / "cardessa" / "extract.py")
            llm_pin = {
                "model_id": extractor_model_id,
                "temperature": 0,
                "decoding": "guided_json",
                "enable_thinking": False,
            }
        elif round_no >= 3:
            code_files.append(ROOT / "cardessa" / "features_rounds.py")
        tokenizer = Tokenizer(
            tokenizer_id="cardessa_qpsi",
            version=f"round{round_no}",
            kind="algorithm" if round_no == 1 else "composite",
            features=specs,
            pin=DeterminismPin(
                code_sha256=sha256_text(
                    "".join(sha256_file(f) for f in code_files)
                ),
                seeds={"master": MASTER_SEED},
                llm=llm_pin,
            ),
        )
        (ROOT / "runs" / "manifests" / f"tokenizer_round{round_no}.json").write_text(
            tokenizer.model_dump_json(indent=2), encoding="utf-8"
        )
        qpsi_path = WORKING / f"qpsi_round{round_no}.parquet"
        features.write_parquet(qpsi_path)
        pins = [
            {
                "path": f"data/working/cardessa_sim/qpsi_round{round_no}.parquet",
                "sha256": sha256_file(qpsi_path),
            }
        ]
        (ROOT / "runs" / "manifests" / f"tokenizer_round{round_no}_files.json").write_text(
            json.dumps(pins, indent=2), encoding="utf-8"
        )
        print("tokenizer manifest + Q_PSI file written and pinned")

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    rounds.append(
        {
            "round": round_no,
            "proposal": report["proposal"],
            "mean_oof_auroc": mean_auroc,
            "previous_auroc": previous,
            "gain": gain,
            "kept": kept,
            "n_features": len(specs),
        }
    )
    STATE_FILE.write_text(json.dumps(rounds, indent=2), encoding="utf-8")
    print(f"round state appended: {STATE_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
