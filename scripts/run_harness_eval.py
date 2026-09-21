"""P8 evaluation: run the M1/M2 loop over the dev slice + 20 impossible cases.

Pass = status executed with all checks green. Escalations count against the
pass rate on the dev slice; on the impossible set they ARE the correct
answer. Results append to runs/state/harness_rounds.json (one entry per
optimisation round; rounds may only change harness/prompts/).
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
from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.runtime import RetrievalContext, retrieve, run_case  # noqa: E402
from ammonix_core.schema import (  # noqa: E402
    EscalationRecord,
    ExecutionTrace,
    HarnessArtefacts,
    RetrievalResult,
    Skill,
)

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.engine import PayerEngine  # noqa: E402
from cardessa.harness import (  # noqa: E402
    CONFIDENCE_FLOOR,
    BasisRuntime,
    M1Provider,
    close_out_choice,
    expected_values,
    label_coordinate,
    make_rule_evaluator,
    prompt_fields,
    route_case,
    score_spread,
)
from cardessa.c2_fallback import FeatIndex, c2_action  # noqa: E402

# v0.5: frozen case-based-fallback config (chosen on dev tranche 92, spec s.C2)
C2_CONFIG = dict(k=25, s_min=3, tau=0.45, m=2)
_FEAT_INDEX = None


def _feat_index() -> FeatIndex:
    global _FEAT_INDEX
    if _FEAT_INDEX is None:
        _FEAT_INDEX = FeatIndex(str(ROOT / "basis" / "feat_index"))
    return _FEAT_INDEX
from cardessa.impossible import build_impossible_cases  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

_EVALUATOR = None


def get_rule_evaluator():
    """Shared M2 evaluator: the deterministic rules."""
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = make_rule_evaluator()
    return _EVALUATOR

WORKING = ROOT / "data" / "working" / "cardessa_sim"
STATE_FILE = ROOT / "runs" / "state" / "harness_rounds.json"


def resolve_and_run(rt, skills, harness, provider, case_row, features, x_scaled):
    from datetime import UTC, datetime

    scores, known, forced, ambiguous_override = route_case(rt, case_row, features)
    by_cluster = {s.scope.ref: s for s in skills if s.scope.level == "cluster"}
    default = next(s for s in skills if s.scope.level == "default")

    if forced is not None:
        skill = by_cluster.get(forced, default)
        retrieval = RetrievalResult(
            case_id=case_row["state_id"], features=features, scores_cal={},
            neighbours=[], tribe_id=f"{forced}::cluster",
            recommended_action_id=forced, ambiguous=False,
            skill_id=skill.skill_id,
            expected_result=skill.expected_result.model_dump(),
        )
    elif not scores:
        skill = default
        retrieval = RetrievalResult(
            case_id=case_row["state_id"], features=features, scores_cal={},
            neighbours=[], tribe_id="none::cluster",
            recommended_action_id="write_off", ambiguous=True,
            skill_id=default.skill_id,
            expected_result=default.expected_result.model_dump(),
        )
    else:
        # v0.4 s.3b: retrieval happens in LABEL space - the coordinate is
        # the calibrated score vector, not the scaled features
        u_live = label_coordinate(rt, scores)
        distances, indices = rt.index.kneighbors(u_live[None, :], n_neighbors=25)
        retrieval = retrieve(
            RetrievalContext(
                case_id=case_row["state_id"], features=features,
                scores_cal=scores, known=known,
                scores_std=score_spread(rt, features, set(scores)),
            ),
            rt.tribes, skills, u_live,
            [rt.index_state_ids[i] for i in indices[0]],
            list(map(float, distances[0])),
            rt.trained_actions,
            ambiguous_override=ambiguous_override,
        )
        skill = next(s for s in skills if s.skill_id == retrieval.skill_id)

    if known and forced is None and scores:
        best = max(scores.values())
        if best < CONFIDENCE_FLOOR:
            # v0.4 s.3: close-out by expected value - a 40% shot at a real
            # balance beats a certain zero. Only when every close-out is
            # worthless does the case still go to a human.
            ev = expected_values(case_row, scores)
            # v0.5: outcome-weighted case-based fallback over the feature-space
            # universe, tried before giving up; else the EV close-out as before.
            _c2, _ = c2_action(_feat_index(), features, set(scores), ev,
                               exclude_episode=case_row.get("episode_id"),
                               **C2_CONFIG)
            choice = _c2 if _c2 is not None else close_out_choice(ev, scores)
            if choice is not None:
                by_cluster_all = {
                    sk.scope.ref: sk for sk in skills
                    if sk.scope.level == "cluster" and sk.kind == "execute"
                }
                choice_skill = by_cluster_all.get(choice)
                if choice_skill is not None:
                    retrieval = retrieval.model_copy(update={
                        "recommended_action_id": choice,
                        "ambiguous": False,
                        "skill_id": choice_skill.skill_id,
                        "expected_result":
                            choice_skill.expected_result.model_dump(),
                    })
                    skill = choice_skill
                    best = None  # close-out path proceeds to execution
            if best is not None:
                now = datetime.now(UTC)
                return ExecutionTrace(
                    case_id=retrieval.case_id, basis_id=rt.manifest.basis_id,
                    harness_id=harness.harness_id, retrieval=retrieval,
                    iterations=[], status="escalated",
                    escalation=EscalationRecord(
                        reason="unresolvable",
                        question=(
                            f"best calibrated score {best:.3f} below floor "
                            f"{CONFIDENCE_FLOOR}; every close-out expected "
                            "value is zero; write_off would be the "
                            "bookkeeping default, flagged low-confidence"
                        ),
                        handed_to="human_queue",
                    ),
                    final=None, started_at=now, finished_at=now,
                )

    m1_prompt = harness.m1_prompts.get(retrieval.skill_id)
    if skill.kind == "execute" and m1_prompt is None:
        now = datetime.now(UTC)
        return ExecutionTrace(
            case_id=retrieval.case_id, basis_id=rt.manifest.basis_id,
            harness_id=harness.harness_id, retrieval=retrieval, iterations=[],
            status="escalated",
            escalation=EscalationRecord(reason="unresolvable", handed_to="human_queue"),
            final=None, started_at=now, finished_at=now,
        )
    # run_case handles escalate / ask_before_deciding kinds and the OOD guard
    # itself; the generate callable is only invoked for execute skills
    return run_case(
        retrieval, skill, case_row,
        m1_prompt or harness.m2_prompt, harness.m2_prompt, harness.llm,
        provider.generate, get_rule_evaluator(), rt.manifest.basis_id,
        harness.harness_id, known=known, prompt_fields=prompt_fields,
    )


def main() -> int:
    rt = BasisRuntime.load(ROOT)
    harness = HarnessArtefacts.model_validate_json(
        (ROOT / "runs" / "manifests" / "harness.json").read_text(encoding="utf-8")
    )
    skills_doc = json.loads(
        (ROOT / "runs" / "manifests" / "skills.json").read_text(encoding="utf-8")
    )
    skills = [Skill.model_validate(s) for s in skills_doc["skills"]]
    provider = M1Provider(cache_dir=ROOT / "data" / "m1_cache")
    print(f"vLLM serving pinned model: {provider.model_id}")

    working_states = pl.read_parquet(WORKING / "states.parquet")
    dev_slice = json.loads(
        (ROOT / "runs" / "manifests" / "dev_slice.json").read_text(encoding="utf-8")
    )
    dev_states = working_states.filter(
        pl.col("episode_id").is_in(dev_slice["example_ids"])
    )
    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    impossible = build_impossible_cases(world, engine, MASTER_SEED, working_states)
    category_of = dict(
        zip(
            impossible["state_id"].to_list(),
            impossible["_impossible_category"].to_list(),
            strict=True,
        )
    )
    print(f"dev slice: {dev_states.height} states; impossible: {impossible.height}")

    # tokenise everything against the FULL working frame so categorical
    # columns match the training layout exactly (extra categories from the
    # impossible rows add columns we simply do not select)
    combined = pl.concat(
        [working_states, impossible.drop("_impossible_category")],
        how="vertical_relaxed",
    )
    from cardessa.features_kept import build_kept_features
    frame, _kept_names, _, _ = build_kept_features(ROOT, combined)
    names = rt.feature_names
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise SystemExit(f"tokeniser drift: training features missing {missing[:5]}")
    eval_ids = set(dev_states["state_id"].to_list()) | set(category_of)
    eval_frame = frame.filter(pl.col("state_id").is_in(sorted(eval_ids)))
    features_by_state = {
        row["state_id"]: {n: row[n] for n in names}
        for row in eval_frame.to_dicts()
    }
    rows_by_state = {
        row["state_id"]: row
        for row in pl.concat(
            [dev_states, impossible.drop("_impossible_category")],
            how="vertical_relaxed",
        ).to_dicts()
    }
    x_scaled_all = rt.scaler.transform(
        np.array([[features_by_state[s][n] for n in names] for s in sorted(eval_ids)])
    )
    scaled_by_state = dict(zip(sorted(eval_ids), x_scaled_all, strict=True))

    def one(state_id: str) -> ExecutionTrace:
        return resolve_and_run(
            rt, skills, harness, provider,
            rows_by_state[state_id], features_by_state[state_id],
            scaled_by_state[state_id],
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        traces = dict(zip(sorted(eval_ids), pool.map(one, sorted(eval_ids)), strict=True))

    dev_ids = [s for s in sorted(eval_ids) if s not in category_of]
    executed = [s for s in dev_ids if traces[s].status == "executed"]
    # v0.2 metric split: the harness pass rate measures M1/M2 on the cases
    # the POLICY routes to execution (executed + iteration-cap failures);
    # policy escalations (clock blown, sub-floor confidence, OOD, escalate
    # skills - all decided BEFORE or WITHOUT M1 succeeding being possible)
    # are correct behaviour under outcome v2 and are reported separately
    attempted = [
        s for s in dev_ids
        if traces[s].status == "executed"
        or (traces[s].escalation and traces[s].escalation.reason == "iteration_cap")
    ]
    pass_rate = len(executed) / max(1, len(attempted))
    pass_rate_all_states = len(executed) / len(dev_ids)
    mean_iterations = float(
        np.mean([max(1, len(traces[s].iterations)) for s in attempted])
    )
    impossible_results = [
        {
            "state_id": s,
            "category": category_of[s],
            "status": traces[s].status,
            "reason": traces[s].escalation.reason if traces[s].escalation else None,
        }
        for s in sorted(category_of)
    ]
    impossible_all_escalated = all(
        r["status"] == "escalated" for r in impossible_results
    )
    escalation_reasons: dict[str, int] = {}
    for s in dev_ids:
        if traces[s].status == "escalated":
            reason = traces[s].escalation.reason
            escalation_reasons[reason] = escalation_reasons.get(reason, 0) + 1

    print(f"pass_rate {pass_rate:.4f} ({len(executed)}/{len(dev_ids)}), "
          f"mean_iterations {mean_iterations:.3f}")
    print(f"dev escalations by reason: {escalation_reasons}")
    print(f"impossible all escalated: {impossible_all_escalated}")
    for r in impossible_results:
        print(f"  {r['category']}: {r['status']} ({r['reason']})")
    print(f"M1 provider calls this run: {provider.calls_to_provider}")

    prompt_hashes = {
        p.name: sha256_file(p) for p in sorted((ROOT / "harness" / "prompts").glob("*.txt"))
    }
    rounds = (
        json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if STATE_FILE.is_file()
        else []
    )
    rounds.append(
        {
            "round": len(rounds) + 1,
            "pass_rate": pass_rate,
            "pass_rate_all_states": pass_rate_all_states,
            "n_attempted": len(attempted),
            "mean_iterations": mean_iterations,
            "impossible_all_escalated": impossible_all_escalated,
            "n_dev": len(dev_ids),
            "n_impossible": len(impossible_results),
            "prompt_hashes": prompt_hashes,
        }
    )
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(rounds, indent=2), encoding="utf-8", newline="\n"
    )
    (ROOT / "runs" / "reports" / "harness_eval.json").write_text(
        json.dumps(
            {
                "milestone": "P8",
                "round": rounds[-1]["round"],
                "pass_rate": pass_rate,
                "pass_rate_all_states": pass_rate_all_states,
                "pass_rate_definition": (
                    "executed / (executed + iteration_cap): the harness is "
                    "measured on cases the policy routes to execution; "
                    "policy escalations are correct behaviour, reported in "
                    "dev_escalations_by_reason"
                ),
                "n_attempted": len(attempted),
                "mean_iterations": mean_iterations,
                "n_dev_states": len(dev_ids),
                "n_executed": len(executed),
                "dev_escalations_by_reason": escalation_reasons,
                "impossible_results": impossible_results,
                "impossible_all_escalated": impossible_all_escalated,
                "llm_model_served": provider.model_id,
                "llm_pin": harness.llm.model_dump(),
                "m1_calls": provider.calls_to_provider,
                "prompt_hashes": prompt_hashes,
            },
            indent=2,
        ),
        encoding="utf-8", newline="\n",
    )
    print("report written: runs/reports/harness_eval.json; round state appended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
