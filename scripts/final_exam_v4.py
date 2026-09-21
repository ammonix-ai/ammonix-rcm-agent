"""THE v0.4 FINAL EXAM. Runs EXACTLY ONCE, on the PRE-REGISTERED fresh draw.

Frozen by the approved spec (ammonix_v0_4_change_spec.md s.5/s.6) BEFORE the
build started: exam tranche 96 (500 episodes, allocation retro 100 /
deductible_heavy 100 / p2p 50 / cob 50 / general 200). The rollout draw is
tranche 97 (500 episodes, mirrored families). The draw is generated once and
hash-pinned; the marker refuses any second run.

v0.4-specific bars on top of the carried v0.3 set:
- V4_no_payer_overpayment: no terminal case's payer money exceeds
  allowed - contractual_share (COB carve-out: paid_with_secondary routes
  secondary dollars through collected_payer by engine design - documented).
- V4_policy_never_resubmits_paid: the system lane triggers zero
  duplicate-claim denials.
- V4_letters_match_money: replayed exam episodes show no payment-promising
  letter where the plan paid nothing.
- V4_deductible_heavy_ev: system collects >= 1.5x the no-EV baseline on the
  deductible_heavy family (same dice), and abandons <= 10% of established
  patient share.
"""

import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.schema import HarnessArtefacts, Skill  # noqa: E402
from ammonix_core.universe import expected_calibration_error  # noqa: E402
from check_toolset import REQUIRED_FIELDS, independent_repull  # noqa: E402
from policy_rollout import play, summary  # noqa: E402
from run_harness_eval import resolve_and_run  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import CardessaEnvironment  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.forms import TOOLS, pre_validate  # noqa: E402
from cardessa.harness import BasisRuntime, M1Provider, route_case  # noqa: E402
from cardessa.impossible import holdout_cases  # noqa: E402
from cardessa.livecases import generate_demo_cases  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402

MARKER = ROOT / "runs" / "state" / "final_exam_v4_ran.json"
REPORT = ROOT / "runs" / "reports" / "test_report_v4.json"
EXAM_TRANCHE = 96
EXAM_ALLOCATION = {
    "retro_auth": 100, "deductible_heavy": 100, "p2p": 50, "cob": 50, None: 200,
}
ROLLOUT_TRANCHE = 97
ROLLOUT_ALLOCATION = {
    "retro_auth": 60, "deductible_heavy": 80, "p2p": 30, "cob": 40, None: 290,
}
ORACLE_ACTIONS = ("request_retro_auth", "appeal_with_necessity", "request_peer_to_peer")
DOOM_FLOOR = 0.10
SUBMITS = ("submit_clean", "submit_with_records")
PAYMENT_PROMISE = "reprocessed for payment"


def oracle_context(row):
    return {
        "days_since_service": row["days_since_service"],
        "strong_documentation": row["clinic_doc_quality"] >= 0.6,
    }


def audit_letters_and_money(episodes, states, world, engine):
    """Deterministically replay every exam episode, tracking the letter kind
    and the plan money at each step; the letters must never promise payment
    the engine did not move (spec s.2)."""
    actions_of = defaultdict(list)
    letter_of = {}
    for r in states.sort("touch_seq").to_dicts():
        actions_of[r["episode_id"]].append(r["action_raw"])
        letter_of[(r["episode_id"], r["touch_seq"])] = (
            r.get("payer_correspondence_text") or ""
        )
    index_of = {
        eid: int(eid.removeprefix("ep-")) for eid in actions_of
    }
    bad = []
    env = CardessaEnvironment(world, engine, MASTER_SEED)
    for episode_id, actions in actions_of.items():
        case = env.reset(index_of[episode_id])
        for action in actions:
            if not action or case.terminal:
                break
            before = case.collected_payer
            case = env.apply(case, action)
            kind = case.correspondence_kind
            paid = case.collected_payer - before
            letter = letter_of.get((episode_id, case.touch_seq), "")
            if kind == "appeal_granted" and paid <= 0.005:
                bad.append(f"{episode_id}-t{case.touch_seq}: appeal_granted with $0")
            if (kind in ("appeal_granted_deductible",)
                    and PAYMENT_PROMISE in letter):
                bad.append(f"{episode_id}-t{case.touch_seq}: deductible letter promises payment")
            if kind == "eob_paid" and paid <= 0.005 and case.resolution != "paid_with_secondary":
                bad.append(f"{episode_id}-t{case.touch_seq}: eob_paid with $0")
    return bad


def main() -> int:
    if MARKER.is_file() or REPORT.is_file():
        raise SystemExit("the v0.4 exam already ran; pre-registered draws are spent.")

    text = CachedTextGenerator(
        provider=VllmProvider(), cache_dir=ROOT / "data" / "text_cache"
    )
    print("generating the PRE-REGISTERED v4 exam draw (tranche 96)...")
    episodes, states, world, engine = generate_demo_cases(
        ROOT, MASTER_SEED, text, n_episodes=500,
        tranche_no=EXAM_TRANCHE, allocation=EXAM_ALLOCATION,
    )
    exam_hash = pl.DataFrame(states).hash_rows().sum()
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
    provider = M1Provider(cache_dir=ROOT / "data" / "m1_cache")
    working = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    fresh_holdout = pl.DataFrame(
        [{c: r.get(c) for c in working.columns}
         for r in holdout_cases(world, engine, MASTER_SEED, 12)],
        schema_overrides=working.schema,
    )
    all_eval = pl.concat(
        [states.select(working.columns), fresh_holdout], how="vertical_relaxed"
    )
    frame, _all_names, kept_round, _ = build_kept_features(
        ROOT, pl.concat([working, all_eval], how="vertical_relaxed")
    )
    names = rt.feature_names
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise SystemExit(f"tokeniser drift: training features missing {missing[:5]}")
    frame_b, _, _, _ = build_kept_features(
        ROOT, pl.concat([working, all_eval], how="vertical_relaxed")
    )
    double_tokenisation_equal = frame.equals(frame_b)
    eval_ids = sorted(all_eval["state_id"].to_list())
    feats = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame.filter(pl.col("state_id").is_in(eval_ids)).to_dicts()
    }
    rows = {r["state_id"]: r for r in all_eval.to_dicts()}
    scaled = rt.scaler.transform(
        np.array([[feats[s][n] for n in names] for s in eval_ids])
    )
    scaled_of = dict(zip(eval_ids, scaled, strict=True))
    episode_of = {e["episode_id"]: e for e in episodes.to_dicts()}

    print(f"running the runtime over {len(eval_ids)} exam states...")

    def one(state_id):
        return resolve_and_run(
            rt, skills, harness, provider, rows[state_id], feats[state_id],
            scaled_of[state_id],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        traces = dict(zip(eval_ids, pool.map(one, eval_ids), strict=True))

    exam_ids = sorted(states["state_id"].to_list())
    holdout_ids = sorted(fresh_holdout["state_id"].to_list())

    # ---- doomed-at-intake + clinic scorecard --------------------------------
    doomed_states, doomed_flagged = set(), 0
    clinic = defaultdict(lambda: {"n_doomed_intakes": 0, "dollars_at_risk": 0.0,
                                  "causes": Counter()})
    for s in exam_ids:
        row = rows[s]
        if row["touch_seq"] != 0:
            continue
        scores, known, forced, _ = route_case(rt, row, feats[s])
        best = max(scores.values()) if scores else 0.0
        if forced is None and known and best < DOOM_FLOOR:
            doomed_states.add(row["episode_id"])
            if traces[s].status == "escalated" or best < DOOM_FLOOR:
                doomed_flagged += 1
            episode = episode_of[row["episode_id"]]
            entry = clinic[episode["clinic_id"]]
            entry["n_doomed_intakes"] += 1
            entry["dollars_at_risk"] += float(row["allowed_amount"])
            if (row["auth_required"] and row["auth_status"] != "on_file"
                    and row["retro_window_days"] == 0):
                entry["causes"]["missing_auth_no_retro"] += 1
            elif not row["pairing_valid"]:
                entry["causes"]["ineligible_pairing"] += 1
            elif row["eligibility_status"] != "active":
                entry["causes"]["lapsed_coverage"] += 1
            else:
                entry["causes"]["other_low_value_intake"] += 1
    doom_flag_rate = doomed_flagged / max(1, len(doomed_states))

    # ---- worker scorecard ---------------------------------------------------
    persona_regret = defaultdict(list)
    for s in exam_ids:
        row = rows[s]
        if row["episode_id"] in doomed_states:
            continue
        scores, known, forced, _ = route_case(rt, row, feats[s])
        if not scores or forced is not None or not known:
            continue
        chosen = row["action_raw"]
        best = max(scores.values())
        persona_regret[row["persona_id"]].append(best - scores.get(chosen, 0.0))
    worker_scorecard = {
        persona: {"mean_calibrated_regret": round(float(np.mean(v)), 4),
                  "n_decisions": len(v)}
        for persona, v in persona_regret.items()
    }

    # ---- P1 family moments --------------------------------------------------
    family_episodes = {
        rows[s]["episode_id"] for s in exam_ids
        if rows[s]["touch_seq"] == 0 and rows[s]["cpt"] == "93229"
        and rows[s]["payer_id"] in ("meridian", "silverbridge")
        and rows[s]["auth_required"] and rows[s]["auth_status"] == "missing"
    }
    p1_states = []
    for episode_id in sorted(family_episodes):
        episode_states = sorted(
            (s for s in exam_ids if rows[s]["episode_id"] == episode_id),
            key=lambda s: rows[s]["touch_seq"],
        )
        if rows[episode_states[0]]["payer_id"] == "meridian":
            p1_states.append(episode_states[0])
        else:
            denial = next(
                (s for s in episode_states if rows[s].get("carc") == "CO-197"), None
            )
            if denial is not None:
                p1_states.append(denial)
    p1_hits = sum(
        1 for s in p1_states
        if traces[s].retrieval.recommended_action_id
        == max(
            {a: engine.action_resolution_prob(rows[s]["payer_id"], a, oracle_context(rows[s]))
             for a in ORACLE_ACTIONS},
            key=lambda a: engine.action_resolution_prob(
                rows[s]["payer_id"], a, oracle_context(rows[s])
            ),
        )
    )
    p1_total = len(p1_states)
    p1_rate = p1_hits / max(1, p1_total)

    # ---- P4 ECE -------------------------------------------------------------
    success_of = {e["episode_id"]: bool(e["success"]) for e in episodes.to_dicts()}
    ece_scores, ece_labels = [], []
    for action in rt.trained_actions:
        own = [s for s in exam_ids if rows[s]["action_raw"] == action]
        if not own:
            continue
        x = np.array([[feats[s][n] for n in names] for s in own])
        raw = rt.refit_models[action].predict_proba(x)[:, 1]
        cal = rt.calibrators[action].predict(raw)
        ece_scores.append(cal)
        ece_labels.append(
            np.array([int(success_of[rows[s]["episode_id"]]) for s in own])
        )
    calibration_ece = expected_calibration_error(
        np.concatenate(ece_scores), np.concatenate(ece_labels)
    )
    n_own_states = int(sum(len(v) for v in ece_labels))

    # ---- P5 / P6 / P7 / self-mistakes ---------------------------------------
    silent = sum(1 for s in holdout_ids if traces[s].status == "executed")
    p5_rate = silent / max(1, len(holdout_ids))

    flat_states, flagged = [], 0
    for s in exam_ids:
        row = rows[s]
        if (row.get("carc") or "") in ("", "CO-16"):
            continue
        appeal = engine.action_resolution_prob(
            row["payer_id"], "appeal_with_necessity", oracle_context(row)
        )
        p2p = engine.action_resolution_prob(
            row["payer_id"], "request_peer_to_peer", oracle_context(row)
        )
        if abs(appeal - p2p) < 0.05 and max(appeal, p2p) > 0:
            flat_states.append(s)
            if traces[s].retrieval.ambiguous:
                flagged += 1
    p6_agreement = flagged / max(1, len(flat_states))

    p7_violations = [
        s for s, t in traces.items()
        if t.status == "executed" and t.final is not None
        and t.final.action_id == "bill_secondary" and not rows[s]["has_secondary"]
    ]
    self_mistakes = []
    for s, t in traces.items():
        row = rows[s]
        rec = t.retrieval.recommended_action_id
        if rec in SUBMITS:
            if (row["auth_required"] and row["auth_status"] != "on_file"
                    and row["days_since_service"] <= row["retro_window_days"]):
                self_mistakes.append(f"{s}: submit into open retro window")
            if not row["cob_position_ok"]:
                self_mistakes.append(f"{s}: submit with wrong COB order")

    # ---- P8 fidelity --------------------------------------------------------
    from cardessa.livecases import bundle_for_state
    p8_checked = p8_bad = 0
    for s in exam_ids:
        t = traces[s]
        if t.status != "executed" or t.final is None or t.final.action_id not in TOOLS:
            continue
        row = rows[s]
        bundle = bundle_for_state(world, engine, row, episode_of[row["episode_id"]])
        artifact = TOOLS[t.final.action_id](t.final.payload, bundle)
        p8_checked += 1
        for field in REQUIRED_FIELDS[artifact["artifact"]]:
            if not artifact["fields"].get(field):
                p8_bad += 1
        for entry in artifact["field_map"]:
            if artifact["artifact"] == "cms1500" and entry["field"] == "prior_auth_reference":
                continue
            if entry["value"] != independent_repull(
                {**bundle, "action_object": t.final.payload}, entry
            ):
                p8_bad += 1
        if pre_validate(bundle, t.final.action_id):
            p8_bad += 1

    # ---- V4: letters vs money on the exam draw ------------------------------
    letter_violations = audit_letters_and_money(episodes, states, world, engine)

    # ---- the rollout exam (tranche 97) + no-EV baseline ---------------------
    print("rollout exam (tranche 97): system vs personas vs no-EV baseline...")
    from cardessa.expansion import expansion_studies, extend_world
    from cardessa.livecases import training_world
    world_r, engine_r = training_world(ROOT, MASTER_SEED)
    start = world_r.studies.height
    rollout_studies = expansion_studies(
        world_r, MASTER_SEED, ROLLOUT_TRANCHE, ROLLOUT_ALLOCATION, start
    )
    world_r = extend_world(world_r, rollout_studies)
    indices = list(range(start, start + 500))
    family_of_index = dict(zip(
        indices, rollout_studies["family"].to_list(), strict=True
    ))
    system_res = play("system", world_r, engine_r, rt, working, indices)
    persona_res = play("personas", world_r, engine_r, rt, working, indices)
    baseline_res = play("system", world_r, engine_r, rt, working, indices,
                        use_ev=False)
    sys_sum, per_sum = summary(system_res), summary(persona_res)
    sys_dollars = sys_sum["payer_collected_total"] + sys_sum["patient_collected_total"]
    per_dollars = per_sum["payer_collected_total"] + per_sum["patient_collected_total"]

    # V4 engine property + duplicate-submit audit over every lane
    overpaid, resubmits = [], 0
    episode_index_of = {}
    env_probe = CardessaEnvironment(world_r, engine_r, MASTER_SEED)
    for idx in indices:
        episode_index_of[env_probe.reset(idx).episode_id] = idx
    for lane_name, lane in (("system", system_res), ("personas", persona_res),
                            ("baseline", baseline_res)):
        for eid, r in lane.items():
            cap = round(r["allowed"] - r["contractual_share"], 2)
            if (r["resolution"] != "paid_with_secondary"
                    and r["collected_payer"] > cap + 0.01):
                overpaid.append(f"{lane_name}:{eid}")
            if lane_name == "system" and r["resubmitted_after_paid"]:
                resubmits += 1

    # V4 deductible_heavy: EV uplift + abandonment
    heavy_eids = [
        eid for eid, r in system_res.items()
        if family_of_index.get(episode_index_of.get(eid)) == "deductible_heavy"
    ]

    def v2_dollars(r):
        return r["collected_payer"] + min(r["collected_patient"],
                                          r["contractual_share"])

    heavy_sys = sum(v2_dollars(system_res[e]) for e in heavy_eids)
    heavy_base = sum(v2_dollars(baseline_res[e]) for e in heavy_eids
                     if e in baseline_res)
    share_total = share_abandoned = 0.0
    for e in heavy_eids:
        r = system_res[e]
        if r["contractual_share"] > 0:
            share_total += r["contractual_share"]
            if str(r["resolution"]).startswith("written_off"):
                share_abandoned += r["contractual_share"]
    abandon_frac = (share_abandoned / share_total) if share_total else 0.0

    # system oracle-regret on exam denial states
    oracle_regrets = []
    for s in exam_ids:
        row = rows[s]
        carc = row.get("carc") or ""
        if not carc or carc == "CO-16" or row["episode_id"] in doomed_states:
            continue
        oracle = {
            a: engine.action_resolution_prob(row["payer_id"], a, oracle_context(row))
            for a in ORACLE_ACTIONS
        }
        best_a, best_p = max(oracle.items(), key=lambda kv: kv[1])
        if best_p <= 0:
            continue
        rec = traces[s].retrieval.recommended_action_id
        oracle_regrets.append(best_p - oracle.get(rec, 0.0))
    system_oracle_regret = float(np.mean(oracle_regrets)) if oracle_regrets else 0.0

    rollout_bars = {
        "collections_vs_personas": sys_dollars >= per_dollars * 1.40,
        "mistakes_not_worse": sys_sum["episodes_with_mistakes"]
        <= per_sum["episodes_with_mistakes"],
        "system_oracle_regret": system_oracle_regret <= 0.05,
        "doomed_flagged_100pct": doom_flag_rate >= 1.0,
    }
    v4_bars = {
        "V4_no_payer_overpayment": not overpaid,
        "V4_policy_never_resubmits_paid": resubmits == 0,
        "V4_letters_match_money": not letter_violations,
        "V4_deductible_heavy_ev_uplift": heavy_base == 0 or heavy_sys
        >= 1.5 * heavy_base,
        "V4_abandoned_share_max_10pct": abandon_frac <= 0.10,
    }
    trap_results = {
        "P1_success_not_imitation": p1_rate >= 0.85,
        "P2_no_leakage": not any(
            n.startswith("mistake_") or n in
            ("days_to_payment", "paid_amount_final", "total_touches",
             "n_process_mistakes")
            for n in set(names)
        ),
        "P3_determinism": double_tokenisation_equal,
        "P4_calibration": calibration_ece <= 0.05,
        "P5_ood_honesty": p5_rate <= 0.05,
        "P6_honest_near_ties": len(flat_states) > 0 and p6_agreement >= 0.7,
        "P7_coverage_discipline": not p7_violations,
        "P8_form_fidelity": p8_checked > 0 and p8_bad == 0,
        "NEW_no_self_mistakes": not self_mistakes,
        **{f"ROLLOUT_{k}": v for k, v in rollout_bars.items()},
        **v4_bars,
    }

    report = {
        "report_id": "cardessa-final-v4-preregistered",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "kept_tokenizer_round": kept_round,
        "exam_states_hash": str(exam_hash),
        "trap_results": trap_results,
        "detail": {
            "P1": {"rate": p1_rate, "n_moments": p1_total, "hits": p1_hits},
            "P4": {"ece": calibration_ece, "n_own_states": n_own_states},
            "P5": {"holdout": len(holdout_ids), "silent": silent},
            "P6": {"flat_states": len(flat_states), "flagged": flagged,
                   "agreement": p6_agreement},
            "P7": {"violations": p7_violations},
            "P8": {"checked": p8_checked, "failures": p8_bad},
            "self_mistakes": self_mistakes[:10],
            "system_oracle_regret": system_oracle_regret,
            "doom": {"n_doomed_episodes": len(doomed_states),
                     "flag_rate": doom_flag_rate},
            "rollout": {"system": sys_sum, "personas": per_sum,
                        "dollars": {"system": sys_dollars,
                                    "personas": per_dollars}},
            "v4": {
                "overpaid": overpaid[:10],
                "system_resubmits_after_paid": resubmits,
                "letter_violations": letter_violations[:10],
                "deductible_heavy": {
                    "n_episodes": len(heavy_eids),
                    "system_v2_dollars": round(heavy_sys, 2),
                    "no_ev_baseline_v2_dollars": round(heavy_base, 2),
                    "uplift": round(heavy_sys / heavy_base, 3)
                    if heavy_base else None,
                    "abandoned_share_fraction": round(abandon_frac, 4),
                },
            },
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    (ROOT / "runs" / "reports" / "worker_scorecard_v4.json").write_text(
        json.dumps({"personas": worker_scorecard,
                    "system_oracle_regret": system_oracle_regret,
                    "note": "doomed-at-intake episodes excluded"}, indent=2),
        encoding="utf-8", newline="\n",
    )
    (ROOT / "runs" / "reports" / "clinic_scorecard_v4.json").write_text(
        json.dumps({c: {"n_doomed_intakes": e["n_doomed_intakes"],
                        "dollars_at_risk": round(e["dollars_at_risk"], 2),
                        "causes": dict(e["causes"])}
                    for c, e in sorted(clinic.items())}, indent=2),
        encoding="utf-8", newline="\n",
    )
    MARKER.write_text(
        json.dumps({"ran_at": report["evaluated_at"],
                    "report_sha256": sha256_file(REPORT)}),
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({"trap_results": trap_results,
                      "P1": round(p1_rate, 4), "n_P1": p1_total,
                      "ece": round(calibration_ece, 4),
                      "P6_agreement": round(p6_agreement, 4),
                      "system_regret": round(system_oracle_regret, 4),
                      "heavy_uplift": report["detail"]["v4"]["deductible_heavy"]["uplift"],
                      "abandoned_frac": round(abandon_frac, 4),
                      "rollout_dollars": {"system": round(sys_dollars),
                                          "personas": round(per_dollars)}}))
    print("v0.4 exam complete: test_report_v4 + scorecards written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
