"""THE v0.2 FINAL EVALUATION on RESERVE B. Runs EXACTLY ONCE.

Reserve B (the 100 reserve-side episodes of tranche 0) was sealed at W1 and
never read by any build loop. It is materialised by deterministic
regeneration; the HARD GATE of change-spec section 4.2 aborts before any
scoring unless the regenerated tranche hash equals the pinned corpus hash
(the v0.1 text-cache caveat is thereby structurally impossible here).

Acceptance criteria per ammonix_v0_2_change_spec.md section 6. Where reserve
B cannot carry a trap population by construction, fresh simulated cases from
the same world stand in, and the report says so:
- P5 holdout-payer cases: holdout patients live in quarantine A only, so
  OOD honesty is measured on freshly simulated holdout cases (never data).
- P6 near-tie cases: measured on reserve cornerstone strong-doc bundling
  states, supplemented with fresh simulated bundling cases if fewer than 10.
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.hashing import sha256_file  # noqa: E402
from ammonix_core.platform import TestReport  # noqa: E402
from ammonix_core.schema import FoldMetrics, HarnessArtefacts, Skill  # noqa: E402
from ammonix_core.universe import expected_calibration_error  # noqa: E402
from build_corpus import PromptCollector, warm_cache  # noqa: E402
from check_toolset import REQUIRED_FIELDS, independent_repull  # noqa: E402
from run_harness_eval import resolve_and_run  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import TRANCHE_SIZE_INITIAL, generate_tranche  # noqa: E402
from cardessa.engine import PayerEngine  # noqa: E402
from cardessa.features import build_round1_features  # noqa: E402
from cardessa.forms import TOOLS, pre_validate  # noqa: E402
from cardessa.harness import BasisRuntime, M1Provider  # noqa: E402
from cardessa.impossible import holdout_cases  # noqa: E402
from cardessa.livecases import bundle_for_state, generate_demo_cases  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

MARKER = ROOT / "runs" / "state" / "final_eval_reserve_ran.json"
REPORT = ROOT / "runs" / "reports" / "test_report_v2.json"
HOLDOUT = {"granite", "pelican"}
ORACLE_ACTIONS = ("request_retro_auth", "appeal_with_necessity", "request_peer_to_peer")
SUBMITS = ("submit_clean", "submit_with_records")


def oracle_context(row: dict) -> dict:
    return {
        "days_since_service": row["days_since_service"],
        "strong_documentation": row["clinic_doc_quality"] >= 0.6,
    }


def main() -> int:
    if MARKER.is_file() or REPORT.is_file():
        raise SystemExit(
            "the reserve-B evaluation already ran; the last sealed draw is "
            "spent. Only fresh simulated data remains for any v0.3."
        )

    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    text = CachedTextGenerator(
        provider=VllmProvider(), cache_dir=ROOT / "data" / "text_cache"
    )

    print("regenerating tranche 0 (text from cache)...")
    collector = PromptCollector()
    generate_tranche(world, engine, MASTER_SEED, collector, 0, TRANCHE_SIZE_INITIAL)
    warm_cache(collector.prompts, text)
    first = generate_tranche(world, engine, MASTER_SEED, text, 0, TRANCHE_SIZE_INITIAL)
    second = generate_tranche(world, engine, MASTER_SEED, text, 0, TRANCHE_SIZE_INITIAL)
    double_generation_equal = first.content_sha256() == second.content_sha256()

    # HARD GATE (spec 4.2): the regenerated corpus must BE the pinned corpus
    pinned = json.loads(
        (ROOT / "runs" / "reports" / "corpus_tranche_0.json").read_text(
            encoding="utf-8"
        )
    )["tranche_content_sha256"]
    if first.content_sha256() != pinned:
        raise SystemExit(
            f"HARD GATE: regenerated tranche hash {first.content_sha256()[:16]} "
            f"!= pinned {pinned[:16]}; the lineage is not byte-faithful - "
            "refusing to score"
        )
    print(f"hard gate passed: tranche hash matches the pin ({pinned[:16]}...)")

    episodes = pl.DataFrame(first.episodes_rows)
    states = pl.DataFrame(first.states_rows)
    r_episodes = episodes.filter(pl.col("split") == "reserve_b")
    r_ids = set(r_episodes["episode_id"].to_list())
    r_states = states.filter(pl.col("episode_id").is_in(sorted(r_ids)))
    print(f"reserve B: {r_episodes.height} episodes, {r_states.height} states")

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

    # stand-in populations reserve B cannot carry by construction
    working = pl.read_parquet(
        ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    fresh_holdout = pl.DataFrame(
        [
            {c: r.get(c) for c in working.columns}
            for r in holdout_cases(world, engine, MASTER_SEED, 12)
        ],
        schema_overrides=working.schema,
    )
    demo_eps, demo_states, demo_world, _ = generate_demo_cases(
        ROOT, MASTER_SEED, text, n_episodes=40, tranche_no=91,
        allocation={"p2p": 40},
    )
    fresh_p2p = demo_states.select(working.columns)
    print(f"stand-ins: {fresh_holdout.height} holdout, {fresh_p2p.height} bundling states")

    all_eval = pl.concat(
        [r_states.select(working.columns), fresh_holdout, fresh_p2p],
        how="vertical_relaxed",
    )
    frame_a, _ = build_round1_features(
        pl.concat([working, all_eval], how="vertical_relaxed")
    )
    frame_b, _ = build_round1_features(
        pl.concat([working, all_eval], how="vertical_relaxed")
    )
    double_tokenisation_equal = frame_a.equals(frame_b)
    names = rt.feature_names
    eval_ids = sorted(all_eval["state_id"].to_list())
    feats = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame_a.filter(pl.col("state_id").is_in(eval_ids)).to_dicts()
    }
    rows = {r["state_id"]: r for r in all_eval.to_dicts()}
    scaled = rt.scaler.transform(
        np.array([[feats[s][n] for n in names] for s in eval_ids])
    )
    scaled_of = dict(zip(eval_ids, scaled, strict=True))
    episode_of = {e["episode_id"]: e for e in r_episodes.to_dicts()}

    print(f"running the runtime over {len(eval_ids)} states...")

    def one(state_id):
        return resolve_and_run(
            rt, skills, harness, provider, rows[state_id], feats[state_id],
            scaled_of[state_id],
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        traces = dict(zip(eval_ids, pool.map(one, eval_ids), strict=True))

    reserve_ids = sorted(r_ids and set(r_states["state_id"].to_list()))
    holdout_ids = sorted(fresh_holdout["state_id"].to_list())
    p2p_ids = sorted(fresh_p2p["state_id"].to_list())

    # ---- per-action metrics + P4 calibration ECE on reserve ---------------
    success_of = {e["episode_id"]: bool(e["success"]) for e in r_episodes.to_dicts()}
    per_action_metrics: dict[str, FoldMetrics] = {}
    ece_scores, ece_labels = [], []
    for action in rt.trained_actions:
        own = [s for s in reserve_ids if rows[s]["action_raw"] == action]
        if not own:
            continue
        x = np.array([[feats[s][n] for n in names] for s in own])
        raw = rt.refit_models[action].predict_proba(x)[:, 1]
        cal = rt.calibrators[action].predict(raw)
        y = np.array([int(success_of[rows[s]["episode_id"]]) for s in own])
        ece_scores.append(cal)
        ece_labels.append(y)
        if len(set(y)) == 2:
            per_action_metrics[action] = FoldMetrics(
                auroc=float(roc_auc_score(y, cal)),
                auprc=float(average_precision_score(y, cal)),
                n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
            )
    calibration_ece = expected_calibration_error(
        np.concatenate(ece_scores), np.concatenate(ece_labels)
    )

    # ---- P1: family decision moments on reserve ---------------------------
    family_episodes = {
        rows[s]["episode_id"] for s in reserve_ids
        if rows[s]["touch_seq"] == 0 and rows[s]["cpt"] == "93229"
        and rows[s]["payer_id"] in ("meridian", "silverbridge")
        and rows[s]["auth_required"] and rows[s]["auth_status"] == "missing"
    }
    p1_states = []
    for episode_id in sorted(family_episodes):
        episode_states = sorted(
            (s for s in reserve_ids if rows[s]["episode_id"] == episode_id),
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
    p1_hits = 0
    for s in p1_states:
        row = rows[s]
        oracle = {
            a: engine.action_resolution_prob(row["payer_id"], a, oracle_context(row))
            for a in ORACLE_ACTIONS
        }
        if traces[s].retrieval.recommended_action_id == max(oracle, key=oracle.get):
            p1_hits += 1
    p1_total = len(p1_states)
    p1_rate = p1_hits / max(1, p1_total)

    w_frame = frame_a.filter(pl.col("state_id").is_in(working["state_id"].to_list()))
    joined = w_frame.join(working.select("state_id", "action_raw"), on="state_id")
    mimic = HistGradientBoostingClassifier(random_state=MASTER_SEED).fit(
        joined.select(names).to_numpy(), joined["action_raw"].to_list()
    )
    mimic_hits = 0
    if p1_states:
        preds = mimic.predict(np.array([[feats[s][n] for n in names] for s in p1_states]))
        for s, pred in zip(p1_states, preds, strict=True):
            row = rows[s]
            oracle = {
                a: engine.action_resolution_prob(row["payer_id"], a, oracle_context(row))
                for a in ORACLE_ACTIONS
            }
            if pred == max(oracle, key=oracle.get):
                mimic_hits += 1
    imitation_rate = mimic_hits / max(1, p1_total)

    # ---- P5: OOD honesty on fresh holdout cases ----------------------------
    silent_errors = sum(1 for s in holdout_ids if traces[s].status == "executed")
    p5_rate = silent_errors / max(1, len(holdout_ids))

    # ---- P6: near-ties on reserve + fresh cornerstone bundling states ------
    p6_pool = [
        s for s in (*reserve_ids, *p2p_ids)
        if (rows[s].get("carc") or "") == "CO-97"
        and rows[s]["clinic_doc_quality"] >= 0.6
        and rows[s]["payer_id"] == "cornerstone"
    ]
    flat_states, flagged = [], 0
    for s in p6_pool:
        row = rows[s]
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

    # ---- P7 coverage discipline + the NEW no-self-mistakes criterion -------
    p7_violations = [
        s for s, t in traces.items()
        if t.status == "executed" and t.final is not None
        and t.final.action_id == "bill_secondary" and not rows[s]["has_secondary"]
    ]
    self_mistakes = []
    for s, t in traces.items():
        rec = t.retrieval.recommended_action_id
        row = rows[s]
        if rec in SUBMITS:
            if (
                row["auth_required"] and row["auth_status"] != "on_file"
                and row["days_since_service"] <= row["retro_window_days"]
            ):
                self_mistakes.append(f"{s}: submitted into open retro window")
            if not row["cob_position_ok"]:
                self_mistakes.append(f"{s}: submitted with wrong COB order")

    # ---- P8 form fidelity on executed reserve artifact cases ----------------
    p8_checked = p8_bad = 0
    for s in reserve_ids:
        t = traces[s]
        if t.status != "executed" or t.final is None or t.final.action_id not in TOOLS:
            continue
        row = rows[s]
        bundle = bundle_for_state(world, engine, row, episode_of[row["episode_id"]])
        artifact = TOOLS[t.final.action_id](t.final.payload, bundle)
        bundle_action = {**bundle, "action_object": t.final.payload}
        p8_checked += 1
        for field in REQUIRED_FIELDS[artifact["artifact"]]:
            if not artifact["fields"].get(field):
                p8_bad += 1
        for entry in artifact["field_map"]:
            if artifact["artifact"] == "cms1500" and entry["field"] == "prior_auth_reference":
                continue
            if entry["value"] != independent_repull(bundle_action, entry):
                p8_bad += 1
        if pre_validate(bundle, t.final.action_id):
            p8_bad += 1

    # ---- recommendation accuracy -------------------------------------------
    rec_hits = rec_total = 0
    for s in reserve_ids:
        row = rows[s]
        carc = row.get("carc") or ""
        if not carc or carc == "CO-16":
            continue
        oracle = {
            a: engine.action_resolution_prob(row["payer_id"], a, oracle_context(row))
            for a in ORACLE_ACTIONS
        }
        best, best_p = max(oracle.items(), key=lambda kv: kv[1])
        if best_p <= 0.0:
            continue
        rec_total += 1
        if traces[s].retrieval.recommended_action_id == best:
            rec_hits += 1
    recommendation_accuracy = rec_hits / max(1, rec_total)

    tokenizer = json.loads(
        (ROOT / "runs" / "manifests" / "tokenizer_round1.json").read_text(encoding="utf-8")
    )
    feature_names_set = {s["name"] for s in tokenizer["features"]}
    p2_clean = not (
        {"days_to_payment", "paid_amount_final", "total_touches", "n_process_mistakes"}
        & feature_names_set
    ) and not any(n.startswith("mistake_") for n in feature_names_set)

    trap_results = {
        "P1_success_not_imitation": p1_rate >= 0.85 and imitation_rate < p1_rate,
        "P2_no_leakage": p2_clean,
        "P3_determinism": double_generation_equal and double_tokenisation_equal,
        "P4_calibration": calibration_ece <= 0.05,
        "P5_ood_honesty": p5_rate <= 0.05,
        "P6_honest_near_ties": len(flat_states) > 0 and p6_agreement >= 0.7,
        "P7_coverage_discipline": not p7_violations,
        "P8_form_fidelity": p8_checked > 0 and p8_bad == 0,
        "NEW_no_self_mistakes": not self_mistakes,
    }
    status_counts = Counter(traces[s].status for s in reserve_ids)
    quarantine_manifest = json.loads(
        (ROOT / "runs" / "manifests" / "quarantine_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = TestReport(
        report_id="cardessa-final-v2-reserveB",
        evaluated_at=datetime.now(UTC),
        basis_manifest_sha256=sha256_file(ROOT / "basis" / "manifest.json"),
        quarantine_manifest_sha256=quarantine_manifest["content_sha256"],
        per_action_metrics=per_action_metrics,
        calibration_ece=calibration_ece,
        recommendation_accuracy=recommendation_accuracy,
        harness_stats={
            "n_reserve_states": len(reserve_ids),
            "status_counts": dict(status_counts),
            "m1_calls": provider.calls_to_provider,
        },
        trap_results=trap_results,
        tainted=False,
    )
    detail = {
        "eval_side": "reserve_b (quarantine A spent by v0.1)",
        "hard_gate_tranche_hash_matched_pin": True,
        "world_content_sha256": world.content_sha256(),
        "corpus_tranche0_sha256": first.content_sha256(),
        "P1": {"rate": p1_rate, "n_moments": p1_total, "hits": p1_hits,
               "imitation_rate": imitation_rate},
        "P4": {"ece": calibration_ece},
        "P5": {"fresh_holdout_states": len(holdout_ids), "silent_errors": silent_errors},
        "P6": {"pool": len(p6_pool), "flat_states": len(flat_states),
               "flagged_ambiguous": flagged, "agreement": p6_agreement},
        "P7": {"violations": p7_violations},
        "P8": {"artifacts_checked": p8_checked, "failures": p8_bad},
        "self_mistakes": self_mistakes[:10],
        "recommendation_accuracy_n": rec_total,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"test_report": json.loads(report.model_dump_json()),
                    "detail": detail}, indent=2),
        encoding="utf-8", newline="\n",
    )
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(
        json.dumps({"ran_at": report.evaluated_at.isoformat(),
                    "report_sha256": sha256_file(REPORT)}),
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({"trap_results": trap_results,
                      "P1_rate": round(p1_rate, 4),
                      "calibration_ece": round(calibration_ece, 4),
                      "recommendation_accuracy": round(recommendation_accuracy, 4)}))
    print("v0.2 TestReport written EXACTLY ONCE: runs/reports/test_report_v2.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
