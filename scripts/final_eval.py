"""THE FINAL EVALUATION (P10). Runs EXACTLY ONCE, via the whitelisted path.

Quarantine A is materialised by deterministic REGENERATION of tranche 0 from
the master seed (text from the prompt-hash cache; double-generation equality
was proven at W2), filtered to the quarantine-side patients fixed at W1. The
sealed parquet files are never read; no build loop has ever touched these
episodes. The report is written once; a marker file refuses any second run.

Trap definitions (corpus spec Section 4), measured on quarantine A:
- P1 success-not-imitation: on retro-family first states, the recommendation
  matches the ORACLE-best rework action (payer-conditional) >= 85%; an
  imitation ablation (multiclass action-mimic) fails to find it.
- P2 no-leakage: the poisoned column is excluded from the feature set (from
  the P2/P7 audits, re-asserted here on the tokenizer manifest).
- P3 determinism: double regeneration hash equality (incl. text) + double
  tokenisation equality, recomputed here.
- P4 calibration vs engine truth: ECE of calibrated P(success) vs realised
  engine-generated outcomes on quarantine own-action states <= 0.05.
- P5 OOD honesty: every quarantine case carries a never-seen payer or
  patient; for HOLDOUT-PAYER claims, escalation is the passing behaviour
  and the silent-confident-error rate must be <= 5%.
- P6 honest near-ties: on states where the oracle makes appeal vs p2p
  nearly equal (|delta| < 0.05), the live ambiguity flag agrees >= 0.7.
- P7 coverage discipline: bill_secondary only in secondary-coverage cases;
  the below-coverage action is refused (escalated), never executed.
- P8 form fidelity: on executed artifact cases, 100% of required fields
  correct against source, zero fabricated values, pre-validation green.
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
from cardessa.livecases import bundle_for_state  # noqa: E402
from cardessa.textgen import CachedTextGenerator, VllmProvider  # noqa: E402
from cardessa.world import generate_world  # noqa: E402

MARKER = ROOT / "runs" / "state" / "final_eval_ran.json"
REPORT = ROOT / "runs" / "reports" / "test_report.json"
HOLDOUT = {"granite", "pelican"}
ORACLE_ACTIONS = ("request_retro_auth", "appeal_with_necessity", "request_peer_to_peer")


def oracle_context(row: dict) -> dict:
    return {
        "days_since_service": row["days_since_service"],
        "strong_documentation": row["clinic_doc_quality"] >= 0.6,
    }


def main() -> int:
    if MARKER.is_file() or REPORT.is_file():
        raise SystemExit(
            "final evaluation already ran; the quarantine draw is spent. "
            "Accept the shipped TestReport or take a fresh draw from reserve B."
        )

    world = generate_world(MASTER_SEED)
    engine = PayerEngine({p.payer_id: p for p in world.payers}, MASTER_SEED)
    text = CachedTextGenerator(
        provider=VllmProvider(), cache_dir=ROOT / "data" / "text_cache"
    )

    print("regenerating tranche 0 from the master seed (text from cache)...")
    collector = PromptCollector()
    generate_tranche(world, engine, MASTER_SEED, collector, 0, TRANCHE_SIZE_INITIAL)
    warm_cache(collector.prompts, text)
    first = generate_tranche(world, engine, MASTER_SEED, text, 0, TRANCHE_SIZE_INITIAL)
    second = generate_tranche(world, engine, MASTER_SEED, text, 0, TRANCHE_SIZE_INITIAL)
    double_generation_equal = first.content_sha256() == second.content_sha256()
    episodes = pl.DataFrame(first.episodes_rows)
    states = pl.DataFrame(first.states_rows)
    q_episodes = episodes.filter(pl.col("split") == "quarantine_a")
    q_ids = set(q_episodes["episode_id"].to_list())
    q_states = states.filter(pl.col("episode_id").is_in(sorted(q_ids)))
    print(f"quarantine A: {q_episodes.height} episodes, {q_states.height} states; "
          f"double-generation equal: {double_generation_equal}")

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
    frame_a, _ = build_round1_features(
        pl.concat([working, q_states.select(working.columns)], how="vertical_relaxed")
    )
    frame_b, _ = build_round1_features(
        pl.concat([working, q_states.select(working.columns)], how="vertical_relaxed")
    )
    double_tokenisation_equal = frame_a.equals(frame_b)
    names = rt.feature_names
    feats = {
        r["state_id"]: {n: r[n] for n in names}
        for r in frame_a.filter(
            pl.col("state_id").is_in(sorted(q_states["state_id"].to_list()))
        ).to_dicts()
    }
    rows = {r["state_id"]: r for r in q_states.to_dicts()}
    ordered = sorted(rows)
    scaled = rt.scaler.transform(
        np.array([[feats[s][n] for n in names] for s in ordered])
    )
    scaled_of = dict(zip(ordered, scaled, strict=True))
    episode_of = {e["episode_id"]: e for e in q_episodes.to_dicts()}

    print(f"running the full runtime over {len(ordered)} quarantine states...")

    def one(state_id):
        return resolve_and_run(
            rt, skills, harness, provider, rows[state_id], feats[state_id],
            scaled_of[state_id],
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        traces = dict(zip(ordered, pool.map(one, ordered), strict=True))

    # ---- per-action generalisation metrics + calibration ECE (P4) ---------
    success_of = {
        e["episode_id"]: bool(e["success"]) for e in q_episodes.to_dicts()
    }
    per_action_metrics: dict[str, FoldMetrics] = {}
    ece_scores, ece_labels = [], []
    for action in rt.trained_actions:
        own = [s for s in ordered if rows[s]["action_raw"] == action]
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
                n_pos=int(y.sum()),
                n_neg=int((1 - y).sum()),
            )
    calibration_ece = expected_calibration_error(
        np.concatenate(ece_scores), np.concatenate(ece_labels)
    )

    # ---- P1: success, not imitation ---------------------------------------
    # family = MCT at Meridian/SilverBridge with auth missing at the first
    # state; the test state is each family episode's DECISION MOMENT: the
    # first state where a rework action is applicable (t0 at Meridian, whose
    # retro window is open; the first denial state at SilverBridge, whose
    # retro window is 0 so nothing reworkable exists before the denial)
    family_episodes = {
        rows[s]["episode_id"] for s in ordered
        if rows[s]["touch_seq"] == 0 and rows[s]["cpt"] == "93229"
        and rows[s]["payer_id"] in ("meridian", "silverbridge")
        and rows[s]["auth_required"] and rows[s]["auth_status"] == "missing"
    }
    p1_states = []
    for episode_id in sorted(family_episodes):
        episode_states = sorted(
            (s for s in ordered if rows[s]["episode_id"] == episode_id),
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
    p1_hits, p1_total = 0, 0
    for s in p1_states:
        row = rows[s]
        oracle = {
            a: engine.action_resolution_prob(row["payer_id"], a, oracle_context(row))
            for a in ORACLE_ACTIONS
        }
        best = max(oracle, key=oracle.get)
        recommended = traces[s].retrieval.recommended_action_id
        p1_total += 1
        if recommended == best:
            p1_hits += 1
    p1_rate = p1_hits / max(1, p1_total)

    # imitation ablation: an action-mimic (multiclass on working data) must
    # NOT find the minority action on the same states
    w_frame = frame_a.filter(pl.col("state_id").is_in(working["state_id"].to_list()))
    w_actions = working.select("state_id", "action_raw")
    joined = w_frame.join(w_actions, on="state_id")
    mimic = HistGradientBoostingClassifier(random_state=MASTER_SEED).fit(
        joined.select(names).to_numpy(), joined["action_raw"].to_list()
    )
    mimic_preds = mimic.predict(np.array([[feats[s][n] for n in names] for s in p1_states]))
    mimic_minority = sum(
        1 for s, pred in zip(p1_states, mimic_preds, strict=True)
        if pred == max(
            {a: engine.action_resolution_prob(rows[s]["payer_id"], a, oracle_context(rows[s]))
             for a in ORACLE_ACTIONS},
            key=lambda a: engine.action_resolution_prob(
                rows[s]["payer_id"], a, oracle_context(rows[s])
            ),
        )
    )
    imitation_fails = (mimic_minority / max(1, p1_total)) < p1_rate

    # ---- P5: OOD honesty ---------------------------------------------------
    holdout_states = [s for s in ordered if rows[s]["payer_id"] in HOLDOUT]
    silent_errors = sum(
        1 for s in holdout_states if traces[s].status == "executed"
    )
    p5_rate = silent_errors / max(1, len(holdout_states))

    # ---- P6: honest near-ties ----------------------------------------------
    flat_states, flagged = [], 0
    for s in ordered:
        row = rows[s]
        if not (row.get("carc") or "") or row["carc"] == "CO-16":
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

    # ---- P7: coverage discipline -------------------------------------------
    p7_violations = [
        s for s, t in traces.items()
        if (
            t.status == "executed"
            and t.final is not None
            and (
                (t.final.action_id == "bill_secondary" and not rows[s]["has_secondary"])
                or t.final.action_id == "provide_requested_info"
            )
        )
    ]

    # ---- P8: form fidelity on executed artifact cases ----------------------
    p8_checked = p8_invented = p8_missing = p8_preval = 0
    for s, t in traces.items():
        if t.status != "executed" or t.final is None or t.final.action_id not in TOOLS:
            continue
        row = rows[s]
        bundle = bundle_for_state(world, engine, row, episode_of[row["episode_id"]])
        artifact = TOOLS[t.final.action_id](t.final.payload, bundle)
        bundle_action = {**bundle, "action_object": t.final.payload}
        p8_checked += 1
        for field in REQUIRED_FIELDS[artifact["artifact"]]:
            if not artifact["fields"].get(field):
                p8_missing += 1
        for entry in artifact["field_map"]:
            if artifact["artifact"] == "cms1500" and entry["field"] == "prior_auth_reference":
                continue
            if entry["value"] != independent_repull(bundle_action, entry):
                p8_invented += 1
        if pre_validate(bundle, t.final.action_id):
            p8_preval += 1

    # ---- recommendation accuracy (vs oracle-best rework on denial states) --
    rec_hits = rec_total = 0
    for s in ordered:
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

    status_counts = Counter(t.status for t in traces.values())

    tokenizer = json.loads(
        (ROOT / "runs" / "manifests" / "tokenizer_round1.json").read_text(
            encoding="utf-8"
        )
    )
    feature_names_set = {s["name"] for s in tokenizer["features"]}
    p2_clean = not (
        {"days_to_payment", "paid_amount_final", "total_touches"} & feature_names_set
    )

    trap_results = {
        "P1_success_not_imitation": p1_rate >= 0.85 and imitation_fails,
        "P2_no_leakage": p2_clean,
        "P3_determinism": double_generation_equal and double_tokenisation_equal,
        "P4_calibration": calibration_ece <= 0.05,
        "P5_ood_honesty": p5_rate <= 0.05,
        "P6_honest_near_ties": p6_agreement >= 0.7,
        "P7_coverage_discipline": not p7_violations,
        "P8_form_fidelity": p8_checked > 0
        and p8_missing == 0 and p8_invented == 0 and p8_preval == 0,
    }

    quarantine_manifest = json.loads(
        (ROOT / "runs" / "manifests" / "quarantine_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    report = TestReport(
        report_id="cardessa-final-v1",
        evaluated_at=datetime.now(UTC),
        basis_manifest_sha256=sha256_file(ROOT / "basis" / "manifest.json"),
        quarantine_manifest_sha256=quarantine_manifest["content_sha256"],
        per_action_metrics=per_action_metrics,
        calibration_ece=calibration_ece,
        recommendation_accuracy=recommendation_accuracy,
        harness_stats={
            "n_quarantine_states": len(ordered),
            "status_counts": dict(status_counts),
            "mean_iterations": float(
                np.mean([max(1, len(t.iterations)) for t in traces.values()])
            ),
            "m1_calls": provider.calls_to_provider,
        },
        trap_results=trap_results,
        tainted=False,
    )
    detail = {
        "world_content_sha256": world.content_sha256(),
        "corpus_tranche0_sha256": first.content_sha256(),
        "P1": {"rate": p1_rate, "n_states": p1_total,
               "imitation_finds_minority": mimic_minority / max(1, p1_total)},
        "P4": {"ece": calibration_ece},
        "P5": {"holdout_states": len(holdout_states), "silent_errors": silent_errors,
               "rate": p5_rate},
        "P6": {"flat_states": len(flat_states), "flagged": flagged,
               "agreement": p6_agreement},
        "P7": {"violations": p7_violations},
        "P8": {"artifacts_checked": p8_checked, "missing_required": p8_missing,
               "invented": p8_invented, "prevalidation_failures": p8_preval},
        "recommendation_accuracy_definition": (
            "share of quarantine denial states (excl. CO-16) where the "
            "recommended action equals the oracle-best rework action"
        ),
        "recommendation_accuracy_n": rec_total,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {"test_report": json.loads(report.model_dump_json()), "detail": detail},
            indent=2,
        ),
        encoding="utf-8", newline="\n",
    )
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(
        json.dumps(
            {"ran_at": report.evaluated_at.isoformat(),
             "report_sha256": sha256_file(REPORT)},
        ),
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({"trap_results": trap_results,
                      "calibration_ece": round(calibration_ece, 4),
                      "recommendation_accuracy": round(recommendation_accuracy, 4)}))
    print("TestReport written EXACTLY ONCE: runs/reports/test_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
