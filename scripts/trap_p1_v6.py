"""Recompute the planted-trap decision rate (P1) under the v0.6 kernel decision
field, on the same tranche-96 exam moments the v0.4 report used. Analysis only;
tranche 96 is not sealed data. Compares route_case's top action to the engine's
best action at each trap moment."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for e in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")): sys.path.insert(0, e)
import polars as pl
from cardessa import MASTER_SEED
from cardessa.harness import BasisRuntime, route_case
from cardessa.features_kept import build_kept_features
from cardessa.textgen import CachedTextGenerator, VllmProvider
from ui.server import generate_demo_cases
from final_exam_v4 import EXAM_TRANCHE, EXAM_ALLOCATION, ORACLE_ACTIONS, oracle_context

text = CachedTextGenerator(provider=VllmProvider(), cache_dir=ROOT / "data" / "text_cache")
episodes, states, world, engine = generate_demo_cases(ROOT, MASTER_SEED, text, n_episodes=500,
                                                       tranche_no=EXAM_TRANCHE, allocation=EXAM_ALLOCATION)
rt = BasisRuntime.load(ROOT); print("decision_field:", rt.decision_field)
working = pl.read_parquet(ROOT / "data/working/cardessa_sim/states.parquet")
frame, _, _, _ = build_kept_features(ROOT, pl.concat([working, states.select(working.columns)], how="vertical_relaxed"))
names = rt.feature_names
rows = {r["state_id"]: r for r in states.to_dicts()}
feats = {r["state_id"]: {n: r[n] for n in names} for r in frame.filter(pl.col("state_id").is_in(list(rows))).to_dicts()}
exam_ids = sorted(rows)
family = {rows[s]["episode_id"] for s in exam_ids if rows[s]["touch_seq"] == 0 and rows[s]["cpt"] == "93229"
          and rows[s]["payer_id"] in ("meridian", "silverbridge") and rows[s]["auth_required"] and rows[s]["auth_status"] == "missing"}
p1 = []
for ep in sorted(family):
    es = sorted((s for s in exam_ids if rows[s]["episode_id"] == ep), key=lambda s: rows[s]["touch_seq"])
    if rows[es[0]]["payer_id"] == "meridian": p1.append(es[0])
    else:
        d = next((s for s in es if rows[s].get("carc") == "CO-197"), None)
        if d is not None: p1.append(d)
hits = 0; recs = {}
for s in p1:
    scores, known, forced, _ = route_case(rt, rows[s], feats[s])
    rec = forced or (max(scores, key=scores.get) if scores else None)
    best = max(ORACLE_ACTIONS, key=lambda a: engine.action_resolution_prob(rows[s]["payer_id"], a, oracle_context(rows[s])))
    hits += (rec == best); recs[(rows[s]["payer_id"], rec == best)] = recs.get((rows[s]["payer_id"], rec == best), 0) + 1
print(json.dumps({"P1_v6_rate": round(hits / max(1, len(p1)), 4), "hits": hits, "n_moments": len(p1), "by_payer": {f"{k[0]}:{'hit' if k[1] else 'miss'}": v for k, v in recs.items()}}))
