"""Smoke-test the WIRED production path: resolve_and_run with C2.

Replays a few tranche-99 episodes to reach states where the swarm is below the
floor and C2 fires, then calls the real resolve_and_run (route_case + gates + C2
+ skill routing + M1) and asserts it routes the C2 action and executes without
error. Confirms the deployed decision path actually uses C2.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTIC = Path(os.environ.get("AGENTIC_BASELINE_ROOT", ROOT / "agentic_baseline"))  # co-located
for e in (str(AGENTIC), str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts")):
    sys.path.insert(0, e)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from ammonix_core.schema import HarnessArtefacts, Skill  # noqa: E402

from cardessa import MASTER_SEED  # noqa: E402
from cardessa.corpus import CardessaEnvironment, state_tabular  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.harness import (  # noqa: E402
    CONFIDENCE_FLOOR, BasisRuntime, M1Provider, expected_values, route_case)
from cardessa.livecases import training_world  # noqa: E402
from cardessa.c2_fallback import FeatIndex, c2_action  # noqa: E402
from run_harness_eval import resolve_and_run, C2_CONFIG  # noqa: E402

LINEAGE = 5000


def main():
    rt = BasisRuntime.load(ROOT)
    harness = HarnessArtefacts.model_validate_json(
        (ROOT / "runs" / "manifests" / "harness.json").read_text(encoding="utf-8"))
    skills = [Skill.model_validate(s) for s in json.loads(
        (ROOT / "runs" / "manifests" / "skills.json").read_text(encoding="utf-8"))["skills"]]
    provider = M1Provider(cache_dir=ROOT / "data" / "m1_cache")
    idx = FeatIndex(str(ROOT / "basis" / "feat_index"))
    working = pl.read_parquet(ROOT / "data" / "working" / "cardessa_sim" / "states.parquet")
    print(f"vLLM: {provider.model_id}")

    we, engine = training_world(ROOT, MASTER_SEED)
    world = extend_world(we, expansion_studies(we, MASTER_SEED, 99, {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}, LINEAGE))
    names = rt.feature_names

    def feats_for(rows):
        snaps = pl.DataFrame([{c: r.get(c) for c in working.columns} for r in rows],
                             schema_overrides=working.schema)
        frame, _n, _r, _p = build_kept_features(ROOT, pl.concat([working, snaps], how="vertical_relaxed"))
        return {r["state_id"]: {n: r[n] for n in names}
                for r in frame.filter(pl.col("state_id").is_in(snaps["state_id"].to_list())).to_dicts()}

    env = CardessaEnvironment(world, engine, MASTER_SEED)
    tested = 0
    # replay episodes; at each touch, if C2 would fire, call resolve_and_run there
    for ep_index in range(LINEAGE, LINEAGE + 300):
        case = env.reset(ep_index)
        for _ in range(8):
            if case.terminal:
                break
            payer = engine.payers[case.payer_id]
            row = state_tabular(case, payer, 0.0)
            row.update({"episode_id": case.episode_id,
                        "state_id": f"{case.episode_id}-r{case.touch_seq}",
                        "touch_seq": case.touch_seq, "action_raw": "",
                        "clinical_indication_text": "", "payer_correspondence_text": ""})
            f = feats_for([row]).get(row["state_id"])
            if f is None:
                break
            scores, known, forced, _ = route_case(rt, row, f)
            fires = None
            if forced is None and known and scores:
                best = max(scores.values())
                if best < CONFIDENCE_FLOOR:
                    ev = expected_values(row, scores)
                    fires, _ = c2_action(idx, f, set(scores), ev,
                                         exclude_episode=row["episode_id"], **C2_CONFIG)
            if fires is not None:
                x_scaled = rt.scaler.transform(np.array([[f[n] for n in names]]))[0]
                trace = resolve_and_run(rt, skills, harness, provider, row, f, x_scaled)
                ok = (trace.status == "executed"
                      and trace.retrieval.recommended_action_id == fires)
                print(f"  {row['state_id']}: C2 picks {fires} | trace action "
                      f"{trace.retrieval.recommended_action_id} status {trace.status} "
                      f"-> {'OK' if ok else 'MISMATCH'}")
                tested += 1
                if not ok:
                    print("  SMOKE FAIL: wired path did not route the C2 action")
                    return 1
                if tested >= 3:
                    print(f"\nSMOKE PASS: resolve_and_run routed C2 on {tested} states, executed cleanly")
                    return 0
            # advance under the same C2 policy
            a = fires
            if a is None:
                a, _ = _adv(rt, row, f, idx, scores, known, forced)
            if a is None:
                break  # would escalate; stop this episode
            case = env.apply(case, a)
    print(f"\nSMOKE INCONCLUSIVE: only {tested} C2-firing states reached in 60 episodes")
    return 0 if tested else 3


def _adv(rt, row, f, idx, scores, known, forced):
    from cardessa.harness import close_out_choice
    if forced is not None:
        return forced, "forced"
    if not scores or not known:
        return None, "esc"
    ba, best = max(scores.items(), key=lambda kv: kv[1])
    if best >= CONFIDENCE_FLOOR:
        return ba, "argmax"
    ev = expected_values(row, scores)
    return close_out_choice(ev, scores), "close_out"


if __name__ == "__main__":
    sys.exit(main())
