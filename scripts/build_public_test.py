"""Build the public demo's data: the one public test, replayed move by move.

The public test is the registered three-way comparison on 900 fresh claims
(comparison_gpt6_tranche103 + comparison_gpt6_tranches104_106, draws 103-105):
the simulated billers, GPT-6 on its own (gpt6_tuned: decides every move and
writes its own paperwork) and the shipped Ammonix (writer ornith_ft). This
script replays each arm KEYLESS from the run caches through the registered
protocol (scripts/rollout_paperwork.py, unchanged) and records every move -
who made it, what the payer answered, the filed paperwork, its checks, the
model's reasoning and the tokens each step cost - for the claim pages.

Gates (the build fails rather than ship a number that differs):
  - every model call must hit a cache (no key, no server: a miss raises);
  - every claim's moves and money must equal the stored per-episode reports;
  - the pooled money must equal the published recount.

  .venv/Scripts/python.exe scripts/build_public_test.py
Writes data/ui_public_test.json.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "ammonix_core"), str(ROOT / "scripts"),
              str(ROOT / "agentic_baseline")):
    sys.path.insert(0, entry)

# a replay must never reach a provider
os.environ["OPENAI_API_KEY"] = "replay-must-not-call"
import cardessa.payer_reader as payer_reader  # noqa: E402


def _no_reader_key(_data_dir):
    raise RuntimeError("replay must not call the payer reviewer (cache miss)")


payer_reader._load_api_key = _no_reader_key

import polars as pl  # noqa: E402

import rollout_paperwork_qwen38  # noqa: E402,F401  (the registered runner's server table)
import rollout_paperwork as rp  # noqa: E402
from cardessa import MASTER_SEED  # noqa: E402
from cardessa.expansion import expansion_studies, extend_world  # noqa: E402
from cardessa.features_kept import build_kept_features  # noqa: E402
from cardessa.forms import TOOLS  # noqa: E402
from cardessa.harness import route_case  # noqa: E402
from cardessa.livecases import bundle_for_state, training_world  # noqa: E402
from cardessa.textgen import CachedTextGenerator  # noqa: E402
from ui.server import _before_facts, _step_note  # noqa: E402

rp.LLM_WORKERS = 1    # one call at a time: each step's token bill is exact
rp.WRITE_WORKERS = 1

DRAWS = {103: 1, 104: 2, 105: 3}   # registered draw -> public draw number
ARMS = (  # (public lane, arm, writer, stored report for the arm in each draw)
    ("human", "personas", "ornith_ft",
     {103: "episodes_personas_rollout_ammonix_gpt6_t103.json",
      104: "episodes_personas_rollout_ammonix_ornith_t104.json",
      105: "episodes_personas_rollout_ammonix_ornith_t105.json"}),
    ("llm", "gpt6_tuned", "gpt6",
     {t: f"episodes_gpt6_tuned_rollout_gpt6_tuned_t{t}.json" for t in DRAWS}),
    ("system", "ammonix", "ornith_ft",
     {t: f"episodes_ammonix_rollout_ammonix_ornith_t{t}.json" for t in DRAWS}),
)
OUT = ROOT / "data" / "ui_public_test.json"
TAG = "public-test-1"


# --------------------------------------------------------------------------
# recording hooks around the unchanged registered protocol
# --------------------------------------------------------------------------
REC: dict = {}
CTX: dict = {}


def _reset_rec():
    REC.clear()
    REC.update(steps={}, handed=set(), decided={}, written={}, share={}, detail={}, rows={})


class RecEnv(rp.CardessaEnvironment):
    def apply(self, case, action_raw, payload=None, records=None):
        before = _before_facts(case)
        rej0 = case.paperwork_rejections
        ep, touch = case.episode_id, case.touch_seq
        after = super().apply(case, action_raw, payload=payload, records=records)
        step = {"touch": touch, "action": action_raw,
                "note": _step_note(before, after, action_raw)}
        if after.paperwork_rejections > rej0:
            reason = after.paperwork_rejection_log[-1].split(":", 1)[-1]
            step["note"] = "paperwork rejected by the payer"
            step["rejection"] = reason
        REC["steps"].setdefault(ep, []).append(step)
        return after


rp.CardessaEnvironment = RecEnv

_clerk = rp.clerk_action


def clerk(env, case, payer):
    REC["handed"].add((case.episode_id, case.touch_seq))
    return _clerk(env, case, payer)


rp.clerk_action = clerk

_outcome = rp.outcome_v2


def outcome(env, case, pending):
    REC["share"][case.episode_id] = case.contractual_share
    return _outcome(env, case, pending)


rp.outcome_v2 = outcome

_make_baseline = rp.make_baseline


def make_baseline(arm):
    pol = _make_baseline(arm)
    inner = pol.decide

    def decide(row, legal):
        llm = pol.llm
        p0, c0 = llm.prompt_tokens, llm.completion_tokens
        d = inner(row, legal)
        REC["decided"][row["state_id"]] = {
            "action": d.action, "reasoning": (d.reasoning or "")[:1200],
            "tokens": {"prompt_tokens": llm.prompt_tokens - p0,
                       "completion_tokens": llm.completion_tokens - c0, "calls": 1}}
        return d

    pol.decide = decide
    return pol


rp.make_baseline = make_baseline


class RecDecider(rp.AmmonixDecider):
    """The shipped decision path, unchanged, plus what it saw: the
    calibrated scores and the facts behind each decision."""

    def decide(self, snap_rows):
        out = super().decide(snap_rows)
        snaps = pl.DataFrame([{c: r.get(c) for c in self.working.columns} for r in snap_rows],
                             schema_overrides=self.working.schema)
        frame, _, _, _ = build_kept_features(
            ROOT, pl.concat([self.working, snaps], how="vertical_relaxed"))
        names = self.rt.feature_names
        feats = {r["state_id"]: {n: r[n] for n in names}
                 for r in frame.filter(pl.col("state_id").is_in(snaps["state_id"].to_list())).to_dicts()}
        for row in snap_rows:
            scores, known, forced, _amb = route_case(self.rt, row, feats[row["state_id"]])
            REC["detail"][row["state_id"]] = {
                "scores": {a: round(v, 4) for a, v in sorted((scores or {}).items(), key=lambda kv: -kv[1])},
                "known_payer": bool(known), "forced": forced,
                "facts": {"carc": row["carc"] or None,
                          "days_since_service": row["days_since_service"],
                          "days_to_filing_deadline": row["days_to_filing_deadline"],
                          "auth_status": row["auth_status"], "balance": row["balance"]},
            }
            REC["decided"][row["state_id"]] = {"action": out[row["state_id"]]}
            # what Ammonix saw, kept so the demo can open each moment exactly
            REC["rows"][row["state_id"]] = {k: (v if not isinstance(v, float) or v == v else None)
                                            for k, v in row.items()}
        return out


rp.AmmonixDecider = RecDecider

_write = rp.write_paperwork


def write_paperwork(decider, writer, action, row):
    llm = writer.llm
    p0, c0 = llm.prompt_tokens, llm.completion_tokens
    payload, trace = _write(decider, writer, action, row)
    entry = {
        "status": trace.status,
        "checks": [{"n": it.n, "passed": it.check.passed, "failures": it.check.failures}
                   for it in trace.iterations],
        "tokens": {"prompt_tokens": llm.prompt_tokens - p0,
                   "completion_tokens": llm.completion_tokens - c0,
                   "calls": len(trace.iterations)},
    }
    if payload is not None:
        world, engine = CTX["world"], CTX["engine"]
        study = world.studies.row(int(row["episode_id"].removeprefix("ep-")), named=True)
        episode_row = {"patient_id": study["patient_id"], "clinic_id": study["clinic_id"],
                       "payer_id": row["payer_id"]}
        if action in TOOLS:
            art = TOOLS[action](payload, bundle_for_state(world, engine, row, episode_row))
            entry["paperwork"] = {"kind": art["artifact"], "field_map": art["field_map"],
                                  "gaps": art["gaps"]}
        else:
            entry["filed_payload"] = payload
    REC["written"][row["state_id"]] = entry
    return payload, trace


rp.write_paperwork = write_paperwork


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def lane_of(public, arm, ep, result, steps):
    out_steps, decide_t, write_t = [], [0, 0, 0], [0, 0, 0]
    by_self = {"human": "human", "llm": "llm", "system": "system"}[public]
    for st in steps:
        sid = f"{ep}-r{st['touch']}"
        s = dict(st)
        if public == "human":
            s["by"] = "human"
        elif (ep, st["touch"]) in REC["handed"]:
            s["by"] = "handed to human"
            dec = REC["decided"].get(sid) or {}
            wr = REC["written"].get(sid)
            if dec.get("action") and wr is not None:
                # it chose a move but its paperwork failed the checks twice;
                # the writing it tried still cost tokens, and counts
                s["handoff_reason"] = "paperwork failed its checks"
                s["checks"] = wr["checks"]
                s["write_status"] = wr["status"]
                s["write_tokens"] = wr["tokens"]
                for i, k in enumerate(("prompt_tokens", "completion_tokens", "calls")):
                    write_t[i] += wr["tokens"][k]
            else:
                s["handoff_reason"] = "not confident enough"
        elif sid not in REC["decided"]:
            s["by"] = "rule: step cap"  # the protocol writes off claims still open at the cap
        else:
            s["by"] = by_self
        if public == "llm" and sid in REC["decided"]:
            dec = REC["decided"][sid]
            if dec.get("reasoning"):
                s["reasoning"] = dec["reasoning"]
            s["tokens"] = dec["tokens"]
            for i, k in enumerate(("prompt_tokens", "completion_tokens", "calls")):
                decide_t[i] += dec["tokens"][k]
        if public == "system" and sid in REC["detail"]:
            s["detail"] = REC["detail"][sid]
        if s["by"] in ("llm", "system") and sid in REC["written"]:
            wr = REC["written"][sid]
            for k in ("paperwork", "filed_payload", "checks"):
                if k in wr:
                    s[k] = wr[k]
            s["write_status"] = wr["status"]
            s["write_tokens"] = wr["tokens"]
            for i, k in enumerate(("prompt_tokens", "completion_tokens", "calls")):
                write_t[i] += wr["tokens"][k]
        out_steps.append(s)
    share = REC["share"][ep]
    patient_valid = round(min(result["collected_patient"], share), 2)
    payer_valid = round(min(result["collected_payer"], result["allowed"] - patient_valid), 2)
    lane = {
        "steps": out_steps, "resolution": result["resolution"],
        "collected": round(payer_valid + patient_valid, 2),
        "payer_collected": round(result["collected_payer"], 2), "payer_counted": payer_valid,
        "patient_collected": round(result["collected_patient"], 2), "patient_counted": patient_valid,
        "touches": result["touches"], "mistakes": result["mistakes"],
        "success": bool(result["success"]),
        "autonomous": public != "human" and all(x["by"] == by_self for x in out_steps),
        "paperwork_rejected": result["paperwork_rejected"],
    }
    tok = {}
    if decide_t[2]:
        tok["decide"] = dict(zip(("prompt_tokens", "completion_tokens", "calls"), decide_t))
    if write_t[2]:
        tok["write"] = dict(zip(("prompt_tokens", "completion_tokens", "calls"), write_t))
    if tok:
        lane["tokens"] = tok
    return lane


def verdict(a, b):
    """Money decides; equal money is a win only on fewer process mistakes."""
    delta = round(a["collected"] - b["collected"], 2)
    if abs(delta) > 0.005:
        return ("a" if delta > 0 else "b"), delta
    if a["mistakes"] != b["mistakes"]:
        return ("a" if a["mistakes"] < b["mistakes"] else "b"), delta
    return "s", delta


def main() -> int:
    world_ext, engine = training_world(ROOT, MASTER_SEED)
    start = world_ext.studies.height
    text = CachedTextGenerator(provider=rp._LazyQwen(), cache_dir=ROOT / "data" / "text_cache")
    reader = rp.PayerReader(cache_dir=ROOT / "data" / "payer_reader")
    claims: dict = {}
    mismatches: list = []
    for tranche, draw in DRAWS.items():
        world = extend_world(world_ext, expansion_studies(world_ext, MASTER_SEED, tranche,
                                                          rp.ALLOCATION, start))
        CTX.update(world=world, engine=engine)
        limit = int(os.environ.get("PUBLIC_TEST_LIMIT", rp.N_EPISODES))  # smoke runs only
        indices = list(range(start, start + limit))
        for public, arm, writer, stored_name in ARMS:
            _reset_rec()
            results, _stats, llm = rp.play([arm], writer, world, engine, indices, text,
                                           reader, swap=False)
            res = results[arm]
            stored = json.loads((ROOT / "runs" / "reports" / stored_name[tranche]).read_text(
                encoding="utf-8"))
            for ep, r in res.items():
                s = stored[ep]
                if (abs(s["collected_payer"] - r["collected_payer"]) > .005
                        or abs(s["collected_patient"] - r["collected_patient"]) > .005
                        or s["actions"] != r["actions"]):
                    mismatches.append((tranche, arm, ep))
                pid = f"{draw}-{ep.removeprefix('ep-')}"
                if pid not in claims:
                    case = RecEnv(world, engine, MASTER_SEED).reset(int(ep.removeprefix("ep-")))
                    payer = engine.payers[case.payer_id]
                    claims[pid] = {"meta": {
                        "draw": draw, "episode": ep, "payer_id": case.payer_id,
                        "payer_name": payer.name, "cpt": case.study["cpt"],
                        "allowed": round(case.allowed, 2), "persona": case.persona,
                        "carc": case.carc}}
                claims[pid][public] = lane_of(public, arm, ep, r, REC["steps"][ep])
                if public == "system":
                    claims[pid]["system_rows"] = {
                        str(st["touch"]): REC["rows"][f"{ep}-r{st['touch']}"]
                        for st in REC["steps"][ep] if f"{ep}-r{st['touch']}" in REC["rows"]}
            if arm != "personas":
                lanes = [claims[f"{draw}-{ep.removeprefix('ep-')}"][public] for ep in res]
                got = sum(sum(v.get(k, 0) for part in (ln.get("tokens") or {}).values()
                              for k in ("prompt_tokens", "completion_tokens") for v in [part])
                          for ln in lanes)
                report = json.loads((ROOT / "runs" / "reports" / (
                    f"rollout_gpt6_tuned_t{tranche}.json" if arm == "gpt6_tuned"
                    else f"rollout_ammonix_ornith_t{tranche}.json")).read_text(encoding="utf-8"))
                want = sum(report["llm"]["writer"][k] for k in ("prompt_tokens", "completion_tokens"))
                if arm == "gpt6_tuned":
                    want += sum(report["policies"]["gpt6_tuned"]["llm"][k]
                                for k in ("prompt_tokens", "completion_tokens"))
                if got != want:
                    mismatches.append((tranche, arm, f"tokens {got} != report {want}"))
            print(f"draw {draw} ({tranche}) {arm:<11} replayed {len(res)} claims; "
                  f"provider calls writer {llm.get('writer', {}).get('calls_to_provider', 0)}, "
                  f"reader {llm.get('reader_calls', 0)}", flush=True)
    if mismatches:
        print(f"FAIL: {len(mismatches)} claims differ from the stored reports: {mismatches[:5]}")
        return 1

    for pid, c in claims.items():
        v, d = verdict(c["system"], c["human"])
        c["verdict"] = {"a": "ammonix", "b": "human", "s": "same"}[v]
        c["delta"] = d
        v2, d2 = verdict(c["system"], c["llm"])
        c["vs_llm"] = {"a": "ammonix", "b": "llm", "s": "same"}[v2]
        c["delta_llm"] = d2

    def total(lane):
        return round(sum(c[lane]["collected"] for c in claims.values()), 2)

    def tok_total(lane, part):
        t = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        for c in claims.values():
            for k in t:
                t[k] += ((c[lane].get("tokens") or {}).get(part) or {}).get(k, 0)
        return t

    published = json.loads((Path(__file__).resolve().parents[3] / "Marketing" / "rcm-post"
                            / "analysis" / "draws_104_106" / "NUMBERS.json").read_text(encoding="utf-8")
                           ) if (Path(__file__).resolve().parents[3] / "Marketing").is_dir() else None
    sb = {
        "claims": len(claims),
        "system_collected": total("system"), "human_collected": total("human"),
        "llm_collected": total("llm"),
        "ammonix_wins": sum(1 for c in claims.values() if c["verdict"] == "ammonix"),
        "human_wins": sum(1 for c in claims.values() if c["verdict"] == "human"),
        "same": sum(1 for c in claims.values() if c["verdict"] == "same"),
        "ammonix_vs_llm": {k: sum(1 for c in claims.values() if c["vs_llm"] == v)
                           for k, v in (("ammonix", "ammonix"), ("llm", "llm"), ("same", "same"))},
        "system_mistake_claims": sum(1 for c in claims.values() if c["system"]["mistakes"] > 0),
        "human_mistake_claims": sum(1 for c in claims.values() if c["human"]["mistakes"] > 0),
        "llm_mistake_claims": sum(1 for c in claims.values() if c["llm"]["mistakes"] > 0),
        "system_autonomous": sum(1 for c in claims.values() if c["system"]["autonomous"]),
        "resolved": {lane: round(sum(c[lane]["success"] for c in claims.values()) / len(claims), 4)
                     for lane in ("system", "llm", "human")},
        "llm_tokens": {"decide": tok_total("llm", "decide"), "write": tok_total("llm", "write"),
                       "claims": len(claims)},
        "system_tokens": {"write": tok_total("system", "write"), "claims": len(claims),
                          "writer_model": "local/Ornith-1.5-9B-rcm-r2"},
        "llm_model_label": "GPT-6",
        "note": ("the one public test: 900 fresh claims in three registered draws; dollars = "
                 "insurer + patient, the patient counted up to the contractual share, the "
                 "insurer up to allowed minus that"),
    }
    if os.environ.get("PUBLIC_TEST_LIMIT"):
        print(json.dumps(claims[next(iter(claims))], indent=1)[:6000])
        print("smoke run: nothing written")
        return 0
    if published:
        want = published["pooled"]["money"]
        got = {"billers": sb["human_collected"], "gpt6_tuned": sb["llm_collected"],
               "ammonix_shipped_writer": sb["system_collected"]}
        if any(abs(want[k] - got[k]) > 0.005 for k in want):
            print(f"FAIL: pooled money {got} differs from the published recount {want}")
            return 1
        print("pooled money equals the published recount", flush=True)
    OUT.write_text(json.dumps({"tag": TAG, "scoreboard": sb, "claims": claims}),
                   encoding="utf-8", newline="\n")
    print(f"written {OUT} ({len(claims)} claims): Ammonix ${sb['system_collected']:,.2f} | "
          f"GPT-6 ${sb['llm_collected']:,.2f} | billers ${sb['human_collected']:,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
