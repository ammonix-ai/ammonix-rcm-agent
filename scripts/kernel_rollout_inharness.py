"""V6-M3: rollout re-check through the real in-harness kernel path.

Same three-arm setup as scripts/kernel_ether_rollout.py (300 fresh tranche-92
episodes, identical dice, persona escalation), but the kernel arm is now the
production path: BasisRuntime loaded with the decision-field switch on and
route_case unchanged. Analysis only; no sealed data; no LLM.
"""
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")
import json, sys
from pathlib import Path
WT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("AMMONIX_BASIS_ROOT",
    WT.parents[1] / "Ammonix_Generic" / "Cardessa_Admin_factory"))
if not DATA_ROOT.is_dir():
    DATA_ROOT = WT.parents[2] / "Ammonix_Generic" / "Cardessa_Admin_factory"
for e in (str(WT), str(WT / "ammonix_core"), str(WT / "scripts")):
    sys.path.insert(0, e)
import polars as pl
import policy_rollout as PR
from cardessa import MASTER_SEED
from cardessa.expansion import expansion_studies, extend_world
from cardessa.harness import BasisRuntime
from cardessa.livecases import training_world

def main() -> int:
    PR.ROOT = DATA_ROOT  # policy_rollout reads working data relative to ROOT
    world_ext, engine = training_world(DATA_ROOT, MASTER_SEED)
    start = world_ext.studies.height
    studies = expansion_studies(world_ext, MASTER_SEED, PR.ROLLOUT_TRANCHE,
                                {"retro_auth": 60, "p2p": 30, "cob": 40, None: 170}, start)
    world = extend_world(world_ext, studies)
    indices = list(range(start, start + PR.N_EPISODES))
    working = pl.read_parquet(DATA_ROOT / "data/working/cardessa_sim/states.parquet")
    os.environ.pop("AMMONIX_DECISION_FIELD", None)
    rt_model = BasisRuntime.load(DATA_ROOT)
    os.environ["AMMONIX_DECISION_FIELD"] = "kernel"
    rt_kernel = BasisRuntime.load(DATA_ROOT)
    os.environ.pop("AMMONIX_DECISION_FIELD", None)
    assert rt_model.decision_field == "model" and rt_kernel.decision_field == "kernel"
    system = PR.play("system", world, engine, rt_model, working, indices)
    kernel = PR.play("system", world, engine, rt_kernel, working, indices)
    personas = PR.play("personas", world, engine, rt_model, working, indices)
    def dollars(res):
        return round(sum(r["collected_payer"] + r["collected_patient"] for r in res.values()), 2)
    report = {
        "analysis": "V6-M3 in-harness kernel rollout (analysis only)",
        "n_episodes": PR.N_EPISODES, "tranche": PR.ROLLOUT_TRANCHE,
        "system": PR.summary(system), "kernel": PR.summary(kernel), "personas": PR.summary(personas),
        "dollars": {"system": dollars(system), "kernel": dollars(kernel), "personas": dollars(personas)},
        "escalated_touches": {"system": sum(r["escalated_touches"] for r in system.values()),
                              "kernel": sum(r["escalated_touches"] for r in kernel.values())},
        "divergent_episodes": sum(1 for e in system if system[e]["actions"] != kernel[e]["actions"]),
    }
    (WT / "runs/reports/kernel_rollout_inharness.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"loop": "V6-M3", "verdict": True,
                      "dollars_system": report["dollars"]["system"],
                      "dollars_kernel": report["dollars"]["kernel"],
                      "dollars_personas": report["dollars"]["personas"],
                      "success_system": report["system"]["success_rate_v2"],
                      "success_kernel": report["kernel"]["success_rate_v2"],
                      "mistakes_system": report["system"]["episodes_with_mistakes"],
                      "mistakes_kernel": report["kernel"]["episodes_with_mistakes"],
                      "divergent": report["divergent_episodes"]}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
