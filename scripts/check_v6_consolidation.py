"""V6-M2 checker: consolidation policy + runtime switch.

  A. temporal firewall: consolidate() refuses a non-terminal or
     non-adjudicated record;
  B. snapshot round-trip: snapshot() then load_snapshot() restores a field
     whose content_hash equals the manifest entry, and decisions match;
  C. recency window: with MAX_RECORDS forced small, the oldest consolidated
     records age out first and basis records (ordinal 0) survive;
  D. runtime switch: unset -> "kernel" (the approved v0.6 default, field
     attached); AMMONIX_DECISION_FIELD=model -> rollback with no field.

Prints: {"loop": "V6-M2", "verdict": true|false, ...}
"""
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")
import json, sys, tempfile
from pathlib import Path
WT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("AMMONIX_BASIS_ROOT",
    WT.parents[1] / "Ammonix_Generic" / "Cardessa_Admin_factory"))
if not DATA_ROOT.is_dir():
    DATA_ROOT = WT.parents[2] / "Ammonix_Generic" / "Cardessa_Admin_factory"
for e in (str(WT), str(WT / "ammonix_core"), str(WT / "scripts")):
    sys.path.insert(0, e)
import numpy as np
from cardessa.harness import BasisRuntime
from cardessa import consolidation as C
from cardessa.kernel_field import KernelField, attach_kernel_field

def main() -> int:
    checks = {}
    # v0.6 default is "kernel" (approved 2026-08-14, runs/APPROVALS.md);
    # "model" must remain reachable as the rollback via the env switch.
    os.environ.pop("AMMONIX_DECISION_FIELD", None)
    rt = BasisRuntime.load(DATA_ROOT)
    checks["D_default_is_kernel"] = (
        rt.decision_field == "kernel" and isinstance(rt.kernel_field, KernelField)
    )
    os.environ["AMMONIX_DECISION_FIELD"] = "model"
    rt_rb = BasisRuntime.load(DATA_ROOT)
    checks["D_rollback_to_model"] = (
        rt_rb.decision_field == "model" and rt_rb.kernel_field is None
    )
    os.environ.pop("AMMONIX_DECISION_FIELD", None)
    field = rt.kernel_field
    n0 = len(field.state_ids)
    log = C.ConsolidationLog(n0)
    u = field.U[0].copy()
    # A
    try:
        C.consolidate(field, log, u_row=u, action=field.actions[0], success=True,
                      state_id="live-x", terminal=False, adjudicated=False)
        checks["A_firewall_refuses"] = False
    except ValueError:
        checks["A_firewall_refuses"] = True
    ok = C.consolidate(field, log, u_row=u, action=field.actions[0], success=True,
                       state_id="adj-1", terminal=True, adjudicated=True)
    checks["A_accepts_adjudicated"] = ok == 1 and len(field.state_ids) == n0 + 1
    # B
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entry = C.snapshot(field, log, tmp, "v0.6-test")
        f2, l2 = C.load_snapshot(tmp, entry["content_hash"])
        checks["B_snapshot_roundtrip"] = (
            f2.content_hash() == entry["content_hash"] == field.content_hash()
            and entry["n_consolidated"] == 1
        )
        x = np.array([[0.0] * len(rt.feature_names)])
        # compare scores at a real coordinate: use field.U row via a stub live_u
        class _RT:  # minimal shim so scores() uses a fixed u
            pass
        want = {a for a in rt.trained_actions}
        u_live = field.U[5:6]
        s1 = {a: v for a, v in _scores_at(field, u_live, want).items()}
        s2 = {a: v for a, v in _scores_at(f2, u_live, want).items()}
        checks["B_decisions_match"] = all(abs(s1[a] - s2[a]) < 1e-12 for a in want)
    # C
    C.MAX_RECORDS = n0 + 2
    for i in range(4):
        C.consolidate(field, log, u_row=u, action=field.actions[0], success=False,
                      state_id=f"adj-{i+2}", terminal=True, adjudicated=True)
    checks["C_window_bounded"] = len(field.state_ids) == n0 + 2
    checks["C_basis_survives"] = int((log.ordinals == 0).sum()) == n0
    checks["C_oldest_aged_out"] = "adj-1" not in field.state_ids and "adj-5" in field.state_ids
    C.MAX_RECORDS = 50_000
    # D
    os.environ["AMMONIX_DECISION_FIELD"] = "kernel"
    rt2 = BasisRuntime.load(DATA_ROOT)
    checks["D_env_switch_attaches"] = rt2.decision_field == "kernel" and isinstance(rt2.kernel_field, KernelField)
    os.environ.pop("AMMONIX_DECISION_FIELD", None)
    verdict = all(checks.values())
    print(json.dumps({"loop": "V6-M2", "verdict": verdict, **checks}))
    return 0 if verdict else 1

def _scores_at(field, u, wanted):
    _, idx = field._nn.kneighbors(u)
    na, ns = field.actions[idx[0]], field.success[idx[0]]
    out = {}
    for a in wanted:
        take = np.where(na == a)[0][:200]
        out[a] = float(ns[take].sum() / (len(take) + 2)) if len(take) >= 3 else field.global_rate.get(a, 0.0)
    return out

if __name__ == "__main__":
    sys.exit(main())
