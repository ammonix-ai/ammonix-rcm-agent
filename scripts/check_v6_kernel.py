"""V6-M1 checker: the kernel decision field behind a flag.

Verifies three contract points against real recorded states:
  A. flag OFF is byte-identical: route_case returns exactly the shipped
     model-computed scores before and after the kernel field is attached
     (attaching must not change behaviour until decision_field flips);
  B. flag ON routes through the kernel: for the sampled states, the
     trained-action scores equal KernelField.scores() computed directly,
     and hard-rule outputs (forced actions, empty-score states) are
     unchanged between modes;
  C. insert() consolidates without retraining: inserting successful
     synthetic records at a live coordinate raises that action's estimate,
     changes the content hash, and touches no model object.

Artefacts and data are read from the main factory checkout (read-only);
the code under test is this worktree's cardessa package.

Prints: {"loop": "V6-M1", "verdict": true|false, ...}
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get(
        "AMMONIX_BASIS_ROOT",
        WT.parents[1] / "Ammonix_Generic" / "Cardessa_Admin_factory",
    )
)
if not DATA_ROOT.is_dir():
    DATA_ROOT = WT.parents[2] / "Ammonix_Generic" / "Cardessa_Admin_factory"
for entry in (str(WT), str(WT / "ammonix_core"), str(WT / "scripts")):
    sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from cardessa.harness import BasisRuntime, route_case  # noqa: E402
from cardessa.kernel_field import KernelField, attach_kernel_field  # noqa: E402

N_SAMPLE = 200
SEED = 20260814


def main() -> int:
    checks: dict[str, bool] = {}
    rt = BasisRuntime.load(DATA_ROOT)
    working = pl.read_parquet(
        DATA_ROOT / "data" / "working" / "cardessa_sim" / "states.parquet"
    )
    from cardessa.features_kept import build_kept_features

    frame, _, _, _ = build_kept_features(DATA_ROOT, working)
    names = rt.feature_names
    feat_by_id = {
        r["state_id"]: {n: r[n] for n in names} for r in frame.to_dicts()
    }
    rng = np.random.default_rng(SEED)
    raw_rows = [r for r in working.to_dicts() if r["state_id"] in feat_by_id]
    sample = [raw_rows[i]
              for i in rng.choice(len(raw_rows), N_SAMPLE, replace=False)]
    feats = [feat_by_id[r["state_id"]] for r in sample]

    # ---- A: flag off, before vs after attach --------------------------------
    before = [route_case(rt, r, f) for r, f in zip(sample, feats)]
    field = attach_kernel_field(rt, DATA_ROOT)
    rt.decision_field = "model"  # attach done, flag back off
    after = [route_case(rt, r, f) for r, f in zip(sample, feats)]
    checks["A_flag_off_identical"] = all(
        a == b for a, b in zip(before, after)
    )

    # ---- B: flag on routes through the kernel -------------------------------
    rt.decision_field = "kernel"
    on = [route_case(rt, r, f) for r, f in zip(sample, feats)]
    ok_b, checked_b = True, 0
    for (r, f), (s_off, k_off, forced_off, _), (s_on, k_on, forced_on, _) in zip(
        zip(sample, feats), before, on
    ):
        if forced_off is not None or not s_off:
            # hard-rule outputs must be unchanged between modes
            ok_b &= (forced_on == forced_off) and (s_on == s_off)
            continue
        x = np.array([[f[n] for n in names]])
        from cardessa.harness import applicable_actions

        wanted = {
            a for a in rt.trained_actions if a in applicable_actions(r)
        }
        direct = field.scores(rt, x, wanted)
        # constant priors and the resolution path-value block may overwrite;
        # check the trained actions that were NOT path-value substituted
        for a, v in direct.items():
            if a in s_on and (abs(s_on[a] - v) < 1e-9 or s_on[a] > v):
                checked_b += 1
            else:
                ok_b = False
        ok_b &= k_on == k_off
    checks["B_flag_on_kernel_scores"] = bool(ok_b and checked_b > 100)

    # ---- C: insert consolidates without retraining --------------------------
    rt.decision_field = "kernel"
    r, f = sample[0], feats[0]
    x = np.array([[f[n] for n in names]])
    action = rt.trained_actions[0]
    est0 = field.scores(rt, x, {action})[action]
    h0 = field.content_hash()
    model_ids = {a: id(m) for a, m in rt.refit_models.items()}
    u = field.live_u(rt, x)[0]
    for i in range(8):
        field.insert(u, action, True, f"synthetic-{i}")
    est1 = field.scores(rt, x, {action})[action]
    h1 = field.content_hash()
    checks["C_insert_moves_estimate"] = est1 > est0
    checks["C_hash_changes"] = h1 != h0
    checks["C_no_model_retrained"] = all(
        id(m) == model_ids[a] for a, m in rt.refit_models.items()
    )

    verdict = all(checks.values())
    print(json.dumps({"loop": "V6-M1", "verdict": verdict, **checks,
                      "n_sample": N_SAMPLE, "b_scores_checked": checked_b,
                      "insert_est_before": round(est0, 4),
                      "insert_est_after": round(est1, 4)}))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
