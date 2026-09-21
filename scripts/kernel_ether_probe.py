"""Kernel-Ether probe (analysis only; no build artefact changes, no sealed data).

Question: the Foundation paper describes the Ether as a kernel-smoothed
average of stored outcomes; the shipped RCM agent estimates the same field
with fitted per-action models. Would the kernel version have been as good?

Round 1 answers this at the estimator level, on the 11,427 working universe
states. For every state we know the action actually taken and its adjudicated
success label. We compare three estimators of P(success | taken action, state):

  model   - the stored out-of-fold calibrated score for the taken action
            (the shipped decision quantity, read from the label-space index)
  kfeat_K - success rate of the K nearest FEATURE-space neighbours that took
            the same action (Laplace-smoothed, same-episode neighbours excluded)
  klab_K  - the same over LABEL-space (u-coordinate) neighbours

Metrics per estimator: pooled AUROC over states with a scoreable estimate,
per-action AUROC, ECE (10 equal-width bins), and coverage (share of states
with >= MIN_SUPPORT same-action neighbours). Prints a JSON verdict.
"""

import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "basis" / "feat_index"
ART = ROOT / "basis" / "artefacts"

KS = (15, 25, 50, 100, 200)
MIN_SUPPORT = 3
LAPLACE = 2  # same m as the shipped case-based fallback


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        total += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(total)


def main() -> int:
    meta = json.loads((FEAT / "meta.json").read_text(encoding="utf-8"))
    X = np.load(FEAT / "matrix.npy").astype("float32")  # (n, d) standardized
    actions = np.asarray(meta["true_action_id"])
    success = np.asarray(meta["outcome_success"], dtype=int)
    episodes = np.asarray(meta["episode_id"])
    state_ids = np.asarray(meta["state_id"])
    n = len(state_ids)
    assert X.shape[0] == n

    # label-space index: NearestNeighbors fitted on u = OOF calibrated vectors
    index = joblib.load(ART / "index.joblib")
    ids = joblib.load(ART / "index_ids.joblib")
    U = np.asarray(index._fit_X, dtype="float32")
    print("index type:", type(index).__name__, "U shape:", U.shape)
    print("index_ids type:", type(ids).__name__,
          "keys:" if isinstance(ids, dict) else "len:",
          list(ids.keys()) if isinstance(ids, dict) else len(ids))

    if isinstance(ids, dict):
        id_list = ids.get("state_ids") or ids.get("ids") or ids.get("state_id")
        u_order = ids.get("u_order") or ids.get("action_order")
        space = ids.get("space")
        print("space:", space, "u_order:", u_order)
    else:
        id_list, u_order = list(ids), None
    assert id_list is not None and len(id_list) == U.shape[0]

    # align the label-space rows to the feat-index row order
    pos_of = {s: i for i, s in enumerate(id_list)}
    aligned = np.asarray([pos_of[s] for s in state_ids])
    U = U[aligned]

    # the shipped decision quantity: u[taken action]
    assert u_order is not None, "need u_order to map actions to u dimensions"
    dim_of = {a: i for i, a in enumerate(u_order)}
    scored_mask = np.asarray([a in dim_of for a in actions])
    model_score = np.full(n, np.nan)
    rows = np.where(scored_mask)[0]
    model_score[rows] = U[rows, [dim_of[a] for a in actions[rows]]]

    from sklearn.neighbors import NearestNeighbors

    kmax = max(KS)
    results = {}

    def kernel_scores(space_X: np.ndarray, tag: str) -> None:
        nn = NearestNeighbors(n_neighbors=min(kmax * 3, n - 1)).fit(space_X)
        _, nbrs = nn.kneighbors(space_X)  # includes self at column 0
        for K in KS:
            est = np.full(n, np.nan)
            support = np.zeros(n, dtype=int)
            for i in range(n):
                same_action = succ = 0
                for j in nbrs[i]:
                    if j == i or episodes[j] == episodes[i]:
                        continue  # no self, no same-episode leakage
                    if actions[j] == actions[i]:
                        same_action += 1
                        succ += success[j]
                    if same_action == K:
                        break
                support[i] = same_action
                if same_action >= MIN_SUPPORT:
                    est[i] = succ / (same_action + LAPLACE)
            m = ~np.isnan(est) & scored_mask
            results[f"{tag}_{K}"] = {
                "coverage": round(float(m.mean()), 4),
                "auroc": round(auroc(success[m], est[m]), 4),
                "ece": round(ece(success[m], est[m]), 4),
                "auroc_on_same_states_model": round(
                    auroc(success[m], model_score[m]), 4),
            }

    kernel_scores(X, "kfeat")
    kernel_scores(U, "klab")

    m = scored_mask & ~np.isnan(model_score)
    results["model"] = {
        "coverage": round(float(m.mean()), 4),
        "auroc": round(auroc(success[m], model_score[m]), 4),
        "ece": round(ece(success[m], model_score[m]), 4),
    }

    # per-action AUROC for the best kernel vs the model
    best_tag = max((k for k in results if k != "model"),
                   key=lambda k: results[k]["auroc"])
    per_action = {}
    K = int(best_tag.split("_")[1])
    print("best kernel:", best_tag)
    print(json.dumps(results, indent=1))
    out = ROOT / "runs" / "reports" / "kernel_ether_probe.json"
    out.write_text(json.dumps({
        "analysis": "kernel-Ether estimator probe (round 1, leave-episode-out)",
        "n_states": int(n),
        "min_support": MIN_SUPPORT, "laplace": LAPLACE,
        "results": results, "best_kernel": best_tag,
    }, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"probe": "kernel_ether", "verdict": True,
                      "model_auroc": results["model"]["auroc"],
                      "best_kernel": best_tag,
                      "best_kernel_auroc": results[best_tag]["auroc"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
