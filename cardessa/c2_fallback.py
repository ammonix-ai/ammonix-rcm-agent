"""V5-M2 (C2): outcome-weighted case-based fallback over the feature-space index.

Engages only below the confidence floor. Given a live case's feature vector (in
the feature-index column space), the applicable actions, and the existing
expected-value estimates, it consults the K nearest FEATURE-space neighbours in
the Knowledge Universe and returns the applicable action whose neighbours most
often SUCCEEDED (outcome-weighted, Laplace-smoothed), provided it clears a
success threshold and carries positive EV. Otherwise returns None and the caller
falls back to the v0.4 behaviour (EV close-out / escalate).

Deterministic: pure numpy over the committed basis/feat_index. No model, no LLM.
"""
import json
import os
from collections import defaultdict

import numpy as np


class FeatIndex:
    def __init__(self, out_dir: str):
        self.X = np.load(os.path.join(out_dir, "matrix.npy"))  # (n,d) f32, standardized
        meta = json.loads(open(os.path.join(out_dir, "meta.json"),
                                encoding="utf-8").read())
        self.cols = meta["feature_columns"]
        self.mean = np.asarray(meta["mean"], dtype="float64")
        self.std = np.asarray(meta["std"], dtype="float64")
        self.action = meta["true_action_id"]
        self.success = np.asarray(meta["outcome_success"], dtype=bool)
        self.episode = meta["episode_id"]
        self.state_id = meta["state_id"]

    def query_vec(self, feat: dict) -> np.ndarray:
        v = np.asarray([float(feat[c]) for c in self.cols], dtype="float64")
        return ((v - self.mean) / self.std).astype("float32")

    def nearest(self, feat: dict, k: int, exclude_episode=None,
                exclude_state=None):
        q = self.query_vec(feat)
        d = np.sqrt(((self.X - q) ** 2).sum(axis=1))
        picked = []
        for j in np.argsort(d):
            if exclude_state is not None and self.state_id[j] == exclude_state:
                continue
            if exclude_episode is not None and self.episode[j] == exclude_episode:
                continue
            picked.append(int(j))
            if len(picked) == k:
                break
        return picked


def c2_action(index: FeatIndex, feat: dict, applicable, ev: dict,
              k: int = 25, s_min: int = 3, tau: float = 0.55, m: int = 2,
              exclude_episode=None, exclude_state=None):
    """Return (action, detail) or (None, detail)."""
    nbrs = index.nearest(feat, k, exclude_episode, exclude_state)
    support = defaultdict(int)
    succ = defaultdict(int)
    for j in nbrs:
        a = index.action[j]
        support[a] += 1
        if index.success[j]:
            succ[a] += 1
    scored = {a: succ[a] / (support[a] + m)
              for a in support if support[a] >= s_min and a in applicable}
    if not scored:
        return None, {"reason": "no supported applicable neighbour action"}
    a_star = max(scored, key=lambda a: (scored[a], support[a]))
    w = scored[a_star]
    detail = {"a_star": a_star, "w": round(w, 3), "support": support[a_star],
              "succ": succ[a_star], "ev": round(float(ev.get(a_star, 0.0)), 2)}
    if w >= tau and ev.get(a_star, 0.0) > 0.005:
        return a_star, detail
    detail["reason"] = "below tau or non-positive EV"
    return None, detail
