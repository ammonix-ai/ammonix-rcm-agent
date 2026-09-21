"""v0.6: the kernel decision field.

The Foundation paper defines the Ether as an average of stored outcome labels
over the neighbourhood of the live case's coordinate, with the classifier
swarm only placing the case. The shipped v0.4/v0.5 agent estimates the same
field with fitted per-action models. The 2026-08-14 experiments
(runs/reports/kernel_ether_probe.json, kernel_ether_rollout.json) measured
the two forms as equal, and the kernel form additionally absorbs newly
adjudicated records without retraining any model. This module is that form.

Behaviour contract:
- default OFF: BasisRuntime.decision_field == "model" leaves every shipped
  code path untouched, byte-identical behaviour;
- ON (attach_kernel_field): route_case scores each applicable trained action
  by the Laplace-smoothed success rate of up to K same-action neighbours at
  the live coordinate; below MIN_SUPPORT neighbours the corpus-level success
  rate for the action is used (the wide-bandwidth limit of the kernel);
- insert() appends an adjudicated record and refits only the neighbour index
  (seconds; no model retraining) - the consolidation step of [1] made
  incremental;
- content_hash() fingerprints the stored arrays so a grown universe can be
  pinned in a manifest exactly like any other artefact.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

K_NEIGHBOURS = 200
MIN_SUPPORT = 3
LAPLACE = 2
QUERY_N = 2000


class KernelField:
    def __init__(self, U, state_ids, actions, success, u_order):
        from sklearn.neighbors import NearestNeighbors

        self.U = np.asarray(U, dtype="float64")
        self.state_ids = list(state_ids)
        self.actions = np.asarray(actions)
        self.success = np.asarray(success, dtype=int)
        self.u_order = list(u_order)
        self._nn_cls = NearestNeighbors
        self._refit()

    @classmethod
    def from_basis(cls, root: Path, rt) -> "KernelField":
        """Build from the shipped basis artefacts: the label-space index rows
        plus the outcome labels recorded in the feature index metadata."""
        meta = json.loads(
            (Path(root) / "basis" / "feat_index" / "meta.json").read_text(
                encoding="utf-8"
            )
        )
        by_id = {
            s: (a, int(y))
            for s, a, y in zip(
                meta["state_id"], meta["true_action_id"], meta["outcome_success"]
            )
        }
        actions = [by_id[s][0] for s in rt.index_state_ids]
        success = [by_id[s][1] for s in rt.index_state_ids]
        return cls(
            U=np.asarray(rt.index._fit_X),
            state_ids=rt.index_state_ids,
            actions=actions,
            success=success,
            u_order=rt.u_order,
        )

    def _refit(self) -> None:
        self._nn = self._nn_cls(
            n_neighbors=min(QUERY_N, len(self.state_ids))
        ).fit(self.U)
        self.global_rate = {
            a: float(self.success[self.actions == a].mean())
            for a in set(self.actions.tolist())
        }

    def live_u(self, rt, x: np.ndarray) -> np.ndarray:
        """Lambda places the case: the calibrated score vector in u_order."""
        u = np.empty((1, len(self.u_order)), dtype="float64")
        for j, action in enumerate(self.u_order):
            raw = rt.refit_models[action].predict_proba(x)[0, 1]
            u[0, j] = float(rt.calibrators[action].predict([raw])[0])
        return u

    def scores(self, rt, x: np.ndarray, wanted: set) -> dict:
        """Neighbour-count success estimates for the wanted actions."""
        u = self.live_u(rt, x)
        _, idx = self._nn.kneighbors(u)
        nbr_actions = self.actions[idx[0]]
        nbr_success = self.success[idx[0]]
        out = {}
        for action in wanted:
            take = np.where(nbr_actions == action)[0][:K_NEIGHBOURS]
            if len(take) >= MIN_SUPPORT:
                out[action] = float(
                    nbr_success[take].sum() / (len(take) + LAPLACE)
                )
            else:
                out[action] = self.global_rate.get(action, 0.0)
        return out

    def insert(self, u_row, action: str, success: bool, state_id: str) -> None:
        """Consolidate one adjudicated record: append and refit the neighbour
        index only. No model is retrained."""
        self.U = np.vstack([self.U, np.asarray(u_row, dtype="float64")])
        self.state_ids.append(state_id)
        self.actions = np.append(self.actions, action)
        self.success = np.append(self.success, int(bool(success)))
        self._refit()

    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.U.tobytes())
        h.update("|".join(self.state_ids).encode("utf-8"))
        h.update("|".join(self.actions.tolist()).encode("utf-8"))
        h.update(self.success.tobytes())
        return h.hexdigest()


def attach_kernel_field(rt, root: Path) -> "KernelField":
    """Switch a loaded BasisRuntime to the kernel decision field."""
    field = KernelField.from_basis(root, rt)
    rt.kernel_field = field
    rt.decision_field = "kernel"
    return field
