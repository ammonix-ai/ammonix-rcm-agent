"""Retrieval-augmented agentic baseline (pre-registered as
preregistration/comparison_rag_tranche100.json, arm "agentic_rag").

The v6 agentic loop with ONE addition to the user-turn briefing: a
SIMILAR PAST CLAIMS block. For the live state we compute the agent's own 83
kept features (the same mechanistic world model), query the agent's own
feature-space index (cardessa.c2_fallback.FeatIndex: 11,427 training-lineage
states, standardized), and render the k nearest neighbours as the action taken
and whether that claim was eventually paid, followed by a per-action tally.
This hands the LLM, as text, the same recorded experience the Ether counts.

Nothing from the live episode or its future enters the block: neighbours are
training-lineage records only (the index holds no evaluation-tranche states),
and same-episode exclusion is moot because tranche-100 episodes are not in it.

Everything else (system prompt, schema, temperature 0, one applicability
retry, persona fallback on escalation) is inherited from AgenticPolicy.
"""

from collections import defaultdict
from pathlib import Path

import polars as pl

from agentic.briefing import build_briefing
from agentic.policy import AgenticPolicy

K_NEIGHBOURS = 25


class RagPolicy(AgenticPolicy):
    def __init__(self, llm, factory_root: Path, k: int = K_NEIGHBOURS,
                 system_prompt: str | None = None):
        super().__init__(llm, system_prompt=system_prompt)
        self._k = k
        from cardessa.c2_fallback import FeatIndex
        from cardessa.harness import BasisRuntime

        self._root = Path(factory_root)
        self._index = FeatIndex(str(self._root / "basis" / "feat_index"))
        rt = BasisRuntime.load(self._root)
        self._names = rt.feature_names
        self._working = pl.read_parquet(
            self._root / "data" / "working" / "cardessa_sim" / "states.parquet"
        )
        self._feats: dict[str, dict] = {}
        self.retrieval_blocks = 0

    def prepare(self, snap_rows: list[dict]) -> None:
        """Batch-compute the kept features for this step's live states."""
        from cardessa.features_kept import build_kept_features

        snaps = pl.DataFrame(
            [{c: r.get(c) for c in self._working.columns} for r in snap_rows],
            schema_overrides=self._working.schema,
        )
        frame, _, _, _ = build_kept_features(
            self._root, pl.concat([self._working, snaps], how="vertical_relaxed")
        )
        ids = snaps["state_id"].to_list()
        self._feats = {
            r["state_id"]: {n: r[n] for n in self._names}
            for r in frame.filter(pl.col("state_id").is_in(ids)).to_dicts()
        }

    def _retrieval_block(self, snap: dict) -> str:
        feats = self._feats.get(snap.get("state_id"))
        if feats is None:
            return ""
        nbrs = self._index.nearest(feats, self._k)
        taken = defaultdict(int)
        paid = defaultdict(int)
        rows = []
        for j in nbrs:
            a = self._index.action[j]
            y = bool(self._index.success[j])
            taken[a] += 1
            paid[a] += int(y)
            rows.append(f"  - took {a}: {'paid' if y else 'not paid'}")
        tally = ", ".join(
            f"{a}: {paid[a]}/{taken[a]} paid" for a in sorted(taken, key=lambda x: -taken[x])
        )
        self.retrieval_blocks += 1
        return (
            "\nSIMILAR PAST CLAIMS (the "
            f"{len(nbrs)} most similar claims in this provider's history, at the same "
            "kind of decision point; whether each was eventually paid):\n"
            + "\n".join(rows)
            + f"\n  tally by action: {tally}\n"
        )

    def _briefing(self, snap: dict, legal_actions: list[str]) -> str:
        return build_briefing(snap, legal_actions) + self._retrieval_block(snap)
