"""Toy world (milestone M1): planted feature-Action-Outcome structure.

The factory's permanent regression fixture. A synthetic environment with a
known optimal policy: the pipeline must recover the planted truth end to end
(mean OOF AUROC >= 0.95, recommended action matches the oracle optimum in
>= 95% of test states).

Planted structure over features x1, x2 in [0,1], x3 boolean, x4 noise:

    act_a is right when x1 > 0.5
    act_b is right when x1 <= 0.5 and x3
    act_c is right when x1 <= 0.5 and not x3
    x2 and x4 carry no signal (decoys the swarm must ignore)

True P(success | state, action) is a sharp sigmoid on that margin, so outcomes
are stochastic but the optimal policy is crisp and analytic.

The logging policy is deliberately biased (habitual act_a half the time): an
imitation learner would recommend act_a everywhere, a success learner must not.
Episodes are single-state; episodic history machinery is exercised from P4 on.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from ammonix_core.hashing import sha256_json
from ammonix_core.schema import Example, Outcome, RawData, RawState

ACTIONS = ("act_a", "act_b", "act_c")
FEATURE_NAMES = ("x1", "x2", "x3", "x4")

_SLOPE = 20.0  # sharpness of the planted boundary
_P_FLOOR, _P_SPAN = 0.02, 0.96  # success probability range [0.02, 0.98]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def true_success_prob(features: dict, action: str) -> float:
    """The planted truth: P(success | state, action), known analytically."""
    x1, x3 = float(features["x1"]), bool(features["x3"])
    if action == "act_a":
        margin = _SLOPE * (x1 - 0.5)
    elif action == "act_b":
        margin = _SLOPE * (0.5 - x1) - (0.0 if x3 else 2.0 * _SLOPE)
    elif action == "act_c":
        margin = _SLOPE * (0.5 - x1) - (2.0 * _SLOPE if x3 else 0.0)
    else:
        raise ValueError(f"unknown toy action: {action}")
    return _P_FLOOR + _P_SPAN * _sigmoid(margin)


def optimal_action(features: dict) -> str:
    """The oracle policy: argmax of the planted success probabilities."""
    return max(ACTIONS, key=lambda a: true_success_prob(features, a))


@dataclass
class ToyState:
    """StateHandle for the toy environment."""

    features: dict
    action_taken: str | None = None
    success: bool | None = None


@dataclass
class ToyWorld:
    """EnvironmentProtocol implementation. One decision per episode."""

    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def reset(self, seed: int) -> ToyState:
        self._rng = np.random.default_rng(seed)
        return ToyState(features=self._draw_features())

    def legal_actions(self, s: ToyState) -> list[str]:
        return [] if s.action_taken is not None else list(ACTIONS)

    def apply(self, s: ToyState, action_raw: str) -> ToyState:
        if action_raw not in ACTIONS:
            raise ValueError(f"illegal toy action: {action_raw}")
        p = true_success_prob(s.features, action_raw)
        return ToyState(
            features=s.features,
            action_taken=action_raw,
            success=bool(self._rng.random() < p),
        )

    def outcome(self, s: ToyState) -> Outcome | None:
        if s.action_taken is None:
            return None
        return Outcome(success=bool(s.success))

    def _draw_features(self) -> dict:
        return {
            "x1": float(self._rng.random()),
            "x2": float(self._rng.random()),
            "x3": bool(self._rng.random() < 0.5),
            "x4": float(self._rng.random()),
        }


def logging_action(features: dict, rng: np.random.Generator) -> str:
    """The biased historical policy: 50% habitual act_a, 50% uniform random."""
    if rng.random() < 0.5:
        return "act_a"
    return ACTIONS[int(rng.integers(len(ACTIONS)))]


@dataclass
class ToyCorpus:
    examples: list[Example]
    states: list[RawState]

    def content_sha256(self) -> str:
        """Order-stable hash over the full corpus content (double-generation check)."""
        rows = [
            (
                s.state_id,
                s.example_id,
                {k: s.data.inline[k] for k in sorted(s.data.inline)},
                s.action_raw,
            )
            for s in self.states
        ] + [(e.example_id, e.outcome.success) for e in self.examples]
        return sha256_json(rows)


def generate_corpus(n_episodes: int, seed: int) -> ToyCorpus:
    """Roll the world under the logging policy; emit schema-shaped Examples."""
    world = ToyWorld()
    policy_rng = np.random.default_rng(seed + 1)
    examples: list[Example] = []
    states: list[RawState] = []
    for i in range(n_episodes):
        s0 = world.reset(seed * 1_000_003 + i)
        action = logging_action(s0.features, policy_rng)
        terminal = world.apply(s0, action)
        outcome = world.outcome(terminal)
        assert outcome is not None
        example_id = f"toy-e{i:06d}"
        state_id = f"toy-s{i:06d}-0"
        data = RawData(
            modality="tabular",
            inline=dict(s0.features),
            sha256=sha256_json({k: s0.features[k] for k in sorted(s0.features)}),
        )
        states.append(
            RawState(
                state_id=state_id,
                example_id=example_id,
                seq=0,
                data=data,
                action_raw=action,
                next_state_id=None,
            )
        )
        examples.append(
            Example(example_id=example_id, state_ids=[state_id], outcome=outcome)
        )
    return ToyCorpus(examples=examples, states=states)


def generate_test_states(n_states: int, seed: int) -> list[dict]:
    """Fresh evaluation states with the oracle's optimal action attached.

    Kept out of data/quarantine/ by design: the toy oracle is analytic, so no
    sealed set is needed at M1.
    """
    world = ToyWorld()
    rows = []
    for i in range(n_states):
        s = world.reset(seed * 2_000_003 + i)
        rows.append({"features": dict(s.features), "optimal_action": optimal_action(s.features)})
    return rows
