"""The agentic decision policy: briefing -> LLM -> guardrail -> (retry) -> action.

Guardrails enforce APPLICABILITY only (facts derivable from the published
payer rules in the snapshot) — never strategy. One feedback retry, mirroring
the factory harness's M2 verifiable-reward loop; a second violation or an
explicit "escalate" hands the touch to the human.
"""

import json
import threading
from dataclasses import dataclass, field

from agentic.briefing import SYSTEM_PROMPT, build_briefing, decision_schema


@dataclass
class Decision:
    action: str | None  # None = escalate to the human for this touch
    reasoning: str = ""
    retried: bool = False
    violations: list[str] = field(default_factory=list)


def applicability_violation(action: str, snap: dict) -> str | None:
    """Published-rules applicability check; None when the action is applicable."""
    carc = snap.get("carc") or ""
    denial_on_table = carc != "" and carc != "CO-16"
    if action == "request_retro_auth":
        if not snap["auth_required"]:
            return "this CPT does not require prior authorization at this payer"
        if snap["auth_status"] == "on_file":
            return "an authorization is already on file"
        if snap["retro_window_days"] <= 0:
            return "this payer does not offer retroactive authorization"
        if snap["days_since_service"] > snap["retro_window_days"]:
            return (
                f"the retro-auth window ({snap['retro_window_days']} days) closed "
                f"{snap['days_since_service'] - snap['retro_window_days']} days ago"
            )
    if action == "appeal_with_necessity" and not denial_on_table:
        return "there is no denial on the table to appeal"
    if action == "request_peer_to_peer":
        if not snap["p2p_available"]:
            return "this payer does not offer peer-to-peer review"
        if not denial_on_table:
            return "there is no denial on the table to review"
    if action == "provide_requested_info" and carc != "CO-16":
        return "the payer has not requested information (no CO-16 on the table)"
    if action == "bill_secondary" and not snap["has_secondary"]:
        return "the patient has no secondary payer"
    return None


class AgenticPolicy:
    """One decision per call; deterministic given the LLM cache."""

    def __init__(self, llm, system_prompt: str | None = None):
        self.llm = llm
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._lock = threading.Lock()
        self.decisions = 0
        self.retries = 0
        self.escalations_explicit = 0
        self.escalations_forced = 0
        self.decision_errors = 0  # infrastructure failures, counted by the runner

    def _count(self, counter: str) -> None:
        with self._lock:
            setattr(self, counter, getattr(self, counter) + 1)

    def _briefing(self, snap: dict, legal_actions: list[str]) -> str:
        """The user-turn briefing; subclasses (e.g. the retrieval-augmented
        arm) may append blocks. The default is the plain briefing."""
        return build_briefing(snap, legal_actions)

    def prepare(self, snap_rows: list[dict]) -> None:
        """Optional per-step batch hook; no-op for the plain policy."""

    def decide(self, snap: dict, legal_actions: list[str]) -> Decision:
        self._count("decisions")
        allowed = [*legal_actions, "escalate"]
        schema = decision_schema(allowed)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._briefing(snap, legal_actions)},
        ]
        violations: list[str] = []
        for attempt in (0, 1):
            raw = self.llm.complete(messages, schema)
            action, reasoning, problem = None, "", None
            try:
                parsed = json.loads(raw)
                reasoning = str(parsed.get("reasoning", ""))
                action = parsed.get("action")
                if action not in allowed:
                    problem = f"'{action}' is not one of the available actions"
            except (json.JSONDecodeError, AttributeError):
                problem = "the reply was not valid JSON"
            if problem is None:
                if action == "escalate":
                    self._count("escalations_explicit")
                    return Decision(None, reasoning, attempt > 0, violations)
                problem = applicability_violation(action, snap)
                if problem is None:
                    return Decision(action, reasoning, attempt > 0, violations)
            violations.append(f"{action}: {problem}")
            if attempt == 0:
                self._count("retries")
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That action is not applicable: {problem}. "
                            "Choose a different action from the available list."
                        ),
                    }
                )
        self._count("escalations_forced")
        return Decision(None, "", True, violations)

    def stats(self) -> dict:
        return {
            "decisions": self.decisions,
            "retries": self.retries,
            "escalations_explicit": self.escalations_explicit,
            "escalations_forced": self.escalations_forced,
            "decision_errors": self.decision_errors,
        }
