"""Stage 10 runtime: retrieval, skill resolution and the M1/M2 loop.

Deterministic, hash-pinned, no external API: the shipped product's loop. M1
(the pinned local model) emits a structured ActionObject under constrained
decoding; M2 is deterministic code checking the payload against the Skill's
ExpectedResultSpec (schema subset + named rules supplied by the domain);
failures feed back as an adjustment and the loop retries up to
max_iterations, then escalates with a full ExecutionTrace.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from ammonix_core.hashing import sha256_text
from ammonix_core.schema import (
    ActionObject,
    AmmonixConfig,
    CheckResult,
    EscalationRecord,
    ExecutionTrace,
    IterationRecord,
    LLMPin,
    Neighbour,
    PromptTemplate,
    RetrievalResult,
    Skill,
    Tribe,
)

_CONFIG = AmmonixConfig()  # single source for shared thresholds (schema defaults)

MAX_ITERATIONS = _CONFIG.max_iterations
AMBIGUITY_MARGIN = _CONFIG.ambiguity_margin

# every validation keyword this validator understands; anything else in a
# schema raises instead of being silently ignored (a check that does not run
# must not look like a check that passed)
_KNOWN_KEYWORDS = {
    "type", "enum", "properties", "required", "additionalProperties", "items",
    "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum",
    "pattern", "title", "description", "default",
}


def validate_payload(schema: dict, payload: object) -> list[str]:
    """Validate against the JSON-schema subset the payload schemas use:
    object/required/additionalProperties, type, enum, array items, string
    bounds and pattern, numeric bounds. Unknown validation keywords raise
    NotImplementedError so no constraint is silently skipped. Returns
    failure paths (empty = valid)."""
    failures: list[str] = []

    def check(node_schema: dict, node: object, path: str) -> None:
        unknown = set(node_schema) - _KNOWN_KEYWORDS
        if unknown:
            raise NotImplementedError(
                f"{path}: schema keywords {sorted(unknown)} are not supported "
                "by this validator; supported: " + ", ".join(sorted(_KNOWN_KEYWORDS))
            )
        kind = node_schema.get("type")
        if "enum" in node_schema:
            if node not in node_schema["enum"]:
                failures.append(f"{path}: {node!r} not in enum")
            return
        if kind == "object":
            if not isinstance(node, dict):
                failures.append(f"{path}: expected object")
                return
            properties = node_schema.get("properties", {})
            for key in node_schema.get("required", []):
                if key not in node:
                    failures.append(f"{path}.{key}: required field missing")
            if node_schema.get("additionalProperties") is False:
                for key in node:
                    if key not in properties:
                        failures.append(f"{path}.{key}: unexpected field")
            for key, sub in properties.items():
                if key in node:
                    check(sub, node[key], f"{path}.{key}")
        elif kind == "array":
            if not isinstance(node, list):
                failures.append(f"{path}: expected array")
                return
            if len(node) < node_schema.get("minItems", 0):
                failures.append(f"{path}: fewer than minItems items")
            if "maxItems" in node_schema and len(node) > node_schema["maxItems"]:
                failures.append(f"{path}: more than maxItems items")
            items = node_schema.get("items")
            if items:
                for i, item in enumerate(node):
                    check(items, item, f"{path}[{i}]")
        elif kind == "string":
            if not isinstance(node, str):
                failures.append(f"{path}: expected string")
                return
            if len(node) < node_schema.get("minLength", 0):
                failures.append(f"{path}: shorter than minLength")
            if "maxLength" in node_schema and len(node) > node_schema["maxLength"]:
                failures.append(f"{path}: longer than maxLength")
            if "pattern" in node_schema and not re.search(node_schema["pattern"], node):
                failures.append(f"{path}: does not match pattern")
        elif kind in ("number", "integer"):
            if (
                not isinstance(node, (int, float))
                or isinstance(node, bool)
                or (kind == "integer" and not isinstance(node, int))
            ):
                failures.append(f"{path}: expected {kind}")
                return
            if node < node_schema.get("minimum", float("-inf")):
                failures.append(f"{path}: below minimum")
            if node > node_schema.get("maximum", float("inf")):
                failures.append(f"{path}: above maximum")
        elif kind == "boolean":
            if not isinstance(node, bool):
                failures.append(f"{path}: expected boolean")

    check(schema, payload, "$")
    return failures


@dataclass
class RetrievalContext:
    """Everything the domain resolved for one live case before the loop."""

    case_id: str
    features: dict[str, float]
    scores_cal: dict[str, float]  # applicable actions only, calibrated
    known: bool  # False = out of distribution (an entity outside the Basis): escalate
    scores_std: dict[str, float] | None = None  # fold-ensemble spread per
    # action, when the basis carries fold models; enables uncertainty-aware
    # ambiguity (a margin smaller than the models' own disagreement is a tie)


def retrieve(
    context: RetrievalContext,
    tribes: list[Tribe],
    skills: list[Skill],
    scaled_point: np.ndarray,
    neighbour_ids: list[str],
    neighbour_distances: list[float],
    trained_actions: list[str],
    ambiguity_margin: float = AMBIGUITY_MARGIN,
    ambiguous_override: bool | None = None,
) -> RetrievalResult:
    """Resolve recommendation, tribe, ambiguity and skill for one case."""
    ranked = sorted(context.scores_cal.items(), key=lambda kv: (-kv[1], kv[0]))
    recommended = ranked[0][0]
    trained_ranked = [kv for kv in ranked if kv[0] in trained_actions]
    live_margin = (
        trained_ranked[0][1] - trained_ranked[1][1]
        if len(trained_ranked) >= 2
        else 1.0
    )

    action_tribes = [t for t in tribes if t.action_id == recommended]
    tribe_id = f"{recommended}::cluster"  # no tribe structure: the parent cluster
    if action_tribes:
        centroid_matrix = np.array(
            [
                [t.centroid[k] for k in sorted(t.centroid)]
                for t in action_tribes
            ]
        )
        point = scaled_point  # already in the same sorted-feature order
        nearest = int(
            np.linalg.norm(centroid_matrix - point[None, :], axis=1).argmin()
        )
        tribe_id = action_tribes[nearest].tribe_id

    if ambiguous_override is not None:
        ambiguous = ambiguous_override
    else:
        # v0.4: an ambiguous TRIBE routes to its ask skill only when the
        # LIVE margin/noise says this case is a coin flip; region membership
        # alone no longer flags the case (label-space tribes are large)
        ambiguous = live_margin < ambiguity_margin
        # uncertainty-aware tie: when the fold models' own disagreement about
        # the top two actions exceeds the gap between them, the ranking is
        # inside the swarm's noise floor and honesty requires the flag
        if not ambiguous and context.scores_std and len(trained_ranked) >= 2:
            top, rival = trained_ranked[0][0], trained_ranked[1][0]
            # noise of a difference of independent estimates: root-sum-of-
            # squares, not the sum (the sum over-flags ~half the dev slice
            # on the first basis that actually persists fold models)
            spread = float(np.hypot(
                context.scores_std.get(top, 0.0),
                context.scores_std.get(rival, 0.0),
            ))
            ambiguous = live_margin < spread

    by_tribe = {
        s.scope.ref: s for s in skills if s.scope.level == "tribe"
    }
    by_cluster = {
        s.scope.ref: s for s in skills if s.scope.level == "cluster"
    }
    default = next((s for s in skills if s.scope.level == "default"), None)
    if default is None:
        raise ValueError(
            "no default skill configured: the skill list must always carry a "
            "scope.level == 'default' fallback"
        )
    # v0.4 s.3b: a tribe-scoped ask skill marks coin-flip TERRITORY; it takes
    # the case only when the live case itself is ambiguous. Confident cases
    # inside a flagged region fall through to the action's cluster skill.
    tribe_skill = by_tribe.get(tribe_id)
    if (tribe_skill is not None and tribe_skill.kind == "ask_before_deciding"
            and not ambiguous):
        tribe_skill = None
    skill = tribe_skill or by_cluster.get(recommended) or default

    return RetrievalResult(
        case_id=context.case_id,
        features=context.features,
        scores_cal=context.scores_cal,
        neighbours=[
            Neighbour(state_id=s, distance=float(d))
            for s, d in zip(neighbour_ids, neighbour_distances, strict=True)
        ],
        tribe_id=tribe_id,
        recommended_action_id=recommended,
        ambiguous=ambiguous,
        skill_id=skill.skill_id,
        expected_result=skill.expected_result.model_dump(),
    )


def context_prompt_fields(
    case_row: dict, retrieval: RetrievalResult, skill: Skill
) -> dict:
    """Generic M1 prompt fields: the recommended action plus every case_record
    field the Skill's own context_sources declare. Domain packages that need
    richer prompts (extra fields, truncation) pass their own builder to
    run_case; this default keeps the runtime free of domain knowledge."""
    fields = {"action_id": retrieval.recommended_action_id}
    for source in skill.context_sources:
        if source.source == "case_record":
            value = case_row.get(source.locator, "")
            fields[source.field] = "" if value is None else str(value)
    return fields


def run_case(
    retrieval: RetrievalResult,
    skill: Skill,
    case_row: dict,
    m1_prompt: PromptTemplate,
    m2_prompt: PromptTemplate,
    llm_pin: LLMPin,
    generate: Callable[[str, dict], str],  # (prompt, payload_schema) -> JSON text
    evaluate_rule: Callable[[str, dict, dict], tuple[bool, str]],
    basis_id: str,
    harness_id: str,
    known: bool = True,
    max_iterations: int = MAX_ITERATIONS,
    prompt_fields: Callable[[dict, RetrievalResult], dict] | None = None,
) -> ExecutionTrace:
    """The verifiable-reward loop for one case."""
    import json as _json

    started = datetime.now(UTC)

    def finish(status, iterations, escalation=None, final=None):
        return ExecutionTrace(
            case_id=retrieval.case_id,
            basis_id=basis_id,
            harness_id=harness_id,
            retrieval=retrieval,
            iterations=iterations,
            status=status,
            escalation=escalation,
            final=final,
            started_at=started,
            finished_at=datetime.now(UTC),
        )

    if not known:
        return finish(
            "escalated", [],
            EscalationRecord(
                reason="needs_outside_information",
                question="case is outside the Basis: no tribe, no calibrated scores",
                handed_to="human_queue",
            ),
        )
    if skill.kind == "escalate":
        return finish(
            "escalated", [],
            EscalationRecord(
                reason="unresolvable",
                question=(
                    f"skill {skill.skill_id} routes "
                    f"{retrieval.recommended_action_id} to a human"
                ),
                handed_to="human_queue",
            ),
        )
    if skill.kind == "ask_before_deciding":
        return finish(
            "escalated", [],
            EscalationRecord(
                reason="needs_outside_information",
                question=skill.question.text_template if skill.question else None,
                handed_to="human_queue",
            ),
        )

    schema = skill.expected_result.payload_schema or {}
    rules = skill.expected_result.rules or []
    iterations: list[IterationRecord] = []
    feedback = ""
    if prompt_fields is None:
        fields = context_prompt_fields(case_row, retrieval, skill)
    else:
        fields = prompt_fields(case_row, retrieval)
    for n in range(1, max_iterations + 1):
        prompt = m1_prompt.text.format(feedback=feedback, **fields)
        raw = generate(prompt, schema)
        try:
            payload = _json.loads(raw)
        except (ValueError, TypeError):
            check = CheckResult(passed=False, failures=["$: output was not valid JSON"])
            iterations.append(
                IterationRecord(
                    n=n, m1_prompt_sha256=sha256_text(prompt), action=None, check=check,
                    adjustment=m2_prompt.text.format(failures="output was not valid JSON"),
                )
            )
            feedback = iterations[-1].adjustment
            continue
        action = ActionObject(
            action_id=retrieval.recommended_action_id, payload=payload, llm=llm_pin
        )
        failures = validate_payload(schema, payload)
        for rule in rules:
            passed, detail = evaluate_rule(rule, payload, case_row)
            if not passed:
                failures.append(f"rule {rule}: {detail}")
        check = CheckResult(passed=not failures, failures=failures)
        if check.passed:
            iterations.append(
                IterationRecord(
                    n=n, m1_prompt_sha256=sha256_text(prompt), action=action,
                    check=check, adjustment=None,
                )
            )
            return finish("executed", iterations, final=action)
        adjustment = m2_prompt.text.format(failures="; ".join(failures))
        iterations.append(
            IterationRecord(
                n=n, m1_prompt_sha256=sha256_text(prompt), action=action,
                check=check, adjustment=adjustment,
            )
        )
        feedback = adjustment

    return finish(
        "escalated", iterations,
        EscalationRecord(reason="iteration_cap", handed_to="human_queue"),
    )
