"""P8 harness glue: skills, basis runtime, routing masks, M1 provider.

Routing policy (all deterministic, from published payer rules and case
facts - the structural preconditions the M2 rules would fail anyway):
- an action is APPLICABLE only when its deterministic preconditions hold
  (filing window open for submissions, retro window open and auth missing
  for retro requests, a live denial for appeals/p2p, a pending balance for
  patient billing, a secondary payer with a clean pending share for
  bill_secondary);
- a CO-16 denial names its own responsive action (provide_requested_info);
  covered and executable since v0.2, with grounding rules that force an
  escalation whenever the letter demands something the record lacks;
- a payer outside the Basis (holdout) is out of distribution: escalate;
- if the best applicable calibrated score is below CONFIDENCE_FLOOR the
  system does not pretend: escalate as unresolvable (write_off would be
  the bookkeeping default, flagged low-confidence, decided by a human).
"""

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from ammonix_core.hashing import sha256_text
from ammonix_core.schema import (
    BasisManifest,
    ContextSource,
    ExpectedResultSpec,
    LLMPin,
    Scope,
    Skill,
    Tribe,
)

from cardessa.payloads import PAYLOAD_SCHEMAS, RULES
from cardessa.textgen import DEFAULT_BASE_URL, PINNED_MODEL_SUBSTRING

CONFIDENCE_FLOOR = 0.05
COVERED_ACTIONS = (
    "submit_clean", "submit_with_records", "request_retro_auth",
    "correct_and_resubmit", "appeal_with_necessity", "request_peer_to_peer",
    "provide_requested_info",  # trained in v0.2 (coverage 108 >= 100)
    "bill_secondary", "bill_patient", "write_off",
)
ESCALATE_ACTIONS = ()  # v0.2: every canonical action is covered


def build_skills(tribes: list[Tribe]) -> list[Skill]:
    skills: list[Skill] = [
        Skill(
            skill_id="default-escalate",
            version="v1",
            scope=Scope(level="default", ref=None),
            kind="escalate",
            context_schema={},
            context_sources=[],
            m1_prompt_ref="",
            expected_result=ExpectedResultSpec(kind="schema"),
        )
    ]
    for action in COVERED_ACTIONS:
        skills.append(
            Skill(
                skill_id=f"skill-{action}",
                version="v1",
                scope=Scope(level="cluster", ref=action),
                kind="execute",
                context_schema={
                    "type": "object",
                    "properties": {"episode_id": {"type": "string"}},
                    "required": ["episode_id"],
                },
                context_sources=[
                    ContextSource(
                        field="episode_id", source="case_record", locator="episode_id"
                    )
                ],
                m1_prompt_ref=f"harness/prompts/m1_{action}.txt",
                expected_result=ExpectedResultSpec(
                    kind="composite",
                    payload_schema=PAYLOAD_SCHEMAS[action],
                    rules=RULES[action],
                ),
            )
        )
    for action in ESCALATE_ACTIONS:
        skills.append(
            Skill(
                skill_id=f"skill-{action}-escalate",
                version="v1",
                scope=Scope(level="cluster", ref=action),
                kind="escalate",
                context_schema={},
                context_sources=[],
                m1_prompt_ref="",
                expected_result=ExpectedResultSpec(kind="schema"),
            )
        )
    # v0.4: the label-space basis surfaces ambiguous tribes; each gets a
    # tribe-scoped ask_before_deciding skill (the paper's regional skill).
    # retrieval prefers tribe scope, so inside an ambiguous region the
    # system asks the DecisionQuestion instead of executing a coin flip.
    for tribe in tribes:
        if tribe.stats.ambiguous:
            skills.append(
                Skill(
                    skill_id=f"skill-ask-{tribe.tribe_id}",
                    version="v1",
                    scope=Scope(level="tribe", ref=tribe.tribe_id),
                    kind="ask_before_deciding",
                    context_schema={},
                    context_sources=[],
                    m1_prompt_ref="",
                    expected_result=ExpectedResultSpec(kind="schema"),
                )
            )
    return skills


@dataclass
class BasisRuntime:
    manifest: BasisManifest
    tribes: list[Tribe]
    scaler: object
    index: object
    index_state_ids: list[str]
    calibrators: dict[str, object]
    refit_models: dict[str, object]
    trained_actions: list[str]
    constant_priors: dict[str, float]
    working_payers: set[str]
    feature_names: list[str]
    resolution_models: dict = None
    resolution_calibrators: dict = None
    success_given_granted: dict = None
    u_order: list[str] | None = None  # v0.4 s.3b: index's action order
    fold_models: dict = None  # action -> [fold models]; empty on bases built
    # before v0.4 (the shipped v0.3 basis has none: the uncertainty path is
    # dormant until the next basis build persists them)
    # v0.6: the kernel decision field (cardessa.kernel_field). Default OFF:
    # decision_field == "model" leaves every shipped path byte-identical.
    # attach_kernel_field(rt, root) switches the episode-success scores in
    # route_case to neighbour counts over the stored label-space index.
    kernel_field: object = None
    decision_field: str = "model"

    @classmethod
    def load(cls, root: Path) -> "BasisRuntime":
        basis = root / "basis"
        manifest = BasisManifest.model_validate_json(
            (basis / "manifest.json").read_text(encoding="utf-8")
        )
        tribes = [
            Tribe.model_validate(t)
            for t in json.loads((basis / "tribes.json").read_text(encoding="utf-8"))
        ]
        rounds = json.loads(
            (root / "runs" / "state" / "swarm_rounds.json").read_text(
                encoding="utf-8"
            )
        )
        kept = [r["round"] for r in rounds if r["kept"]][-1]
        swarm = json.loads(
            (root / "runs" / "manifests" / f"swarm_round{kept}.json").read_text(
                encoding="utf-8"
            )
        )
        artefacts = basis / "artefacts"
        trained = swarm["trained_actions"]
        working_eps = pl.read_parquet(
            root / "data" / "working" / "cardessa_sim" / "episodes.parquet"
        )
        rt = cls(
            manifest=manifest,
            tribes=tribes,
            scaler=joblib.load(artefacts / "scaler.joblib"),
            index=joblib.load(artefacts / "index.joblib"),
            index_state_ids=joblib.load(artefacts / "index_ids.joblib")["state_ids"],
            u_order=joblib.load(artefacts / "index_ids.joblib").get("u_order"),
            calibrators={
                a: joblib.load(basis / f"artefacts/calibrator_{a}.joblib")
                for a in trained
            },
            refit_models={
                a: joblib.load(artefacts / f"refit_{a}.joblib") for a in trained
            },
            trained_actions=trained,
            constant_priors={
                a: info["prior"] for a, info in swarm["constant_prior"].items()
            },
            working_payers=set(working_eps["payer_id"].unique().to_list()),
            feature_names=[s.name for s in manifest.tokenizer.features],
            fold_models={
                a: [joblib.load(f) for f in sorted(artefacts.glob(f"fold_{a}_*.joblib"))]
                for a in trained
                if list(artefacts.glob(f"fold_{a}_*.joblib"))
            },
            **cls._load_resolution(root),
        )
        # v0.6 runtime switch. Default "kernel": the kernel decision field,
        # activation approved at gate P8, 2026-08-14 (runs/APPROVALS.md, "v0.6
        # kernel decision field: activation"). "model" is the rollback, set
        # via the AMMONIX_DECISION_FIELD env var or basis/decision_field.json
        # {"decision_field": "model"}.
        choice = os.environ.get("AMMONIX_DECISION_FIELD")
        cfg = basis / "decision_field.json"
        if choice is None and cfg.is_file():
            choice = json.loads(cfg.read_text(encoding="utf-8")).get(
                "decision_field"
            )
        if (choice or "kernel").lower() == "kernel":
            from cardessa.kernel_field import attach_kernel_field
            attach_kernel_field(rt, root)
        return rt

    @staticmethod
    def _load_resolution(root: Path) -> dict:
        """v0.3 resolution heads (spec s.4); absent on older lineages."""
        manifest_path = root / "runs" / "manifests" / "resolution_heads.json"
        if not manifest_path.is_file():
            return {"resolution_models": {}, "resolution_calibrators": {},
                    "success_given_granted": {}}
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        artefacts = root / "basis" / "artefacts"
        return {
            "resolution_models": {
                a: joblib.load(artefacts / f"resolution_{a}.joblib")
                for a in doc["actions"]
            },
            "resolution_calibrators": {
                a: joblib.load(artefacts / f"resolution_cal_{a}.joblib")
                for a in doc["actions"]
            },
            "success_given_granted": doc["success_given_granted"],
        }


def applicable_actions(case_row: dict) -> set[str]:
    """Deterministic structural preconditions from published rules + case facts."""
    applicable = {"write_off"}
    carc = case_row.get("carc") or ""
    prior = (case_row.get("prior_actions") or "").split(",")
    # balance < allowed means the payer already paid its share: resubmitting
    # an adjudicated-and-paid claim is duplicate billing, never a plan (the
    # engine world does not deny duplicates, so the mask must)
    already_paid = case_row["balance"] < case_row.get(
        "allowed_amount", case_row["balance"]
    ) - 0.005
    if case_row["days_to_filing_deadline"] > 0 and not already_paid:
        # a submission the engine will reject on published rules is not a
        # plan: pairing must be legal and the COB order correct
        if case_row["pairing_valid"] and case_row["cob_position_ok"]:
            applicable |= {"submit_clean", "submit_with_records"}
            # adjudication is deterministic: with a denial standing and no
            # claim fact changed, an identical resubmission is guaranteed the
            # identical denial (the ledger's resubmitted_unchanged mistake)
            if carc:
                if any(a in prior for a in (
                    "submit_clean", "submit_with_records", "correct_and_resubmit",
                )):
                    applicable.discard("submit_clean")
                if "submit_with_records" in prior:
                    applicable.discard("submit_with_records")
        # the correctable-on-resubmission situation is the COB order
        # (CO-16 info requests route to provide_requested_info; pairing and
        # authorization cannot be fixed by editing claim fields)
        if not case_row["cob_position_ok"] and case_row["pairing_valid"]:
            applicable.add("correct_and_resubmit")
    if (
        case_row["auth_required"]
        and case_row["auth_status"] != "on_file"
        and case_row["retro_window_days"] >= case_row["days_since_service"]
    ):
        applicable.add("request_retro_auth")
    if carc:
        applicable.add("appeal_with_necessity")
        if case_row["p2p_available"]:
            applicable.add("request_peer_to_peer")
    if case_row["balance"] > 0 and case_row["touches_so_far"] >= 1:
        # a patient is only billed after the payer adjudicated at least once
        # (the corpus has zero touch-0 bill_patient states: outside support)
        applicable.add("bill_patient")
    if (
        case_row["has_secondary"]
        and case_row["touches_so_far"] >= 1
        and not carc
        and case_row["balance"] > 0
    ):
        applicable.add("bill_secondary")
    # contest decisions are seeded per EPISODE, not per touch (engine.py):
    # a repeated identical request gets the identical answer, so a contest
    # that already failed once is off the menu - same paperwork, same draw
    for act in (
        "request_retro_auth", "appeal_with_necessity",
        "request_peer_to_peer", "provide_requested_info",
    ):
        if act in prior:
            applicable.discard(act)
    return applicable


def score_spread(
    rt: BasisRuntime, features: dict[str, float], actions: set[str]
) -> dict[str, float] | None:
    """Fold-ensemble disagreement per action for one live case (calibrated
    scale), or None when the basis carries no fold models. Feeds
    RetrievalContext.scores_std: a recommendation margin smaller than this
    spread is inside the swarm's own noise and must be flagged ambiguous."""
    if not rt.fold_models:
        return None
    x = np.array([[features[n] for n in rt.feature_names]])
    spread: dict[str, float] = {}
    for action, models in rt.fold_models.items():
        if action not in actions or not models:
            continue
        cal = [
            float(rt.calibrators[action].predict([m.predict_proba(x)[0, 1]])[0])
            for m in models
        ]
        spread[action] = float(np.std(cal))
    return spread


def route_case(rt: BasisRuntime, case_row: dict, features: dict[str, float]):
    """Returns (scores_cal over applicable actions, known, forced_action,
    ambiguous_override).

    v0.3 (spec s.4 + s.5.2): resolution heads score "will this request be
    granted"; the ambiguity flag lives on resolution margins between
    applicable rework rivals; a rework action whose resolution probability
    beats every episode-success score is surfaced at its PATH VALUE
    (resolution x historical success-given-granted).

    forced_action: CO-16 names its responsive action; it resolves to the
    provide_requested_info skill (execute since v0.2, grounding-ruled)."""
    known = case_row["payer_id"] in rt.working_payers
    if (case_row.get("carc") or "") == "CO-16":
        # forced only on the FIRST attempt: the resolution draw is seeded per
        # episode, so a failed info request would fail identically again
        if "provide_requested_info" not in (
            case_row.get("prior_actions") or ""
        ).split(","):
            return {}, known, "provide_requested_info", None
    if case_row["cumulative_delay_days"] > 120:
        # the corpus outcome definition's success clock is blown: no action
        # can reach a successful outcome; a human decides the write-off
        return {}, known, None, None
    x = np.array([[features[n] for n in rt.feature_names]])
    applicable = applicable_actions(case_row)
    scores: dict[str, float] = {}
    if rt.decision_field == "kernel" and rt.kernel_field is not None:
        # v0.6: episode-success scores are neighbour counts at the live
        # coordinate (the Ether read as [1] defines it); Lambda still places
        wanted = {a for a in rt.trained_actions if a in applicable}
        scores.update(rt.kernel_field.scores(rt, x, wanted))
    else:
        for action in rt.trained_actions:
            if action not in applicable:
                continue
            raw = rt.refit_models[action].predict_proba(x)[0, 1]
            scores[action] = float(rt.calibrators[action].predict([raw])[0])
    for action, prior in rt.constant_priors.items():
        if action in applicable:
            scores[action] = prior

    # v0.3: resolution scores for applicable rework actions
    res_scores: dict[str, float] = {}
    for action, model in (rt.resolution_models or {}).items():
        if action in applicable:
            raw = model.predict_proba(x)[0, 1]
            res_scores[action] = float(
                rt.resolution_calibrators[action].predict([raw])[0]
            )
    # path-value substitution (spec 5.2): a rework whose resolution chance
    # beats every episode-success score is scored at its full path value
    if scores and res_scores:
        best_episode = max(scores.values())
        for action, p_res in res_scores.items():
            if p_res > best_episode:
                path_value = p_res * rt.success_given_granted.get(action, 0.0)
                scores[action] = max(scores.get(action, 0.0), path_value)
    # ambiguity on the resolution scale (spec s.4): two applicable rework
    # rivals within 0.05 of each other is a genuine coin flip
    ambiguous_override = None
    rivals = sorted(res_scores.values(), reverse=True)
    if len(rivals) >= 2:
        ambiguous_override = (rivals[0] - rivals[1]) < 0.05
    return scores, known, None, ambiguous_override


CLOSE_OUT_ACTIONS = ("bill_secondary", "bill_patient", "write_off")

# Limitations experiment (2026-08-19): AMMONIX_EV_CONSTANTS=estimated swaps the
# three shipped close-out constants for the values re-estimated by deterministic
# replay of the 5,000 recorded raw-tranche claims
# (runs/reports/ev_constants_estimate.json). Default: shipped v0.4 values,
# byte-identical behavior. Same opt-in pattern as AMMONIX_DECISION_FIELD.
EV_CONSTANTS = {
    "shipped": {"patient_small": 0.7, "patient_large": 0.4, "secondary": 0.9955},
    "estimated": {"patient_small": 0.6945, "patient_large": 0.3859, "secondary": 1.0},
}[os.environ.get("AMMONIX_EV_CONSTANTS", "shipped")]


def expected_values(case_row: dict, scores: dict) -> dict[str, float]:
    """v0.4 s.3: expected collected dollars per applicable action, under
    the outcome-v2 money rules. Honesty note (audit 2026-07-22): the three
    payment-probability constants below (0.7 / 0.4 patient pay, 0.9955
    secondary acceptance) are NOT "the world's published constants" as
    this docstring once claimed - they were copied from the simulator's
    hidden parameters (corpus.py bill_patient; corpus.py bill_secondary
    in fact pays unconditionally, so 0.9955 matches no simulator constant
    at all). Estimated empirically from the recorded raw-tranche episodes
    (5,000 replayed), all three fall inside Wilson 95% CIs (0.6945
    [0.675, 0.714] on n=2131; 0.3859 [0.346, 0.427] on n=552; 1.0
    [0.983, 1.0] on n=228) - so the values ARE recoverable from training
    data, what a billing operation would learn from its own ledger - but
    as shipped they were oracle-sourced. Full audit:
    runs/reports/ev_constants_estimate.json. Values deliberately
    unchanged: shipped v0.4 behavior is frozen. Patient and secondary
    dollars are capped at the contractual share; a standing denial makes
    secondary value zero (nothing is owed until adjudication says so -
    the ep-05004 lesson, encoded)."""
    from cardessa.engine import patient_share

    share = patient_share(
        float(case_row["allowed_amount"]),
        float(case_row["deductible_remaining"]),
        float(case_row["coinsurance_pct"]),
    )
    balance = float(case_row["balance"])
    denial_standing = bool(case_row.get("carc"))
    ev: dict[str, float] = {}
    for action, p in scores.items():
        if action == "write_off":
            ev[action] = 0.0
        elif action == "bill_patient":
            dollars = min(balance, share)
            pay_p = (EV_CONSTANTS["patient_small"] if dollars <= 300
                     else EV_CONSTANTS["patient_large"])
            ev[action] = round(pay_p * dollars, 2)
        elif action == "bill_secondary":
            dollars = 0.0 if denial_standing else min(balance, share)
            ev[action] = round(EV_CONSTANTS["secondary"] * dollars, 2)
        else:
            # payer-facing: success probability x outstanding dollars
            ev[action] = round(float(p) * balance, 2)
    return ev


def close_out_choice(ev: dict[str, float], scores: dict) -> str | None:
    """At a close-out moment (nothing clears the confidence floor), the
    best positive-EV close-out action; None when every option is worthless
    (then escalation remains the honest answer)."""
    close = {a: ev[a] for a in CLOSE_OUT_ACTIONS if a in scores}
    if not close:
        return None
    best = max(close, key=lambda a: (close[a], a))
    return best if close[best] > 0.005 else None


def label_coordinate(rt, scores_cal: dict) -> "np.ndarray":
    """v0.4 s.3b: the live case's Universe coordinate u = calibrated score
    vector over the trained actions, in the index's stored order."""
    import numpy as np

    return np.array([float(scores_cal.get(a, 0.0)) for a in rt.u_order])


@dataclass
class M1Provider:
    """Pinned-model-only guided-JSON generation with a prompt-hash cache."""

    cache_dir: Path
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = 400
    model_id: str = field(default="", init=False)
    calls_to_provider: int = 0

    def __post_init__(self) -> None:
        with urllib.request.urlopen(f"{self.base_url}/models", timeout=10) as response:
            served = json.loads(response.read())["data"]
        ids = [entry["id"] for entry in served]
        matches = [i for i in ids if PINNED_MODEL_SUBSTRING in i]
        if not matches:
            raise RuntimeError(
                f"vLLM at {self.base_url} serves {ids}, not the pinned model."
            )
        self.model_id = matches[0]

    def generate(self, prompt: str, schema: dict) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = sha256_text(prompt + json.dumps(schema, sort_keys=True))
        cached = self.cache_dir / f"{key}.json"
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                # this vLLM ignores the legacy guided_json param SILENTLY;
                # response_format json_schema is the enforced path (verified
                # live: guided_json let "cpt_code" through, this does not)
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "payload", "schema": schema, "strict": True},
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        body = None
        for attempt in range(3):  # transient server stalls: retry, same request
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    body = json.loads(response.read())
                break
            except (TimeoutError, OSError):
                if attempt == 2:
                    raise
        assert body is not None
        content = body["choices"][0]["message"]["content"]
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        content = content.strip()
        cached.write_text(content, encoding="utf-8")
        self.calls_to_provider += 1
        return content


def prompt_fields(case_row: dict, retrieval) -> dict:
    """Cardessa M1 prompt fields (moved verbatim from the generic runtime in
    v0.4 so the core stays domain-free; the rendered prompt bytes - and hence
    the M1 prompt-hash cache - are unchanged). Pass to run_case."""
    return {
        "action_id": retrieval.recommended_action_id,
        "episode_id": case_row.get("episode_id", ""),
        "cpt": case_row.get("cpt", ""),
        "dx_codes": case_row.get("dx_codes", ""),
        "carc": case_row.get("carc") or "none",
        "payer_id": case_row.get("payer_id", ""),
        "balance": case_row.get("balance", ""),
        "days_since_service": case_row.get("days_since_service", ""),
        "days_to_filing_deadline": case_row.get("days_to_filing_deadline", ""),
        "retro_window_days": case_row.get("retro_window_days", ""),
        "auth_status": case_row.get("auth_status", ""),
        "correspondence": (case_row.get("payer_correspondence_text") or "")[:800],
        "indication": (case_row.get("clinical_indication_text") or "")[:600],
    }


def make_rule_evaluator():
    """The M2 rule evaluator: the deterministic predicates of payloads.py.
    Code only; nothing in the check depends on a model."""
    from cardessa.payloads import evaluate_rule

    return evaluate_rule


def make_llm_pin(shard_hashes: dict[str, str]) -> LLMPin:
    combined = sha256_text(
        "".join(shard_hashes[k] for k in sorted(shard_hashes))
    )
    return LLMPin(
        name="Qwen3.6-27B-AWQ-INT4",
        weights_sha256=combined,
        quantisation="AWQ-INT4",
        temperature=0.0,
        constrained_decoding=True,
    )
