"""Case briefing: everything the agent may see, nothing it may not.

Input is the factory's `state_tabular` snapshot (the admin-visible channel).
Only genuinely post-hoc/outcome fields are dropped; no engine-truth or
post-hoc field ever reaches the prompt. Fairness contract (v0.4 parity
audit): the briefing renders every decision-time fact the shipped Ammonix
basis' 83 features read — including persona_id and subscriber_relationship,
which the tranche-92 run wrongly withheld from the agent.
Descriptions are factual/neutral — applicability facts, no strategy hints.
"""

# snapshot fields the agent must never see -> one-word reason.
# (engine_truth_* fields never enter state_tabular; listed defensively.)
EXCLUDED_FIELDS = {
    "days_to_payment": "post-hoc",  # also the planted poisoned column (P2)
    "paid_amount_final": "outcome",
    "total_touches": "outcome",
}

CARC_LEGEND = {
    "CO-197": "prior authorization missing at the time of service",
    "CO-16": "claim lacks information; the payer requested a specific document",
    "CO-97": "service bundled into another paid service (not separately payable)",
    "CO-50": "not deemed medically necessary by the payer",
    "CO-22": "coordination of benefits: another payer is primary for this claim",
    "CO-29": "timely filing window expired",
    "CO-18": "exact duplicate: this claim was already adjudicated and paid",
    "PR-204": "service not covered under the patient's plan",
}

ACTION_CATALOG = {
    "submit_clean": "submit the claim to the payer as-is",
    "submit_with_records": "submit the claim with supporting medical records attached",
    "request_retro_auth": (
        "ask the payer for retroactive prior authorization "
        "(only possible while the payer's retro-auth window is open "
        "and the required authorization is not on file)"
    ),
    "correct_and_resubmit": (
        "fix claim data (e.g. billing order / codes) and resubmit"
    ),
    "appeal_with_necessity": (
        "file a formal appeal with a medical-necessity argument "
        "(requires a denial on the table)"
    ),
    "request_peer_to_peer": (
        "request a clinician-to-medical-director review "
        "(requires a denial on the table and a payer that offers P2P)"
    ),
    "provide_requested_info": (
        "send the specific document the payer asked for "
        "(only meaningful on a CO-16 information request)"
    ),
    "bill_secondary": "bill the patient's secondary payer for the open balance",
    "bill_patient": "bill the remaining balance to the patient",
    "write_off": "close the remaining balance as uncollectible",
    "escalate": "hand this touch to a human billing specialist",
}

SYSTEM_PROMPT = (
    "You are an autonomous revenue-cycle agent for a cardiac monitoring "
    "provider. You work ONE claim touch at a time: given the case briefing, "
    "decide the single next action.\n"
    "Published objective: an episode succeeds when at least 98% of the "
    "allowed amount is collected within 120 days of first submission. Every "
    "extra touch costs score, and actions that violate the payer's published "
    "rules at decision time are penalised as process mistakes. Patient money "
    "beyond the contractual patient share does not count.\n"
    "Escalation hands the touch to a scarce human specialist: use it only "
    "when no available action can properly progress the claim.\n"
    "Reason step by step about the payer's published rules and the case "
    "facts, in 2-4 sentences, then commit to exactly one action. "
    "Respond with JSON only."
)


# The tuned prompt (pre-registered as sol_tuning winner P2, 2026-08-19):
# corrected success threshold plus close-out and payer-weighting guidance.
TUNED_SYSTEM_PROMPT = SYSTEM_PROMPT.replace("at least 98%", "at least 90%") + (
    "\nWhen the payer path is exhausted, money the patient or a secondary "
    "plan properly owes should be billed, not written off: write off only "
    "when nothing is owed or nothing is collectible."
    "\nPayers differ in which rework succeeds. When several rework actions "
    "are applicable, weight the SIMILAR PAST CLAIMS tally heavily: prefer "
    "the action that actually got similar claims paid at this payer."
)


def decision_schema(actions: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "action": {"type": "string", "enum": list(actions)},
        },
        "required": ["reasoning", "action"],
        "additionalProperties": False,
    }


def _money(x) -> str:
    return f"${float(x):,.2f}"


def build_briefing(snap: dict, legal_actions: list[str]) -> str:
    """The user-turn briefing for one decision, from an admin-visible snapshot.

    Snapshots may carry EXCLUDED_FIELDS; this renderer never reads them.
    """
    retro_open = (
        snap["retro_window_days"] > 0
        and snap["days_since_service"] <= snap["retro_window_days"]
    )
    lines = [
        f"CASE (touch {snap['touches_so_far']}):",
        f"- procedure CPT {snap['cpt']}, diagnoses {snap['dx_codes']} "
        f"(payer considers this pairing {'valid' if snap['pairing_valid'] else 'INVALID'})",
        f"- allowed amount {_money(snap['allowed_amount'])}, open balance {_money(snap['balance'])}",
        f"- patient plan: deductible remaining {_money(snap['deductible_remaining'])}, "
        f"coinsurance {round(snap['coinsurance_pct'] * 100)}%; "
        f"patient's relationship to plan subscriber: {snap['subscriber_relationship']}",
        f"- eligibility {snap['eligibility_status']}; assignment of benefits "
        f"{'on file' if snap['aob_on_file'] else 'NOT on file'}",
        f"- secondary payer: {'yes' if snap['has_secondary'] else 'none'}",
        f"- billing (COB) order: {'correct' if snap['cob_position_ok'] else 'WRONG - another payer is primary'}",
        f"- days since service: {snap['days_since_service']}; "
        f"days since first submission: {snap['cumulative_delay_days']}",
        f"- clinic documentation quality: {snap['clinic_doc_quality']:.2f} (0-1)",
        f"- billing specialist assigned to this account (works escalated "
        f"touches): {snap['persona_id']}",
        "",
        f"PAYER {snap['payer_id']} ({snap['payer_archetype']} archetype), published rules:",
        f"- timely filing: {snap['timely_filing_days']} days after service "
        f"({snap['days_to_filing_deadline']} days left)",
        f"- prior authorization for this CPT: "
        f"{'REQUIRED' if snap['auth_required'] else 'not required'}; "
        f"authorization currently {snap['auth_status'].upper()}",
        "- retroactive authorization window: "
        + (
            f"{snap['retro_window_days']} days after service "
            f"({'OPEN' if retro_open else 'CLOSED'} now)"
            if snap["retro_window_days"] > 0
            else "not offered by this payer"
        ),
        f"- peer-to-peer review offered: {'yes' if snap['p2p_available'] else 'no'}",
        "- duplicate claims: a claim already adjudicated and paid must not be "
        "resubmitted (a resubmission is denied CO-18)",
    ]
    carc = snap.get("carc") or ""
    lines.append("")
    if carc:
        meaning = CARC_LEGEND.get(carc, "unlisted reason code")
        lines.append(f"PAYER RESPONSE ON THE TABLE: denial/notice {carc} = {meaning}.")
    else:
        lines.append("PAYER RESPONSE ON THE TABLE: none.")
    prior_actions = snap.get("prior_actions") or ""
    prior_carcs = snap.get("prior_carcs") or ""
    lines.append(
        f"HISTORY: actions so far [{prior_actions or 'none'}]; "
        f"prior payer codes [{prior_carcs or 'none'}]."
    )
    if prior_carcs:
        seen = [c for c in dict.fromkeys(prior_carcs.split(",")) if c and c != carc]
        for code in seen:
            if code in CARC_LEGEND:
                lines.append(f"  ({code} = {CARC_LEGEND[code]})")
    lines.append("")
    lines.append("AVAILABLE ACTIONS (choose exactly one):")
    for action in [*legal_actions, "escalate"]:
        lines.append(f"- {action}: {ACTION_CATALOG[action]}")
    return "\n".join(lines)
