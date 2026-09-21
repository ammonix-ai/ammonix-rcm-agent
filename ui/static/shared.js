/* Plain-language vocabulary shared by every screen. */
const ACTION_LABELS = {
  submit_clean: 'Submit the claim',
  submit_with_records: 'Submit with medical records',
  correct_and_resubmit: 'Correct the claim and resubmit',
  bill_secondary: 'Bill the secondary insurer',
  request_retro_auth: 'Request retroactive authorization',
  appeal_with_necessity: 'Appeal with a medical-necessity letter',
  request_peer_to_peer: 'Request a peer-to-peer review',
  provide_requested_info: 'Send the information the payer asked for',
  bill_patient: 'Bill the patient',
  write_off: 'Write off the balance',
};

const STATUS_LABELS = {
  executed: 'Ready to send',
  escalated: 'With a human',
  adjudicated: 'Adjudicated',
  sent_to_human: 'With a human',
};

const ESCALATION_LABELS = {
  needs_outside_information: 'Payer never seen in training — needs a human',
  unresolvable: 'No path can resolve this — needs a human',
  iteration_cap: 'Paperwork failed its checks — needs a human',
  low_confidence: 'Confidence too low — needs a human',
};

const FACT_LABELS = {
  dx_codes: 'Diagnosis codes',
  days_since_service: 'Days since the study',
  days_to_filing_deadline: 'Days left to file',
  auth_status: 'Prior authorization',
  retro_window_days: 'Retro-auth window (days)',
  touches_so_far: 'Attempts so far',
  prior_actions: 'Previous actions',
  prior_carcs: 'Previous denial codes',
  allowed_amount: 'Allowed amount',
};

const RESOLUTION_LABELS = {
  paid: 'Paid',
  paid_on_appeal: 'Paid on appeal',
  paid_with_secondary: 'Paid with secondary',
  patient_billed: 'Patient billed',
  written_off: 'Written off',
  written_off_at_cap: 'Written off (gave up)',
};
function resolutionLabel(r) { return RESOLUTION_LABELS[r] || prettify(r); }

// one-line plain meaning for each end state, shown on hover
const RESOLUTION_PLAIN = {
  paid: 'the insurer paid its share of the claim',
  paid_on_appeal: 'the insurer paid after its denial was contested',
  paid_with_secondary: 'a second insurer covered part of the balance',
  patient_billed: "the remaining balance became the patient's bill",
  written_off: 'the balance was abandoned — nobody is billed, the clinic takes the loss',
  written_off_at_cap: 'gave up after too many attempts — the clinic takes the loss',
};
function resolutionAttr(r) {
  return RESOLUTION_PLAIN[r] ? ` title="${esc(RESOLUTION_PLAIN[r])}"` : '';
}
function laneOutcomeAttr(l) {
  if (l && l.resolution === 'paid_with_secondary' && !(l.collected > 0)) {
    return ' title="the second insurer was billed but owed nothing on this claim"';
  }
  return resolutionAttr(l && l.resolution);
}
function laneOutcomeLabel(l) {
  // a success-sounding resolution with zero dollars is an abandonment
  if (l && l.resolution === 'paid_with_secondary' && !(l.collected > 0)) {
    return 'Closed with nothing — secondary had nothing to pay';
  }
  return resolutionLabel(l.resolution);
}

// plain names for the paperwork rules (M2 checks)
const RULE_PLAIN = {
  cpt_matches_case: 'billing code matches the case',
  pairing_legal: 'diagnosis fits the procedure',
  filing_window_open: 'filing deadline not passed',
  retro_window_open: 'retro-auth window still open',
  auth_actually_missing: 'authorization really is missing',
  addresses_current_carc: 'fix addresses the denial reason',
  cites_denial_carc: 'letter cites the denial code',
  paragraph_grounded_in_case: 'necessity paragraph grounded in the record',
  letter_names_requested_item: 'letter names the requested item',
  requested_item_in_record: 'requested document exists in the record',
  amount_within_balance: 'amount within the outstanding balance',
  p2p_offered: 'payer offers peer-to-peer review',
  secondary_exists: 'a secondary insurer exists',
};
function rulePlain(r) { return RULE_PLAIN[r] || prettify(r); }

// short plain-language reason for each denial code, shown on hover
const CARC_PLAIN = {
  'CO-197': 'no prior authorization was on file',
  'CO-50': 'payer says it was not medically necessary',
  'CO-16': 'claim is missing information or has an error',
  'CO-97': 'payment was bundled into another service',
  'CO-29': 'filed after the deadline',
  'PR-204': 'not covered by this plan',
  'CO-22': 'another insurer should pay first',
  'CO-18': 'duplicate of an already-paid claim',
};
function carcAttr(text) {
  const m = String(text || '').match(/(?:CO|PR)-\d+/);
  return m && CARC_PLAIN[m[0]] ? ` title="${esc(m[0])}: ${esc(CARC_PLAIN[m[0]])}"` : '';
}

// the four study types (real CPT codes), plain meaning shown on hover
const CPT_PLAIN = {
  '93229': 'mobile cardiac telemetry: a wearable heart monitor streaming continuously for 2 to 4 weeks',
  '93271': 'cardiac event monitoring: a wearable recorder the patient triggers when symptoms strike',
  '93247': 'long-term ECG: a stick-on heart recorder worn 1 to 2 weeks',
  '93226': 'Holter monitor: a 24 to 48 hour heart recording',
};
function cptAttr(c) {
  return CPT_PLAIN[c] ? ` title="${esc(c)}: ${esc(CPT_PLAIN[c])}" style="cursor:help"` : '';
}

// action families share the universe's colors everywhere in the app
const ACTION_FAMILY_UI = {
  submit_clean: 'submit', submit_with_records: 'submit',
  correct_and_resubmit: 'submit', bill_secondary: 'submit',
  request_retro_auth: 'contest', appeal_with_necessity: 'contest',
  request_peer_to_peer: 'contest', provide_requested_info: 'contest',
  bill_patient: 'close', write_off: 'close',
};
const FAMILY_UI_COLOR = {submit: '#2a78d6', contest: '#008300', close: '#e87ba4'};
function actionColor(a) {
  return FAMILY_UI_COLOR[ACTION_FAMILY_UI[a]] || 'var(--ink-soft)';
}
function actionDot(a) {
  return `<span class="dot" style="background:${actionColor(a)}"></span>`;
}
// one-line plain meaning for each move, shown on hover
const ACTION_PLAIN = {
  submit_clean: 'send the claim to the insurer and wait for its decision',
  submit_with_records: 'send the claim with the medical records already attached',
  correct_and_resubmit: 'fix the error on the claim and send it again',
  bill_secondary: "send the unpaid part to the patient's second insurer",
  request_retro_auth: 'ask the insurer to grant the missing authorization after the fact',
  appeal_with_necessity: 'contest the denial with a letter arguing the care was medically needed',
  request_peer_to_peer: "book a doctor-to-doctor call with the insurer's reviewer",
  provide_requested_info: 'send the document or detail the insurer asked for',
  bill_patient: 'send the remaining balance to the patient as their bill',
  write_off: 'stop chasing the money — the clinic absorbs the balance as a loss',
};
function actionAttr(a) {
  return ACTION_PLAIN[a] ? ` title="${esc(ACTION_PLAIN[a])}"` : '';
}
function actionSpan(a) {
  return `<b style="color:${actionColor(a)}"${actionAttr(a)}>${esc(actionLabel(a))}</b>`;
}

const PERSONA_NOTES = {
  diligent: 'Careful biller (40% of claims). Fixes a missing authorization before submitting while the window is open, attaches records when documentation is weak, uses peer-to-peer review and secondary billing correctly. Slips into a random mistake about 5% of the time.',
  hasty: 'Rushed biller (35% of claims). Submits without checking authorization, resubmits unchanged once after a denial, then appeals whether or not the appeal makes sense.',
  conservative: 'Risk-averse biller (25% of claims). Requests records excessively up front, and gives up on hard denials: writes off balances over $300 instead of fighting, bills the patient below that.',
};
function personaNote(id) { return PERSONA_NOTES[id] || ''; }

function llmBeatsAmmonix(c) {
  // the same substance rule the scoreboard uses, client-side
  if (!c.llm) return false;
  const delta = c.llm.collected - c.system.collected;
  if (Math.abs(delta) > 0.005) return delta > 0;
  return c.llm.mistakes < c.system.mistakes;
}

function verdictPillHtml(c) {
  // money wording only when money actually differed; tie-breaks say why
  if (c.verdict === 'same') {
    const abandoned = c.human && c.system
      && String(c.human.resolution).startsWith('written_off')
      && String(c.system.resolution).startsWith('written_off');
    return `<span class="pill plain">same${abandoned ? ' · both abandoned' : ''}</span>`;
  }
  const who = c.verdict === 'ammonix' ? 'Ammonix' : 'Biller';
  const cls = c.verdict === 'ammonix' ? 'green' : 'red';
  if (Math.abs(c.delta) > 0.005) {
    const amt = c.verdict === 'ammonix' ? c.delta : -c.delta;
    return `<span class="pill ${cls}">${who} +${money(amt)}</span>`;
  }
  return `<span class="pill ${cls}">${who} · fewer mistakes</span>`;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}
function actionLabel(a) { return ACTION_LABELS[a] || prettify(a); }
function prettify(s) {
  s = String(s ?? '').replaceAll('_', ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function money(v) {
  return '$' + Number(v).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
function pct(v) { return Math.round(Number(v) * 100) + '%'; }
function statusLabel(c) {
  if (c.status === 'escalated' || c.status === 'sent_to_human') {
    return ESCALATION_LABELS[c.escalation_reason] || STATUS_LABELS[c.status];
  }
  return STATUS_LABELS[c.status] || prettify(c.status);
}

/* Paper-document renderer, shared by case.html (Ammonix moments) and
   claim.html (the LLM lane's filings). `art` is the artifact object the
   server ships: {kind, field_map, gaps}. */
/* A field whose source is the action payload was WRITTEN by the writer
   model; every other field is pulled from the case record by the harness
   tools. The document renders that split visibly (owner rule: input vs
   agent-written must be unmistakable). */
function isModelField(entry) { return entry.source_table === 'action_object'; }

function sourceSpan(entry) {
  const cols = Array.isArray(entry.source_column)
    ? entry.source_column.join('+') : entry.source_column;
  const gap = entry.value === '' || entry.value == null;
  const val = gap ? '<span class="gapfield">— left empty —</span>' : esc(entry.value);
  const model = isModelField(entry);
  return `<span${model ? ' class="modelfield"' : ''} data-source-table="${esc(entry.source_table)}"
    data-source-column="${esc(cols)}" data-row-id="${esc(entry.row_id)}"
    title="${model ? 'written by the writer model'
      : `from the case record: ${esc(entry.source_table)}.${esc(cols)} (row ${esc(entry.row_id)})`}">${val}</span>`;
}

function buildDocument(art) {
  const kinds = {
    cms1500: {title: 'Health Insurance Claim Form', sub: 'CMS-1500 · simulated', cls: 'doc cms'},
    prior_auth_request: {title: 'Retroactive Authorization Request', sub: 'payer decision request · simulated', cls: 'doc'},
    appeal_letter: {title: 'Appeal of Claim Denial', sub: 'formal appeal · simulated', cls: 'doc'},
  };
  const k = kinds[art.kind] || {title: prettify(art.kind), sub: 'simulated', cls: 'doc'};
  const prose = [];
  const rows = [];
  for (const entry of art.field_map) {
    if (entry.field === 'necessity_paragraph' && art.kind === 'appeal_letter') {
      prose.push(`<div class="prose"><h4>Statement of medical necessity — <span class="modeltag">written by the writer model</span></h4>${sourceSpan(entry)}</div>`);
      continue;
    }
    if (entry.field === 'clinical_indication' && art.kind === 'prior_auth_request') {
      prose.push(`<div class="prose"><h4>Clinical indication — quoted verbatim from the case record</h4>${sourceSpan(entry)}</div>`);
      continue;
    }
    const gap = entry.value === '' || entry.value == null;
    rows.push(`<tr${gap ? ' class="gaprow"' : ''}>
      <td class="lbl">${esc(prettify(entry.field))}</td>
      <td class="val">${sourceSpan(entry)}</td></tr>`);
  }
  const gapNote = art.gaps.length
    ? `<p class="sim-note">No source, left empty: ${art.gaps.map(prettify).join(', ')}.</p>`
    : '';
  const anyModel = art.field_map.some(isModelField);
  const legend = anyModel
    ? '<p class="doc-legend"><span class="swatch modelfield">written by the model</span> · the rest comes from the case record</p>'
    : '';
  return `<div class="${k.cls}">
    <p class="doc-title">${esc(k.title)}</p><p class="doc-sub">${esc(k.sub)}</p>
    ${legend}
    <table>${rows.join('')}</table>${prose.join('')}${gapNote}
    <p class="sim-note">Simulated; no real patient, payer or provider exists.</p>
  </div>`;
}
