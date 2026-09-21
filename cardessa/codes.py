"""Real public code sets used by the simulation (corpus spec Section 1).

CPT and ICD-10 codes are real so the demo is credible to healthcare insiders;
allowed-amount ranges are invented.
"""

# Study types: CPT -> (label, invented allowed range USD, monitoring days range)
STUDIES = {
    "93229": ("MCT", (550.0, 950.0), (14, 30)),  # mobile cardiac telemetry, technical
    "93271": ("CEM", (210.0, 420.0), (14, 30)),  # cardiac event monitoring
    "93247": ("LTM", (240.0, 480.0), (7, 15)),  # long-term ECG 7-15 days
    "93226": ("HOLTER", (90.0, 180.0), (1, 2)),  # Holter up to 48h
}
CPT_CODES = tuple(STUDIES)

# Diagnosis pool (real ICD-10); each patient carries 2-3
DX_POOL = (
    "I48.0",  # paroxysmal AF
    "I48.91",  # AF unspecified
    "I47.1",  # SVT
    "I49.5",  # sick sinus syndrome
    "I49.9",  # arrhythmia unspecified
    "R00.2",  # palpitations
    "R55",  # syncope
    "R42",  # dizziness
    "G45.9",  # TIA (cryptogenic stroke workup)
    "I25.10",  # CAD
    "I34.0",  # mitral insufficiency
    "Z86.73",  # personal history of TIA/stroke
)

# Dx that justify each study under strict (LCD-style) code pairing.
# Arrhythmia/syncope/TIA diagnoses justify continuous monitoring; palpitations
# or dizziness alone justify only the short studies.
_ARRHYTHMIA_DX = {"I48.0", "I48.91", "I47.1", "I49.5", "I49.9", "R55", "G45.9", "Z86.73"}
COVERED_DX = {
    "93229": _ARRHYTHMIA_DX,  # MCT: strongest indication required
    "93247": _ARRHYTHMIA_DX | {"R00.2", "R42"},
    "93271": _ARRHYTHMIA_DX | {"R00.2", "R42"},
    "93226": set(DX_POOL) - {"I25.10", "I34.0"},  # Holter: any rhythm-adjacent complaint
}

DX_NAMES = {
    "I48.0": "paroxysmal atrial fibrillation",
    "I48.91": "atrial fibrillation, unspecified",
    "I47.1": "supraventricular tachycardia",
    "I49.5": "sick sinus syndrome",
    "I49.9": "cardiac arrhythmia, unspecified",
    "R00.2": "palpitations",
    "R55": "syncope and collapse",
    "R42": "dizziness and giddiness",
    "G45.9": "transient cerebral ischemic attack, unspecified",
    "I25.10": "atherosclerotic heart disease of native coronary artery",
    "I34.0": "nonrheumatic mitral valve insufficiency",
    "Z86.73": "personal history of TIA and cerebral infarction",
}

# CARC denial codes (real) used by the payer engine
CARC = {
    "CO-197": "Precertification/authorization absent",
    "CO-50": "Not deemed a medical necessity by the payer",
    "CO-16": "Claim lacks information or has submission error",
    "CO-97": "Payment bundled into another service",
    "CO-29": "Time limit for filing has expired",
    "PR-204": "Service not covered under the patient's current benefit plan",
    "CO-22": "Care may be covered by another payer per coordination of benefits",
    "CO-18": "Exact duplicate claim or service",
}
