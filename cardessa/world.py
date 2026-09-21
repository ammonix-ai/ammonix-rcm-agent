"""World generation (corpus spec Sections 1.4 and 2): 25 clinics, exactly
1,000 patients with coverage records, and the initial studies, all drawn from
the master seed and regenerate-identical.

Split assignment (quarantine A / reserve B / working) is fixed here, once, at
world creation: all patients of the two holdout payers go to quarantine A by
construction, and every future episode inherits its patient's side, so the
vault wall is never crossed. Curated scenario families are marked on studies
so the corpus generator (W2) can guarantee coverage of rare actions.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from ammonix_core.hashing import sha256_json

from cardessa.codes import COVERED_DX, DX_POOL, STUDIES
from cardessa.engine import stable_seed
from cardessa.payers import PayerConfig, build_payer_configs

N_CLINICS = 25
N_PATIENTS = 1000

QUARANTINE_A_TARGET = 150  # 15% of patients, includes ALL holdout-payer patients
RESERVE_B_TARGET = 100  # 10% sealed reserve

# payer_id -> share of patients (holdouts small; all their patients go to A)
PAYER_MIX = {
    "meridian": 0.14,
    "atlas": 0.10,
    "cornerstone": 0.10,
    "medicare": 0.20,
    "silverbridge": 0.08,
    "lakeshore": 0.07,
    "prairie": 0.07,
    "harborview": 0.07,
    "bluesummit": 0.08,
    "keystone": 0.05,
    "granite": 0.02,
    "pelican": 0.02,
}

# curated family -> target study count (30% of the initial 1000)
FAMILY_TARGETS = {
    "retro_auth": 80,  # the P1 trap family: MCT at Meridian/SilverBridge, auth missing
    "p2p": 60,  # Cornerstone bundling denials
    "cob": 60,  # secondary-coverage episodes
    "medicaid_enrollment": 50,  # Harborview + non-enrolled ordering clinic
    "timely_filing": 50,  # near-miss filing windows
}

_FIRST_NAMES = (
    "Ana Bruno Carla Dmitri Elena Farid Greta Hugo Iris Jonas Katja Liam Mara Nils "
    "Oona Pavel Quinn Rosa Sven Tessa Umar Vera Wim Xenia Yusuf Zoe Adam Bea Carl "
    "Dora Emil Freya Gil Hana Ivo Judit Kai Lena Marc Nora"
).split()
_LAST_NAMES = (
    "Albrecht Baumann Castellan Dorn Eberhart Falk Gruber Hartmann Imhof Jansen "
    "Keller Lindqvist Moreau Novak Oberli Petrov Quist Rossi Steiner Tanner Urban "
    "Vogel Weber Xanthos Ypsilanti Zimmermann Arnold Brunner Conti Dietrich Egger "
    "Fischer Graf Huber Isler Jung Koch Lang Maier Nussbaum"
).split()
_CITIES = ("Fairview", "Lakewood", "Riverton", "Cedar Falls", "Brookfield")
_STREETS = ("Maple Ave", "Oak St", "Harbor Rd", "Summit Blvd", "Willow Ln")
_CLINIC_STEMS = (
    "Northgate", "Riverbend", "Summit", "Lakeside", "Cedar Grove", "Fairmont",
    "Harborview", "Westfield", "Eastbrook", "Oakland", "Pinecrest", "Silver Creek",
    "Maple Ridge", "Stonebridge", "Clearwater", "Highland", "Brookstone", "Redwood",
    "Meadowbrook", "Ironwood", "Bayside", "Foxglove", "Larkspur", "Amberfield",
    "Crestline",
)


@dataclass
class World:
    payers: list[PayerConfig]
    clinics: pl.DataFrame
    patients: pl.DataFrame
    coverage: pl.DataFrame
    studies: pl.DataFrame

    def content_sha256(self) -> str:
        payload = {
            "payers": [p.model_dump() for p in sorted(self.payers, key=lambda p: p.payer_id)],
            "clinics": self.clinics.rows(named=True),
            "patients": self.patients.rows(named=True),
            "coverage": self.coverage.rows(named=True),
            "studies": self.studies.rows(named=True),
        }
        return sha256_json(_dates_to_iso(payload))


def _dates_to_iso(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _dates_to_iso(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dates_to_iso(v) for v in obj]
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _generate_clinics(rng: np.random.Generator) -> pl.DataFrame:
    rows = []
    for i, stem in enumerate(_CLINIC_STEMS):
        rows.append(
            {
                "clinic_id": f"cl-{i:03d}",
                "name": f"{stem} Cardiology Associates",
                "npi": f"9{rng.integers(10**8, 10**9 - 1):09d}",  # 9-prefixed: clearly invented
                "medicaid_enrolled": bool(i % 5 != 0),  # 5 of 25 not enrolled
                "doc_quality": round(float(rng.uniform(0.3, 0.95)), 3),
            }
        )
    return pl.DataFrame(rows)


def _draw_diagnoses(rng: np.random.Generator, weak_indication: bool) -> list[str]:
    """2-3 dx per patient; weak_indication patients carry no MCT-justifying code."""
    strong = sorted(COVERED_DX["93229"])
    weak = sorted(set(DX_POOL) - COVERED_DX["93229"])
    n = int(rng.integers(2, 4))
    if weak_indication:
        picks = rng.choice(len(weak), size=min(n, len(weak)), replace=False)
        return [weak[i] for i in sorted(picks)]
    first = strong[int(rng.integers(len(strong)))]
    rest_pool = sorted(set(DX_POOL) - {first})
    picks = rng.choice(len(rest_pool), size=n - 1, replace=False)
    return [first] + [rest_pool[i] for i in sorted(picks)]


def _generate_patients(
    rng: np.random.Generator, payer_ids: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    mix_payers = list(PAYER_MIX)
    mix_probs = np.array([PAYER_MIX[p] for p in mix_payers])
    mix_probs = mix_probs / mix_probs.sum()
    assert set(mix_payers) == set(payer_ids)

    patient_rows, coverage_rows = [], []
    for i in range(N_PATIENTS):
        patient_id = f"pt-{i:04d}"
        payer_id = mix_payers[int(rng.choice(len(mix_payers), p=mix_probs))]
        weak_indication = rng.random() < 0.15
        first = _FIRST_NAMES[int(rng.integers(len(_FIRST_NAMES)))]
        last = _LAST_NAMES[int(rng.integers(len(_LAST_NAMES)))]
        dependent = rng.random() < 0.25
        annual_deductible = float(rng.choice([0.0, 250.0, 500.0, 1000.0, 2500.0]))
        patient_rows.append(
            {
                "patient_id": patient_id,
                "first_name": first,
                "last_name": last,
                "dob": date(
                    int(rng.integers(1936, 1991)),
                    int(rng.integers(1, 13)),
                    int(rng.integers(1, 29)),
                ),
                "sex": "F" if rng.random() < 0.5 else "M",
                "address_street": f"{int(rng.integers(1, 999))} "
                f"{_STREETS[int(rng.integers(len(_STREETS)))]}",
                "address_city": _CITIES[int(rng.integers(len(_CITIES)))],
                "address_state": "IL",
                "address_zip": f"6{int(rng.integers(10**3, 10**4 - 1)):04d}",
                "diagnoses": _draw_diagnoses(rng, weak_indication),
                "stability": round(float(rng.uniform(0.0, 0.12)), 4),
                "payer_id": payer_id,
            }
        )
        has_secondary = rng.random() < 0.20
        coverage_rows.append(
            {
                "patient_id": patient_id,
                "payer_id": payer_id,
                "plan_variant": "plus" if rng.random() < 0.4 else "standard",
                "member_id": f"M{rng.integers(10**7, 10**8 - 1):08d}",
                "subscriber_name": f"{first} {last}"
                if not dependent
                else f"{_FIRST_NAMES[int(rng.integers(len(_FIRST_NAMES)))]} {last}",
                "subscriber_relationship": "self"
                if not dependent
                else ("spouse" if rng.random() < 0.5 else "child"),
                "aob_on_file": bool(rng.random() >= 0.02),  # ~2% missing: denial pathway
                "annual_deductible": annual_deductible,
                "deductible_remaining": round(
                    float(rng.uniform(0.0, annual_deductible)), 2
                ),
                "coinsurance_pct": float(rng.choice([0.0, 0.1, 0.2])),
                "copay": float(rng.choice([0.0, 25.0, 40.0])),
                "secondary_payer_id": (
                    [p for p in ("prairie", "bluesummit", "keystone") if p != payer_id][
                        int(rng.integers(2))
                    ]
                    if has_secondary
                    else None
                ),
                "has_medicare_primary": payer_id in ("prairie", "harborview")
                and rng.random() < 0.3,
            }
        )
    patients = pl.DataFrame(patient_rows)
    coverage = pl.DataFrame(coverage_rows)
    return patients, coverage


def _assign_splits(rng: np.random.Generator, patients: pl.DataFrame) -> pl.DataFrame:
    """Fixed once at world creation. All holdout-payer patients -> quarantine A."""
    holdout_mask = patients["payer_id"].is_in(["granite", "pelican"])
    holdout_ids = patients.filter(holdout_mask)["patient_id"].to_list()
    others = patients.filter(~holdout_mask)["patient_id"].to_list()
    order = rng.permutation(len(others))
    shuffled = [others[i] for i in order]
    n_more_a = max(0, QUARANTINE_A_TARGET - len(holdout_ids))
    quarantine_a = set(holdout_ids) | set(shuffled[:n_more_a])
    reserve_b = set(shuffled[n_more_a : n_more_a + RESERVE_B_TARGET])
    split = [
        "quarantine_a"
        if pid in quarantine_a
        else ("reserve_b" if pid in reserve_b else "working")
        for pid in patients["patient_id"].to_list()
    ]
    return patients.with_columns(pl.Series("split", split))


def _pick_family_members(
    rng: np.random.Generator,
    patients: pl.DataFrame,
    coverage: pl.DataFrame,
) -> dict[str, str]:
    """patient_id -> family, disjoint, per FAMILY_TARGETS, from eligible pools."""
    payer_of = dict(
        zip(patients["patient_id"].to_list(), patients["payer_id"].to_list(), strict=True)
    )
    secondary = {
        r["patient_id"]
        for r in coverage.rows(named=True)
        if r["secondary_payer_id"] is not None
    }
    pools = {
        "retro_auth": [p for p, payer in payer_of.items() if payer in ("meridian", "silverbridge")],
        "p2p": [p for p, payer in payer_of.items() if payer == "cornerstone"],
        "cob": sorted(secondary),
        "medicaid_enrollment": [p for p, payer in payer_of.items() if payer == "harborview"],
        "timely_filing": list(payer_of),
    }
    assigned: dict[str, str] = {}
    for family, target in FAMILY_TARGETS.items():
        eligible = [p for p in pools[family] if p not in assigned]
        if len(eligible) < target:
            raise ValueError(f"family {family}: only {len(eligible)} eligible for {target}")
        order = rng.permutation(len(eligible))
        for i in order[:target]:
            assigned[eligible[i]] = family
    return assigned


def _generate_studies(
    rng: np.random.Generator,
    patients: pl.DataFrame,
    clinics: pl.DataFrame,
    families: dict[str, str],
    payers: dict[str, PayerConfig],
) -> pl.DataFrame:
    cpts = list(STUDIES)
    cpt_probs = np.array([0.30, 0.25, 0.25, 0.20])  # MCT, CEM, LTM, Holter
    enrolled = clinics.filter(pl.col("medicaid_enrolled"))["clinic_id"].to_list()
    non_enrolled = clinics.filter(~pl.col("medicaid_enrolled"))["clinic_id"].to_list()
    all_clinics = clinics["clinic_id"].to_list()

    rows = []
    for i, patient in enumerate(patients.rows(named=True)):
        family = families.get(patient["patient_id"])
        if family == "retro_auth":
            cpt = "93229"
        elif family == "p2p":
            cpt = "93271"
        else:
            cpt = cpts[int(rng.choice(len(cpts), p=cpt_probs))]
        if family == "medicaid_enrollment":
            clinic_id = non_enrolled[int(rng.integers(len(non_enrolled)))]
        elif patient["payer_id"] in ("harborview", "pelican"):
            clinic_id = enrolled[int(rng.integers(len(enrolled)))]
        else:
            clinic_id = all_clinics[int(rng.integers(len(all_clinics)))]

        order_date = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 330)))
        service_start = order_date + timedelta(days=int(rng.integers(2, 15)))
        dur_lo, dur_hi = STUDIES[cpt][2]
        service_end = service_start + timedelta(days=int(rng.integers(dur_lo, dur_hi + 1)))
        report_delay = (
            int(rng.integers(55, 76))
            if family == "timely_filing"
            else int(rng.integers(1, 6))
        )
        report_delivered = service_end + timedelta(days=report_delay)

        payer = payers[patient["payer_id"]]
        auth_needed = cpt in payer.auth_required_cpts
        if family == "retro_auth":
            auth_obtained = False  # the planted P1 situation
        else:
            auth_obtained = bool(auth_needed and rng.random() < 0.65)
        prior_holter = (
            int(rng.integers(10, 26))
            if family == "p2p" or (cpt == "93271" and rng.random() < 0.05)
            else None
        )
        rows.append(
            {
                "study_id": f"st-{i:05d}",
                "patient_id": patient["patient_id"],
                "clinic_id": clinic_id,
                "cpt": cpt,
                "order_date": order_date,
                "service_start": service_start,
                "service_end": service_end,
                "report_delivered": report_delivered,
                "auth_required": auth_needed,
                "auth_obtained": auth_obtained,
                "eligibility_verified": bool(rng.random() < 0.9),
                "coverage_lapsed": bool(rng.random() < patient["stability"]),
                "prior_holter_within_days": prior_holter,
                "family": family,
            }
        )
    return pl.DataFrame(rows)


def generate_world(master_seed: int) -> World:
    payers = build_payer_configs()
    payer_map = {p.payer_id: p for p in payers}

    clinics = _generate_clinics(np.random.default_rng(stable_seed(master_seed, "clinics")))
    patients, coverage = _generate_patients(
        np.random.default_rng(stable_seed(master_seed, "patients")),
        list(payer_map),
    )
    patients = _assign_splits(
        np.random.default_rng(stable_seed(master_seed, "splits")), patients
    )
    families = _pick_family_members(
        np.random.default_rng(stable_seed(master_seed, "families")), patients, coverage
    )
    studies = _generate_studies(
        np.random.default_rng(stable_seed(master_seed, "studies")),
        patients,
        clinics,
        families,
        payer_map,
    )
    return World(
        payers=payers, clinics=clinics, patients=patients, coverage=coverage, studies=studies
    )


def write_world(world: World, out_dir: str | Path) -> dict[str, str]:
    """Write the snapshot files; returns file name -> sha256 of file content."""
    from ammonix_core.hashing import sha256_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "payers.yaml").write_text(
        yaml.safe_dump(
            [p.model_dump() for p in sorted(world.payers, key=lambda p: p.payer_id)],
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    world.clinics.write_parquet(out / "clinics.parquet")
    world.patients.write_parquet(out / "patients.parquet")
    world.coverage.write_parquet(out / "coverage.parquet")
    world.studies.write_parquet(out / "studies.parquet")
    return {
        name: sha256_file(out / name)
        for name in (
            "payers.yaml",
            "clinics.parquet",
            "patients.parquet",
            "coverage.parquet",
            "studies.parquet",
        )
    }


def load_payers_yaml(path: str | Path) -> list[PayerConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [PayerConfig.model_validate(entry) for entry in raw]

