"""Showcase data: learners at a South African skills-development provider. Entirely synthetic.

Not engine code. Nothing under domain/ or application/ imports this module; the engine treats these datasets
like any other upload. The generator exists so the demo and the evaluations have realistic data with a known
truth behind it.

How the learner data is made (the causal story the model card and the documentation describe):

* Hidden traits come first: engagement, financial and travel pressure, the host employer's mentoring quality
  and how often that employer pays stipends late.
* What the provider observes in a learner's first six weeks follows from those traits: attendance, formative
  assessment average, LMS logins, missed submissions and late stipend payments.
* Whether a learner drops out follows from the hidden traits too, plus noise. Gender, population group and
  disability have no direct effect. Disability raises travel distance, which raises pressure, so it acts only
  through that proxy: the fairness audit is meant to find it.
* Learners who started in 2026 log in more (blended delivery), so their LMS logins drift.

The monthly dataset is the provider network's progress by month, programme, province and SETA, with seasonal
drop-out peaks in December and January and one planted anomaly (see TRUTH).

Only the standard library is used, so the data is identical on every machine and quick to regenerate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any

FIRST_MONTH = "2024-01"
CUTOFF = "2026-10"  # the month the extract was taken; outcomes after it are unknown
LAST_START = "2026-09"

PROGRAMMES: dict[str, dict[str, Any]] = {
    # share of starts, duration in months, drop-out tendency, monthly drop-out hazard (network data)
    "Learnership": {"share": 0.55, "months": 12, "effect": 0.0, "hazard": 0.022},
    "Skills programme": {"share": 0.30, "months": 5, "effect": -0.45, "hazard": 0.030},
    "Apprenticeship": {"share": 0.15, "months": 18, "effect": 0.15, "hazard": 0.014},
}
_NQF = {
    "Learnership": (["NQF 2", "NQF 3", "NQF 4", "NQF 5"], [0.2, 0.3, 0.35, 0.15]),
    "Skills programme": (["NQF 2", "NQF 3", "NQF 4"], [0.35, 0.4, 0.25]),
    "Apprenticeship": (["NQF 4"], [1.0]),
}
_STIPEND = {"Learnership": (4500, 600), "Skills programme": (2800, 400), "Apprenticeship": (6000, 800)}

PROVINCES: dict[str, float] = {
    "Gauteng": 0.30,
    "KwaZulu-Natal": 0.18,
    "Western Cape": 0.13,
    "Eastern Cape": 0.10,
    "Limpopo": 0.08,
    "Mpumalanga": 0.07,
    "North West": 0.06,
    "Free State": 0.05,
    "Northern Cape": 0.03,
}
SETAS: dict[str, float] = {
    "MICT SETA": 0.22,
    "Services SETA": 0.20,
    "W&RSETA": 0.18,
    "merSETA": 0.16,
    "BANKSETA": 0.12,
    "HWSETA": 0.12,
}
# Fictional host employers: (name, SETA, home province).
EMPLOYERS: list[tuple[str, str, str]] = [
    ("Lesedi Digital Works", "MICT SETA", "Gauteng"),
    ("Amandla Telecom Services", "MICT SETA", "Western Cape"),
    ("Khanyisa Contact Centres", "Services SETA", "KwaZulu-Natal"),
    ("Imbali Hospitality Group", "Services SETA", "Western Cape"),
    ("Kopano Facilities", "Services SETA", "Gauteng"),
    ("Ilanga Fresh Stores", "W&RSETA", "KwaZulu-Natal"),
    ("Ubuntu Value Retail", "W&RSETA", "Eastern Cape"),
    ("Sizwe Precision Engineering", "merSETA", "Gauteng"),
    ("Masakhane Motor Components", "merSETA", "Eastern Cape"),
    ("Bokamoso Community Finance", "BANKSETA", "Limpopo"),
    ("Thuthuka Care Clinics", "HWSETA", "Mpumalanga"),
    ("Naledi Health Partners", "HWSETA", "Free State"),
]
GENDERS = (["Female", "Male"], [0.52, 0.48])
POPULATION_GROUPS = (["African", "Coloured", "White", "Indian/Asian"], [0.81, 0.085, 0.075, 0.03])
AGE_BANDS = (["18-24", "25-29", "30-35"], [0.45, 0.35, 0.20])
PRIOR = {
    "Grade 12": (0.55, 0.0),
    "Grade 11 or lower": (0.20, -4.0),
    "Certificate": (0.15, 3.0),
    "Diploma": (0.10, 5.0),
}

# Seasonality by calendar month (index 0 = January).
_INTAKE_SEASON = [1.2, 1.8, 1.0, 0.9, 0.8, 0.8, 1.6, 1.1, 0.9, 0.8, 0.6, 0.3]
_DROPOUT_SEASON = [1.40, 1.05, 0.95, 1.0, 0.95, 1.10, 1.0, 0.95, 0.95, 1.0, 1.05, 1.50]

DRIFT_FROM = "2026-01"
DRIFT_LOGINS = 1.6

ANOMALY: dict[str, Any] = {
    "month": "2026-08",
    "province": "KwaZulu-Natal",
    "seta": "W&RSETA",
    "employer": "Ilanga Fresh Stores",
    "factor": 4.5,
    "reason": "Stipend payments at one host employer were late for two months.",
}

TRUTH: dict[str, Any] = {
    "anomalies": [{**ANOMALY, "metric": "monthly_dropouts"}],
    "drift": {"feature": "lms_logins", "cohortsFrom": DRIFT_FROM, "factor": DRIFT_LOGINS},
    "protectedDirectEffect": {"gender": 0.0, "population_group": 0.0, "disability": 0.0},
    "proxy": {"disability": "raises travel_km by about 50%, which raises financial and travel pressure"},
}

_FIRST_NAMES = [
    "Thabo",
    "Lerato",
    "Sipho",
    "Ayanda",
    "Naledi",
    "Kagiso",
    "Zanele",
    "Bongani",
    "Palesa",
    "Mpho",
    "Nomsa",
    "Tshepo",
    "Anele",
    "Lindiwe",
    "Sibusiso",
    "Refilwe",
    "Karabo",
    "Thandeka",
    "Musa",
    "Busisiwe",
    "Johan",
    "Annelie",
    "Pieter",
    "Chantel",
    "Ruan",
    "Fatima",
    "Yusuf",
    "Priya",
    "Kevin",
    "Shireen",
]
_LAST_NAMES = [
    "Dlamini",
    "Nkosi",
    "Mokoena",
    "Khumalo",
    "Ndlovu",
    "Mahlangu",
    "Mthembu",
    "Zulu",
    "Molefe",
    "Naidoo",
    "Botha",
    "van der Merwe",
    "Pillay",
    "Maseko",
    "Radebe",
    "Sithole",
    "Baloyi",
    "Mabena",
    "Petersen",
    "Adams",
]


@dataclass
class Simulation:
    learners: list[dict[str, Any]]
    monthly: list[dict[str, Any]]
    truth: dict[str, Any] = field(default_factory=lambda: dict(TRUTH))


# ------------------------------------------------------------------ helpers


def month_index(month: str) -> int:
    year, mon = month.split("-")
    return int(year) * 12 + int(mon) - 1


def month_label(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 40:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit, k, product = math.exp(-lam), 0, rng.random()
    while product > limit:
        k += 1
        product *= rng.random()
    return k


def _binomial(rng: random.Random, n: int, p: float) -> int:
    return sum(1 for _ in range(n) if rng.random() < p)


def _pick(rng: random.Random, options: tuple[list[str], list[float]]) -> str:
    values, weights = options
    return rng.choices(values, weights)[0]


def _sa_id(rng: random.Random, birth: date) -> str:
    """A South African ID number with a valid birth date and Luhn check digit (not a real person's)."""
    body = f"{birth:%y%m%d}{rng.randint(0, 9999):04d}08"
    digits = [int(c) for c in body]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return body + str((10 - total % 10) % 10)


def _cell(rng: random.Random) -> str:
    return f"0{rng.choice('678')}{rng.randint(1, 9)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"


# ------------------------------------------------------------------ learners


def _employer_traits(rng: random.Random) -> dict[str, tuple[float, float]]:
    """Mentoring quality and late-stipend rate per host employer."""
    traits: dict[str, tuple[float, float]] = {}
    for name, _, _ in EMPLOYERS:
        traits[name] = (rng.gauss(0.0, 0.5), rng.uniform(0.02, 0.30))
    traits[ANOMALY["employer"]] = (traits[ANOMALY["employer"]][0], 0.34)
    return traits


def learners(n: int, seed: int = 20261119) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    traits = _employer_traits(rng)
    first, last_start, cutoff = month_index(FIRST_MONTH), month_index(LAST_START), month_index(CUTOFF)
    start_months = list(range(first, last_start + 1))
    start_weights = [_INTAKE_SEASON[m % 12] * (1 + 0.01 * (m - first)) for m in start_months]
    employer_weights = [
        1.0 + (0.6 if seta in ("MICT SETA", "Services SETA") else 0.0) for _, seta, _ in EMPLOYERS
    ]
    provinces, province_weights = list(PROVINCES), list(PROVINCES.values())
    prior_labels = list(PRIOR)
    prior_weights = [share for share, _ in PRIOR.values()]

    rows: list[dict[str, Any]] = []
    for i in range(n):
        start = rng.choices(start_months, start_weights)[0]
        programme = rng.choices(list(PROGRAMMES), [p["share"] for p in PROGRAMMES.values()])[0]
        spec = PROGRAMMES[programme]
        employer, seta, home = rng.choices(EMPLOYERS, employer_weights)[0]
        province = home if rng.random() < 0.7 else rng.choices(provinces, province_weights)[0]
        gender = _pick(rng, GENDERS)
        population_group = _pick(rng, POPULATION_GROUPS)
        disability = "Yes" if rng.random() < 0.03 else "No"
        age_band = _pick(rng, AGE_BANDS)
        prior = rng.choices(prior_labels, prior_weights)[0]
        mean, sd = _STIPEND[programme]
        stipend = max(1500, round(rng.gauss(mean, sd) / 50) * 50)

        # Hidden traits.
        engagement = rng.gauss(0.0, 1.0)
        travel = min(120.0, math.exp(rng.gauss(math.log(12), 0.65)) * (1.5 if disability == "Yes" else 1.0))
        pressure = (
            0.6 * (math.log(travel) - math.log(12)) / 0.65
            + 0.35 * -(stipend - mean) / sd
            + rng.gauss(0.0, 0.6)
        )
        mentoring, late_rate = traits[employer]
        late = _binomial(rng, 6, late_rate)

        # What the provider sees in the first six weeks.
        attendance = 100 * _sigmoid(
            1.4 + 0.85 * engagement - 0.4 * pressure + 0.5 * mentoring - 0.15 * late + rng.gauss(0, 0.3)
        )
        formative = min(100.0, max(5.0, 57 + 8.5 * engagement + PRIOR[prior][1] + rng.gauss(0, 6.5)))
        drift = DRIFT_LOGINS if start >= month_index(DRIFT_FROM) else 1.0
        logins = _poisson(rng, math.exp(2.2 + 0.35 * engagement) * drift)
        missed = _poisson(rng, math.exp(-0.7 - 0.55 * engagement + 0.3 * pressure))

        # The outcome. Protected attributes have no direct effect.
        logit = (
            -1.75
            - 1.0 * engagement
            + 0.6 * pressure
            - 0.55 * mentoring
            + 0.28 * late
            + spec["effect"]
            + (0.3 if age_band == "30-35" else 0.0)
            + rng.gauss(0, 0.35)
        )
        drops = rng.random() < _sigmoid(logit)
        offset = 1 + int((spec["months"] - 1) * rng.random() ** 1.4)
        planned_end = start + spec["months"]
        if drops and start + offset <= cutoff:
            outcome = "Dropped out"
        elif not drops and planned_end <= cutoff:
            outcome = "Completed"
        else:
            outcome = "In progress"

        age = {"18-24": rng.randint(18, 24), "25-29": rng.randint(25, 29), "30-35": rng.randint(30, 35)}[
            age_band
        ]
        start_year = start // 12
        birth = date(start_year - age, rng.randint(1, 12), rng.randint(1, 28))
        rows.append(
            {
                "learner_id": f"LR-{200001 + i}",
                "full_name": f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
                "id_number": _sa_id(rng, birth),
                "cell_number": _cell(rng),
                "programme_type": programme,
                "nqf_level": rng.choices(*_NQF[programme])[0],
                "seta": seta,
                "employer": employer,
                "province": province,
                "prior_qualification": prior,
                "gender": gender,
                "population_group": population_group,
                "disability": disability,
                "age_band": age_band,
                "start_month": month_label(start),
                "stipend_rand_avg": stipend,
                "travel_km": round(travel, 1),
                "attendance_pct": round(attendance, 1),
                # About 2% of learners missed the first formative assessment.
                "formative_avg": None if rng.random() < 0.02 else round(formative, 1),
                "lms_logins": logins,
                "missed_submissions": missed,
                "late_stipend_payments": late,
                "outcome": outcome,
                "starts": 1,
                "completed": int(outcome == "Completed"),
                "dropped": int(outcome == "Dropped out"),
            }
        )
    return rows


# ------------------------------------------------------------------ monthly network progress


def monthly(seed: int = 20261120, starts_per_month: float = 2600.0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    first, last = month_index(FIRST_MONTH), month_index(CUTOFF)
    anomaly_month = month_index(ANOMALY["month"])
    rows: list[dict[str, Any]] = []
    for programme, spec in PROGRAMMES.items():
        for province, province_share in PROVINCES.items():
            for seta, seta_share in SETAS.items():
                share = spec["share"] * province_share * seta_share
                # Start near the steady state, so the first months are not a ramp-up.
                stock = starts_per_month * share * spec["months"] * 0.8
                for m in range(first, last + 1):
                    season = m % 12
                    intake = starts_per_month * share * _INTAKE_SEASON[season] * (1 + 0.004 * (m - first))
                    starts = _poisson(rng, intake)
                    hazard = spec["hazard"] * _DROPOUT_SEASON[season]
                    if m == anomaly_month and province == ANOMALY["province"] and seta == ANOMALY["seta"]:
                        hazard *= float(ANOMALY["factor"])
                    dropped = min(round(stock), _poisson(rng, stock * hazard))
                    graduated = min(round(stock) - dropped, _poisson(rng, stock / spec["months"] * 0.85))
                    rows.append(
                        {
                            "month": month_label(m),
                            "programme_type": programme,
                            "province": province,
                            "seta": seta,
                            "new_starts": starts,
                            "enrolled_learners": round(stock),
                            "dropouts": dropped,
                            "graduates": graduated,
                        }
                    )
                    stock = max(0.0, stock + starts - dropped - graduated)
    rows.sort(key=lambda r: (r["month"], r["programme_type"], r["province"], r["seta"]))
    return rows


def simulate(n: int = 5000, seed: int = 20261119) -> Simulation:
    return Simulation(learners=learners(n, seed), monthly=monthly(seed + 1))
