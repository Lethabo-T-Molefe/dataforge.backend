"""Guided, explainable scoring.

A model version is a documented, deterministic heuristic: a set of factors, each read from a shared metric
and compared with the population, with weights that always add up to one. A trained model can later
replace a heuristic behind the same contract (score, band, ranked factors, model version).
"""

import math
from dataclasses import dataclass
from typing import Any

from .analysis.format import format_value

WEIGHT_TOLERANCE = 1e-6
WINDOWS = ("recent", "trend", "all")
DIRECTIONS = ("up", "down")


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class FactorSpec:
    key: str
    name: str
    weight: float
    direction: str  # up: a higher value raises the score · down: a higher value lowers it
    measure: str  # a measure key on the model's dataset
    window: str = "recent"  # recent: the last three months · trend: change against the three before · all
    why: str = ""  # the documented reason this factor matters

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "weight": self.weight,
            "direction": self.direction,
            "measure": self.measure,
            "window": self.window,
            "why": self.why,
        }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: str
    outcome: str
    dataset_id: str
    factors: tuple[FactorSpec, ...]
    description: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} v{self.version}"


def factor_from_dict(data: dict[str, Any]) -> FactorSpec:
    try:
        return FactorSpec(
            key=str(data["key"]),
            name=str(data["name"]),
            weight=float(data["weight"]),
            direction=str(data["direction"]),
            measure=str(data["measure"]),
            window=str(data.get("window", "recent")),
            why=str(data.get("why", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModelError(f"A factor is missing a field or has a wrong value: {error}") from error


def validate_model(spec: ModelSpec, measures: set[str] | None = None) -> ModelSpec:
    if not spec.factors:
        raise ModelError("A model needs at least one factor.")
    keys = [f.key for f in spec.factors]
    if len(set(keys)) != len(keys):
        raise ModelError("Factor keys must be unique.")
    for factor in spec.factors:
        if factor.weight <= 0:
            raise ModelError(f"{factor.name}: weights must be positive.")
        if factor.direction not in DIRECTIONS:
            raise ModelError(f"{factor.name}: direction must be up or down.")
        if factor.window not in WINDOWS:
            raise ModelError(f"{factor.name}: window must be one of {', '.join(WINDOWS)}.")
        if measures is not None and factor.measure not in measures:
            raise ModelError(f"{factor.name}: {factor.measure} is not a measure of the model's dataset.")
    weights = sum(f.weight for f in spec.factors)
    if abs(weights - 1.0) > WEIGHT_TOLERANCE:
        raise ModelError(f"Factor weights must add up to 1. They add up to {weights:.4f}.")
    return spec


@dataclass(frozen=True)
class FactorInput:
    """A factor's value for the cohort, and the population's typical value and spread across units."""

    cohort: float
    centre: float
    spread: float
    units: int
    unit: str  # count | money | rate | decimal, for wording


def band_for(score: int) -> str:
    if score >= 80:
        return "High"
    if score >= 65:
        return "Moderate to high"
    if score >= 50:
        return "Moderate"
    if score >= 35:
        return "Low to moderate"
    return "Low"


def factor_score(value: FactorInput, direction: str) -> float:
    """Logistic of the robust z-score: 0.5 is typical, towards 1 favours the outcome, towards 0 works against it."""
    if not math.isfinite(value.cohort) or not math.isfinite(value.centre):
        return 0.5
    spread = value.spread if value.spread and math.isfinite(value.spread) and value.spread > 0 else None
    z = 0.0 if spread is None else (value.cohort - value.centre) / spread
    z = max(-6.0, min(6.0, z))
    s = 1.0 / (1.0 + math.exp(-1.7 * z))
    return s if direction == "up" else 1.0 - s


def _window_words(window: str) -> str:
    return {
        "recent": "over the last three months",
        "trend": "against the three months before",
        "all": "over all months",
    }[window]


def score_cohort(spec: ModelSpec, inputs: dict[str, FactorInput], subject: str) -> dict[str, Any]:
    """Score a cohort and explain it. Factors are ranked by how much they moved the score."""
    validate_model(spec)
    rows: list[dict[str, Any]] = []
    total = 0.0
    for factor in spec.factors:
        value = inputs.get(factor.key)
        if value is None:
            raise ModelError(f"No value for factor {factor.name}.")
        s = factor_score(value, factor.direction)
        total += factor.weight * s
        effect = "raises" if s > 0.55 else "lowers" if s < 0.45 else "barely moves"
        if factor.window == "trend":
            shown = (
                format_value(value.cohort, value.unit)
                if value.unit != "rate"
                else f"{value.cohort * 100:+.1f} pts"
            )
            typical = (
                format_value(value.centre, value.unit)
                if value.unit != "rate"
                else f"{value.centre * 100:+.1f} pts"
            )
            reading = f"{subject} changed by {shown} {_window_words(factor.window)}, against a typical {typical}. This {effect} the score."
        else:
            reading = (
                f"{subject} sits at {format_value(value.cohort, value.unit)} {_window_words(factor.window)}, against a "
                f"typical {format_value(value.centre, value.unit)} across {value.units} groups. This {effect} the score."
            )
        rows.append(
            {
                "key": factor.key,
                "name": factor.name,
                "weight": factor.weight,
                # The direction the factor pushes this score, which is what a reader needs.
                "direction": "up" if s >= 0.5 else "down",
                "reading": reading,
                "why": factor.why,
                "factorScore": round(s, 4),
                "contribution": round(factor.weight * (s - 0.5) * 100, 2),
                "value": value.cohort,
                "typical": value.centre,
            }
        )
    score = int(round(total * 100))
    rows.sort(key=lambda r: (-abs(float(r["contribution"])), -float(r["weight"])))
    return {
        "subject": subject,
        "outcome": spec.outcome,
        "score": score,
        "band": band_for(score),
        "modelVersion": spec.label,
        "factors": rows,
        "weightsSum": round(sum(f.weight for f in spec.factors), 6),
    }
