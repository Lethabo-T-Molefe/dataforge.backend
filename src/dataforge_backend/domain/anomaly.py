"""Anomaly detection for monthly metrics: is this month unusually far from where it was expected to be?

The expected value allows for seasonality and trend, and the spread is robust to the very outliers it looks for:

* 24 or more earlier months: seasonal-naive with drift. Expected = the same month last year plus the typical
  year-on-year change (the median of all such changes so far).
* 6 to 23 earlier months: a Theil-Sen trend line (the median of the slopes between every pair of months).
* Either way, the spread is 1.4826 x the median absolute deviation of past residuals, which estimates a
  standard deviation without being pulled by earlier spikes.

A month is unusual when it lies more than k spreads from the expected value (k = 3.5 by default). The older
band rule (mean +- k standard deviations of recent months) is kept for comparison: it ignores seasonality, so
it raises false alarms in seasonal peaks, and it is pulled wide by the spikes it should catch.

Standard library only, so it is cheap to run over every governed metric and easy to verify.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

SEASON = 12
MIN_TREND_HISTORY = 6
MAD_TO_SD = 1.4826
DEFAULT_K = 3.5


@dataclass(frozen=True)
class Detection:
    method: str  # seasonal | trend | band | insufficient
    month: str | None
    value: float | None
    expected: float | None
    low: float | None
    high: float | None
    z: float | None
    unusual: bool
    direction: str | None  # above | below
    spread: float | None
    history: int
    k: float
    points: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "methodLabel": METHOD_LABELS[self.method],
            "month": self.month,
            "value": self.value,
            "expected": _round(self.expected),
            "low": _round(self.low),
            "high": _round(self.high),
            "z": _round(self.z, 2),
            "unusual": self.unusual,
            "direction": self.direction,
            "spread": _round(self.spread),
            "history": self.history,
            "k": self.k,
        }


METHOD_LABELS = {
    "seasonal": "Seasonal baseline (same month last year plus the typical yearly change)",
    "trend": "Trend baseline (Theil-Sen line through the earlier months)",
    "band": "Recent average band (mean and standard deviation of recent months)",
    "insufficient": "Not enough history",
}


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, digits)


def _clean(series: Sequence[tuple[str, float | None]]) -> list[tuple[str, float]]:
    points = [(m, float(v)) for m, v in series if v is not None and math.isfinite(float(v))]
    return sorted(points)


def _mad_spread(residuals: list[float], scale_hint: float) -> float:
    if not residuals:
        return max(0.05 * abs(scale_hint), 1e-9)
    centre = statistics.median(residuals)
    mad = statistics.median([abs(r - centre) for r in residuals])
    spread = MAD_TO_SD * mad
    # A perfectly regular history has no spread; 5% of its level keeps tiny wobbles from counting as unusual.
    return spread if spread > 0 else max(0.05 * abs(scale_hint), 1e-9)


def theil_sen(values: list[float]) -> tuple[float, float]:
    """(slope, intercept) of the Theil-Sen line through (0, v0), (1, v1), ..."""
    slopes = [
        (values[j] - values[i]) / (j - i) for i in range(len(values)) for j in range(i + 1, len(values))
    ]
    slope = statistics.median(slopes) if slopes else 0.0
    intercept = statistics.median([v - slope * i for i, v in enumerate(values)])
    return slope, intercept


def _seasonal(values: list[float]) -> tuple[list[float | None], float, float]:
    """In-sample expected values, the drift, and the robust spread for the seasonal-naive model."""
    diffs = [values[i] - values[i - SEASON] for i in range(SEASON, len(values))]
    drift = statistics.median(diffs)
    expected: list[float | None] = [None] * SEASON + [
        values[i - SEASON] + drift for i in range(SEASON, len(values))
    ]
    residuals = [values[i] - (values[i - SEASON] + drift) for i in range(SEASON, len(values))]
    return expected, drift, _mad_spread(residuals, statistics.median(values))


def detect(series: Sequence[tuple[str, float | None]], k: float = DEFAULT_K) -> Detection:
    """Judge the latest month of a monthly series against the months before it."""
    points = _clean(series)
    if not points:
        return Detection("insufficient", None, None, None, None, None, None, False, None, None, 0, k)
    months = [m for m, _ in points]
    values = [v for _, v in points]
    month, value = months[-1], values[-1]
    history = values[:-1]
    if len(history) >= 2 * SEASON:
        in_sample, drift, spread = _seasonal(history)
        expected = history[-SEASON] + drift
        method = "seasonal"
        fitted: list[float | None] = [*in_sample, expected]
    elif len(history) >= MIN_TREND_HISTORY:
        slope, intercept = theil_sen(history)
        residuals = [v - (intercept + slope * i) for i, v in enumerate(history)]
        spread = _mad_spread(residuals, statistics.median(history))
        expected = intercept + slope * len(history)
        method = "trend"
        fitted = [intercept + slope * i for i in range(len(values))]
    else:
        return Detection(
            "insufficient", month, value, None, None, None, None, False, None, None, len(history), k
        )
    z = (value - expected) / spread
    low, high = expected - k * spread, expected + k * spread
    unusual = abs(z) > k
    chart = [
        {
            "month": m,
            "value": v,
            "expected": _round(e),
            "low": _round(e - k * spread) if e is not None else None,
            "high": _round(e + k * spread) if e is not None else None,
            "anomaly": unusual and i == len(values) - 1,
        }
        for i, (m, v, e) in enumerate(zip(months, values, fitted, strict=True))
    ]
    return Detection(
        method,
        month,
        value,
        expected,
        low,
        high,
        z,
        unusual,
        ("above" if z > 0 else "below") if unusual else None,
        spread,
        len(history),
        k,
        chart,
    )


def band(series: Sequence[tuple[str, float | None]], k: float = 3.0, window: int = 12) -> Detection:
    """The older rule, kept for comparison: mean +- k standard deviations of the last `window` months."""
    points = _clean(series)
    if len(points) < 4:
        return Detection(
            "insufficient", None, None, None, None, None, None, False, None, None, len(points), k
        )
    month, value = points[-1]
    history = [v for _, v in points[:-1]][-window:]
    centre = statistics.fmean(history)
    spread = statistics.stdev(history) if len(history) > 1 else 0.0
    spread = spread if spread > 0 else max(0.05 * abs(centre), 1e-9)
    z = (value - centre) / spread
    unusual = abs(z) > k
    return Detection(
        "band",
        month,
        value,
        centre,
        centre - k * spread,
        centre + k * spread,
        z,
        unusual,
        ("above" if z > 0 else "below") if unusual else None,
        spread,
        len(history),
        k,
    )


# ------------------------------------------------------------------ evaluation


@dataclass
class BacktestScore:
    method: str
    flagged: list[str]
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 1.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "flagged": self.flagged,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "falseNegatives": self.false_negatives,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
        }


def backtest(
    series: Sequence[tuple[str, float | None]],
    anomalies: set[str],
    *,
    k: float = DEFAULT_K,
    band_k: float = 3.0,
    band_window: int = 12,
    start: int = 12,
) -> dict[str, BacktestScore]:
    """Replay the series month by month, judging each month only on the months before it, as a monitor would.

    Returns precision and recall against the months known to be anomalous, for the robust detector and the
    band rule. Months judged are those after `start` earlier months.
    """
    points = _clean(series)
    judged = [m for m, _ in points[start:]]
    results: dict[str, BacktestScore] = {}
    for method in ("robust", "band"):
        flagged = []
        for t in range(start, len(points)):
            window = points[: t + 1]
            verdict = detect(window, k) if method == "robust" else band(window, band_k, band_window)
            if verdict.unusual:
                flagged.append(points[t][0])
        truth = anomalies & set(judged)
        tp = len(truth & set(flagged))
        results[method] = BacktestScore(
            method, flagged, tp, len(set(flagged) - truth), len(truth - set(flagged))
        )
    return results


def synthetic_series(seed: int, months: int = 36) -> tuple[list[tuple[str, float]], set[str]]:
    """A seasonal series with trend and noise, and one or two planted spikes or dips (the known truth)."""
    rng = random.Random(seed)
    level = rng.uniform(50, 500)
    trend = rng.uniform(-0.01, 0.015) * level
    amplitude = rng.uniform(0.05, 0.35) * level
    phase = rng.randint(0, 11)
    noise = rng.uniform(0.02, 0.06) * level
    labels = [f"{2024 + (i // 12)}-{i % 12 + 1:02d}" for i in range(months)]
    values = [
        level + trend * i + amplitude * math.sin(2 * math.pi * (i + phase) / 12) + rng.gauss(0, noise)
        for i in range(months)
    ]
    planted: set[str] = set()
    for _ in range(rng.choice([1, 1, 2])):
        i = rng.randint(24, months - 1)
        values[i] += rng.choice([-1, 1]) * rng.uniform(6, 10) * noise
        planted.add(labels[i])
    return list(zip(labels, values, strict=True)), planted


def benchmark(count: int = 200, k: float = DEFAULT_K, band_k: float = 3.0) -> dict[str, Any]:
    """Precision and recall of both methods over `count` seeded synthetic series, judged from month 24."""
    totals = {m: [0, 0, 0] for m in ("robust", "band")}
    for seed in range(count):
        series, planted = synthetic_series(seed)
        for method, score in backtest(series, planted, k=k, band_k=band_k, start=24).items():
            totals[method][0] += score.true_positives
            totals[method][1] += score.false_positives
            totals[method][2] += score.false_negatives
    out: dict[str, Any] = {"series": count, "judgedFrom": "month 25 of 36", "k": k, "bandK": band_k}
    for method, (tp, fp, fn) in totals.items():
        out[method] = {
            "truePositives": tp,
            "falsePositives": fp,
            "falseNegatives": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 1.0,
        }
    return out
