"""Anomaly detection: a robust, seasonal expected level, and why it beats the older band rule."""

import math

import pytest

from dataforge_backend.domain import anomaly
from dataforge_backend.domain.monitoring import MonitorError, MonitorRule, evaluate, validate_rule
from dataforge_backend.seed_data import learners as showcase


def _months(n: int, start_year: int = 2024) -> list[str]:
    return [f"{start_year + i // 12}-{i % 12 + 1:02d}" for i in range(n)]


def _seasonal(n: int, spike: dict[int, float] | None = None) -> list[tuple[str, float]]:
    """A steady level with a strong December peak and a little deterministic noise."""
    values = []
    for i in range(n):
        base = 100 + 0.5 * i + (60 if i % 12 == 11 else 0) + 3 * math.sin(i * 1.7)
        values.append(base + (spike or {}).get(i, 0.0))
    return list(zip(_months(n), values, strict=True))


def test_a_spike_against_the_seasonal_baseline_is_unusual() -> None:
    series = _seasonal(31, spike={30: 45})
    verdict = anomaly.detect(series)
    assert verdict.method == "seasonal" and verdict.unusual and verdict.direction == "above"
    assert verdict.z is not None and verdict.z > anomaly.DEFAULT_K
    assert verdict.points[-1]["anomaly"] is True and verdict.points[-1]["expected"] is not None


def test_a_seasonal_peak_is_expected_by_the_robust_detector_but_trips_the_band_rule() -> None:
    december = _seasonal(36)  # the last month is a December peak
    assert december[-1][0].endswith("-12")
    assert anomaly.detect(december).unusual is False
    assert anomaly.band(december, k=3.0).unusual is True


def test_a_short_history_uses_a_robust_trend_and_too_short_is_said_plainly() -> None:
    steady = [(m, 50 + 2 * i) for i, m in enumerate(_months(10))]
    verdict = anomaly.detect(steady)
    assert verdict.method == "trend" and not verdict.unusual
    assert verdict.expected == pytest.approx(68, abs=0.01)
    jumped = [*steady[:-1], (steady[-1][0], 140)]
    assert anomaly.detect(jumped).unusual
    assert anomaly.detect(steady[:5]).method == "insufficient"


def test_a_perfectly_flat_history_still_has_a_sensible_spread() -> None:
    flat = [(m, 200.0) for m in _months(12)]
    assert anomaly.detect([*flat, ("2025-01", 203.0)]).unusual is False
    assert anomaly.detect([*flat, ("2025-01", 400.0)]).unusual is True


def test_theil_sen_ignores_a_single_outlier() -> None:
    values = [float(i * 3) for i in range(12)]
    values[5] = 500.0
    slope, intercept = anomaly.theil_sen(values)
    assert slope == pytest.approx(3.0) and intercept == pytest.approx(0.0)


def test_the_backtest_finds_the_planted_showcase_anomaly_and_the_band_rule_misses_it() -> None:
    planted = showcase.ANOMALY
    totals: dict[str, float] = {}
    for row in showcase.monthly():
        if row["province"] == planted["province"]:
            totals[row["month"]] = totals.get(row["month"], 0.0) + row["dropouts"]
    scores = anomaly.backtest(sorted(totals.items()), {planted["month"]}, start=24)
    assert scores["robust"].flagged == [planted["month"]]
    assert scores["band"].recall == 0.0


def test_over_many_synthetic_series_the_robust_detector_finds_far_more_with_few_false_alarms() -> None:
    result = anomaly.benchmark(60)
    assert result["robust"]["recall"] > result["band"]["recall"] + 0.3
    assert result["robust"]["precision"] >= 0.8


def test_the_unusual_month_rule_is_validated_and_explained() -> None:
    validate_rule(MonitorRule("anomaly", 3.5))
    with pytest.raises(MonitorError):
        validate_rule(MonitorRule("anomaly", 9))
    series = _seasonal(31, spike={30: 45})
    result = evaluate(series, MonitorRule("anomaly", 3.5), "count")
    assert result.triggered and "robust standard deviations above" in result.message
    assert result.detector and result.detector["method"] == "seasonal"
    assert len(result.points) == 31 and result.expected is not None
    calm = evaluate(_seasonal(31), MonitorRule("anomaly", 3.5), "count")
    assert not calm.triggered and calm.message.startswith("Within its expected range")
