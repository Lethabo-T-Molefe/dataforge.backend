"""Metric monitors: when a tracked metric moves in a meaningful way, raise an alert and explain why."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .analysis.format import format_delta, format_percent, format_value, js_fixed
from .analysis.stats import mean, standard_deviation
from .values import thousands

RULE_KINDS = ("change", "above", "below", "band")
CADENCES = {"hourly": 3600, "daily": 86_400, "weekly": 604_800}


class MonitorError(ValueError):
    pass


@dataclass(frozen=True)
class MonitorRule:
    kind: str
    threshold: (
        float  # change: per cent · above/below: a value in the metric's unit · band: standard deviations
    )
    window: int = 6  # band: months of history that define the expected level

    def describe(self, unit: str) -> str:
        if self.kind == "change":
            return f"Alert if month-on-month change exceeds {self.threshold:g}%"
        if self.kind == "above":
            return f"Alert if the value crosses above {format_value(self.threshold, unit)}"
        if self.kind == "below":
            return f"Alert if the value drops below {format_value(self.threshold, unit)}"
        return f"Alert if the value leaves its expected band (±{self.threshold:g} standard deviations of the last {self.window} months)"


def validate_rule(rule: MonitorRule) -> MonitorRule:
    if rule.kind not in RULE_KINDS:
        raise MonitorError(f"Rule must be one of {', '.join(RULE_KINDS)}.")
    if not math.isfinite(rule.threshold) or (rule.kind in ("change", "band") and rule.threshold <= 0):
        raise MonitorError("The threshold must be a positive number.")
    if rule.kind == "band" and not 3 <= rule.window <= 36:
        raise MonitorError("The band needs between 3 and 36 months of history.")
    return rule


@dataclass(frozen=True)
class Evaluation:
    triggered: bool
    message: str
    severity: str  # high | low
    month: str | None
    previous_month: str | None
    value: float | None
    previous: float | None
    expected: tuple[float, float] | None = None


def evaluate(
    series: list[tuple[str, float]], rule: MonitorRule, unit: str, month_label: Callable[[str], str] = str
) -> Evaluation:
    """Check the latest month of a monthly series against the rule."""
    points = [(month, value) for month, value in series if value is not None and math.isfinite(value)]
    if len(points) < 2:
        return Evaluation(False, "Not enough months of data to evaluate yet.", "low", None, None, None, None)
    (prev_month, prev), (month, latest) = points[-2], points[-1]
    name = month_label(month)

    def fmt(v: float) -> str:
        return format_value(v, unit)

    if rule.kind == "change":
        pct = math.nan if prev == 0 else (latest - prev) / abs(prev) * 100
        triggered = math.isfinite(pct) and abs(pct) > rule.threshold
        verb = "Rose" if latest > prev else "Fell"
        message = (
            f"{verb} {js_fixed(abs(pct), 0)}% from {month_label(prev_month)} to {name}, beyond the {rule.threshold:g}% limit"
            if triggered
            else f"Moved {format_percent(pct)} in {name}, within the {rule.threshold:g}% limit"
        )
        severity = "high" if triggered and (latest < prev or abs(pct) >= 1.5 * rule.threshold) else "low"
        return Evaluation(triggered, message, severity, month, prev_month, latest, prev)

    if rule.kind in ("above", "below"):
        above = rule.kind == "above"
        crossed = (latest > rule.threshold >= prev) if above else (latest < rule.threshold <= prev)
        earlier = [v for _, v in points[:-1]]
        first_time = crossed and all(
            (v <= rule.threshold) if above else (v >= rule.threshold) for v in earlier
        )
        target = fmt(rule.threshold)
        if crossed:
            message = (
                f"Crossed above its {target} target{' for the first time' if first_time else ''} in {name}"
                if above
                else f"Dropped below its {target} floor{' for the first time' if first_time else ''} in {name}"
            )
        else:
            side = "above" if latest > rule.threshold else "below"
            message = f"At {fmt(latest)} in {name}, {side} the {target} {'target' if above else 'floor'}"
        return Evaluation(crossed, message, "low" if above else "high", month, prev_month, latest, prev)

    history = [v for _, v in points[:-1]][-rule.window :]
    if len(history) < 3:
        return Evaluation(
            False,
            "The expected band needs at least three earlier months.",
            "low",
            month,
            prev_month,
            latest,
            prev,
        )
    centre, spread = mean(history), standard_deviation(history)
    low, high = centre - rule.threshold * spread, centre + rule.threshold * spread
    triggered = latest < low or latest > high
    pct = math.nan if centre == 0 else (latest - centre) / abs(centre) * 100
    if triggered:
        direction = "below" if latest < low else "above"
        message = f"{'Fell' if direction == 'below' else 'Rose'} {js_fixed(abs(pct), 0)}% {direction} the expected band in {name}"
    else:
        message = f"Within the expected band ({fmt(low)} – {fmt(high)}) in {name}"
    severity = "high" if triggered and latest < low else "low"
    return Evaluation(triggered, message, severity, month, prev_month, latest, prev, (low, high))


@dataclass(frozen=True)
class GroupChange:
    group: str
    before: float  # the metric's total (additive) or numerator (ratio) in the earlier month
    after: float
    den_before: float = 0.0  # ratio metrics: the denominator
    den_after: float = 0.0


@dataclass
class RootCauseInput:
    label: str
    unit: str
    is_ratio: bool
    month_before: str
    month_after: str
    total_before: float
    total_after: float
    # Dimension label → group changes. Totals for additive metrics; numerator and denominator for ratios.
    breakdowns: dict[str, list[GroupChange]] = field(default_factory=dict)
    # Additive metrics that are a volume × a rate (revenue = enrolments × average fee).
    volume_label: str | None = None
    rate_label: str | None = None
    rate_unit: str = "money"
    volume_before: float | None = None
    volume_after: float | None = None


def root_cause(data: RootCauseInput, limit: int = 4) -> list[dict[str, str]]:
    """The main contributing factors to a change, largest first, with the supporting figures."""
    factors: list[tuple[float, dict[str, str]]] = []
    a, b = data.month_before, data.month_after

    if data.is_ratio:
        # Change in a pooled rate = Σ over groups of (share_after × rate_after − share_before × rate_before).
        for dim, groups in data.breakdowns.items():
            den_a = sum(g.den_before for g in groups) or math.nan
            den_b = sum(g.den_after for g in groups) or math.nan
            for g in groups:
                effect = (g.after / den_b) - (g.before / den_a)
                if not math.isfinite(effect) or effect == 0:
                    continue
                rate_a = g.before / g.den_before if g.den_before else math.nan
                rate_b = g.after / g.den_after if g.den_after else math.nan
                detail = f"{dim} {g.group}: {format_value(rate_a, data.unit)} in {a}, {format_value(rate_b, data.unit)} in {b}"
                factors.append(
                    (
                        abs(effect),
                        {
                            "factor": f"{g.group}",
                            "contribution": format_delta(effect, data.unit),
                            "detail": detail,
                        },
                    )
                )
        factors.sort(key=lambda f: -f[0])
        return [f for _, f in factors[:limit]]

    base = data.total_before
    if not base:
        return []

    def pct(v: float) -> str:
        return f"{'+' if v > 0 else '−' if v < 0 else ''}{js_fixed(abs(v), 0)}%"

    if data.volume_label and data.volume_before and data.volume_after and data.rate_label:
        rate_a = data.total_before / data.volume_before
        rate_b = data.total_after / data.volume_after
        volume_effect = (data.volume_after - data.volume_before) * rate_a
        rate_effect = (rate_b - rate_a) * data.volume_before
        interaction = (data.volume_after - data.volume_before) * (rate_b - rate_a)
        moved = data.volume_after - data.volume_before
        factors.append(
            (
                abs(volume_effect),
                {
                    "factor": data.volume_label,
                    "contribution": pct(volume_effect / base * 100),
                    "detail": f"{thousands(round(abs(moved)))} {'fewer' if moved < 0 else 'more'} {data.volume_label.lower()} than {a}",
                },
            )
        )
        factors.append(
            (
                abs(rate_effect),
                {
                    "factor": data.rate_label,
                    "contribution": pct(rate_effect / base * 100),
                    "detail": f"Moved from {format_value(rate_a, data.rate_unit)} to {format_value(rate_b, data.rate_unit)}",
                },
            )
        )
        if abs(interaction) / abs(base) >= 0.01:
            factors.append(
                (
                    abs(interaction),
                    {
                        "factor": f"{data.volume_label} and {data.rate_label.lower()} together",
                        "contribution": pct(interaction / base * 100),
                        "detail": "Both moved at once, which adds this on top of each effect alone",
                    },
                )
            )

    # Where the change came from: the groups that moved most, in the dimension that concentrates it most.
    best: list[GroupChange] = []
    best_dim = ""
    best_share = -1.0
    total_change = data.total_after - data.total_before
    for dim, groups in data.breakdowns.items():
        if not groups or total_change == 0:
            continue
        top = max(groups, key=lambda g: abs(g.after - g.before))
        share = abs(top.after - top.before) / abs(total_change)
        if share > best_share:
            best, best_dim, best_share = groups, dim, share
    for g in sorted(best, key=lambda g: -abs(g.after - g.before))[:2]:
        delta = g.after - g.before
        if delta == 0:
            continue
        amount = format_value(abs(delta), data.unit) if data.unit == "money" else thousands(round(abs(delta)))
        detail = (
            f"{amount} {'less' if delta < 0 else 'more'} than {a}"
            if data.unit == "money"
            else f"{amount} {'fewer' if delta < 0 else 'more'} {data.label.lower()} than {a}"
        )
        factors.append(
            (
                abs(delta) * 0.999,
                {
                    "factor": f"{data.label} in {g.group}",
                    "contribution": pct(delta / base * 100),
                    "detail": f"{best_dim} · {detail}",
                },
            )
        )
    factors.sort(key=lambda f: -f[0])
    return [f for _, f in factors[:limit]]
