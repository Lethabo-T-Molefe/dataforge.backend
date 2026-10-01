"""Small statistics used by the analysis engine, matching the frontend and PostgreSQL definitions."""

import math
from collections.abc import Sequence

NAN = float("nan")


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def total(values: Sequence[float]) -> float:
    return float(sum(values))


def mean(values: Sequence[float]) -> float:
    return total(values) / len(values) if values else NAN


def quantile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics, as PERCENTILE_CONT, R and NumPy do."""
    if not values:
        return NAN
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def standard_deviation(values: Sequence[float]) -> float:
    """Sample standard deviation (n − 1), as STDDEV_SAMP."""
    if len(values) < 2:
        return NAN
    centre = mean(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / (len(values) - 1))


def tukey_fences(values: Sequence[float]) -> tuple[float, float]:
    q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return NAN
    mx, my = mean(xs[:n]), mean(ys[:n])
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    vx = sum((xs[i] - mx) ** 2 for i in range(n))
    vy = sum((ys[i] - my) ** 2 for i in range(n))
    return NAN if vx == 0 or vy == 0 else cov / math.sqrt(vx * vy)


def ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            result[order[k]] = average
        i = j + 1
    return result


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """Ordinary least squares: (slope, intercept)."""
    if not xs:
        return None
    mx, my = mean(xs), mean(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / denominator
    return slope, my - slope * mx


def correlation_strength(r: float) -> str:
    """Cohen's conventional reading of a correlation's size."""
    size = abs(r)
    if size < 0.1:
        return "no meaningful"
    if size < 0.3:
        return "a weak"
    if size < 0.5:
        return "a moderate"
    return "a strong"
