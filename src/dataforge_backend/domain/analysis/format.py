"""Number formatting in each measure's unit, worded the same way as the frontend: 48 213, R1 214, 66.3%."""

import math
from decimal import ROUND_HALF_UP, Decimal

from ..values import thousands

MINUS = "−"


def _ok(value: float) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def js_fixed(value: float, digits: int) -> str:
    """Number.prototype.toFixed: exact halves round away from zero (1.25 → "1.3"), unlike Python's format."""
    if not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def js_round(value: float) -> int:
    """Math.round: halves round up, so 2.5 → 3 and −2.5 → −2."""
    return math.floor(value + 0.5)


def format_number(value: float) -> str:
    rounded = js_round(value)
    return f"{MINUS}{thousands(abs(rounded))}" if rounded < 0 else thousands(rounded)


def format_value(value: float, unit: str) -> str:
    if not _ok(value):
        return "—"
    if unit == "money":
        rounded = js_round(value)
        return f"{MINUS}R{thousands(abs(rounded))}" if rounded < 0 else f"R{thousands(rounded)}"
    if unit == "rate":
        return f"{js_fixed(value * 100, 1)}%"
    if unit == "decimal":
        return js_fixed(value, 2)
    return format_number(value)


def format_delta(value: float, unit: str) -> str:
    if not _ok(value):
        return "—"
    sign = "+" if value > 0 else MINUS if value < 0 else ""
    if unit == "rate":
        return f"{sign}{js_fixed(abs(value * 100), 1)} pts"
    return f"{sign}{format_value(abs(value), unit)}"


def format_percent(value: float, digits: int = 1) -> str:
    if not _ok(value):
        return "—"
    sign = "+" if value > 0 else MINUS if value < 0 else ""
    return f"{sign}{js_fixed(abs(value), digits)}%"


def format_compact(value: float, unit: str) -> str:
    if not _ok(value):
        return "—"
    if unit == "rate":
        return f"{js_fixed(value * 100, 1)}%"
    if unit == "decimal":
        return js_fixed(value, 2)
    prefix = "R" if unit == "money" else ""
    size = abs(value)
    if size >= 1_000_000:
        return f"{prefix}{js_fixed(value / 1_000_000, 1)}m"
    if size >= 10_000:
        return f"{prefix}{js_round(value / 1_000)}k"
    if size >= 1_000:
        return f"{prefix}{js_fixed(value / 1_000, 1)}k"
    return f"{prefix}{js_round(value)}"
