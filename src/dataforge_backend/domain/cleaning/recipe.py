"""Running a cleaning recipe: previews on a sample, the full ordered run, and preparing the silver table."""

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from ..pii import PiiKind, detect_pii, mask_value
from ..values import is_blank, parse_bool, parse_date, parse_number
from .catalogue import METHODS_BY_ID, Method
from .operations import CleaningContext, OperationError, StepOutcome, run_operation


@dataclass(frozen=True)
class RecipeStepSpec:
    method_id: str
    params: dict[str, Any]
    status: str  # accepted | skipped


def method_or_error(method_id: str) -> Method:
    method = METHODS_BY_ID.get(method_id)
    if method is None:
        raise OperationError(f"Unknown cleaning method {method_id!r}.")
    return method


def apply_recipe(
    frame: pd.DataFrame, steps: list[RecipeStepSpec], ctx: CleaningContext
) -> tuple[pd.DataFrame, list[StepOutcome]]:
    outcomes: list[StepOutcome] = []
    for step in steps:
        if step.status != "accepted":
            continue
        method = method_or_error(step.method_id)
        outcome = run_operation(method.operation, frame, step.params, ctx)
        frame = outcome.frame
        outcomes.append(outcome)
    return frame, outcomes


def _render(value: Any) -> str:
    if is_blank(value):
        return "(empty)"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".") if value != int(value) else f"{int(value)}"
    return str(value)


def _input_columns(params: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in (
        "columns",
        "subset",
        "dates",
        "numbers",
        "text_columns",
        "phone_columns",
        "quasi_identifiers",
        "value_columns",
    ):
        value = params.get(key)
        if isinstance(value, list):
            names.extend(str(v) for v in value)
    for key in ("column", "dependent"):
        if params.get(key):
            names.append(str(params[key]))
    if isinstance(params.get("types"), dict):
        names.extend(params["types"].keys())
    return list(dict.fromkeys(names))


def preview_step(
    sample: pd.DataFrame, method: Method, params: dict[str, Any], ctx: CleaningContext, limit: int = 6
) -> dict[str, Any]:
    """Run one step on a sample and show a few rows before and after."""
    outcome = run_operation(method.operation, sample.copy(), params, ctx)
    after = outcome.frame
    inputs = [c for c in _input_columns(params) if c in sample.columns]
    shown_before = inputs[:3] or [c for c in outcome.affected if c in sample.columns][:3]
    shown_after = list(
        dict.fromkeys(
            [c for c in shown_before if c in after.columns]
            + [c for c in outcome.affected if c in after.columns]
        )
    )[:4]
    before_rows: list[str] = []
    after_rows: list[str] = []

    removed = [i for i in sample.index if i not in after.index]
    if removed:
        for i in removed[:limit]:
            before_rows.append(" · ".join(_render(sample.at[i, c]) for c in shown_before) or f"row {i}")
            after_rows.append("(row removed)")
    else:
        changed: list[Any] = []
        for i in sample.index:
            if i not in after.index:
                continue
            old = tuple(_render(sample.at[i, c]) for c in shown_before)
            new = tuple(_render(after.at[i, c]) for c in shown_after)
            if (
                old != new[: len(old)]
                or len(new) > len(old)
                and any(v not in ("False", "(empty)") for v in new[len(old) :])
            ):
                changed.append(i)
            if len(changed) >= limit:
                break
        for i in changed or list(sample.index[:limit]):
            before_rows.append(" · ".join(_render(sample.at[i, c]) for c in shown_before) or f"row {i}")
            after_rows.append(" · ".join(_render(after.at[i, c]) for c in shown_after) or f"row {i}")

    return {
        "before": before_rows,
        "after": after_rows,
        "columnsBefore": shown_before,
        "columnsAfter": shown_after,
        "summary": outcome.summary,
        "rowsBefore": outcome.rows_before,
        "rowsAfter": outcome.rows_after,
        "cellsChanged": outcome.cells_changed,
        "affected": outcome.affected,
        "sampleSize": len(sample),
    }


def _looks_masked(series: pd.Series) -> bool:
    values = [str(v) for v in series.dropna().head(200)]
    return bool(values) and sum("•" in v for v in values) / len(values) > 0.8


def enforce_masking(frame: pd.DataFrame, pii_columns: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Mask every personal column still present in the clear, including ones a residual scan finds."""
    out = frame.copy()
    masked: list[str] = []
    residual = {f.column: f.kind.value for f in detect_pii(out) if f.column not in pii_columns}
    for column, kind in {**pii_columns, **residual}.items():
        if column not in out.columns or _looks_masked(out[column]):
            continue
        out[column] = pd.Series(
            out[column].map(lambda v, k=kind: mask_value(v, k or PiiKind.NAME)), index=out.index, dtype=object
        )
        masked.append(column)
    return out, masked


def _column_type(series: pd.Series) -> str:
    values = [v for v in series if not is_blank(v)]
    if not values:
        return "text"
    if all(isinstance(v, bool) for v in values):
        return "boolean"
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return "integer" if all(float(v).is_integer() for v in values) else "numeric"
    if all(isinstance(v, date) for v in values):
        return "date"
    texts = [str(v).strip() for v in values]
    sample = texts[:5000]
    if all(t.lower() in {"true", "false"} for t in sample):
        return "boolean"
    numbers = [parse_number(t) for t in sample]
    if all(n is not None for n in numbers) and not any(
        t.startswith("0") and len(t) > 1 and t[1].isdigit() for t in sample
    ):
        return (
            "integer"
            if all(float(n).is_integer() for n in numbers if n is not None)
            and not any("." in t for t in sample)
            else "numeric"
        )
    if all(len(t) == 10 and t[4] == "-" and parse_date(t) is not None for t in sample):
        return "date"
    return "text"


def type_for_silver(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Give each column its real type so silver holds numbers, dates and booleans, not text."""
    out = pd.DataFrame(index=frame.index)
    types: dict[str, str] = {}
    for column in frame.columns:
        kind = _column_type(frame[column])
        types[str(column)] = kind
        convert = _CONVERTERS[kind]
        # A list keeps Python ints and None as they are; Series.map would turn them into floats and NaN.
        converted = [None if is_blank(v) else convert(v) for v in frame[column].tolist()]
        out[column] = pd.Series(converted, index=frame.index, dtype=object)
    return out, types


def _to_int(value: Any) -> int | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return int(round(value))
    number = parse_number(value)
    return None if number is None else int(round(number))


def _to_float(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return parse_number(value)


_CONVERTERS: dict[str, Any] = {
    "integer": _to_int,
    "numeric": _to_float,
    "date": lambda v: v if isinstance(v, date) else parse_date(v),
    "boolean": lambda v: v if isinstance(v, bool) else parse_bool(v),
    "text": str,
}
