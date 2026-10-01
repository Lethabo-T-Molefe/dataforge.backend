"""The semantic layer: shared metric definitions and the analysis fields they give each dataset.

A metric is defined once, in one place, over column names. It applies to every approved dataset that has
those columns, so every answer, dashboard and notebook computes it the same way. A dataset's remaining
numeric columns are offered as plain (ungoverned) totals, and its categorical columns as dimensions.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .analysis.model import UNITS, AnalysisDataset, DimensionDef, MeasureDef, is_identifier
from .analysis.sql import ident

NUMERIC_TYPES = frozenset({"integer", "numeric", "decimal", "bigint", "double precision", "real", "smallint"})
TEXT_TYPES = frozenset({"text", "string", "character varying", "boolean"})
DATE_TYPES = frozenset({"date", "timestamp", "timestamp without time zone", "timestamp with time zone"})
MAX_DIMENSION_VALUES = 60
_MONEY_WORDS = (
    "fee",
    "revenue",
    "amount",
    "price",
    "cost",
    "paid",
    "spend",
    "salary",
    "income",
    "rand",
    "value_zar",
)
_RATE_WORDS = ("rate", "ratio", "share", "pct", "percent")


class MetricError(ValueError):
    pass


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    unit: str
    kind: str  # additive | count | distinct | ratio
    description: str = ""
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    synonyms: tuple[str, ...] = ()
    metric_id: str | None = None
    governed: bool = True

    def columns(self) -> list[str]:
        return [c for c in (self.column, self.numerator, self.denominator) if c and c != "*"]

    def to_measure(self) -> MeasureDef:
        return MeasureDef(
            id=self.key,
            label=self.name,
            unit=self.unit,
            kind=self.kind,
            description=self.description,
            column=self.column,
            numerator=self.numerator,
            denominator=self.denominator,
            governed=self.governed,
            metric_id=self.metric_id,
        )


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str  # integer | numeric | text | date | boolean | ...
    distinct: int = 0
    pii: bool = False
    values: tuple[str, ...] = ()  # most common first, for categorical columns
    month_like: bool = False  # text written YYYY-MM


@dataclass
class DatasetFields:
    measures: dict[str, MeasureDef] = field(default_factory=dict)
    dimensions: dict[str, DimensionDef] = field(default_factory=dict)
    time_column: str | None = None
    time_kind: str | None = None


def humanise(name: str) -> str:
    words = re.sub(r"[_\-]+", " ", name).strip()
    return words[:1].upper() + words[1:]


def expression_for(spec: MetricSpec) -> str:
    """The metric's formula as SQL, shown on the metric's definition."""
    if spec.kind == "additive":
        return f"SUM({ident(spec.column or '')})"
    if spec.kind == "count":
        return "COUNT(*)"
    if spec.kind == "distinct":
        return f"COUNT(DISTINCT {ident(spec.column or '')})"
    num = f"SUM({ident(spec.numerator or '')})"
    den = "COUNT(*)" if spec.denominator == "*" else f"SUM({ident(spec.denominator or '')})"
    return f"{num}::numeric / NULLIF({den}, 0)"


def validate_metric(spec: MetricSpec, columns: dict[str, str] | None = None) -> MetricSpec:
    """Check a metric is well formed and, when a table is given, that its columns exist and are numeric."""
    if not is_identifier(spec.key):
        raise MetricError(
            "A metric key is lower case letters, digits and underscores, starting with a letter."
        )
    if not spec.name.strip():
        raise MetricError("Give the metric a name.")
    if spec.unit not in UNITS:
        raise MetricError(f"Unit must be one of {', '.join(UNITS)}.")
    if spec.kind in ("additive", "distinct"):
        if not spec.column:
            raise MetricError("Choose the column the metric is computed from.")
        if spec.numerator or spec.denominator:
            raise MetricError("Only ratio metrics have a numerator and denominator.")
    elif spec.kind == "ratio":
        if not spec.numerator or not spec.denominator:
            raise MetricError("A ratio needs a numerator and a denominator column.")
        if spec.column:
            raise MetricError("A ratio is computed from its numerator and denominator, not one column.")
    elif spec.kind == "count":
        if spec.column or spec.numerator or spec.denominator:
            raise MetricError("A row count does not use a column.")
    else:
        raise MetricError("Kind must be additive, count, distinct or ratio.")
    for name in spec.columns():
        if not is_identifier(name):
            raise MetricError(f"{name!r} is not a valid column name.")
    if columns is not None:
        for name in spec.columns():
            if name not in columns:
                raise MetricError(f"{name} is not a column of this dataset.")
            needs_number = not (spec.kind == "distinct")
            if needs_number and columns[name] not in NUMERIC_TYPES:
                raise MetricError(f"{name} holds {columns[name]} values, so it cannot be added up.")
    return spec


def applies_to(spec: MetricSpec, columns: dict[str, str]) -> bool:
    try:
        validate_metric(spec, columns)
    except MetricError:
        return False
    return True


def _guess_unit(column: ColumnInfo) -> str:
    lowered = column.name.lower()
    if any(word in lowered for word in _MONEY_WORDS):
        return "money"
    if any(word in lowered for word in _RATE_WORDS):
        return "rate"
    return "count" if column.type in ("integer", "bigint", "smallint") else "decimal"


def _looks_like_identifier(column: ColumnInfo, rows: int) -> bool:
    lowered = column.name.lower()
    return (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered.endswith("_number")
        or (rows > 20 and column.distinct >= rows * 0.98 and column.type in ("integer", "bigint"))
    )


def derive_fields(columns: list[ColumnInfo], rows: int, metrics: list[MetricSpec]) -> DatasetFields:
    """The measures, dimensions and time column a dataset offers to the analysis tools."""
    out = DatasetFields()
    types = {c.name: c.type for c in columns}

    for column in columns:
        if column.type in DATE_TYPES and not column.pii:
            out.time_column, out.time_kind = column.name, "date"
            break
    if out.time_column is None:
        for column in columns:
            if column.month_like and not column.pii:
                out.time_column, out.time_kind = column.name, "month"
                break

    for column in columns:
        if column.pii or column.name == out.time_column:
            continue
        if (
            column.type in TEXT_TYPES
            and 2 <= column.distinct <= MAX_DIMENSION_VALUES
            and is_identifier(column.name)
        ):
            out.dimensions[column.name] = DimensionDef(
                id=column.name,
                label=humanise(column.name),
                column=column.name,
                plural=_plural(humanise(column.name).lower()),
                values=column.values,
            )

    # Governed metrics come first so the default measure is a shared definition.
    for spec in metrics:
        if applies_to(spec, types):
            out.measures[spec.key] = spec.to_measure()
    for column in columns:
        if column.pii or column.type not in NUMERIC_TYPES or column.name in out.measures:
            continue
        if _looks_like_identifier(column, rows) or not is_identifier(column.name):
            continue
        covered = any(m.kind == "additive" and m.column == column.name for m in out.measures.values())
        if covered:
            continue
        out.measures[column.name] = MeasureDef(
            id=column.name,
            label=humanise(column.name),
            unit=_guess_unit(column),
            kind="additive",
            description=f"Sum of {column.name}. Not a shared metric definition.",
            column=column.name,
        )
    if not any(m.kind == "count" for m in out.measures.values()) and "row_count" not in out.measures:
        out.measures["row_count"] = MeasureDef(
            id="row_count", label="Rows", unit="count", kind="count", description="Number of rows."
        )
    return out


def _plural(noun: str) -> str:
    if noun.endswith("y") and not noun.endswith(("ay", "ey", "oy", "uy")):
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def build_analysis_dataset(
    dataset_id: str, name: str, table: str, fields: DatasetFields, months: list[str]
) -> AnalysisDataset:
    governed = [m.id for m in fields.measures.values() if m.governed]
    default = governed[0] if governed else next(iter(fields.measures))
    return AnalysisDataset(
        id=dataset_id,
        name=name,
        table=table,
        measures=fields.measures,
        dimensions=fields.dimensions,
        default_measure=default,
        time_column=fields.time_column,
        time_kind=fields.time_kind,
        months=tuple(months),
    )


def metric_to_dict(spec: MetricSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "name": spec.name,
        "unit": spec.unit,
        "kind": spec.kind,
        "column": spec.column,
        "numerator": spec.numerator,
        "denominator": spec.denominator,
        "expression": expression_for(spec),
        "synonyms": list(spec.synonyms),
    }
