"""What an analysis can read (a dataset's measures, dimensions and months) and the settings of one analysis."""

import re
from dataclasses import dataclass, field, fields, replace
from datetime import date
from typing import Any

MONTH_DIMENSION = "month"
METHODS = ("summarise", "compare", "trend", "relationship", "explain")
UNITS = ("count", "money", "rate", "decimal")
MEASURE_KINDS = ("additive", "count", "distinct", "ratio")
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class AnalysisError(ValueError):
    """Settings that do not fit the dataset, such as an unknown measure."""


def is_identifier(name: str) -> bool:
    return bool(_IDENT.match(name))


@dataclass(frozen=True)
class MeasureDef:
    """A number the analysis can compute.

    additive: SUM(column) · count: COUNT(*) · distinct: COUNT(DISTINCT column) ·
    ratio: numerator ÷ denominator, where the denominator may be "*" for the row count.
    """

    id: str
    label: str
    unit: str
    kind: str
    description: str = ""
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    governed: bool = False
    metric_id: str | None = None

    @property
    def additive(self) -> bool:
        """Adds up across groups, so shares and contributions make sense."""
        return self.kind in ("additive", "count")

    @property
    def is_ratio(self) -> bool:
        return self.kind == "ratio"

    def columns(self) -> list[str]:
        names = [self.column, self.numerator, self.denominator]
        return [name for name in names if name and name != "*"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "unit": self.unit,
            "kind": "additive" if self.additive else self.kind,
            "description": self.description,
            "column": self.column,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "governed": self.governed,
            "metricId": self.metric_id,
        }


@dataclass(frozen=True)
class DimensionDef:
    id: str
    label: str
    column: str
    plural: str
    # Values in their natural order (largest group first), used to sort groups and fill filter lists.
    values: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "column": self.column, "values": list(self.values)}


@dataclass(frozen=True)
class AnalysisDataset:
    id: str
    name: str
    table: str  # schema-qualified, e.g. silver.course_completions
    measures: dict[str, MeasureDef]
    dimensions: dict[str, DimensionDef]
    default_measure: str
    time_column: str | None = None
    time_kind: str | None = None  # date | month (text written YYYY-MM)
    months: tuple[str, ...] = ()

    @property
    def has_time(self) -> bool:
        return self.time_column is not None and bool(self.months)

    def dimension_ids(self) -> list[str]:
        ids = list(self.dimensions)
        return [*ids, MONTH_DIMENSION] if self.has_time else ids

    def dimension_label(self, dim: str) -> str:
        return "Month" if dim == MONTH_DIMENSION else self.dimensions[dim].label

    def dimension_plural(self, dim: str) -> str:
        return "months" if dim == MONTH_DIMENSION else self.dimensions[dim].plural

    def dimension_order(self, dim: str) -> tuple[str, ...]:
        return self.months if dim == MONTH_DIMENSION else self.dimensions[dim].values

    def grains(self) -> list[str]:
        """Units of observation: pairs first (finer, more dots), then single dimensions."""
        dims = list(self.dimensions)
        out: list[str] = []
        if self.has_time:
            out += [f"{d}-{MONTH_DIMENSION}" for d in dims]
        out += [f"{a}-{b}" for i, a in enumerate(dims) for b in dims[i + 1 :]]
        out += dims
        if self.has_time:
            out.append(MONTH_DIMENSION)
        return out

    def month_label(self, month: str, short: bool = False) -> str:
        if not _MONTH.match(month):
            return month
        year, number = int(month[:4]), int(month[5:])
        name = _MONTH_NAMES[number - 1]
        if not short:
            return f"{name} {year}"
        # A short label is only unambiguous when the months span less than a year.
        spans_years = len({m[5:] for m in self.months}) < len(self.months)
        return f"{name} {str(year)[2:]}" if spans_years else name

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "table": self.table,
            "measures": [m.to_dict() for m in self.measures.values()],
            "dimensions": [d.to_dict() for d in self.dimensions.values()],
            "hasTime": self.has_time,
            "months": [
                {"id": m, "label": self.month_label(m), "short": self.month_label(m, True)}
                for m in self.months
            ],
            "grains": [{"id": g, "label": grain_label(self, g)} for g in self.grains()],
            "defaultMeasure": self.default_measure,
        }


def grain_dims(grain: str) -> list[str]:
    return grain.split("-")


def grain_label(dataset: AnalysisDataset, grain: str) -> str:
    dims = grain_dims(grain)
    if len(dims) == 1:
        return f"Each {_singular(dataset, dims[0])}"
    a, b = dims
    if b == MONTH_DIMENSION:
        return f"Each {_singular(dataset, a)} in each month"
    return f"Each {_singular(dataset, b)} in each {_singular(dataset, a)}"


def grain_column_label(dataset: AnalysisDataset, grain: str) -> str:
    return " · ".join(dataset.dimension_label(d) for d in grain_dims(grain))


def grain_nouns(dataset: AnalysisDataset, grain: str) -> tuple[str, str]:
    """(one, many): 'a region in one month', 'region-months'."""
    dims = grain_dims(grain)
    if len(dims) == 1:
        return f"a {_singular(dataset, dims[0])}", dataset.dimension_plural(dims[0])
    a, b = dims
    if b == MONTH_DIMENSION:
        return f"a {_singular(dataset, a)} in one month", f"{_singular(dataset, a)}-months"
    return (
        f"a {_singular(dataset, b)} in one {_singular(dataset, a)}",
        f"{_singular(dataset, b)}-{_singular(dataset, a)} pairs",
    )


def _singular(dataset: AnalysisDataset, dim: str) -> str:
    return "month" if dim == MONTH_DIMENSION else dataset.dimensions[dim].label.lower()


@dataclass(frozen=True)
class BuildConfig:
    dataset_id: str
    method: str
    measure: str
    measure_y: str
    dimension: str
    grain: str
    split: str = "none"
    date_from: str = ""
    date_to: str = ""
    filters: dict[str, str] = field(default_factory=dict)
    aggregation: str = "total"  # total | monthly-mean | monthly-median
    rate_method: str = "pooled"  # pooled | average
    display: str = "value"  # value | share | index
    sort: str = "desc"  # desc | asc | name
    centre: str = "median"  # median | mean
    outliers: str = "include"  # include | exclude
    change: str = "level"  # level | mom | cumulative
    smoothing: str = "none"  # none | ma3
    correlation: str = "pearson"  # pearson | spearman
    period_a: str = ""
    period_b: str = ""
    change_view: str = "amount"  # amount | percent

    def period(self) -> tuple[str, str]:
        return (
            (self.date_from, self.date_to)
            if self.date_from <= self.date_to
            else (self.date_to, self.date_from)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasetId": self.dataset_id,
            "method": self.method,
            "measure": self.measure,
            "measureY": self.measure_y,
            "dimension": self.dimension,
            "grain": self.grain,
            "split": self.split,
            "from": self.date_from,
            "to": self.date_to,
            "filters": dict(self.filters),
            "aggregation": self.aggregation,
            "rateMethod": self.rate_method,
            "display": self.display,
            "sort": self.sort,
            "centre": self.centre,
            "outliers": self.outliers,
            "change": self.change,
            "smoothing": self.smoothing,
            "correlation": self.correlation,
            "periodA": self.period_a,
            "periodB": self.period_b,
            "changeView": self.change_view,
        }


_CHOICES: dict[str, tuple[str, ...]] = {
    "aggregation": ("total", "monthly-mean", "monthly-median"),
    "rate_method": ("pooled", "average"),
    "display": ("value", "share", "index"),
    "sort": ("desc", "asc", "name"),
    "centre": ("median", "mean"),
    "outliers": ("include", "exclude"),
    "change": ("level", "mom", "cumulative"),
    "smoothing": ("none", "ma3"),
    "correlation": ("pearson", "spearman"),
    "change_view": ("amount", "percent"),
}
_CAMEL = {
    "datasetId": "dataset_id",
    "measureY": "measure_y",
    "from": "date_from",
    "to": "date_to",
    "rateMethod": "rate_method",
    "periodA": "period_a",
    "periodB": "period_b",
    "changeView": "change_view",
}


def measures_for(dataset: AnalysisDataset, method: str) -> list[str]:
    """Explaining a change needs a total that adds up, so rates and distinct counts are left out."""
    if method == "explain":
        return [m.id for m in dataset.measures.values() if m.additive]
    return list(dataset.measures)


def _pick_measure(dataset: AnalysisDataset, method: str, wanted: str | None) -> str:
    allowed = measures_for(dataset, method)
    if wanted and wanted in allowed:
        return wanted
    if dataset.default_measure in allowed:
        return dataset.default_measure
    if not allowed:
        raise AnalysisError("This dataset has no measure this method can use.")
    return allowed[0]


def _second_measure(dataset: AnalysisDataset, first: str) -> str:
    """A partner for the relationship method.

    Two totals are trivially related (bigger groups have more of everything), so a total is paired with a
    rate, preferably the rate it is the denominator of (enrolments against average fee), and a rate with
    another rate (sessions per learner against completion rate).
    """
    first_def = dataset.measures[first]
    others = [m for m in dataset.measures.values() if m.id != first]
    ratios = [m for m in others if m.is_ratio]
    if first_def.is_ratio:
        if ratios:
            return ratios[0].id
        denominators = [m for m in others if m.column and m.column == first_def.denominator]
        return (denominators or others or [first_def])[0].id
    per_unit = [m for m in ratios if m.denominator == first_def.column]
    return (per_unit or ratios or others or [first_def])[0].id


def _relationship_pair(dataset: AnalysisDataset) -> tuple[str, str]:
    """(x, y): the likely driver on x and the outcome on y."""
    ratios = [m for m in dataset.measures.values() if m.is_ratio]
    default = dataset.measures[dataset.default_measure]
    if len(ratios) >= 2:
        y = default if default.is_ratio else ratios[0]
        return _second_measure(dataset, y.id), y.id
    if len(ratios) == 1:
        y = ratios[0]
        return _second_measure(dataset, y.id), y.id
    return default.id, _second_measure(dataset, default.id)


def default_config(
    dataset: AnalysisDataset, method: str, presets: dict[str, str | None] | None = None
) -> BuildConfig:
    if method not in METHODS:
        raise AnalysisError(f"Unknown method {method!r}.")
    presets = presets or {}
    dims = list(dataset.dimensions)
    wanted_dim = presets.get("dimension")
    dimension = wanted_dim if wanted_dim in dims else (dims[0] if dims else MONTH_DIMENSION)
    measure = _pick_measure(dataset, method, presets.get("measure"))
    if method == "relationship" and not presets.get("measure"):
        measure, measure_y = _relationship_pair(dataset)
    else:
        measure_y = _second_measure(dataset, measure)
    grains = dataset.grains()
    months = dataset.months
    return BuildConfig(
        dataset_id=dataset.id,
        method=method,
        measure=measure,
        measure_y=measure_y,
        dimension=dimension,
        grain=grains[0] if grains else dimension,
        date_from=months[0] if months else "",
        date_to=months[-1] if months else "",
        period_a=months[-2] if len(months) > 1 else (months[0] if months else ""),
        period_b=months[-1] if months else "",
    )


def config_from_dict(dataset: AnalysisDataset, data: dict[str, Any]) -> BuildConfig:
    """Validate a build configuration (camelCase or snake_case keys) against the dataset's fields."""
    method = str(data.get("method") or "compare")
    base = default_config(
        dataset, method, {"measure": data.get("measure"), "dimension": data.get("dimension")}
    )
    known = {f.name for f in fields(BuildConfig)}
    values: dict[str, Any] = {}
    for key, value in data.items():
        name = _CAMEL.get(key, key)
        if name in known and value is not None and name != "filters":
            values[name] = str(value)
    filters: dict[str, str] = {}
    raw_filters = data.get("filters") or {}
    if not isinstance(raw_filters, dict):
        raise AnalysisError("filters must be an object of dimension → value.")
    # The frontend's region and course selectors arrive as top-level keys.
    for key in dataset.dimensions:
        if key in data and key not in known:
            raw_filters = {**raw_filters, key: data[key]}
    for dim, value in raw_filters.items():
        if value in (None, "", "all"):
            continue
        if dim not in dataset.dimensions:
            raise AnalysisError(f"{dim!r} is not a dimension of {dataset.name}.")
        allowed = dataset.dimensions[dim].values
        if allowed and str(value) not in allowed:
            raise AnalysisError(f"{value!r} is not a known {dataset.dimensions[dim].label.lower()}.")
        filters[dim] = str(value)
    values.pop("dataset_id", None)
    config = replace(base, **values, filters=filters)
    return validate_config(dataset, config)


def validate_config(dataset: AnalysisDataset, config: BuildConfig) -> BuildConfig:
    if config.method not in METHODS:
        raise AnalysisError(f"Unknown method {config.method!r}.")
    for name, allowed in _CHOICES.items():
        value = getattr(config, name)
        if value not in allowed:
            raise AnalysisError(f"{name} must be one of {', '.join(allowed)}.")
    allowed_measures = measures_for(dataset, config.method)
    if config.measure not in allowed_measures:
        if config.method == "explain" and config.measure in dataset.measures:
            raise AnalysisError(
                "Contributions need a total, such as revenue or enrolments. Pick one of those."
            )
        raise AnalysisError(f"{config.measure!r} is not a measure of {dataset.name}.")
    if config.measure_y not in dataset.measures:
        raise AnalysisError(f"{config.measure_y!r} is not a measure of {dataset.name}.")
    if config.dimension not in dataset.dimensions and not (
        config.dimension == MONTH_DIMENSION and dataset.has_time
    ):
        raise AnalysisError(f"{config.dimension!r} is not a dimension of {dataset.name}.")
    if config.grain not in dataset.grains():
        raise AnalysisError(f"{config.grain!r} is not a unit this dataset supports.")
    if config.split != "none" and config.split not in dataset.dimensions:
        raise AnalysisError(f"Cannot split by {config.split!r}.")
    if config.method in ("trend", "explain") and not dataset.has_time:
        raise AnalysisError(f"{dataset.name} has no date column, so it cannot show change over time.")
    if dataset.has_time:
        for name in ("date_from", "date_to", "period_a", "period_b"):
            value = getattr(config, name)
            if not _MONTH.match(value):
                raise AnalysisError(f"{name} must be a month written YYYY-MM.")
    return config


def month_of(value: Any) -> str | None:
    if isinstance(value, date):
        return f"{value.year:04d}-{value.month:02d}"
    text = str(value or "")
    return text[:7] if re.match(r"^\d{4}-\d{2}", text) else None
