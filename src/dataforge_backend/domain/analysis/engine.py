"""The analysis engine: plans the grouped query for a method, then turns its rows into a readable result.

The result carries a title, a plain reading, how the numbers were produced, caveats, a chart, a table,
headline figures and the display SQL. Formatting is described (unit, percent, index) rather than done with
functions, so the frontend formats chart ticks and tooltips the same way it formats everything else.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .format import format_compact, format_delta, format_number, format_percent, format_value, js_fixed
from .model import (
    MONTH_DIMENSION,
    AnalysisDataset,
    AnalysisError,
    BuildConfig,
    MeasureDef,
    grain_column_label,
    grain_dims,
    grain_label,
    grain_nouns,
)
from .sql import GroupQuery, build_sql, group_query
from .stats import (
    correlation_strength,
    finite,
    linear_fit,
    mean,
    median,
    pearson,
    quantile,
    spearman,
    standard_deviation,
    total,
    tukey_fences,
)

NAN = float("nan")


# ------------------------------------------------------------------ planning


def plan(dataset: AnalysisDataset, config: BuildConfig, policy: str | None = None) -> GroupQuery:
    """The one grouped query each method needs."""
    m = dataset.measures[config.measure]
    method = config.method
    if method == "summarise":
        return group_query(
            dataset, config, grain_dims(config.grain), {"v": (m, config.rate_method)}, policy=policy
        )
    if method == "relationship":
        y = dataset.measures[config.measure_y]
        values = {"x": (m, config.rate_method), "y": (y, config.rate_method)}
        return group_query(dataset, config, grain_dims(config.grain), values, policy=policy)
    if method == "trend":
        dims = [MONTH_DIMENSION] + ([config.split] if config.split != "none" else [])
        return group_query(
            dataset, config, dims, {"v": (m, config.rate_method)}, parts={"p": m}, policy=policy
        )
    if method == "explain":
        if not m.additive:
            raise AnalysisError(
                "Contributions need a total, such as revenue or enrolments. Pick one of those."
            )
        return group_query(
            dataset,
            config,
            [config.dimension, MONTH_DIMENSION],
            {"v": (m, "pooled")},
            period="pair",
            policy=policy,
        )
    monthly = (
        m.additive
        and config.aggregation != "total"
        and dataset.has_time
        and config.dimension != MONTH_DIMENSION
    )
    dims = [config.dimension, MONTH_DIMENSION] if monthly else [config.dimension]
    return group_query(dataset, config, dims, {"v": (m, config.rate_method)}, parts={"p": m}, policy=policy)


# ------------------------------------------------------------------ grouping


@dataclass
class _Bar:
    name: str
    value: float
    volume: float
    share: float = NAN
    index: float = NAN


@dataclass
class _LineChange:
    name: str
    first: float
    last: float

    @property
    def delta(self) -> float:
        return self.last - self.first


@dataclass
class _Dot:
    name: str
    x: float
    y: float


@dataclass
class _Move:
    name: str
    a: float
    b: float

    @property
    def delta(self) -> float:
        return self.b - self.a

    @property
    def pct(self) -> float:
        return NAN if self.a == 0 else self.delta / self.a * 100


@dataclass
class _Group:
    parts: tuple[str, ...]
    label: str
    rows: list[dict[str, Any]]


def _float(value: Any) -> float:
    if value is None:
        return NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def _group(
    dataset: AnalysisDataset, rows: list[dict[str, Any]], dims: list[str], positions: list[int]
) -> list[_Group]:
    """Group rows on the dimensions at `positions` of the query, in each dimension's natural order."""
    groups: dict[tuple[str, ...], _Group] = {}
    for row in rows:
        parts = tuple(str(row[f"d{p}"]) for p in positions)
        group = groups.get(parts)
        if group is None:
            label = " · ".join(
                dataset.month_label(part, True) if dims[p] == MONTH_DIMENSION else part
                for part, p in zip(parts, positions, strict=True)
            )
            group = groups[parts] = _Group(parts, label, [])
        group.rows.append(row)

    def key(group: _Group) -> tuple[Any, ...]:
        out: list[Any] = []
        for part, p in zip(group.parts, positions, strict=True):
            order = dataset.dimension_order(dims[p])
            out.append((order.index(part) if part in order else len(order), part))
        return tuple(out)

    return sorted(groups.values(), key=key)


def _scope_text(dataset: AnalysisDataset, config: BuildConfig, period: str) -> str:
    parts: list[str] = []
    if period == "range" and dataset.has_time:
        low, high = config.period()
        parts.append(
            dataset.month_label(low)
            if low == high
            else f"{dataset.month_label(low)} to {dataset.month_label(high)}"
        )
    parts += list(config.filters.values())
    return ", ".join(parts) or "all rows"


def _rate_sentence(dataset: AnalysisDataset, m: MeasureDef, rate_method: str) -> str:
    if m.kind == "distinct":
        return f"counted distinct {m.column}"
    if not m.is_ratio:
        return f"added up {m.label.lower()}"
    denominator = "the row count" if m.denominator == "*" else f"total {m.denominator}"
    if rate_method == "average":
        return f"averaged each row's {m.label.lower()}, giving every row the same weight"
    return f"pooled {m.label.lower()}: total {m.numerator} divided by {denominator}"


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def _list_names(names: list[str], limit: int = 3) -> str:
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} and {len(names) - limit} more"


def _measure_format(unit: str) -> dict[str, Any]:
    return {"type": "measure", "unit": unit}


def _compact_format(unit: str) -> dict[str, Any]:
    return {"type": "compact", "unit": unit}


def _empty(
    dataset: AnalysisDataset, config: BuildConfig, title: str, reading: str | None = None
) -> dict[str, Any]:
    return {
        "title": title,
        "reading": reading or "No rows match these settings. Widen the period or clear a filter.",
        "method": "",
        "notes": [],
        "visual": {"kind": "bars", "points": []},
        "format": {"type": "number"},
        "axisFormat": {"type": "number"},
        "table": {"columns": [], "rows": []},
        "sql": build_sql(dataset, config),
        "figures": [],
        "empty": True,
    }


def _denominator_label(dataset: AnalysisDataset, m: MeasureDef) -> str:
    if m.denominator == "*":
        return "Rows"
    for other in dataset.measures.values():
        if other.column == m.denominator and other.additive:
            return other.label
    return _capitalise((m.denominator or "").replace("_", " "))


# ------------------------------------------------------------------ summarise


def _histogram(values: list[float], unit: str, count: int = 6) -> list[dict[str, Any]]:
    low, high = min(values), max(values)
    if low == high:
        return [{"name": format_compact(low, unit), "value": len(values)}]
    width = (high - low) / count
    bins = [0] * count
    for value in values:
        bins[min(count - 1, math.floor((value - low) / width))] += 1
    return [
        {
            "name": f"{format_compact(low + i * width, unit)}–{format_compact(low + (i + 1) * width, unit)}",
            "value": n,
        }
        for i, n in enumerate(bins)
    ]


def _summarise(dataset: AnalysisDataset, config: BuildConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    m = dataset.measures[config.measure]

    def fmt(v: float) -> str:
        return format_value(v, m.unit)

    dims = grain_dims(config.grain)
    _, many = grain_nouns(dataset, config.grain)
    title = f"{m.label}, {grain_label(dataset, config.grain).lower()}"

    units = [
        (g.label, _float(g.rows[0]["v"]))
        for g in _group(dataset, rows, dims, list(range(len(dims))))
        if finite(_float(g.rows[0]["v"]))
    ]
    if not units:
        return _empty(dataset, config, title)
    low_fence, high_fence = tukey_fences([v for _, v in units])

    def unusual(v: float) -> bool:
        return v < low_fence or v > high_fence

    flagged = [label for label, v in units if unusual(v)]
    used = [(label, v) for label, v in units if not unusual(v)] if config.outliers == "exclude" else units
    values = [v for _, v in used]
    if not values:
        return _empty(dataset, config, title)

    n, excluded = len(values), len(units) - len(used)
    avg, mid, std = mean(values), median(values), standard_deviation(values)
    q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
    centre = mid if config.centre == "median" else avg
    skew = 0 if mid == 0 else (avg - mid) / abs(mid)

    reading = (
        f"Across {n} {many}, the {config.centre} {m.label.lower()} is {fmt(centre)}. "
        f"The middle half sits between {fmt(q1)} and {fmt(q3)}."
    )
    if abs(skew) > 0.1:
        reading += (
            f" The mean ({fmt(avg)}) is {'above' if skew > 0 else 'below'} the median, "
            f"so a few {'large' if skew > 0 else 'small'} values pull it."
        )
    if flagged:
        verb = "value sits" if len(flagged) == 1 else "values sit"
        left_out = " and are left out" if config.outliers == "exclude" else ""
        reading += f" {len(flagged)} {verb} outside the usual range ({_list_names(flagged)}){left_out}."

    notes: list[str] = []
    if n < 10:
        notes.append(f"Only {n} values. The spread and any unusual values are fragile.")
    if m.unit == "money" and config.centre == "mean":
        notes.append("Money is usually skewed. The median is often the safer typical value.")
    if m.is_ratio and config.rate_method == "average":
        notes.append("Averaging rates gives every row the same weight, whatever its size.")

    left = f", leaving out {excluded} beyond the Tukey fences (1.5 × IQR)" if excluded else ""
    figures: list[dict[str, Any]] = [
        {"label": "Median" if config.centre == "median" else "Mean", "value": fmt(centre)},
        {"label": "Middle half", "value": f"{fmt(q1)} – {fmt(q3)}"},
        {"label": "Standard deviation", "value": fmt(std)},
        {
            "label": "Values",
            "value": format_number(n),
            **({"detail": f"{excluded} left out"} if excluded else {}),
        },
    ]
    return {
        "title": title,
        "reading": reading,
        "method": f"Grouped {dataset.name} into {grain_label(dataset, config.grain).lower()} "
        f"({_scope_text(dataset, config, 'range')}). For each, {_rate_sentence(dataset, m, config.rate_method)}. "
        f"Then summarised the {len(units)} values{left}.",
        "notes": notes,
        "visual": {"kind": "histogram", "bins": _histogram(values, m.unit)},
        "format": {"type": "count", "singular": "value", "plural": "values"},
        "axisFormat": {"type": "number"},
        "table": {
            "columns": [grain_column_label(dataset, config.grain), m.label, "Unusual?"],
            "rows": [
                [
                    label,
                    fmt(v),
                    ("Yes · left out" if config.outliers == "exclude" else "Yes") if unusual(v) else "",
                ]
                for label, v in sorted(units, key=lambda u: -u[1])
            ],
        },
        "sql": build_sql(dataset, config),
        "figures": figures,
    }


# ------------------------------------------------------------------ compare

_AGGREGATION_LABELS = {
    "total": "Total",
    "monthly-mean": "Average monthly",
    "monthly-median": "Median monthly",
}


def _compare(dataset: AnalysisDataset, config: BuildConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    m = dataset.measures[config.measure]

    def fmt(v: float) -> str:
        return format_value(v, m.unit)

    dim = config.dimension
    dim_label = dataset.dimension_label(dim)
    additive = m.additive
    monthly = additive and config.aggregation != "total" and dataset.has_time and dim != MONTH_DIMENSION
    value_header = f"{_AGGREGATION_LABELS[config.aggregation]} {m.label.lower()}" if additive else m.label
    title = f"{value_header} by {dim_label.lower()}"
    dims = [dim, MONTH_DIMENSION] if monthly else [dim]

    items: list[_Bar] = []
    for g in _group(dataset, rows, dims, [0]):
        if monthly:
            month_values = [_float(r["v"]) for r in g.rows if finite(_float(r["v"]))]
            value = median(month_values) if config.aggregation == "monthly-median" else mean(month_values)
            volume = total(month_values)
        else:
            value = _float(g.rows[0]["v"])
            volume = _float(g.rows[0]["p_den"]) if m.is_ratio else value
        if finite(value):
            items.append(_Bar(g.label, value, volume if finite(volume) else 0.0))
    if not items:
        return _empty(dataset, config, title)

    grand = total([i.value for i in items])
    average = mean([i.value for i in items])
    display = "value" if (not additive and config.display == "share") else config.display
    for item in items:
        item.share = item.value / grand * 100 if grand else NAN
        item.index = item.value / average * 100 if average else NAN
    if config.sort == "name":
        ordered = sorted(items, key=lambda i: i.name.lower())
    else:
        ordered = sorted(items, key=lambda i: i.value, reverse=config.sort != "asc")
    by_value = sorted(items, key=lambda i: -i.value)
    top, bottom = by_value[0], by_value[-1]

    def shown(item: _Bar) -> float:
        return float(item.share if display == "share" else item.index if display == "index" else item.value)

    if display == "share":
        value_format: dict[str, Any] = {"type": "share"}
    elif display == "index":
        value_format = {"type": "index"}
    else:
        value_format = _measure_format(m.unit)

    plural = dataset.dimension_plural(dim)
    singular = dim_label.lower()
    reading = f"{top.name} is highest at {fmt(top.value)} and {bottom.name} is lowest at {fmt(bottom.value)}."
    if len(items) == 1:
        reading = (
            f"{top.name}: {fmt(top.value)}. Only one {singular} matches the filters, so there is nothing to "
            f"compare it with. Clear the {singular} filter to compare."
        )
    elif additive and config.aggregation == "total":
        reading += f" {top.name} alone accounts for {js_fixed(top.share, 0)}% of the total."
    elif not additive:
        reading += f" The gap between them is {format_delta(top.value - bottom.value, m.unit)}."

    largest = max(i.volume for i in items)
    small = [i.name for i in items if i.volume < largest * 0.15]
    notes: list[str] = []
    if small:
        one = len(small) == 1
        notes.append(
            f"{_list_names(small)} {'is' if one else 'are'} much smaller than the largest {singular}, so "
            f"{'its figure moves' if one else 'their figures move'} more from month to month."
        )
    if m.is_ratio and config.rate_method == "average":
        notes.append(
            "Averaging rates gives small groups the same weight as large ones. Pooling is usually the fairer choice."
        )
    if not additive and config.display == "share":
        notes.append(
            "Shares of a total only make sense for measures that add up, so values are shown instead."
        )
    if additive and config.aggregation != "total" and not monthly:
        notes.append("This dataset has no months to average over, so totals are shown.")

    volume_label = _denominator_label(dataset, m) if m.is_ratio else None
    columns = [dim_label, value_header]
    if additive:
        columns.append("Share of total")
    columns.append("Index (100 = average)")
    if volume_label:
        columns.append(volume_label)
    table_rows: list[list[Any]] = []
    for item in ordered:
        cells: list[Any] = [item.name, fmt(item.value)]
        if additive:
            cells.append(f"{js_fixed(item.share, 1)}%")
        cells.append(f"{js_fixed(item.index, 0)}" if finite(item.index) else "—")
        if volume_label:
            cells.append(format_number(item.volume))
        table_rows.append(cells)

    if monthly:
        how = (
            f"added up {m.label.lower()} per month, then took the "
            f"{'median' if config.aggregation == 'monthly-median' else 'average'} month"
        )
    else:
        how = _rate_sentence(dataset, m, config.rate_method)
    method = f"For each {singular} ({_scope_text(dataset, config, 'range')}), {how}."
    if display == "share":
        method += " Shares are each group's part of the sum of all groups."
    if display == "index":
        method += " The index divides each group by the average group and multiplies by 100."

    if len(items) == 1:
        figures: list[dict[str, Any]] = [{"label": top.name, "value": fmt(top.value)}]
    else:
        third = (
            {
                "label": "Highest ÷ lowest",
                "value": f"{js_fixed(top.value / bottom.value, 1)}×" if bottom.value > 0 else "—",
            }
            if additive
            else {"label": "Gap", "value": format_delta(top.value - bottom.value, m.unit)}
        )
        figures = [
            {"label": "Highest", "value": fmt(top.value), "detail": top.name},
            {"label": "Lowest", "value": fmt(bottom.value), "detail": bottom.name},
            third,
            {"label": _capitalise(plural), "value": str(len(items))},
        ]

    visual: dict[str, Any] = {
        "kind": "bars",
        "points": [{"name": i.name, "value": shown(i)} for i in ordered],
    }
    if display == "index":
        visual["reference"] = 100
    return {
        "title": title,
        "reading": reading,
        "method": method,
        "notes": notes,
        "visual": visual,
        "format": value_format,
        "axisFormat": _compact_format(m.unit) if display == "value" else value_format,
        "table": {"columns": columns, "rows": table_rows},
        "sql": build_sql(dataset, config),
        "figures": figures,
    }


# ------------------------------------------------------------------ trend


def _trend(dataset: AnalysisDataset, config: BuildConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    m = dataset.measures[config.measure]

    def fmt(v: float) -> str:
        return format_value(v, m.unit)

    low, high = config.period()
    months = [month for month in dataset.months if low <= month <= high]
    split = config.split if config.split != "none" else None
    change_label = {"mom": "month-on-month change", "cumulative": "running total"}.get(config.change, "")
    title = (
        f"{m.label}{f', {change_label}' if change_label else ''}"
        f"{f' by {dataset.dimension_label(split).lower()}' if split else ''}, month by month"
    )
    if not rows or not months:
        return _empty(dataset, config, title)

    # cells[(group, month)] = row
    series_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    omitted = 0
    if split:
        sizes: dict[str, float] = defaultdict(float)
        for row in rows:
            size = _float(row["p_den"])
            sizes[str(row["d1"])] += size if finite(size) else 0.0
        order = dataset.dimension_order(split)
        ranked = sorted(sizes, key=lambda g: (-sizes[g], order.index(g) if g in order else len(order), g))
        keep = ranked[:4]
        omitted = max(0, len(ranked) - 4)
        for row in rows:
            if str(row["d1"]) in keep:
                series_rows[str(row["d1"])][str(row["d0"])] = row
        names = keep
    else:
        for row in rows:
            series_rows[m.label][str(row["d0"])] = row
        names = [m.label]

    series: list[tuple[str, list[float | None]]] = []
    for name in names:
        per_month = [series_rows[name].get(month) for month in months]
        level = [_float(r["v"]) if r else NAN for r in per_month]
        values: list[float | None]
        if config.change == "mom":
            values = []
            for i, value in enumerate(level):
                previous = level[i - 1] if i > 0 else NAN
                ok = i > 0 and finite(previous) and previous != 0 and finite(value)
                values.append((value - previous) / previous * 100 if ok else None)
        elif config.change == "cumulative":
            values = []
            if m.is_ratio and config.rate_method == "pooled":
                top_sum = bottom_sum = 0.0
                for r in per_month:
                    if r:
                        top_sum += _float(r["p_num"]) if finite(_float(r["p_num"])) else 0.0
                        bottom_sum += _float(r["p_den"]) if finite(_float(r["p_den"])) else 0.0
                    values.append(None if bottom_sum == 0 else top_sum / bottom_sum)
            else:
                running = 0.0
                for i, value in enumerate(level):
                    running += value if finite(value) else 0.0
                    values.append(running if m.additive else running / (i + 1))
        else:
            values = [value if finite(value) else None for value in level]
        if config.smoothing == "ma3":
            smoothed: list[float | None] = []
            for i in range(len(values)):
                window = values[max(0, i - 2) : i + 1]
                ok = i >= 2 and all(v is not None for v in window)
                smoothed.append(mean([v for v in window if v is not None]) if ok else None)
            values = smoothed
        series.append((name, values))

    is_mom = config.change == "mom"
    value_format = {"type": "percent", "digits": 1} if is_mom else _measure_format(m.unit)
    axis_format = {"type": "percent", "digits": 0} if is_mom else _compact_format(m.unit)

    def render(v: float) -> str:
        return format_percent(v) if is_mom else fmt(v)

    points = []
    for i, month in enumerate(months):
        point: dict[str, Any] = {"name": dataset.month_label(month, True)}
        for name, values in series:
            point[name] = values[i]
        points.append(point)

    def first_last(values: list[float | None]) -> tuple[tuple[float, int] | None, tuple[float, int] | None]:
        indexed = [(v, i) for i, v in enumerate(values) if v is not None]
        return (indexed[0], indexed[-1]) if indexed else (None, None)

    def change_text(first: float, last: float) -> str:
        # Rates change in points; totals change in per cent.
        if m.is_ratio:
            return format_delta(last - first, m.unit)
        return format_percent(NAN if first == 0 else (last - first) / abs(first) * 100)

    running_noun = "total" if m.additive else "figure"

    def month_name(i: int) -> str:
        return dataset.month_label(months[i]) if 0 <= i < len(months) else ""

    figures: list[dict[str, Any]] = []

    if is_mom:
        points_all = [(v, i, name) for name, values in series for i, v in enumerate(values) if v is not None]
        rise = max(points_all, key=lambda p: p[0]) if points_all else None
        fall = min(points_all, key=lambda p: p[0]) if points_all else None

        def who(p: tuple[float, int, str]) -> str:
            return f" ({p[2]})" if split else ""

        high_name = "Smallest fall" if rise and rise[0] < 0 else "Biggest rise"
        low_name = "Smallest rise" if fall and fall[0] >= 0 else "Biggest fall"
        if rise and fall:
            reading = (
                f"{high_name} from one month to the next: {format_percent(rise[0])} in {month_name(rise[1])}{who(rise)}. "
                f"{low_name}: {format_percent(fall[0])} in {month_name(fall[1])}{who(fall)}."
            )
            figures += [
                {
                    "label": high_name,
                    "value": format_percent(rise[0]),
                    "detail": f"{month_name(rise[1])}{who(rise)}",
                },
                {
                    "label": low_name,
                    "value": format_percent(fall[0]),
                    "detail": f"{month_name(fall[1])}{who(fall)}",
                },
            ]
        else:
            reading = "Month-on-month change needs at least two months."
    elif not split:
        values = series[0][1]
        first, last = first_last(values)
        if not first or not last:
            reading = "Not enough months with data to show a trend."
        elif config.change == "cumulative":
            reading = (
                f"By {month_name(last[1])}, the running {running_noun} reached {fmt(last[0])}, "
                f"starting from {fmt(first[0])} in {month_name(first[1])}."
            )
            figures += [
                {"label": f"Running {running_noun} by {month_name(last[1])}", "value": fmt(last[0])},
                {"label": month_name(first[1]), "value": fmt(first[0])},
                {"label": "Months", "value": str(len(months))},
            ]
        else:
            direction = "rose" if last[0] > first[0] else "fell" if last[0] < first[0] else "held steady"
            peak = max(((v, i) for i, v in enumerate(values) if v is not None), key=lambda p: p[0])
            reading = (
                f"{m.label} {direction} from {fmt(first[0])} in {month_name(first[1])} to {fmt(last[0])} in "
                f"{month_name(last[1])} ({change_text(first[0], last[0])}). "
                f"The highest month was {month_name(peak[1])} at {fmt(peak[0])}."
            )
            figures += [
                {"label": month_name(first[1]), "value": fmt(first[0])},
                {"label": month_name(last[1]), "value": fmt(last[0])},
                {"label": "Change", "value": change_text(first[0], last[0])},
                {"label": "Highest month", "value": fmt(peak[0]), "detail": month_name(peak[1])},
            ]
    elif config.change == "cumulative":
        ends = sorted(
            ((name, first_last(values)[1]) for name, values in series if first_last(values)[1]),
            key=lambda e: -(e[1][0] if e[1] else 0),
        )
        if ends:
            (hi_name, hi), (lo_name, lo) = ends[0], ends[-1]
            assert hi and lo
            reading = (
                f"By {month_name(hi[1])}, {hi_name} has the highest running {running_noun} ({fmt(hi[0])}) "
                f"and {lo_name} the lowest ({fmt(lo[0])})."
            )
            for name, end in ends:
                assert end
                figures.append({"label": name, "value": fmt(end[0]), "detail": f"by {month_name(end[1])}"})
        else:
            reading = "Not enough months with data to compare lines."
    else:
        changes: list[_LineChange] = []
        for name, values in series:
            first, last = first_last(values)
            if first and last:
                changes.append(_LineChange(name, first[0], last[0]))
        if changes:
            best = max(changes, key=lambda c: c.delta)
            worst = min(changes, key=lambda c: c.delta)
            reading = (
                f"Of the {len(changes)} largest {dataset.dimension_plural(split)}, {best.name} "
                f"{'fell least' if best.delta < 0 else 'gained most'} ({change_text(best.first, best.last)}) "
                f"and {worst.name} {'fell most' if worst.delta < 0 else 'gained least'} "
                f"({change_text(worst.first, worst.last)})."
            )
            for c in changes:
                figures.append(
                    {
                        "label": c.name,
                        "value": fmt(c.last),
                        "detail": change_text(c.first, c.last),
                    }
                )
        else:
            reading = "Not enough months with data to compare lines."

    notes: list[str] = []
    if len(months) < 4:
        notes.append(f"Only {len(months)} months. That is too short to call a trend.")
    if omitted and split:
        plural = dataset.dimension_plural(split)
        notes.append(
            f"Showing the 4 largest {plural}. {omitted} smaller {'one is' if omitted == 1 else 'ones are'} left out "
            "to keep the chart readable."
        )
    if config.smoothing == "ma3":
        notes.append("The three-month average starts once three months are available.")
    if config.change == "cumulative" and m.is_ratio and config.rate_method == "average":
        notes.append("For averaged rates, the running figure is the mean of the monthly values so far.")

    method = f"For each month ({_scope_text(dataset, config, 'range')}), {_rate_sentence(dataset, m, config.rate_method)}."
    if is_mom:
        method += " Then took the percentage change from the month before."
    if config.change == "cumulative":
        method += " Then added the months up as they went."
    if config.smoothing == "ma3":
        method += " Finally, averaged each month with the two before it."

    return {
        "title": title,
        "reading": reading,
        "method": method,
        "notes": notes,
        "visual": {"kind": "lines", "points": points, "series": [name for name, _ in series]},
        "format": value_format,
        "axisFormat": axis_format,
        "table": {
            "columns": ["Month", *[name for name, _ in series]],
            "rows": [
                [
                    dataset.month_label(month),
                    *["—" if (v := values[i]) is None else render(v) for _, values in series],
                ]
                for i, month in enumerate(months)
            ],
        },
        "sql": build_sql(dataset, config),
        "figures": figures,
    }


# ------------------------------------------------------------------ relationship


def _relationship(
    dataset: AnalysisDataset, config: BuildConfig, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    x_def, y_def = dataset.measures[config.measure], dataset.measures[config.measure_y]
    one, many = grain_nouns(dataset, config.grain)
    title = f"{y_def.label} against {x_def.label}"
    if x_def.id == y_def.id:
        return _empty(dataset, config, title, "Choose two different measures to compare.")
    dims = grain_dims(config.grain)
    units: list[_Dot] = []
    for g in _group(dataset, rows, dims, list(range(len(dims)))):
        x, y = _float(g.rows[0]["x"]), _float(g.rows[0]["y"])
        if finite(x) and finite(y):
            units.append(_Dot(g.label, x, y))
    kept = units
    if config.outliers == "exclude" and len(units) >= 4:
        fx = tukey_fences([u.x for u in units])
        fy = tukey_fences([u.y for u in units])
        kept = [u for u in units if fx[0] <= u.x <= fx[1] and fy[0] <= u.y <= fy[1]]
    excluded = len(units) - len(kept)
    xs, ys = [u.x for u in kept], [u.y for u in kept]
    is_pearson = config.correlation == "pearson"
    r = pearson(xs, ys) if is_pearson else spearman(xs, ys)
    symbol = "r" if is_pearson else "ρ"
    fit = linear_fit(xs, ys) if is_pearson else None

    notes = [
        "Two measures moving together is not proof that one causes the other.",
        f"Each dot is {one}, not a person. The result describes groups, not individual people.",
    ]
    if len(kept) < 10:
        notes.append(f"Only {len(kept)} dots. A correlation from so few points is unstable.")
    if x_def.is_ratio or y_def.is_ratio:
        notes.append(
            "Rates are averaged per row within each dot."
            if config.rate_method == "average"
            else "Rates are pooled within each dot: total numerator divided by total denominator."
        )

    direction = "positive" if r > 0 else "negative"
    strength = correlation_strength(r) if finite(r) else ""
    if not finite(r):
        reading = "There are not enough varied points to measure a relationship. Widen the period or choose a finer unit."
    else:
        reading = (
            f"There is {strength} {direction} relationship between {x_def.label.lower()} and {y_def.label.lower()} "
            f"across {len(kept)} {many} ({symbol} = {js_fixed(r, 2)})."
        )
        if abs(r) >= 0.1:
            reading += f" Where {x_def.label.lower()} is higher, {y_def.label.lower()} tends to be {'higher' if r > 0 else 'lower'}."
            if is_pearson:
                reading += f" A straight line accounts for about {round(r * r * 100)}% of the variation."

    size_word = strength.removeprefix("an ").removeprefix("a ")
    figures: list[dict[str, Any]] = [
        {
            "label": "Pearson r" if is_pearson else "Spearman ρ",
            "value": f"{js_fixed(r, 2)}" if finite(r) else "—",
        },
        {"label": "Strength", "value": _capitalise(f"{size_word}, {direction}") if finite(r) else "—"},
    ]
    if is_pearson and finite(r):
        figures.append({"label": "Shared variation (r²)", "value": f"{round(r * r * 100)}%"})
    figures.append(
        {"label": "Dots", "value": str(len(kept)), **({"detail": f"{excluded} left out"} if excluded else {})}
    )

    visual_fit = None
    if fit and len(kept) > 1:
        slope, intercept = fit
        lo, hi = min(xs), max(xs)
        visual_fit = {"x1": lo, "y1": intercept + slope * lo, "x2": hi, "y2": intercept + slope * hi}
    technique = (
        "Pearson's correlation, which looks for a straight-line pattern"
        if is_pearson
        else "Spearman's rank correlation, which compares their order rather than their size"
    )
    left = f", after leaving out {excluded} dots beyond the Tukey fences" if excluded else ""
    return {
        "title": title,
        "reading": reading,
        "method": f"Grouped {dataset.name} into {grain_label(dataset, config.grain).lower()} "
        f"({_scope_text(dataset, config, 'range')}) and worked out both measures for each. Then measured how closely "
        f"they move together with {technique}{left}.",
        "notes": notes,
        "visual": {
            "kind": "scatter",
            "points": [{"name": u.name, "x": u.x, "y": u.y} for u in kept],
            "fit": visual_fit,
            "xLabel": x_def.label,
            "yLabel": y_def.label,
            "xUnit": x_def.unit,
            "yUnit": y_def.unit,
        },
        "format": _measure_format(y_def.unit),
        "axisFormat": _compact_format(y_def.unit),
        "table": {
            "columns": [grain_column_label(dataset, config.grain), x_def.label, y_def.label],
            "rows": [[u.name, format_value(u.x, x_def.unit), format_value(u.y, y_def.unit)] for u in kept],
        },
        "sql": build_sql(dataset, config),
        "figures": figures,
    }


# ------------------------------------------------------------------ explain


def _explain(dataset: AnalysisDataset, config: BuildConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    m = dataset.measures[config.measure]

    def fmt(v: float) -> str:
        return format_value(v, m.unit)

    dim = config.dimension
    singular = dataset.dimension_label(dim).lower()
    a, b = config.period_a, config.period_b
    la, lb = dataset.month_label(a), dataset.month_label(b)
    percent_view = config.change_view == "percent"
    title = f"What changed {m.label.lower()} from {la} to {lb}, by {singular}"
    if not m.additive:
        return _empty(
            dataset,
            config,
            title,
            "Contributions need a total, such as revenue or enrolments. Pick one of those.",
        )
    if a == b:
        return _empty(dataset, config, title, "Pick two different months to compare.")

    groups: list[_Move] = []
    for g in _group(dataset, rows, [dim, MONTH_DIMENSION], [0]):
        by_month = {str(r["d1"]): _float(r["v"]) for r in g.rows}
        va = by_month.get(a, 0.0)
        vb = by_month.get(b, 0.0)
        groups.append(_Move(g.label, va if finite(va) else 0.0, vb if finite(vb) else 0.0))
    if not groups:
        return _empty(dataset, config, title)

    total_a, total_b = total([g.a for g in groups]), total([g.b for g in groups])
    total_delta = total_b - total_a

    def share(d: float) -> float:
        return NAN if total_delta == 0 else d / total_delta * 100

    by_amount = sorted(groups, key=lambda g: -abs(g.delta))
    by_percent = sorted((g for g in groups if finite(g.pct)), key=lambda g: -abs(g.pct))
    ordered = by_percent if percent_view else by_amount
    top = by_amount[0]
    relative = by_percent[0] if by_percent else None

    def sign(v: float) -> int:
        return (v > 0) - (v < 0)

    opposite = [g for g in groups if g.delta != 0 and sign(g.delta) == -sign(total_delta)]
    pct = NAN if total_a == 0 else total_delta / total_a * 100

    def verb(d: float) -> str:
        return "rose" if d > 0 else "fell" if d < 0 else "did not change"

    if total_delta == 0:
        reading = f"Total {m.label.lower()} did not change between {la} and {lb}."
    else:
        top_share = share(top.delta)
        reading = (
            f"{m.label} {verb(total_delta)} by {fmt(abs(total_delta))} ({format_percent(pct)}) from {la} to {lb}. "
            f"{top.name} {verb(top.delta)} by {fmt(abs(top.delta))}, "
            f"{f'{js_fixed(top_share, 0)}' if finite(top_share) else '—'}% of the change."
        )
        # The biggest amount usually comes from the biggest group. The biggest relative move can be a different story.
        if relative and relative.name != top.name:
            reading += (
                f" Relative to its own size, {relative.name} moved most ({format_percent(relative.pct)})."
            )
        if opposite:
            reading += f" {_list_names([g.name for g in opposite])} moved the other way."

    if percent_view:
        notes = [
            "Percentages compare each group with its own starting value, so small groups can show large swings on small amounts.",
            "Switch to amounts to see which groups make up the total change.",
        ]
    else:
        notes = ["Each group's change adds up to the total change, so the shares sum to 100%."]
        if opposite:
            notes.append("Groups that moved against the total show a negative share of the change.")

    scope = _scope_text(dataset, config, "pair")
    method = f"Added up {m.label.lower()} for each {singular} in {la} and in {lb}"
    method += f" ({scope})" if config.filters else ""
    method += ", then took the difference."
    method += (
        f" Each {singular}'s percentage is its difference divided by its own {la} value."
        if percent_view
        else f" Each {singular}'s share is its difference divided by the total difference."
    )
    top_share = share(top.delta)
    last_figure = (
        {
            "label": "Biggest relative move",
            "value": format_percent(relative.pct),
            "detail": relative.name,
        }
        if percent_view and relative
        else {
            "label": "Biggest driver",
            "value": top.name,
            "detail": f"{f'{js_fixed(top_share, 0)}' if finite(top_share) else '—'}% of the change",
        }
    )
    return {
        "title": title,
        "reading": reading,
        "method": method,
        "notes": notes,
        "visual": {
            "kind": "contribution",
            "points": [{"name": g.name, "value": g.pct if percent_view else g.delta} for g in ordered],
        },
        "format": {"type": "percent", "digits": 1} if percent_view else {"type": "delta", "unit": m.unit},
        "axisFormat": {"type": "percent", "digits": 0} if percent_view else _compact_format(m.unit),
        "table": {
            "columns": [dataset.dimension_label(dim), la, lb, "Change", "Change %", "Share of change"],
            "rows": [
                [
                    g.name,
                    fmt(g.a),
                    fmt(g.b),
                    format_delta(g.delta, m.unit),
                    format_percent(g.pct),
                    f"{js_fixed(share(g.delta), 1)}%" if finite(share(g.delta)) else "—",
                ]
                for g in ordered
            ],
        },
        "sql": build_sql(dataset, config),
        "figures": [
            {"label": la, "value": fmt(total_a)},
            {"label": lb, "value": fmt(total_b)},
            {"label": "Change", "value": format_delta(total_delta, m.unit), "detail": format_percent(pct)},
            last_figure,
        ],
    }


# ------------------------------------------------------------------ entry


def interpret(dataset: AnalysisDataset, config: BuildConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn the grouped query's rows into a result. Pure, so it is easy to test."""
    runners = {"summarise": _summarise, "trend": _trend, "relationship": _relationship, "explain": _explain}
    result = runners.get(config.method, _compare)(dataset, config, rows)
    result["config"] = config.to_dict()
    cleaned: dict[str, Any] = _clean(result)
    return cleaned


def _clean(value: Any) -> Any:
    """JSON has no NaN or infinity: replace them with null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value
