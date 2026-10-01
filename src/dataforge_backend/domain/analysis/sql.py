"""SQL for analyses.

Two kinds of SQL are produced for every analysis:

* the display SQL, the analytical query shown next to the chart. It is written for PostgreSQL, runs as-is
  in the workbench sandbox and gives the same figures as the chart, so a chart built by point and click is
  also a starting point for code;
* the grouped query, which the engine runs to get one value per group. Values are bound as parameters and
  row-level policies are added to its WHERE clause.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .methods import METHOD_NAMES
from .model import MONTH_DIMENSION, AnalysisDataset, BuildConfig, MeasureDef, grain_dims

_BARE = re.compile(r"^[a-z_][a-z0-9_]*$")
_RESERVED = frozenset(
    """all analyse analyze and any array as asc asymmetric both case cast check collate column constraint create
    current_catalog current_date current_role current_time current_timestamp current_user default deferrable desc
    distinct do else end except false fetch for foreign from grant group having in initially intersect into lateral
    leading limit localtime localtimestamp not null offset on only or order placing primary references returning
    select session_user some symmetric table then to trailing true union unique user using variadic when where
    window with""".split()
)


def ident(name: str) -> str:
    """Quote an identifier only when it needs it."""
    if _BARE.match(name) and name not in _RESERVED:
        return name
    return '"' + name.replace('"', '""') + '"'


def qualified(table: str) -> str:
    schema, _, name = table.partition(".")
    return f"{ident(schema)}.{ident(name)}" if name else ident(schema)


def literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def month_expr(dataset: AnalysisDataset, prefix: str = "") -> str:
    column = prefix + ident(dataset.time_column or "month")
    return column if dataset.time_kind == "month" else f"to_char({column}, 'YYYY-MM')"


# ------------------------------------------------------------------ display SQL


def measure_sql(m: MeasureDef, rate_method: str) -> str:
    if m.kind == "additive":
        return f"SUM({ident(m.column or '')})"
    if m.kind == "count":
        return "COUNT(*)"
    if m.kind == "distinct":
        return f"COUNT(DISTINCT {ident(m.column or '')})"
    num = ident(m.numerator or "")
    if m.denominator == "*":
        return (
            f"AVG({num}::numeric)"
            if rate_method == "average"
            else f"SUM({num})::numeric / NULLIF(COUNT(*), 0)"
        )
    den = ident(m.denominator or "")
    if rate_method == "average":
        return f"AVG({num}::numeric / NULLIF({den}, 0))"
    return f"SUM({num})::numeric / NULLIF(SUM({den}), 0)"


def _filtered_total(m: MeasureDef, condition: str) -> str:
    if m.kind == "count":
        return f"COUNT(*) FILTER (WHERE {condition})"
    return f"COALESCE(SUM({ident(m.column or '')}) FILTER (WHERE {condition}), 0)"


@dataclass
class _Display:
    dataset: AnalysisDataset
    config: BuildConfig
    ctes: list[str] = field(default_factory=list)

    @property
    def needs_base(self) -> bool:
        ds = self.dataset
        return ds.has_time and not (ds.time_kind == "month" and ds.time_column == "month")

    @property
    def source(self) -> str:
        return "base" if self.needs_base else qualified(self.dataset.table)

    def col(self, dim: str) -> str:
        return "month" if dim == MONTH_DIMENSION else ident(self.dataset.dimensions[dim].column)

    def where(self, period: str) -> str:
        c, ds = self.config, self.dataset
        conditions: list[str] = []
        if ds.has_time:
            if period == "range":
                low, high = c.period()
                conditions.append(f"month BETWEEN {literal(low)} AND {literal(high)}")
            else:
                conditions.append(f"month IN ({literal(c.period_a)}, {literal(c.period_b)})")
        for dim, value in c.filters.items():
            conditions.append(f"{self.col(dim)} = {literal(value)}")
        return "WHERE  " + "\n  AND  ".join(conditions) if conditions else ""

    def base_cte(self) -> str | None:
        if not self.needs_base:
            return None
        ds = self.dataset
        columns: list[str] = []
        for d in ds.dimensions.values():
            if d.column != ds.time_column:
                columns.append(ident(d.column))
        for m in ds.measures.values():
            for name in m.columns():
                if ident(name) not in columns and name != ds.time_column:
                    columns.append(ident(name))
        columns.append(f"{month_expr(ds)} AS month")
        return f"base AS (\n  SELECT {', '.join(columns)}\n  FROM   {qualified(ds.table)}\n)"

    def compose(self, final: str) -> str:
        method = METHOD_NAMES.get(self.config.method, self.config.method)
        header = f"-- {method} · {self.dataset.name}\n"
        ctes = [cte for cte in [self.base_cte(), *self.ctes] if cte]
        body = f"WITH {',\n'.join(ctes)}\n{final}" if ctes else final
        return header + body


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def _compare(d: _Display) -> str:
    c, ds = d.config, d.dataset
    m = ds.measures[c.measure]
    dim = d.col(c.dimension)
    order = "1" if c.sort == "name" else f"2 {'ASC' if c.sort == 'asc' else 'DESC'}"
    where = d.where("range")
    if m.additive and c.aggregation != "total" and ds.has_time:
        statistic = (
            f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {m.id})"
            if c.aggregation == "monthly-median"
            else f"AVG({m.id})"
        )
        alias = f"{'median' if c.aggregation == 'monthly-median' else 'avg'}_monthly_{m.id}"
        body = _units_body(d, f"{dim}, month, {measure_sql(m, c.rate_method)} AS {m.id}", f"{dim}, month")
        d.ctes.append(f"monthly AS (\n{body}\n)")
        return d.compose(
            f"SELECT {dim},\n       {statistic} AS {alias}\nFROM   monthly\nGROUP  BY {dim}\nORDER  BY {order};"
        )
    expression = measure_sql(m, c.rate_method)
    extra = ""
    if c.display == "share" and m.additive:
        extra = f",\n       ROUND(100.0 * {expression} / SUM({expression}) OVER (), 1) AS share_pct"
    elif c.display == "index":
        extra = f",\n       ROUND(100.0 * {expression} / AVG({expression}) OVER (), 0) AS index_vs_average"
    lines = [f"SELECT {dim},", f"       {expression} AS {m.id}{extra}", f"FROM   {d.source}"]
    if where:
        lines.append(where)
    lines += [f"GROUP  BY {dim}", f"ORDER  BY {order};"]
    return d.compose("\n".join(lines))


def _units_body(d: _Display, select: str, dims: str) -> str:
    where = d.where("range")
    lines = [f"SELECT {select}", f"FROM   {d.source}"]
    if where:
        lines.append(where)
    lines.append(f"GROUP  BY {dims}")
    return _indent("\n".join(lines))


def _summarise(d: _Display) -> str:
    c, ds = d.config, d.dataset
    m = ds.measures[c.measure]
    dims = ", ".join(d.col(x) for x in grain_dims(c.grain))
    d.ctes.append(
        f"units AS (\n{_units_body(d, f'{dims}, {measure_sql(m, c.rate_method)} AS value', dims)}\n)"
    )
    exclude = c.outliers == "exclude"
    if exclude:
        d.ctes.append(
            "fences AS (\n"
            "  SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS q1,\n"
            "         PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS q3\n"
            "  FROM   units\n)"
        )
    source = (
        "FROM   units CROSS JOIN fences\nWHERE  value BETWEEN q1 - 1.5 * (q3 - q1) AND q3 + 1.5 * (q3 - q1)"
        if exclude
        else "FROM   units"
    )
    return d.compose(
        "SELECT COUNT(*) AS n,\n"
        "       AVG(value) AS mean,\n"
        "       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) AS median,\n"
        "       STDDEV_SAMP(value) AS std_dev,\n"
        "       MIN(value) AS min,\n"
        "       MAX(value) AS max,\n"
        "       PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) AS q1,\n"
        "       PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) AS q3\n"
        f"{source};"
    )


def _num_den(m: MeasureDef) -> tuple[str, str]:
    num = f"SUM({ident(m.numerator or '')})"
    den = "COUNT(*)" if m.denominator == "*" else f"SUM({ident(m.denominator or '')})"
    return num, den


def _trend(d: _Display) -> str:
    c, ds = d.config, d.dataset
    m = ds.measures[c.measure]
    split = d.col(c.split) if c.split != "none" else ""
    group_cols = f"month, {split}" if split else "month"
    partition = f"PARTITION BY {split} " if split else ""
    pooled_running = c.change == "cumulative" and m.is_ratio and c.rate_method == "pooled"
    if pooled_running:
        num, den = _num_den(m)
        select = f"{group_cols},\n       {num} AS numerator,\n       {den} AS denominator"
    else:
        select = f"{group_cols}, {measure_sql(m, c.rate_method)} AS value"
    d.ctes.append(f"monthly AS (\n{_units_body(d, select, group_cols)}\n)")

    if pooled_running:
        transform = (
            "SUM(numerator) OVER (w ROWS UNBOUNDED PRECEDING)::numeric\n"
            "         / NULLIF(SUM(denominator) OVER (w ROWS UNBOUNDED PRECEDING), 0)"
        )
    elif c.change == "mom":
        transform = "100.0 * (value - LAG(value) OVER w) / NULLIF(LAG(value) OVER w, 0)"
    elif c.change == "cumulative":
        # A running total only makes sense for totals; averaged rates get a running mean instead.
        transform = f"{'SUM' if m.additive else 'AVG'}(value) OVER (w ROWS UNBOUNDED PRECEDING)"
    else:
        transform = "value"
    value_name = (
        "change_pct" if c.change == "mom" else f"running_{m.id}" if c.change == "cumulative" else m.id
    )
    order = f"{split}, month" if split else "month"

    if c.smoothing == "none" and c.change == "level":
        return d.compose(f"SELECT {group_cols}, value AS {m.id}\nFROM   monthly\nORDER  BY {order};")
    d.ctes.append(
        f"shaped AS (\n  SELECT {group_cols},\n         {transform} AS {value_name}\n  FROM   monthly\n"
        f"  WINDOW w AS ({partition}ORDER BY month)\n)"
    )
    final_value = (
        f"AVG({value_name}) OVER ({partition}ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS {value_name}_3m_avg"
        if c.smoothing == "ma3"
        else value_name
    )
    return d.compose(f"SELECT {group_cols},\n       {final_value}\nFROM   shaped\nORDER  BY {order};")


def _relationship(d: _Display) -> str:
    c, ds = d.config, d.dataset
    x, y = ds.measures[c.measure], ds.measures[c.measure_y]
    dims = ", ".join(d.col(v) for v in grain_dims(c.grain))
    select = (
        f"{dims},\n       {measure_sql(x, c.rate_method)} AS x,  -- {x.label}\n"
        f"       {measure_sql(y, c.rate_method)} AS y   -- {y.label}"
    )
    d.ctes.append(f"units AS (\n{_units_body(d, select, dims)}\n)")
    exclude = c.outliers == "exclude"
    source = "kept" if exclude else "units"
    if exclude:
        d.ctes.append(
            "fences AS (\n"
            "  SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY x) AS x1,\n"
            "         PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY x) AS x3,\n"
            "         PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y) AS y1,\n"
            "         PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y) AS y3\n"
            "  FROM   units\n)"
        )
        d.ctes.append(
            "kept AS (\n  SELECT units.*\n  FROM   units CROSS JOIN fences\n"
            "  WHERE  x BETWEEN x1 - 1.5 * (x3 - x1) AND x3 + 1.5 * (x3 - x1)\n"
            "    AND  y BETWEEN y1 - 1.5 * (y3 - y1) AND y3 + 1.5 * (y3 - y1)\n)"
        )
    if c.correlation == "spearman":
        d.ctes.append(
            "ranked AS (\n  -- Average ranks, so tied values share their rank\n"
            "  SELECT (RANK() OVER (ORDER BY x) + COUNT(*) OVER () + 1 - RANK() OVER (ORDER BY x DESC)) / 2.0 AS rx,\n"
            "         (RANK() OVER (ORDER BY y) + COUNT(*) OVER () + 1 - RANK() OVER (ORDER BY y DESC)) / 2.0 AS ry\n"
            f"  FROM   {source}\n)"
        )
        return d.compose("SELECT CORR(rx, ry) AS spearman_rho,\n       COUNT(*) AS n\nFROM   ranked;")
    return d.compose(
        "SELECT CORR(x, y) AS pearson_r,\n       REGR_SLOPE(y, x) AS slope,\n"
        f"       REGR_INTERCEPT(y, x) AS intercept,\n       COUNT(*) AS n\nFROM   {source};"
    )


def _explain(d: _Display) -> str:
    c, ds = d.config, d.dataset
    m = ds.measures[c.measure]
    dim = d.col(c.dimension)
    a = _filtered_total(m, f"month = {literal(c.period_a)}")
    b = _filtered_total(m, f"month = {literal(c.period_b)}")
    where = d.where("pair")
    lines = [f"SELECT {dim},", f"       {a} AS period_a,", f"       {b} AS period_b", f"FROM   {d.source}"]
    if where:
        lines.append(where)
    lines.append(f"GROUP  BY {dim}")
    d.ctes.append(f"periods AS (\n{_indent(chr(10).join(lines))}\n)")
    order = (
        "ABS((period_b - period_a)::numeric / NULLIF(period_a, 0))"
        if c.change_view == "percent"
        else "ABS(period_b - period_a)"
    )
    return d.compose(
        f"SELECT {dim},\n       period_a,\n       period_b,\n       period_b - period_a AS change,\n"
        "       ROUND(100.0 * (period_b - period_a) / NULLIF(period_a, 0), 1) AS change_pct,\n"
        "       ROUND(100.0 * (period_b - period_a) / NULLIF(SUM(period_b - period_a) OVER (), 0), 1) AS share_of_change_pct\n"
        f"FROM   periods\nORDER  BY {order} DESC;"
    )


def build_sql(dataset: AnalysisDataset, config: BuildConfig) -> str:
    display = _Display(dataset, config)
    builders = {"summarise": _summarise, "trend": _trend, "relationship": _relationship, "explain": _explain}
    return builders.get(config.method, _compare)(display)


# ------------------------------------------------------------------ grouped query (executed)


@dataclass(frozen=True)
class GroupQuery:
    sql: str
    params: dict[str, Any]
    dims: list[str]


def _value_expr(m: MeasureDef, rate_method: str) -> str:
    t = "t."
    if m.kind == "additive":
        return f"SUM({t}{ident(m.column or '')})::float8"
    if m.kind == "count":
        return "COUNT(*)::float8"
    if m.kind == "distinct":
        return f"COUNT(DISTINCT {t}{ident(m.column or '')})::float8"
    num = f"{t}{ident(m.numerator or '')}"
    if m.denominator == "*":
        return (
            f"AVG({num}::float8)" if rate_method == "average" else f"SUM({num})::float8 / NULLIF(COUNT(*), 0)"
        )
    den = f"{t}{ident(m.denominator or '')}"
    if rate_method == "average":
        return f"AVG(CASE WHEN {den} > 0 THEN {num}::float8 / {den} END)"
    return f"SUM({num})::float8 / NULLIF(SUM({den})::float8, 0)"


def _parts_exprs(m: MeasureDef) -> tuple[str, str]:
    """Numerator and denominator totals of a ratio, or the total twice for anything else."""
    if m.is_ratio:
        num = f"SUM(t.{ident(m.numerator or '')})::float8"
        den = "COUNT(*)::float8" if m.denominator == "*" else f"SUM(t.{ident(m.denominator or '')})::float8"
        return num, den
    value = _value_expr(m, "pooled")
    return value, value


def group_query(
    dataset: AnalysisDataset,
    config: BuildConfig,
    dims: list[str],
    values: dict[str, tuple[MeasureDef, str]],
    *,
    period: str = "range",
    parts: dict[str, MeasureDef] | None = None,
    policy: str | None = None,
) -> GroupQuery:
    """One row per combination of `dims`, with each requested value.

    `values` maps an output alias to (measure, rate method). `parts` adds the numerator and denominator
    totals of a measure as `<alias>_num` and `<alias>_den`, used for running pooled rates and group sizes.
    `policy` is a row-level security predicate that has already been validated.
    """
    params: dict[str, Any] = {}
    select: list[str] = []
    for i, dim in enumerate(dims):
        expr = (
            month_expr(dataset, "t.")
            if dim == MONTH_DIMENSION
            else f"t.{ident(dataset.dimensions[dim].column)}"
        )
        select.append(f"{expr}::text AS d{i}")
    for alias, (measure, rate) in values.items():
        select.append(f"{_value_expr(measure, rate)} AS {alias}")
    for alias, measure in (parts or {}).items():
        num, den = _parts_exprs(measure)
        select += [f"{num} AS {alias}_num", f"{den} AS {alias}_den"]

    conditions: list[str] = []
    if dataset.has_time:
        month = month_expr(dataset, "t.")
        if period == "range":
            low, high = config.period()
            conditions.append(f"{month} BETWEEN :p_from AND :p_to")
            params.update(p_from=low, p_to=high)
        else:
            conditions.append(f"{month} IN (:p_a, :p_b)")
            params.update(p_a=config.period_a, p_b=config.period_b)
    for i, (dim, value) in enumerate(config.filters.items()):
        conditions.append(f"t.{ident(dataset.dimensions[dim].column)}::text = :f{i}")
        params[f"f{i}"] = value
    for dim in dims:
        if dim != MONTH_DIMENSION:
            conditions.append(f"t.{ident(dataset.dimensions[dim].column)} IS NOT NULL")
    if policy:
        conditions.append(f"({policy})")
    where = f"\nWHERE {' AND '.join(conditions)}" if conditions else ""
    group = f"\nGROUP BY {', '.join(str(i + 1) for i in range(len(dims)))}" if dims else ""
    sql = f"SELECT {', '.join(select)}\nFROM {qualified(dataset.table)} AS t{where}{group}"
    return GroupQuery(sql, params, dims)
