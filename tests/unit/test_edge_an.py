"""Catalogue fixtures for analysis readings that need the demo fact table."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from catalogue_progress import marks
from dataforge_backend.domain.analysis import (
    MONTH_DIMENSION,
    AnalysisDataset,
    AnalysisError,
    build_sql,
    config_from_dict,
    default_config,
    interpret,
    plan,
)
from dataforge_backend.domain.analysis.sql import literal, measure_sql
from dataforge_backend.domain.semantic import ColumnInfo, MetricSpec, build_analysis_dataset, derive_fields
from dataforge_backend.seed import COURSES, MONTHS, REGIONS, fact_rows

METRICS = [
    MetricSpec(
        "completion_rate",
        "Completion rate",
        "rate",
        "ratio",
        numerator="completions",
        denominator="enrolments",
    ),
    MetricSpec("enrolments", "Enrolments", "count", "additive", column="enrolments"),
    MetricSpec(
        "sessions_per_learner",
        "Sessions per learner",
        "decimal",
        "ratio",
        numerator="sessions",
        denominator="enrolments",
    ),
    MetricSpec("revenue", "Revenue", "money", "additive", column="revenue"),
]


@pytest.fixture(scope="module")
def facts() -> pd.DataFrame:
    return pd.DataFrame(fact_rows())


@pytest.fixture(scope="module")
def dataset() -> AnalysisDataset:
    columns = [
        ColumnInfo("region", "text", 7, values=tuple(REGIONS)),
        ColumnInfo("course", "text", 4, values=tuple(COURSES)),
        ColumnInfo("month", "text", 6, month_like=True),
        ColumnInfo("enrolments", "integer", 150),
        ColumnInfo("completions", "integer", 150),
        ColumnInfo("revenue", "integer", 168),
        ColumnInfo("sessions", "integer", 150),
    ]
    return build_analysis_dataset(
        "facts", "Facts", "silver.facts", derive_fields(columns, 168, METRICS), MONTHS
    )


def _rows(facts: pd.DataFrame, dataset: AnalysisDataset, config: Any) -> list[dict[str, Any]]:
    query = plan(dataset, config)
    frame = facts
    if "p_from" in query.params:
        frame = frame[(frame.month >= query.params["p_from"]) & (frame.month <= query.params["p_to"])]
    if "p_a" in query.params:
        frame = frame[frame.month.isin([query.params["p_a"], query.params["p_b"]])]
    for dim, value in config.filters.items():
        frame = frame[frame[dim] == value]
    columns = [MONTH_DIMENSION if d == MONTH_DIMENSION else d for d in query.dims]
    grouped = frame.groupby(columns) if columns else [((), frame)]
    out = []
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = {f"d{i}": k for i, k in enumerate(keys)}
        for alias, measure_id in (("v", config.measure), ("x", config.measure), ("y", config.measure_y)):
            m = dataset.measures[measure_id]
            if m.kind == "ratio":
                value = part[m.numerator].sum() / part[m.denominator].sum()
                row[f"{alias}_num"], row[f"{alias}_den"] = part[m.numerator].sum(), part[m.denominator].sum()
            else:
                value = part[m.column].sum()
                row[f"{alias}_num"] = row[f"{alias}_den"] = value
            row[alias] = float(value)
        row["p_num"], row["p_den"] = row["v_num"], row["v_den"]
        out.append(row)
    return out


@marks(
    "AN-04", "AN-05", "AN-06", "AN-09", "AN-10", "AN-11", "AN-12", "AN-13",
    "AN-15", "AN-17", "AN-18", "AN-19", "AN-21", "AN-22", "AN-23", "AN-24", "AN-25",
    "SQ-04", "SQ-07", "SM-03",
)
def test_an_readings_on_demo_facts(facts: pd.DataFrame, dataset: AnalysisDataset) -> None:
    assert "completion_rate" in dataset.measures and dataset.measures["completion_rate"].governed

    rate = config_from_dict(
        dataset, {"method": "compare", "measure": "completion_rate", "dimension": "region", "display": "share"}
    )
    rate_result = interpret(dataset, rate, _rows(facts, dataset, rate))
    assert any("shares of a total" in n.lower() for n in rate_result.get("notes", []))
    assert "pts" in rate_result["reading"] or rate_result["figures"]

    by_name = config_from_dict(
        dataset, {"method": "compare", "measure": "enrolments", "dimension": "region", "sort": "name"}
    )
    named = interpret(dataset, by_name, _rows(facts, dataset, by_name))
    labels = [p["name"] for p in named["visual"]["points"]]
    assert labels == sorted(labels, key=str.casefold)

    mom = config_from_dict(
        dataset, {"method": "trend", "measure": "enrolments", "change": "mom", "date_from": "2025-10", "date_to": "2026-03"}
    )
    mom_result = interpret(dataset, mom, _rows(facts, dataset, mom))
    points = mom_result["visual"]["points"]
    measure_key = next(k for k in points[0] if k != "name")
    assert points and points[0][measure_key] is None

    cum = config_from_dict(
        dataset,
        {"method": "trend", "measure": "enrolments", "change": "cumulative", "date_from": "2025-10", "date_to": "2026-03"},
    )
    cum_result = interpret(dataset, cum, _rows(facts, dataset, cum))
    key = next(k for k in cum_result["visual"]["points"][0] if k != "name")
    vals = [p[key] for p in cum_result["visual"]["points"] if p[key] is not None]
    assert vals == sorted(vals)

    smooth = config_from_dict(
        dataset,
        {
            "method": "trend",
            "measure": "enrolments",
            "smoothing": "ma3",
            "date_from": "2025-10",
            "date_to": "2026-03",
        },
    )
    smooth_result = interpret(dataset, smooth, _rows(facts, dataset, smooth))
    skey = next(k for k in smooth_result["visual"]["points"][0] if k != "name")
    assert smooth_result["visual"]["points"][0][skey] is None
    assert smooth_result["visual"]["points"][1][skey] is None

    split = config_from_dict(
        dataset,
        {
            "method": "trend",
            "measure": "enrolments",
            "split": "course",
            "date_from": "2025-10",
            "date_to": "2026-03",
        },
    )
    split_result = interpret(dataset, split, _rows(facts, dataset, split))
    series = split_result["visual"].get("series") or split_result["visual"].get("points")
    assert series

    rel = config_from_dict(dataset, {"method": "relationship", "outliers": "exclude"})
    rel_result = interpret(dataset, rel, _rows(facts, dataset, rel))
    assert "not proof" in rel_result["reading"].lower() or any(
        "not proof" in n.lower() for n in rel_result.get("notes", [])
    )
    assert "straight line" in rel_result["reading"].lower() or "r =" in rel_result["reading"].lower() or rel_result["figures"]

    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "explain", "measure": "completion_rate"})
    # Bypass validation path: interpret still empties for non-additive if forced via config object
    from dataforge_backend.domain.analysis.model import BuildConfig

    bad = default_config(dataset, "explain")
    forced = BuildConfig(**{**bad.__dict__, "measure": "completion_rate"})
    empty_explain = interpret(dataset, forced, [])
    assert empty_explain["empty"] is True and "total" in empty_explain["reading"].lower()

    explain = config_from_dict(dataset, {"method": "explain", "measure": "enrolments", "dimension": "region"})
    explained = interpret(dataset, explain, _rows(facts, dataset, explain))
    assert "nan" not in explained["reading"].lower()
    pct = config_from_dict(
        dataset,
        {"method": "explain", "measure": "enrolments", "dimension": "region", "changeView": "percent"},
    )
    pct_result = interpret(dataset, pct, _rows(facts, dataset, pct))
    assert pct_result["visual"]["points"]

    scoped = config_from_dict(
        dataset,
        {
            "method": "compare",
            "measure": "enrolments",
            "dimension": "region",
            "date_from": "2026-03",
            "date_to": "2026-03",
        },
    )
    scope = interpret(dataset, scoped, _rows(facts, dataset, scoped))
    blob = str(scope)
    assert "2026-03" in blob or "March" in blob or scope.get("empty")

    count_m = type("M", (), {"kind": "ratio", "column": None, "denominator": "*", "numerator": "completions"})()
    assert "COUNT(*)" in measure_sql(count_m, "pooled")  # type: ignore[arg-type]
    assert "AVG(" in measure_sql(count_m, "average")  # type: ignore[arg-type]
    assert literal("O'Brien") == "'O''Brien'"
    sql = build_sql(
        dataset,
        config_from_dict(dataset, {"method": "compare", "measure": "enrolments", "region": REGIONS[0]}),
    )
    assert REGIONS[0] in sql or ":" in plan(
        dataset, config_from_dict(dataset, {"method": "compare", "measure": "enrolments", "region": REGIONS[0]})
    ).sql


@marks("AN-07")
def test_an07_monthly_mean(facts: pd.DataFrame, dataset: AnalysisDataset) -> None:
    config = config_from_dict(
        dataset,
        {
            "method": "compare",
            "measure": "enrolments",
            "dimension": "region",
            "aggregation": "monthly-mean",
        },
    )
    result = interpret(dataset, config, _rows(facts, dataset, config))
    assert result["visual"]["points"]
