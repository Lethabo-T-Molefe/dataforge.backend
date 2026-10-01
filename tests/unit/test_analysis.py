"""The analysis engine on the demo fact table, computing grouped rows in pandas as PostgreSQL would."""

from typing import Any

import pandas as pd
import pytest

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
from dataforge_backend.domain.analysis.format import format_compact, format_delta, format_percent, format_value
from dataforge_backend.domain.semantic import ColumnInfo, MetricSpec, build_analysis_dataset, derive_fields
from dataforge_backend.seed import COURSES, MONTHS, REGIONS, fact_rows

METRICS = [
    MetricSpec("completion_rate", "Completion rate", "rate", "ratio", numerator="completions", denominator="enrolments"),
    MetricSpec("enrolments", "Enrolments", "count", "additive", column="enrolments"),
    MetricSpec("sessions_per_learner", "Sessions per learner", "decimal", "ratio", numerator="sessions", denominator="enrolments"),
    MetricSpec("revenue", "Revenue", "money", "additive", column="revenue"),
    MetricSpec("avg_fee", "Average fee", "money", "ratio", numerator="revenue", denominator="enrolments"),
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
    return build_analysis_dataset("facts", "Facts", "silver.facts", derive_fields(columns, 168, METRICS), MONTHS)


def run_query(facts: pd.DataFrame, dataset: AnalysisDataset, config: Any) -> list[dict[str, Any]]:
    """Evaluate the planned grouped query in pandas: filter, group and aggregate like the SQL does."""
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
    rows = []
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
        rows.append(row)
    return rows


def test_fields_come_from_the_semantic_layer(dataset: AnalysisDataset) -> None:
    assert dataset.default_measure == "completion_rate"
    assert dataset.measures["completion_rate"].governed
    assert set(dataset.dimensions) == {"region", "course"}
    assert dataset.grains()[0] == "region-month"
    relationship = default_config(dataset, "relationship")
    assert (relationship.measure, relationship.measure_y) == ("sessions_per_learner", "completion_rate")


def test_compare_pooled_rate_matches_hand_computation(facts: pd.DataFrame, dataset: AnalysisDataset) -> None:
    config = config_from_dict(dataset, {"method": "compare", "measure": "completion_rate", "dimension": "region"})
    result = interpret(dataset, config, run_query(facts, dataset, config))
    gauteng = facts[facts.region == "Gauteng"]
    expected = gauteng.completions.sum() / gauteng.enrolments.sum()
    point = next(p for p in result["visual"]["points"] if p["name"] == "Gauteng")
    assert point["value"] == pytest.approx(expected)
    assert result["figures"][0]["label"] == "Highest"
    assert "SUM(completions)::numeric / NULLIF(SUM(enrolments), 0)" in result["sql"]


def test_explain_contributions_add_up_to_the_total(facts: pd.DataFrame, dataset: AnalysisDataset) -> None:
    config = config_from_dict(dataset, {"method": "explain", "measure": "enrolments", "dimension": "region"})
    result = interpret(dataset, config, run_query(facts, dataset, config))
    by_month = facts.groupby("month").enrolments.sum()
    expected = by_month["2026-03"] - by_month["2026-02"]
    assert sum(p["value"] for p in result["visual"]["points"]) == pytest.approx(expected)
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "explain", "measure": "completion_rate"})


def test_trend_summarise_and_relationship_read_sensibly(facts: pd.DataFrame, dataset: AnalysisDataset) -> None:
    for body in (
        {"method": "trend", "measure": "completion_rate", "split": "course", "change": "mom"},
        {"method": "trend", "measure": "enrolments", "change": "cumulative", "smoothing": "ma3"},
        {"method": "summarise", "measure": "revenue", "grain": "region-course", "outliers": "exclude"},
        {"method": "relationship", "correlation": "spearman"},
    ):
        config = config_from_dict(dataset, body)
        result = interpret(dataset, config, run_query(facts, dataset, config))
        assert result["reading"] and result["sql"].startswith("-- ")
        assert result["table"]["rows"]


def test_filters_are_validated_and_bound_as_parameters(dataset: AnalysisDataset) -> None:
    config = config_from_dict(dataset, {"method": "compare", "region": "Free State", "dimension": "course"})
    query = plan(dataset, config)
    assert ":f0" in query.sql and query.params["f0"] == "Free State"
    assert "region = 'Free State'" in build_sql(dataset, config)
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "compare", "region": "Atlantis"})
    with pytest.raises(AnalysisError):
        config_from_dict(dataset, {"method": "compare", "measure": "revenue; DROP TABLE x"})


def test_formatting_matches_the_frontend() -> None:
    assert format_value(1214.4, "money") == "R1 214"
    assert format_value(0.6634, "rate") == "66.3%"
    assert format_value(48213, "count") == "48 213"
    assert format_delta(-0.031, "rate") == "−3.1 pts"
    assert format_percent(4.21) == "+4.2%"
    assert format_compact(1_250_000, "money") == "R1.3m"
