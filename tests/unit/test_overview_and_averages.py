"""The assistant describes a dataset when asked, measurements are averaged, and profiling maps once per value."""

import pandas as pd

from dataforge_backend.domain.analysis.model import MeasureDef
from dataforge_backend.domain.analysis.sql import measure_sql
from dataforge_backend.domain.overview import (
    fallback_narrative,
    overview_facts,
    overview_table,
    schema_prompt,
    wants_overview,
)
from dataforge_backend.domain.semantic import ColumnInfo, derive_fields, is_measurement
from dataforge_backend.domain.values import map_distinct, parse_date, parse_number

PROFILE = {
    "rows": 294,
    "quality": {"completeness": 97, "conformance": 99, "duplicates": 1},
    "columns": [
        {"name": "age", "inferredType": "integer", "min": "28", "max": "66", "mean": "47.83", "nullPct": 0},
        {
            "name": "sex",
            "inferredType": "integer",
            "min": "0",
            "max": "1",
            "mean": "0.72",
            "nullPct": 0,
        },
        {
            "name": "region",
            "inferredType": "string",
            "distinct": 3,
            "topValues": [{"value": "Gauteng", "count": 120}, {"value": "Limpopo", "count": 90}],
            "nullPct": 2.5,
        },
        {"name": "email", "inferredType": "string", "masked": True, "piiKind": "email", "topValues": []},
    ],
}


def test_overview_questions_are_recognised() -> None:
    for question in (
        "What is this dataset about?",
        "Describe the data",
        "Give me an overview",
        "Summarise this dataset",
        "What columns are there?",
        "Tell me about my data",
        "What's in this file?",
    ):
        assert wants_overview(question), question
    for question in ("What is age by sex?", "How many rows are there by region?", "How has fee changed?"):
        assert not wants_overview(question), question


def test_overview_never_shows_personal_values() -> None:
    table = overview_table(PROFILE)
    email = next(row for row in table["rows"] if row[0] == "email")
    assert email[1] == "Personal (masked)"
    assert "masked" in email[2]
    prompt = schema_prompt("Heart attack dataset", "", PROFILE)
    assert "Gauteng (120)" in prompt and "28 to 66" in prompt
    assert "@" not in prompt
    assert "294 rows and 4 columns" in fallback_narrative("Heart attack dataset", PROFILE)
    facts = overview_facts(PROFILE, {"email": "email"})
    assert "97% of cells are filled" in facts and "email" in facts


def test_measurements_are_averaged_and_amounts_added_up() -> None:
    for name in ("age", "trestbps", "chol", "email_open_rate", "session_duration_avg", "membership_years"):
        assert is_measurement(name), name
    for name in ("fee", "amount_paid", "total_purchases", "customer_service_calls", "lifetime_value"):
        assert not is_measurement(name), name
    fields = derive_fields(
        [ColumnInfo("age", "integer", 40), ColumnInfo("fee", "numeric", 30), ColumnInfo("city", "text", 2)],
        100,
        [],
    )
    age, fee = fields.measures["age"], fields.measures["fee"]
    assert age.is_average and age.label == "Average age" and age.columns() == ["age"]
    assert fee.kind == "additive"


def test_an_average_ignores_blank_rows() -> None:
    average = MeasureDef(
        id="age", label="Average age", unit="decimal", kind="ratio", numerator="age", denominator="#"
    )
    assert (
        measure_sql(average, "pooled") == 'AVG("age"::numeric)'
        or measure_sql(average, "pooled") == "AVG(age::numeric)"
    )


def test_map_distinct_matches_map() -> None:
    series = pd.Series(["1", "2", None, "1", "x", float("nan"), "2026-03-14", "1"], dtype=object)
    for fn in (parse_number, parse_date):
        assert map_distinct(series, fn).tolist() == series.map(fn).tolist() or all(
            (a is None and b is None) or a == b or (a != a and b != b)
            for a, b in zip(map_distinct(series, fn).tolist(), series.map(fn).tolist(), strict=True)
        )
    assert parse_date("27.4") is None and parse_date("14.03.2026") is not None
